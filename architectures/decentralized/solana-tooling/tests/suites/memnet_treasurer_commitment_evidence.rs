//! A conviction has to name a payload the target signed
//! (wienerlabs/leviathan-net#43).
//!
//! The chain cannot recompute a gradient, so whether a payload was honest
//! stays the verifiers' claim. What the target committed to does not: the
//! treasurer rebuilds the commitment hash from the step, the nonce, the batch
//! and the payload's results digest. The target's p2p key, read from the
//! roster, has to have signed that hash, through an ed25519 precompile
//! instruction in the same transaction. Every conviction here runs on the
//! real programs, signed with real keys, over a real payload dump.

use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::RunState;
use psyche_core::payload_commitment_run;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use psyche_solana_tooling::process_treasurer_instructions::run_slash_instruction;
use psyche_solana_tooling::process_treasurer_instructions::run_submit_audit_verdict_instruction;
use psyche_solana_tooling::signed_commitment;
use psyche_solana_tooling::signed_commitment::ed25519_signature_instruction;
use psyche_solana_tooling::signed_commitment::PayloadHeader;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::state::commitment_run;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::CommitmentEvidence;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::instruction::Instruction;
use solana_sdk::instruction::InstructionError;
use solana_sdk::signer::Signer;
use solana_sdk::transaction::TransactionError;

use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::NODES;
use super::bonded_run::SLASHING_RATE;
use super::bonded_run::SUBJECT;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::payload_digest;
use super::signed_payload::replayed_digest;
use super::signed_payload::HONEST_PAYLOAD;

/// A node of the run other than the subject.
const OTHER: usize = 0;

async fn training_run(index: u64, run_id: &str) -> Harness {
    let mut harness = Harness::new(index, run_id, JoinGate::Bond).await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness
}

/// The run authority's slash of `node`, sent with `proofs` alongside it.
async fn slash_with(
    harness: &mut Harness,
    node: usize,
    commitment: &SignedCommitment,
    proofs: Vec<Instruction>,
) -> anyhow::Result<()> {
    let index = harness.roster_index(node).await;
    let slash = run_slash_instruction(
        &harness.payer.pubkey(),
        &harness.main_authority.pubkey(),
        &harness.run,
        &harness.coordinator_account,
        &harness.run_id,
        &harness.nodes[node].wallet.pubkey(),
        None,
        index,
        commitment.batch_start,
        commitment.batch_end,
        commitment.evidence,
        replayed_digest(),
    );
    let mut instructions = proofs;
    instructions.push(slash);
    harness
        .endpoint
        .process_instructions_with_signers(
            &harness.payer,
            &instructions,
            &[&harness.main_authority],
        )
        .await
        .map(|_| ())
}

#[tokio::test]
pub async fn a_genuine_commitment_convicts() {
    let mut harness = training_run(430, "Leviathan genuine evidence").await;
    harness.convict(SUBJECT).await;
    harness.finish_epoch().await;
    assert_eq!(harness.ledger(SUBJECT).await.1, SLASHING_RATE);
}

#[tokio::test]
pub async fn a_slash_without_the_signature_proof_is_refused() {
    let mut harness = training_run(431, "Leviathan unproven evidence").await;
    let commitment = harness.signed_payload(SUBJECT).await;
    assert_eq!(
        refusal_code(
            slash_with(&mut harness, SUBJECT, &commitment, vec![]).await
        ),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
}

/// Another node's genuine signature over the subject's commitment is not the
/// subject committing to it.
#[tokio::test]
pub async fn a_commitment_signed_by_another_key_is_refused() {
    let mut harness = training_run(432, "Leviathan foreign evidence").await;
    let genuine = harness.signed_payload(SUBJECT).await;
    let foreign = SignedCommitment::sign(
        &harness.nodes[OTHER].p2p,
        genuine.run,
        genuine.header(),
        genuine.evidence.results_digest,
    );
    assert_eq!(
        refusal_code(
            slash_with(
                &mut harness,
                SUBJECT,
                &foreign,
                vec![foreign.proof_instruction()]
            )
            .await
        ),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
    // The same commitment, signed by the subject, convicts.
    harness.try_convict_on(SUBJECT, &genuine).await.unwrap();
}

/// A proof naming the subject's key with a signature the subject did not make
/// fails in the runtime, before the treasurer ever runs.
#[tokio::test]
pub async fn a_forged_signature_naming_the_target_is_refused() {
    let mut harness = training_run(433, "Leviathan forged evidence").await;
    let genuine = harness.signed_payload(SUBJECT).await;
    let forged_signature: [u8; 64] = harness.nodes[OTHER]
        .p2p
        .sign_message(&genuine.committed_hash())
        .into();
    let forged = SignedCommitment {
        signature: forged_signature,
        ..genuine
    };
    let error = slash_with(
        &mut harness,
        SUBJECT,
        &forged,
        vec![forged.proof_instruction()],
    )
    .await
    .expect_err("a forged signature was accepted");
    assert!(
        matches!(
            error.downcast_ref::<TransactionError>(),
            Some(TransactionError::InstructionError(
                0,
                InstructionError::Custom(_)
            ))
        ),
        "expected the precompile at instruction 0 to refuse, got: {error:#}"
    );
    assert_eq!(harness.ledger(SUBJECT).await.1, 0);
}

/// The hash is rebuilt from the batch, step, nonce and digest the conviction
/// names, so a genuine signature over one payload does not convict on any
/// other.
#[tokio::test]
pub async fn a_signature_over_another_payload_is_refused() {
    let mut harness = training_run(434, "Leviathan moved evidence").await;
    let genuine = harness.signed_payload(SUBJECT).await;
    let moved = [
        SignedCommitment {
            batch_end: genuine.batch_end + 1,
            ..genuine
        },
        SignedCommitment {
            batch_start: genuine.batch_start + 1,
            ..genuine
        },
        SignedCommitment {
            evidence: CommitmentEvidence {
                trainer_nonce: genuine.evidence.trainer_nonce + 1,
                ..genuine.evidence
            },
            ..genuine
        },
        SignedCommitment {
            evidence: CommitmentEvidence {
                results_digest: payload_digest(HONEST_PAYLOAD),
                ..genuine.evidence
            },
            ..genuine
        },
    ];
    for claimed in moved {
        assert_eq!(
            refusal_code(
                slash_with(
                    &mut harness,
                    SUBJECT,
                    &claimed,
                    vec![genuine.proof_instruction()]
                )
                .await
            ),
            treasurer_refusal(ProgramError::CommitmentNotSigned),
            "a conviction on {claimed:?} was not refused"
        );
    }
}

/// A payload commitment is bound to the run it was made in, so the subject's
/// genuine signature over the same work in another run convicts it of
/// nothing here.
#[tokio::test]
pub async fn a_commitment_made_in_another_run_is_refused() {
    let mut harness = training_run(443, "Leviathan other run evidence").await;
    let genuine = harness.signed_payload(SUBJECT).await;
    let elsewhere = SignedCommitment::sign(
        &harness.nodes[SUBJECT].p2p,
        payload_commitment_run("Leviathan another run"),
        genuine.header(),
        genuine.evidence.results_digest,
    );
    assert_ne!(elsewhere.run, genuine.run);
    assert_eq!(
        refusal_code(
            slash_with(
                &mut harness,
                SUBJECT,
                &elsewhere,
                vec![elsewhere.proof_instruction()]
            )
            .await
        ),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
    assert_eq!(
        genuine.run,
        payload_commitment_run(&harness.run_id),
        "the run a commitment is bound to is the coordinator's run id"
    );
}

/// Where the proof sits in the transaction does not matter, only that it is
/// there.
#[tokio::test]
pub async fn the_proof_may_follow_the_conviction() {
    let mut harness = training_run(435, "Leviathan trailing evidence").await;
    let commitment = harness.signed_payload(SUBJECT).await;
    let index = harness.roster_index(SUBJECT).await;
    let slash = run_slash_instruction(
        &harness.payer.pubkey(),
        &harness.main_authority.pubkey(),
        &harness.run,
        &harness.coordinator_account,
        &harness.run_id,
        &harness.nodes[SUBJECT].wallet.pubkey(),
        None,
        index,
        commitment.batch_start,
        commitment.batch_end,
        commitment.evidence,
        replayed_digest(),
    );
    harness
        .endpoint
        .process_instructions_with_signers(
            &harness.payer,
            &[slash, commitment.proof_instruction()],
            &[&harness.main_authority],
        )
        .await
        .unwrap();
    harness.finish_epoch().await;
    assert_eq!(harness.ledger(SUBJECT).await.1, SLASHING_RATE);
}

/// A proof whose signature covers more than the commitment hash is a
/// signature over something else.
#[tokio::test]
pub async fn a_signature_over_a_longer_message_is_refused() {
    let mut harness = training_run(436, "Leviathan padded evidence").await;
    let commitment = harness.signed_payload(SUBJECT).await;
    let mut longer = commitment.committed_hash().to_vec();
    longer.extend_from_slice(&commitment.committed_hash());
    let signature: [u8; 64] =
        harness.nodes[SUBJECT].p2p.sign_message(&longer).into();
    let proof =
        ed25519_signature_instruction(&commitment.signer, &signature, &longer);
    assert_eq!(
        refusal_code(
            slash_with(&mut harness, SUBJECT, &commitment, vec![proof]).await
        ),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
}

/// A conviction is for work in the epoch the coordinator is in. A commitment
/// the subject signed in an earlier epoch names work that epoch already
/// charged, and one for a step not yet reached names work nobody has done.
#[tokio::test]
pub async fn a_commitment_outside_the_current_epoch_is_refused() {
    let mut harness = training_run(437, "Leviathan stale evidence").await;
    let earlier = harness.signed_payload(SUBJECT).await;
    harness.finish_epoch().await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    assert_eq!(
        refusal_code(harness.try_convict_on(SUBJECT, &earlier).await),
        treasurer_refusal(ProgramError::CommitmentOutsideEpoch),
    );

    let current = harness.signed_payload(SUBJECT).await;
    let ahead = SignedCommitment::sign(
        &harness.nodes[SUBJECT].p2p,
        current.run,
        PayloadHeader {
            step: current.evidence.step + 1,
            trainer_nonce: current.evidence.trainer_nonce + 1,
            ..current.header()
        },
        current.evidence.results_digest,
    );
    assert_eq!(
        refusal_code(harness.try_convict_on(SUBJECT, &ahead).await),
        treasurer_refusal(ProgramError::CommitmentOutsideEpoch),
    );

    harness.try_convict_on(SUBJECT, &current).await.unwrap();
}

/// A run restarted from an earlier step trains that step again, in a new
/// epoch, and its nodes sign it again. A commitment signed the first time
/// names a round that is over, though its step is back inside the epoch, and
/// convicts nobody now. The same work signed in this epoch does
/// (wienerlabs/leviathan-net#67).
#[tokio::test]
pub async fn a_commitment_from_before_a_restart_is_refused() {
    let mut harness = training_run(443, "Leviathan restarted evidence").await;
    let before = harness.signed_payload(SUBJECT).await;
    let step = before.evidence.step;
    harness.finish_epoch().await;
    assert!(harness.state().await.coordinator.progress.step > step);

    harness.restart_from_step(step).await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    let coordinator = harness.state().await.coordinator;
    assert_eq!(coordinator.progress.step, step);
    assert_eq!(coordinator.epoch_state.start_step, step);

    assert_eq!(
        refusal_code(harness.try_convict_on(SUBJECT, &before).await),
        treasurer_refusal(ProgramError::CommitmentOutsideEpoch),
    );
    let again = harness.signed_payload(SUBJECT).await;
    assert_eq!(again.evidence.step, step);
    harness.try_convict_on(SUBJECT, &again).await.unwrap();
}

/// Batches are assigned by the round's committee, and in a warmup no round
/// has been drawn over the roster, so there is no payload to commit to.
#[tokio::test]
pub async fn no_payload_is_assigned_in_a_warmup() {
    let mut harness =
        Harness::new(442, "Leviathan warmup payload", JoinGate::Bond).await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::Warmup).await;
    let coordinator = harness.state().await.coordinator;
    let identity = harness.nodes[SUBJECT].identity();
    assert!(
        signed_commitment::payload_header(&coordinator, &identity).is_err()
    );
    harness.advance_to(RunState::RoundTrain).await;
    let coordinator = harness.state().await.coordinator;
    assert!(
        signed_commitment::payload_header(&coordinator, &identity).is_ok()
    );
}

/// Six nodes with half the free seats verifying, so each round draws a
/// committee of three and a quorum of two.
async fn committee_run(index: u64, run_id: &str) -> Harness {
    const COMMITTEE_NODES: usize = 6;
    let mut harness = Harness::with(
        index,
        run_id,
        JoinGate::Bond,
        COMMITTEE_NODES,
        |config: &mut CoordinatorConfig| config.verification_percent = 50,
    )
    .await;
    for node in 0..COMMITTEE_NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness
}

/// The verifiers and the trainers drawn for the round under way.
async fn committee(harness: &mut Harness) -> (Vec<usize>, Vec<usize>) {
    let coordinator = harness.state().await.coordinator;
    let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
    let (mut verifiers, mut trainers) = (vec![], vec![]);
    for node in harness.roster().await {
        let index = harness.roster_index(node).await;
        match selection.get_committee(index).committee {
            Committee::Verifier => verifiers.push(node),
            Committee::Trainer => trainers.push(node),
            Committee::TieBreaker => {},
        }
    }
    (verifiers, trainers)
}

async fn vote(
    harness: &mut Harness,
    verifier: usize,
    target: usize,
    commitment: &SignedCommitment,
) -> anyhow::Result<()> {
    vote_replayed(harness, verifier, target, commitment, replayed_digest())
        .await
}

/// `vote`, with the verifier's own replay of the payload hashing to
/// `replayed_hash`.
async fn vote_replayed(
    harness: &mut Harness,
    verifier: usize,
    target: usize,
    commitment: &SignedCommitment,
    replayed_hash: [u8; 32],
) -> anyhow::Result<()> {
    let target_index = harness.roster_index(target).await;
    let target_key = harness.nodes[target].wallet.pubkey();
    process_treasurer_run_submit_audit_verdict(
        &mut harness.endpoint,
        &harness.payer,
        &harness.nodes[verifier].wallet,
        &harness.run,
        &harness.coordinator_account,
        &harness.run_id,
        &target_key,
        target_index,
        commitment,
        replayed_hash,
    )
    .await
}

/// Runs the run on into the next round's training, every node rejoining
/// when the epoch ends first.
async fn next_round(harness: &mut Harness, nodes: usize) {
    let at = |state: &CoordinatorInstanceState| {
        let coordinator = &state.coordinator;
        (
            coordinator.progress.epoch,
            coordinator.current_round().map(|round| round.height),
        )
    };
    let start = at(&harness.state().await);
    let mut rejoined = None;
    for _ in 0..200 {
        harness.step().await;
        let state = harness.state().await;
        match state.coordinator.run_state {
            RunState::RoundTrain if at(&state) != start => return,
            RunState::WaitingForMembers
                if rejoined != Some(state.coordinator.progress.epoch) =>
            {
                rejoined = Some(state.coordinator.progress.epoch);
                for node in 0..nodes {
                    harness.join(node).await.unwrap();
                }
            },
            _ => {},
        }
    }
    panic!("the run never reached another round");
}

async fn verdict_of(harness: &mut Harness, target: usize) -> AuditVerdict {
    let key = find_audit_verdict(
        &harness.run,
        &harness.nodes[target].wallet.pubkey(),
    );
    let account = harness.endpoint.get_account(&key).await.unwrap().unwrap();
    AuditVerdict::from_pda_account(&account.owner, &account.data)
        .unwrap()
        .unwrap()
}

/// A quorum convicts on the target's signed commitment, and the verdict
/// records exactly what was signed, so anyone can rebuild the hash from the
/// account and its run.
#[tokio::test]
pub async fn a_quorum_convicts_on_the_signed_commitment() {
    let mut harness = committee_run(438, "Leviathan committee evidence").await;
    let (verifiers, trainers) = committee(&mut harness).await;
    assert!(verifiers.len() >= 2, "the round drew {verifiers:?}");
    let target = trainers[0];
    let commitment = harness.signed_payload(target).await;

    for verifier in &verifiers[..2] {
        vote(&mut harness, *verifier, target, &commitment)
            .await
            .unwrap();
    }
    let verdict = verdict_of(&mut harness, target).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.evidence, commitment.evidence);
    assert_eq!(verdict.committed_hash, commitment.committed_hash());
    // The account and the run its coordinator trains are all it takes to
    // rebuild the hash.
    let run = commitment_run(&harness.state().await.coordinator);
    assert_eq!(
        verdict.committed_hash,
        verdict.evidence.committed_hash(
            &run,
            verdict.batch_start,
            verdict.batch_end
        )
    );
    assert_eq!(
        (verdict.batch_start, verdict.batch_end),
        (commitment.batch_start, commitment.batch_end)
    );

    harness.finish_epoch().await;
    assert_eq!(harness.ledger(target).await.1, SLASHING_RATE);
}

#[tokio::test]
pub async fn a_vote_without_the_signature_proof_is_refused() {
    let mut harness = committee_run(439, "Leviathan unproven vote").await;
    let (verifiers, trainers) = committee(&mut harness).await;
    let target = trainers[0];
    let commitment = harness.signed_payload(target).await;
    let target_index = harness.roster_index(target).await;
    let verifier = &harness.nodes[verifiers[0]].wallet;
    let ballot = run_submit_audit_verdict_instruction(
        &harness.payer.pubkey(),
        &verifier.pubkey(),
        &harness.run,
        &harness.coordinator_account,
        &harness.run_id,
        &harness.nodes[target].wallet.pubkey(),
        target_index,
        commitment.batch_start,
        commitment.batch_end,
        commitment.evidence,
        replayed_digest(),
    );
    let result = harness
        .endpoint
        .process_instructions_with_signers(
            &harness.payer,
            &[ballot],
            &[verifier],
        )
        .await;
    assert_eq!(
        refusal_code(result),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
}

#[tokio::test]
pub async fn a_vote_on_a_commitment_signed_by_another_key_is_refused() {
    let mut harness = committee_run(440, "Leviathan foreign vote").await;
    let (verifiers, trainers) = committee(&mut harness).await;
    let (target, bystander) = (trainers[0], trainers[1]);
    let genuine = harness.signed_payload(target).await;
    let foreign = SignedCommitment::sign(
        &harness.nodes[bystander].p2p,
        genuine.run,
        genuine.header(),
        genuine.evidence.results_digest,
    );
    assert_eq!(
        refusal_code(vote(&mut harness, verifiers[0], target, &foreign).await),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
}

/// Votes are counted per payload the target signed. A vote citing another
/// payload it really signed, broadcast like every payload, is counted apart:
/// cast first, it no longer fixes the evidence every later vote has to match,
/// so one seat cannot veto what the rest of the committee agrees on. The
/// verdict records the payload the quorum is for.
#[tokio::test]
pub async fn a_vote_on_other_signed_evidence_cannot_veto_the_quorum() {
    let mut harness = committee_run(441, "Leviathan split evidence").await;
    let (verifiers, trainers) = committee(&mut harness).await;
    assert_eq!(verifiers.len(), 3, "a committee of three, a quorum of two");
    let target = trainers[0];
    let first = harness.signed_payload(target).await;
    let other = SignedCommitment::sign(
        &harness.nodes[target].p2p,
        first.run,
        first.header(),
        payload_digest(HONEST_PAYLOAD),
    );
    vote(&mut harness, verifiers[0], target, &other)
        .await
        .unwrap();
    vote(&mut harness, verifiers[1], target, &first)
        .await
        .unwrap();
    assert_eq!(
        verdict_of(&mut harness, target).await.status,
        VerdictStatus::Voting,
        "one vote for each is no quorum"
    );
    vote(&mut harness, verifiers[2], target, &first)
        .await
        .unwrap();

    let verdict = verdict_of(&mut harness, target).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.committed_hash, first.committed_hash());
    assert_eq!(verdict.evidence, first.evidence);
    assert_eq!(verdict.verdict_count, 2);
    assert_eq!(verdict.voters.len(), 3);
    assert!(verdict.dissents(0));
    let voters: Vec<_> = verifiers[1..]
        .iter()
        .map(|node| harness.nodes[*node].wallet.pubkey())
        .collect();
    assert_eq!(
        verdict.bounty_recipients(),
        voters,
        "the bounty is the quorum's, not the dissenting vote's"
    );
}

/// Verifiers replay within the tolerance band of each other, not bit for bit,
/// so two votes on the same payload count together whatever their replays
/// hashed to.
#[tokio::test]
pub async fn votes_whose_replays_hashed_apart_still_reach_a_quorum() {
    let mut harness = committee_run(443, "Leviathan replays apart").await;
    let (verifiers, trainers) = committee(&mut harness).await;
    let target = trainers[0];
    let commitment = harness.signed_payload(target).await;
    vote_replayed(&mut harness, verifiers[0], target, &commitment, [0x11; 32])
        .await
        .unwrap();
    vote_replayed(&mut harness, verifiers[1], target, &commitment, [0x22; 32])
        .await
        .unwrap();
    let verdict = verdict_of(&mut harness, target).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.replayed_hash, [0x11; 32]);
}

/// A later round can draw the target onto the committee judging it, with a
/// payload it signed earlier in the epoch still open to conviction. Its vote
/// on itself is refused: cast first, it fixed the evidence the rest of the
/// round had to agree with.
#[tokio::test]
pub async fn the_target_cannot_vote_on_its_own_verdict() {
    let mut harness = committee_run(442, "Leviathan self vote").await;
    let (_, trainers) = committee(&mut harness).await;
    let target = trainers[0];
    let mut signed = (
        harness.state().await.coordinator.progress.epoch,
        harness.signed_payload(target).await,
    );
    let commitment = loop {
        next_round(&mut harness, 6).await;
        let epoch = harness.state().await.coordinator.progress.epoch;
        let (verifiers, trainers) = committee(&mut harness).await;
        if verifiers.contains(&target) && signed.0 == epoch {
            break signed.1;
        }
        if trainers.contains(&target) {
            signed = (epoch, harness.signed_payload(target).await);
        }
    };
    assert_eq!(
        refusal_code(vote(&mut harness, target, target, &commitment).await),
        treasurer_refusal(ProgramError::VerifierIsTarget),
    );
    let key = find_audit_verdict(
        &harness.run,
        &harness.nodes[target].wallet.pubkey(),
    );
    assert!(
        harness.endpoint.get_account(&key).await.unwrap().is_none(),
        "no dispute was opened by the target's own vote"
    );
}
