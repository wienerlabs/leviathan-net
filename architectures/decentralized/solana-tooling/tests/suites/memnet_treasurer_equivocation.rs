//! A trainer that signs two different payloads for one batch is convicted on
//! the two signatures alone, by anyone, with no committee
//! (wienerlabs/leviathan-net#46).

use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_core::payload_commitment_run;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_solana_tooling::equivocation::process_treasurer_run_commit_equivocation;
use psyche_solana_tooling::equivocation::process_treasurer_run_slash_equivocation;
use psyche_solana_tooling::equivocation::run_slash_equivocation_instruction;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_tooling::signed_commitment::PayloadHeader;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_treasurer::logic::RunUpdateParams;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::appeals_harness::*;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::payload_digest;
use super::signed_payload::replayed_digest;
use super::signed_payload::FRAUD_PAYLOAD;
use super::signed_payload::HONEST_PAYLOAD;

/// What the target signed for its batch this round, re-signed with
/// `results_digest`, `step` and `nonce` changed as asked: another payload for
/// the same work, or a commitment to other work.
pub(super) fn resigned(
    h: &AppealsHarness,
    p2p: &Keypair,
    step_offset: u32,
    nonce_offset: u32,
    batch_offset: u64,
    results_digest: [u8; 32],
) -> SignedCommitment {
    let first = h.target_commitment.header();
    SignedCommitment::sign(
        p2p,
        h.target_commitment.run,
        PayloadHeader {
            step: first.step + step_offset,
            trainer_nonce: first.trainer_nonce + nonce_offset,
            batch_id: BatchId(ClosedInterval::new(
                first.batch_id.0.start + batch_offset,
                first.batch_id.0.end + batch_offset,
            )),
            ..first
        },
        results_digest,
    )
}

/// `reporter` commits to the proof that `target` signed `first` and
/// `second`, and reveals it a slot later, as a reporter has to
/// (wienerlabs/leviathan-net#92).
#[allow(clippy::too_many_arguments)]
pub(super) async fn commit_then_reveal(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    reporter: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target: &Pubkey,
    target_index: u64,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> anyhow::Result<()> {
    process_treasurer_run_commit_equivocation(
        endpoint, payer, reporter, run, target, first, second,
    )
    .await?;
    endpoint.forward_clock_slot(1).await?;
    process_treasurer_run_slash_equivocation(
        endpoint,
        payer,
        reporter,
        run,
        coordinator_account,
        run_id,
        target,
        target_index,
        first,
        second,
    )
    .await
}

/// Anyone sends the proof that the target signed `first` and `second`.
pub(super) async fn report(
    h: &mut AppealsHarness,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> anyhow::Result<()> {
    let reporter = Keypair::new();
    let payer = h.payer.insecure_clone();
    let target_index = h.target_index;
    let run_id = h.run_id.clone();
    let target = h.target_key;
    let (run, coordinator_account) = (h.run, h.coordinator_account);
    commit_then_reveal(
        &mut h.endpoint,
        &payer,
        &reporter,
        &run,
        &coordinator_account,
        &run_id,
        &target,
        target_index,
        first,
        second,
    )
    .await
}

/// Two genuine, different payloads the target signed for one batch convict
/// it, whether they differ in results or in nonce, and the epoch charges the
/// ejection.
#[tokio::test]
pub async fn two_signed_payloads_for_one_batch_convict() {
    // Another payload for the batch; and its own payload, re-signed under
    // another nonce, which is another commitment to the same work.
    for (index, nonce_offset, digest) in [
        (150, 0, payload_digest(HONEST_PAYLOAD)),
        (151, 1, payload_digest(FRAUD_PAYLOAD)),
    ] {
        let mut h =
            setup(&format!("Leviathan equivocation {index}"), index).await;
        let target_key = h.target_key;
        let p2p = h.p2ps[h.target_client_idx].insecure_clone();
        let first = h.target_commitment;
        let second = resigned(&h, &p2p, 0, nonce_offset, 0, digest);
        report(&mut h, &first, &second).await.unwrap();
        assert_eq!(
            h.client_state(&target_key).await,
            Some(ClientState::Ejected)
        );
        h.finish_epoch_with_witnesses().await;
        assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    }
}

/// One commitment, sent twice, is not two.
#[tokio::test]
pub async fn the_same_commitment_twice_does_not_convict() {
    let mut h = setup("Leviathan equivocation same", 152).await;
    let first = h.target_commitment;
    assert_eq!(
        refusal_code(report(&mut h, &first, &first).await),
        treasurer_refusal(ProgramError::EquivocationNotProven),
    );
    let target_key = h.target_key;
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );
}

/// A commitment to another step of the epoch is other work, not a
/// contradiction.
#[tokio::test]
pub async fn commitments_for_different_steps_do_not_convict() {
    let mut h = setup("Leviathan equivocation step", 153).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    // A step later in the same epoch, so both commitments are this epoch's.
    h.advance_to_next_round().await;
    let later = resigned(&h, &p2p, 1, 0, 0, replayed_digest());
    assert_eq!(
        refusal_code(report(&mut h, &first, &later).await),
        treasurer_refusal(ProgramError::EquivocationNotProven),
    );
}

/// A commitment to another batch, sent as if it were for this one, does not
/// prove a signature over this batch's commitment.
#[tokio::test]
pub async fn commitments_for_different_batches_do_not_convict() {
    let mut h = setup("Leviathan equivocation batch", 154).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let elsewhere = resigned(&h, &p2p, 0, 0, 100, replayed_digest());
    // The tooling refuses it outright; the chain is asked directly, for a
    // proof the reporter committed to.
    let reporter = Keypair::new();
    let payer = h.payer.insecure_clone();
    let (run, target_key) = (h.run, h.target_key);
    process_treasurer_run_commit_equivocation(
        &mut h.endpoint,
        &payer,
        &reporter,
        &run,
        &target_key,
        &first,
        &elsewhere,
    )
    .await
    .unwrap();
    h.endpoint.forward_clock_slot(1).await.unwrap();
    let instruction = run_slash_equivocation_instruction(
        &h.payer.pubkey(),
        &reporter.pubkey(),
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &h.target_key,
        h.target_index,
        &first,
        &elsewhere,
    );
    assert_eq!(
        refusal_code(
            h.endpoint
                .process_instructions_with_signers(
                    &payer,
                    &[
                        first.proof_instruction(),
                        elsewhere.proof_instruction(),
                        instruction,
                    ],
                    &[&reporter],
                )
                .await
        ),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
}

/// A second commitment signed by any key but the target's convicts nobody.
#[tokio::test]
pub async fn a_commitment_signed_by_another_key_does_not_convict() {
    let mut h = setup("Leviathan equivocation forged", 155).await;
    let first = h.target_commitment;
    let forged = resigned(&h, &Keypair::new(), 0, 0, 0, replayed_digest());
    assert_eq!(
        refusal_code(report(&mut h, &first, &forged).await),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
}

/// A target already ejected this epoch cannot be ejected again, and the proof
/// has nothing to record a later charge in: it is refused, and the target
/// pays for the epoch's ejection once.
#[tokio::test]
pub async fn a_target_already_ejected_is_refused() {
    let mut h = setup("Leviathan equivocation ejected", 156).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let second = resigned(&h, &p2p, 0, 0, 0, replayed_digest());
    report(&mut h, &first, &second).await.unwrap();
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    let third = resigned(&h, &p2p, 0, 2, 0, replayed_digest());
    assert_eq!(
        refusal_code(report(&mut h, &first, &third).await),
        treasurer_refusal(ProgramError::EquivocationNotLandable),
    );
}

/// The target is the client at the roster index the proof names, and the
/// key the proof names has to be that client's. Naming another seat, with its
/// own key, puts the proof to that seat's p2p key, which signed neither
/// commitment; naming the target's index with another key, or a seat past the
/// roster, names nobody the proof is about.
#[tokio::test]
pub async fn a_proof_naming_another_seat_is_refused() {
    let mut h = setup("Leviathan equivocation seat", 157).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let second = resigned(&h, &p2p, 0, 1, 0, replayed_digest());
    let roster = h.clients.len() as u64;
    let other_seat = (h.target_index + 1) % roster;
    let other_key = Pubkey::new_from_array(
        *h.coordinator_state().await.coordinator.epoch_state.clients
            [other_seat as usize]
            .id
            .signer(),
    );
    let target_key = h.target_key;
    for (wrong_key, wrong, refusal) in [
        (other_key, other_seat, ProgramError::CommitmentNotSigned),
        (other_key, h.target_index, ProgramError::TargetMismatch),
        (target_key, roster + 5, ProgramError::TargetMismatch),
    ] {
        let reporter = Keypair::new();
        let payer = h.payer.insecure_clone();
        let run_id = h.run_id.clone();
        let (run, coordinator_account) = (h.run, h.coordinator_account);
        let refused = commit_then_reveal(
            &mut h.endpoint,
            &payer,
            &reporter,
            &run,
            &coordinator_account,
            &run_id,
            &wrong_key,
            wrong,
            &first,
            &second,
        )
        .await;
        assert_eq!(refusal_code(refused), treasurer_refusal(refusal));
    }
    let target_key = h.target_key;
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );
}

/// Each commitment needs its own proof in the transaction: one proven and
/// the other only claimed convicts nobody.
#[tokio::test]
pub async fn one_proven_signature_is_not_two() {
    let mut h = setup("Leviathan equivocation half", 158).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let second = resigned(&h, &p2p, 0, 1, 0, replayed_digest());
    let reporter = Keypair::new();
    let payer = h.payer.insecure_clone();
    let (run, target_key) = (h.run, h.target_key);
    process_treasurer_run_commit_equivocation(
        &mut h.endpoint,
        &payer,
        &reporter,
        &run,
        &target_key,
        &first,
        &second,
    )
    .await
    .unwrap();
    h.endpoint.forward_clock_slot(1).await.unwrap();
    let instruction = run_slash_equivocation_instruction(
        &h.payer.pubkey(),
        &reporter.pubkey(),
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &h.target_key,
        h.target_index,
        &first,
        &second,
    );
    for proofs in [
        vec![first.proof_instruction()],
        vec![second.proof_instruction()],
    ] {
        let mut instructions = proofs;
        instructions.push(instruction.clone());
        assert_eq!(
            refusal_code(
                h.endpoint
                    .process_instructions_with_signers(
                        &payer,
                        &instructions,
                        &[&reporter],
                    )
                    .await
            ),
            treasurer_refusal(ProgramError::CommitmentNotSigned),
        );
    }
}

/// A commitment made in another run is not a commitment in this one, though
/// the target's key signed it.
#[tokio::test]
pub async fn a_commitment_made_in_another_run_does_not_convict() {
    let mut h = setup("Leviathan equivocation run", 159).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let elsewhere = SignedCommitment::sign(
        &p2p,
        payload_commitment_run("Leviathan another run"),
        PayloadHeader {
            trainer_nonce: first.evidence.trainer_nonce + 1,
            ..first.header()
        },
        replayed_digest(),
    );
    assert_eq!(
        refusal_code(report(&mut h, &first, &elsewhere).await),
        treasurer_refusal(ProgramError::CommitmentNotSigned),
    );
}

/// Two commitments signed in an epoch that is over name work that epoch
/// already settled.
#[tokio::test]
pub async fn commitments_from_an_ended_epoch_do_not_convict() {
    let mut h = setup("Leviathan equivocation stale", 165).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let second = resigned(&h, &p2p, 0, 1, 0, replayed_digest());
    h.finish_epoch_with_witnesses().await;
    h.rejoin_all_and_start_epoch().await;
    let target_key = h.target_key;
    h.target_index = h.roster_index_of(&target_key).await;
    assert_eq!(
        refusal_code(report(&mut h, &first, &second).await),
        treasurer_refusal(ProgramError::CommitmentOutsideEpoch),
    );
}

/// A proof lands only as an ejection the epoch is certain to charge. Once
/// the epoch has stopped charging, with its roster still on the account, and
/// once a pause has withdrawn the target, it is refused: nothing would
/// record it to charge later.
#[tokio::test]
pub async fn a_proof_the_epoch_cannot_charge_is_refused() {
    let mut h = setup("Leviathan equivocation uncharged", 166).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let second = resigned(&h, &p2p, 0, 1, 0, replayed_digest());
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.run_state().await, RunState::WaitingForMembers);
    assert_eq!(
        refusal_code(report(&mut h, &first, &second).await),
        treasurer_refusal(ProgramError::EquivocationNotLandable),
    );

    let main_authority = h.main_authority.insecure_clone();
    let payer = h.payer.insecure_clone();
    let (run, instance, account) =
        (h.run, h.coordinator_instance, h.coordinator_account);
    process_treasurer_run_update(
        &mut h.endpoint,
        &payer,
        &main_authority,
        &run,
        &instance,
        &account,
        RunUpdateParams {
            metadata: None,
            config: None,
            model: None,
            progress: None,
            epoch_earning_rate_total_shared: None,
            epoch_slashing_rate_per_client: None,
            paused: Some(true),
            client_version: None,
        },
    )
    .await
    .unwrap();
    let target_key = h.target_key;
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Withdrawn)
    );
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    assert_eq!(
        refusal_code(report(&mut h, &first, &second).await),
        treasurer_refusal(ProgramError::EquivocationNotLandable),
    );
}
