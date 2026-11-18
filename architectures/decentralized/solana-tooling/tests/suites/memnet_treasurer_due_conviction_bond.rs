//! A slash that has not reached the ledger yet still owes its charge. An
//! ejection is charged only when its epoch ends. A conviction that can no
//! longer be overturned but has not landed is charged at the slashing rate
//! once it lands, in an epoch that charges it or at the target's withdrawal:
//! one recorded as `Due`, or one whose challenge window has closed. Until
//! then the coordinator's ledger knows nothing of either. The bond checks
//! read only the ledger, so a target slashed this way joined, claimed and
//! voted on a bond it was about to lose. A bond posted at the run's minimum
//! backs a seat or a vote; one that its slash leaves short of the minimum
//! does not.

use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::RunState;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_claim;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::signer::Signer;

use super::appeals_harness::setup;
use super::appeals_harness::AppealsHarness;
use super::appeals_harness::CHALLENGE_WINDOW;
use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::SLASHING_RATE;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::replayed_digest;

/// Six nodes with half the free seats verifying, so each round draws a
/// committee of three and a quorum of two.
const COMMITTEE_NODES: usize = 6;

async fn committee_run(index: u64, run_id: &str) -> Harness {
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
    let selection =
        CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
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
        replayed_digest(),
    )
    .await
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

/// Runs the run on into the next round's training, every node rejoining
/// when the epoch ends first.
async fn next_round(harness: &mut Harness) {
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
                for node in 0..COMMITTEE_NODES {
                    harness.join(node).await.unwrap();
                }
            },
            _ => {},
        }
    }
    panic!("the run never reached another round");
}

/// A trainer the run authority ejects, and then the round's committee
/// convicts. The ejection is the first conviction's, so the committee's
/// cannot eject the target again and is recorded as due. The epoch's end
/// charges the authority's ejection to the ledger; the committee's
/// conviction is still to be charged. Returns the run, back in
/// `WaitingForMembers`, and the target.
async fn convicted_due(index: u64, run_id: &str) -> (Harness, usize) {
    let mut harness = committee_run(index, run_id).await;
    let (verifiers, trainers) = committee(&mut harness).await;
    assert!(
        verifiers.len() >= 2 && !trainers.is_empty(),
        "the round drew {verifiers:?} and {trainers:?}"
    );
    let target = trainers[0];
    let commitment = harness.signed_payload(target).await;
    harness.try_convict_on(target, &commitment).await.unwrap();
    for verifier in &verifiers[..2] {
        vote(&mut harness, *verifier, target, &commitment)
            .await
            .unwrap();
    }
    assert_eq!(
        verdict_of(&mut harness, target).await.status,
        VerdictStatus::Due
    );

    harness.finish_epoch().await;
    assert_eq!(harness.ledger(target).await.1, SLASHING_RATE);
    assert_eq!(
        verdict_of(&mut harness, target).await.status,
        VerdictStatus::Due,
        "nobody landed the conviction"
    );
    (harness, target)
}

/// Topping the bond up by what the ledger charged is not enough: the due
/// conviction will take as much again. Until the bond covers that too, the
/// target can neither join the next epoch nor claim what it earned, and once
/// it does, both are open to it.
#[tokio::test]
pub async fn a_due_conviction_counts_against_joining_and_claiming() {
    let (mut harness, target) =
        convicted_due(446, "Leviathan due conviction join").await;

    harness.deposit(target, SLASHING_RATE).await.unwrap();
    assert_eq!(
        refusal_code(harness.join(target).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
    );
    assert_eq!(
        refusal_code(harness.claim(target, 0).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
    );

    // A new slot, so this deposit is not the same transaction as the first.
    harness.endpoint.forward_clock_slot(1).await.unwrap();
    harness.deposit(target, SLASHING_RATE).await.unwrap();
    harness.join(target).await.unwrap();
    harness.claim(target, 0).await.unwrap();
}

/// A target that joined with its due conviction covered, and then asks to
/// withdraw part of its bond, no longer has the minimum behind a vote once the
/// conviction is charged too. Drawn onto a committee, it cannot vote.
#[tokio::test]
pub async fn a_due_conviction_counts_against_a_vote() {
    let (mut harness, target) =
        convicted_due(447, "Leviathan due conviction vote").await;
    harness.deposit(target, 2 * SLASHING_RATE).await.unwrap();
    for node in 0..COMMITTEE_NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;

    let mut drawn = None;
    for _ in 0..40 {
        let (verifiers, trainers) = committee(&mut harness).await;
        if verifiers.contains(&target) && !trainers.is_empty() {
            drawn = Some(trainers[0]);
            break;
        }
        next_round(&mut harness).await;
    }
    let trainer = drawn.expect("the target was never drawn to verify");

    harness.request_withdraw(target, SLASHING_RATE).await;
    let commitment = harness.signed_payload(trainer).await;
    assert_eq!(
        refusal_code(vote(&mut harness, target, trainer, &commitment).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
    );
    assert_eq!(
        verdict_of(&mut harness, target).await.status,
        VerdictStatus::Due,
        "nobody landed the conviction"
    );
}

/// The run authority ejects a trainer mid-epoch. The ledger charges the
/// ejection only when the epoch ends, but the target is not to claim on the
/// bond it is about to lose in the meantime.
#[tokio::test]
pub async fn an_ejection_counts_against_claiming_before_its_epoch_ends() {
    let mut harness = committee_run(448, "Leviathan ejection claim").await;
    let (_, trainers) = committee(&mut harness).await;
    let target = trainers[0];
    harness.claim(target, 0).await.unwrap();

    let commitment = harness.signed_payload(target).await;
    harness.try_convict_on(target, &commitment).await.unwrap();
    assert_eq!(
        harness.ledger(target).await.1,
        0,
        "the ledger charges the ejection when the epoch ends"
    );
    // New slots, so no claim here is the same transaction as another.
    harness.endpoint.forward_clock_slot(1).await.unwrap();
    assert_eq!(
        refusal_code(harness.claim(target, 0).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
    );

    harness.deposit(target, SLASHING_RATE).await.unwrap();
    harness.endpoint.forward_clock_slot(1).await.unwrap();
    harness.claim(target, 0).await.unwrap();
}

async fn claim_nothing(
    h: &mut AppealsHarness,
    client: usize,
) -> anyhow::Result<()> {
    process_treasurer_participant_claim(
        &mut h.endpoint,
        &h.payer,
        &h.clients[client],
        &h.clients_collateral[client],
        &h.collateral_mint,
        &h.run,
        &h.coordinator_account,
        0,
    )
    .await
}

/// A conviction inside its challenge window may yet be overturned, so it
/// does not hold the bond. Once the window closes nobody can challenge it any
/// more and it only waits to be finalised, so it is owed, and the target
/// cannot claim on the bond it will take.
#[tokio::test]
pub async fn a_conviction_past_its_challenge_window_counts_against_claiming() {
    let mut h = setup("Leviathan closed window claim", 449).await;
    h.reach_verifier_quorum().await;
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);
    let target = h.target_client_idx;
    claim_nothing(&mut h, target).await.unwrap();

    h.endpoint
        .forward_clock_unix_timestamp(CHALLENGE_WINDOW as u64 + 1)
        .await
        .unwrap();
    h.endpoint.forward_clock_slot(1).await.unwrap();
    assert_eq!(
        refusal_code(claim_nothing(&mut h, target).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
    );
    assert_eq!(
        h.verdict_status().await,
        VerdictStatus::SlashPending,
        "nobody finalised the conviction"
    );
}
