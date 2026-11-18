//! A conviction recorded as `Due` could not eject its target into an epoch
//! that would charge it (wienerlabs/leviathan-net#36). It used to be settled
//! only through its target: by `run_finalize_slash` once the target was back
//! on the roster of a charging epoch, or by the target's own withdrawal. A
//! target that never came back and never withdrew left it due for good, so
//! its voters were never credited and its bond stayed held by its own
//! conviction. Anyone may now charge it, while the target is on no roster an
//! epoch will still charge (wienerlabs/leviathan-net#74).

use psyche_coordinator::RunState;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_finalize_slash;
use psyche_solana_tooling::verdict_votes::process_treasurer_run_charge_due_conviction;
use psyche_solana_tooling::verdict_votes::settle_open_disputes;
use psyche_solana_treasurer::state::ConvictionCharge;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::state::WithdrawalStanding;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

use super::appeals_harness::*;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

/// What a forfeit of the slashing rate owes the conviction's recipients.
const BOUNTY: u64 =
    (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64;

/// A run whose verifiers convict the target, whose epoch then ends, and whose
/// conviction is finalised once its challenge window has closed: between
/// epochs, so it is recorded as due. The target is still on the ended epoch's
/// roster, which nothing will charge.
async fn due_between_epochs(run_id: &str, index: u64) -> AppealsHarness {
    let mut h = setup(run_id, index).await;
    h.reach_verifier_quorum().await;
    h.finish_epoch_with_witnesses().await;
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    let target_key = h.target_key;
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        h.target_index,
    )
    .await
    .unwrap();
    assert_eq!(h.verdict_status().await, VerdictStatus::Due);
    assert_eq!(h.run_state().await, RunState::WaitingForMembers);
    h
}

/// A wallet of no one in the run, funded to pay for its own transactions.
async fn stranger(h: &mut AppealsHarness) -> Keypair {
    let keeper = Keypair::new();
    h.endpoint
        .request_airdrop(&keeper.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    keeper
}

/// `payer` charges the target's due conviction.
async fn charge_due(
    h: &mut AppealsHarness,
    payer: &Keypair,
) -> anyhow::Result<()> {
    let target_key = h.target_key;
    h.endpoint.forward_clock_slot(1).await.unwrap();
    process_treasurer_run_charge_due_conviction(
        &mut h.endpoint,
        payer,
        &h.run,
        &h.coordinator_account,
        &target_key,
    )
    .await
}

async fn total_bonded(h: &mut AppealsHarness) -> u64 {
    get_run(&mut h.endpoint, &h.run)
        .await
        .unwrap()
        .unwrap()
        .total_bonded_amount
}

/// The target never comes back and never withdraws. A stranger charges its
/// conviction at the slashing rate, and its voters are credited their shares.
/// The ledger never saw the charge, so the bond does not count it as a
/// settled point. Once charged it is never charged again.
#[tokio::test]
pub async fn a_due_conviction_whose_target_never_returns_is_charged_by_anyone()
{
    let mut h = due_between_epochs("Leviathan due charge gone", 740).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let voters = h.quorum_voters();
    let mut voters_before = vec![];
    for voter in &voters {
        voters_before.push(h.holdings_of(*voter).await);
    }
    let bonded_before = total_bonded(&mut h).await;
    assert_eq!(h.participant_of(target).await.bond_amount, BOND);

    let keeper = stranger(&mut h).await;
    charge_due(&mut h, &keeper).await.unwrap();

    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.charge, ConvictionCharge::Charged);
    assert_eq!(verdict.bounty_owed, BOUNTY);
    let participant = h.participant_of(target).await;
    assert_eq!(participant.bond_amount, BOND - SLASHING_RATE);
    assert_eq!(participant.bond_settled_slashed_points, 0);
    assert_eq!(h.slashed_of(&target_key).await, 0);
    assert_eq!(total_bonded(&mut h).await, bonded_before - SLASHING_RATE);

    h.pay_every_seat_of(&target_key).await;
    let share = BOUNTY / voters.len() as u64;
    for (voter, before) in voters.iter().zip(voters_before) {
        assert_eq!(h.holdings_of(*voter).await - before, share);
    }

    // Charged once: neither the charge nor a finalisation reaches it again.
    let another = stranger(&mut h).await;
    assert_eq!(
        refusal_code(charge_due(&mut h, &another).await),
        treasurer_refusal(ProgramError::ConvictionNotDue),
    );
    assert_eq!(
        refusal_code(
            process_treasurer_run_finalize_slash(
                &mut h.endpoint,
                &another,
                &h.run,
                &h.coordinator_account,
                &h.run_id,
                &target_key,
                h.target_index,
            )
            .await
        ),
        treasurer_refusal(ProgramError::VerdictNotPending),
    );
    assert_eq!(
        h.participant_of(target).await.bond_amount,
        BOND - SLASHING_RATE
    );

    // Every vote released, the verdict holds nobody and owes nobody, so a
    // later dispute can take the account over.
    for voter in &voters {
        h.endpoint.forward_clock_slot(1).await.unwrap();
        h.release_vote(None, &target_key, *voter).await.unwrap();
    }
    assert!(h.verdict_of(&target_key).await.is_settled());
}

/// A target back on the roster of an epoch the ledger will charge is left to
/// it: the conviction lands there as an ejection, and is charged through the
/// ledger, never through the bond directly as well.
#[tokio::test]
pub async fn a_due_conviction_waits_while_its_target_is_on_a_charging_roster() {
    let mut h = due_between_epochs("Leviathan due charge back", 741).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let keeper = stranger(&mut h).await;

    h.rejoin_all().await;
    assert_eq!(h.run_state().await, RunState::Warmup);
    assert_eq!(
        refusal_code(charge_due(&mut h, &keeper).await),
        treasurer_refusal(ProgramError::TargetInChargingEpoch),
    );
    let config = h.config().await;
    h.tick_after(config.warmup_time).await;
    assert_eq!(h.run_state().await, RunState::RoundTrain);
    assert_eq!(
        refusal_code(charge_due(&mut h, &keeper).await),
        treasurer_refusal(ProgramError::TargetInChargingEpoch),
    );
    assert_eq!(h.participant_of(target).await.bond_amount, BOND);

    let target_index = h.roster_index_of(&target_key).await;
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &keeper,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .unwrap();
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.charge, ConvictionCharge::Ejected);
    assert_eq!(
        refusal_code(charge_due(&mut h, &keeper).await),
        treasurer_refusal(ProgramError::ConvictionNotDue),
    );

    // The ledger charges the ejection once its epoch ends, and the bond pays
    // it once, as a settled point.
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    h.charge_conviction_of(&target_key).await.unwrap();
    let participant = h.participant_of(target).await;
    assert_eq!(participant.bond_amount, BOND - SLASHING_RATE);
    assert_eq!(participant.bond_settled_slashed_points, SLASHING_RATE);
    assert_eq!(h.verdict_of(&target_key).await.bounty_owed, BOUNTY);
}

/// A target whose due conviction was charged this way, and that then comes
/// back for another epoch, is not charged for it again: not by the ledger,
/// and not at its withdrawal.
#[tokio::test]
pub async fn a_target_charged_this_way_that_comes_back_is_not_charged_again() {
    let mut h = due_between_epochs("Leviathan due charge return", 742).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let keeper = stranger(&mut h).await;
    charge_due(&mut h, &keeper).await.unwrap();

    h.rejoin_all_and_start_epoch().await;
    let target_index = h.roster_index_of(&target_key).await;
    assert_eq!(
        refusal_code(
            process_treasurer_run_finalize_slash(
                &mut h.endpoint,
                &keeper,
                &h.run,
                &h.coordinator_account,
                &h.run_id,
                &target_key,
                target_index,
            )
            .await
        ),
        treasurer_refusal(ProgramError::VerdictNotPending),
    );
    assert_eq!(
        refusal_code(charge_due(&mut h, &keeper).await),
        treasurer_refusal(ProgramError::ConvictionNotDue),
    );
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, 0);

    h.request_withdraw(target, BOND - SLASHING_RATE).await;
    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    assert_eq!(
        h.observed_standing(target).await,
        WithdrawalStanding::Free {
            settle_conviction: false
        }
    );
    let before = h.collateral_of(target).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.participant_of(target).await.bond_amount, 0);
}

/// The target's own withdrawal still charges a due conviction, once: after it
/// nobody can charge the conviction again.
#[tokio::test]
pub async fn the_withdrawal_still_charges_a_due_conviction_once() {
    let mut h = setup("Leviathan due charge withdraw", 743).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target, BOND).await;
    h.reach_verifier_quorum().await;
    h.finish_epoch_with_witnesses().await;
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        h.target_index,
    )
    .await
    .unwrap();
    assert_eq!(h.verdict_status().await, VerdictStatus::Due);
    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();

    let before = h.collateral_of(target).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target).await - before, BOND - SLASHING_RATE);
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.bounty_owed, BOUNTY);

    let keeper = stranger(&mut h).await;
    assert_eq!(
        refusal_code(charge_due(&mut h, &keeper).await),
        treasurer_refusal(ProgramError::ConvictionNotDue),
    );
}

/// The other order: charged by anyone while the target's withdrawal waits,
/// the conviction is not charged again when the withdrawal settles. The
/// pending withdrawal was cut down to what the bond holds.
#[tokio::test]
pub async fn a_due_conviction_charged_first_is_not_charged_again_at_withdrawal()
{
    let mut h = setup("Leviathan due charge first", 744).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target, BOND).await;
    h.reach_verifier_quorum().await;
    h.finish_epoch_with_witnesses().await;
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        h.target_index,
    )
    .await
    .unwrap();
    let keeper = stranger(&mut h).await;
    charge_due(&mut h, &keeper).await.unwrap();
    let participant = h.participant_of(target).await;
    assert_eq!(participant.bond_amount, BOND - SLASHING_RATE);
    assert_eq!(
        participant.bond_withdraw_pending_amount,
        BOND - SLASHING_RATE
    );
    assert_eq!(
        h.observed_standing(target).await,
        WithdrawalStanding::Free {
            settle_conviction: false
        }
    );

    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    let bonded_before = total_bonded(&mut h).await;
    let before = h.collateral_of(target).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.participant_of(target).await.bond_amount, 0);
    assert_eq!(
        total_bonded(&mut h).await,
        bonded_before - (BOND - SLASHING_RATE)
    );
}

/// The verifier daemon's sweep charges a due conviction whose target is gone,
/// and credits its voters, without anyone asking it to.
#[tokio::test]
pub async fn the_dispute_sweep_charges_a_due_conviction_whose_target_is_gone() {
    let mut h = due_between_epochs("Leviathan due charge sweep", 745).await;
    let target_key = h.target_key;
    let voters = h.quorum_voters();
    let mut voters_before = vec![];
    for voter in &voters {
        voters_before.push(h.holdings_of(*voter).await);
    }
    let keeper = stranger(&mut h).await;
    let sweep = settle_open_disputes(
        &mut h.endpoint,
        &keeper,
        &h.run,
        &h.coordinator_account,
    )
    .await
    .unwrap();
    assert!(sweep.failures.is_empty(), "{:?}", sweep.failures);
    assert_eq!(sweep.convictions_charged, 1);
    assert_eq!(sweep.seats_credited, voters.len());
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.charge, ConvictionCharge::Charged);
    let share = BOUNTY / voters.len() as u64;
    for (voter, before) in voters.iter().zip(voters_before) {
        assert_eq!(h.holdings_of(*voter).await - before, share);
    }
}
