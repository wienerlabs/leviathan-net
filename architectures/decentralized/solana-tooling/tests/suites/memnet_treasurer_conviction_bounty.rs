//! A convicted target's forfeit pays its bounty to the conviction's own
//! recipients, whatever happens to the verdict account afterwards, and no
//! matter how many there are (wienerlabs/leviathan-net#64).
//!
//! The bounty used to be paid at the target's withdrawal, in that transaction,
//! to whoever the verdict account named then. A later dispute could take the
//! account over first and receive it, an overturned verdict named its losing
//! verifiers, and a committee too large for one transaction left the target
//! unable to withdraw. Now the conviction's charge is taken from the bond once
//! the ledger has made it, its bounty is recorded on the verdict, and each
//! recipient's bond is credited by seat.

use anchor_spl::token::spl_token;
use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::verdict_votes::settle_open_disputes;
use psyche_solana_tooling::verdict_votes::settle_previous_dispute;
use psyche_solana_tooling::verdict_votes::unsettled_disputes;
use psyche_solana_tooling::verdict_votes::DisputeSettlement;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::state::ConvictionCharge;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signer::Signer;

use super::appeals_harness::setup;
use super::appeals_harness::setup_sized;
use super::appeals_harness::AppealsHarness;
use super::appeals_harness::RunSize;
use super::appeals_harness::APPEAL_WINDOW;
use super::appeals_harness::BOND;
use super::appeals_harness::BOUNTY_BPS;
use super::appeals_harness::CHALLENGE_WINDOW;
use super::appeals_harness::SLASHING_RATE;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::slash_on_signed_payload;

/// The run's bond withdraw delay.
const WITHDRAW_DELAY: u64 = 100;

/// What one conviction's forfeit owes its recipients.
fn bounty() -> u64 {
    SLASHING_RATE * BOUNTY_BPS as u64 / 10_000
}

/// Anyone settles everything about the last dispute against the target that
/// needs nobody's consent.
async fn settle(h: &mut AppealsHarness) -> DisputeSettlement {
    let target = h.target_key;
    settle_of(h, &target).await
}

/// Anyone settles everything about the last dispute against `target` that
/// needs nobody's consent.
async fn settle_of(
    h: &mut AppealsHarness,
    target: &Pubkey,
) -> DisputeSettlement {
    let payer = h.payer.insecure_clone();
    settle_previous_dispute(
        &mut h.endpoint,
        &payer,
        &h.run,
        &h.coordinator_account,
        target,
    )
    .await
    .unwrap()
}

/// The verifiers convict the target, its challenge window closes, and the
/// conviction finalises as an ejection in the epoch under way.
async fn convict_and_finalize(h: &mut AppealsHarness) {
    h.reach_verifier_quorum().await;
    h.endpoint
        .forward_clock_unix_timestamp(CHALLENGE_WINDOW as u64)
        .await
        .unwrap();
    let settled = settle(h).await;
    assert!(settled.conviction_finalized);
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
}

/// Every participant's bond adds up to what the run counts as bonded.
async fn assert_bonds_add_up(h: &mut AppealsHarness) {
    let mut bonds = 0;
    for client_idx in 0..h.clients.len() {
        let key = h.clients[client_idx].pubkey();
        if let Some(participant) =
            get_participant(&mut h.endpoint, &find_participant(&h.run, &key))
                .await
                .unwrap()
        {
            bonds += participant.bond_amount;
        }
    }
    let bonded = get_run(&mut h.endpoint, &h.run)
        .await
        .unwrap()
        .unwrap()
        .total_bonded_amount;
    assert_eq!(bonds, bonded, "the bonds and the run's bonded total agree");
}

/// The charge waits for the epoch that charges the ejection, takes that charge
/// from the bond once, and only then are the voters paid. The target's own
/// withdrawal does not charge it again.
#[tokio::test]
pub async fn a_conviction_is_charged_once_its_epoch_is_and_only_once() {
    let mut h = setup("Leviathan bounty charge once", 311).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    convict_and_finalize(&mut h).await;

    let verdict = h.verdict_of(&target_key).await;
    let epoch_start = h
        .coordinator_state()
        .await
        .coordinator
        .epoch_state
        .start_timestamp;
    assert_eq!(verdict.charge, ConvictionCharge::Ejected);
    assert_eq!(verdict.charge_epoch_start, epoch_start);
    assert_eq!(verdict.charge_rate, SLASHING_RATE);

    // The epoch it ejected the target in is still under way.
    assert_eq!(
        refusal_code(h.charge_conviction_of(&target_key).await),
        treasurer_refusal(ProgramError::EjectionNotCharged),
    );
    assert_eq!(
        refusal_code(h.pay_seat(0).await),
        treasurer_refusal(ProgramError::ConvictionNotCharged),
    );

    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    let bonded_before = get_run(&mut h.endpoint, &h.run)
        .await
        .unwrap()
        .unwrap()
        .total_bonded_amount;
    h.charge_conviction_of(&target_key).await.unwrap();
    let participant = h.participant_of(target).await;
    assert_eq!(participant.bond_amount, BOND - SLASHING_RATE);
    assert_eq!(participant.bond_settled_slashed_points, SLASHING_RATE);
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.charge, ConvictionCharge::Charged);
    assert_eq!(verdict.bounty_owed, bounty());
    assert_eq!(
        get_run(&mut h.endpoint, &h.run)
            .await
            .unwrap()
            .unwrap()
            .total_bonded_amount,
        bonded_before - SLASHING_RATE,
    );
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    assert_eq!(
        refusal_code(h.charge_conviction_of(&target_key).await),
        treasurer_refusal(ProgramError::ConvictionNotChargeable),
    );

    let voters = h.quorum_voters();
    let mut before = vec![];
    for voter in &voters {
        before.push(h.holdings_of(*voter).await);
    }
    h.pay_every_seat_of(&target_key).await;
    for (voter, before) in voters.iter().zip(before) {
        assert_eq!(
            h.holdings_of(*voter).await - before,
            bounty() / voters.len() as u64
        );
    }
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    assert_eq!(
        refusal_code(h.pay_seat(0).await),
        treasurer_refusal(ProgramError::BountySeatPaid),
    );
    assert!(h.verdict_of(&target_key).await.is_settled());

    // The withdrawal finds the conviction paid for: the rest of the bond
    // leaves, and nothing is charged twice.
    h.request_withdraw(target, BOND - SLASHING_RATE).await;
    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY)
        .await
        .unwrap();
    let wallet_before = h.collateral_of(target).await;
    h.try_finalize_withdraw(target).await.unwrap();
    assert_eq!(
        h.collateral_of(target).await - wallet_before,
        BOND - SLASHING_RATE
    );
    assert_eq!(h.participant_of(target).await.bond_amount, 0);
    assert_bonds_add_up(&mut h).await;
}

/// The issue's first case. An upheld target rejoins and trains, and a later
/// dispute against it wants its verdict account. It gets the account only
/// once the conviction is charged and every convicting voter credited, so the
/// target's later withdrawal pays the new dispute's voter nothing.
#[tokio::test]
pub async fn a_later_dispute_cannot_take_the_bounty_from_the_convicting_voters()
{
    let mut h = setup("Leviathan bounty takeover", 312).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let voters = h.quorum_voters();
    convict_and_finalize(&mut h).await;
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);

    // A later epoch, in a round where the target trains. Whoever the round
    // draws to judge it opens the new dispute, one of the convicting voters
    // or not.
    h.rejoin_all_and_start_epoch().await;
    let target_index = h.advance_until_trains(target).await;
    let new_voter = h.committee_now().await.verifiers[0].0;
    let commitment = h.signed_payload_of(target).await;

    // The conviction is not charged, and its voters not paid: no takeover.
    assert!(!h.verdict_of(&target_key).await.is_settled());
    assert_eq!(
        refusal_code(
            h.convict_on(new_voter, target, target_index, &commitment)
                .await
        ),
        treasurer_refusal(ProgramError::PreviousVerdictUnsettled),
    );

    let mut voters_before = vec![];
    for voter in &voters {
        voters_before.push(h.holdings_of(*voter).await);
    }
    assert_eq!(
        settle(&mut h).await,
        DisputeSettlement {
            conviction_charged: true,
            votes_freed: 0,
            seats_credited: voters.len(),
            ..DisputeSettlement::default()
        }
    );
    let share = bounty() / voters.len() as u64;
    for (voter, before) in voters.iter().zip(&voters_before) {
        assert_eq!(h.holdings_of(*voter).await - before, share);
    }

    // Now the account is free, and the new dispute starts from nothing.
    h.convict_on(new_voter, target, target_index, &commitment)
        .await
        .unwrap();
    let taken_over = h.verdict_of(&target_key).await;
    assert_eq!(taken_over.status, VerdictStatus::Voting);
    assert_eq!(taken_over.voters, vec![h.clients[new_voter].pubkey()]);
    assert_eq!(taken_over.charge, ConvictionCharge::Unlanded);
    assert_eq!(taken_over.bounty_owed, 0);
    assert!(taken_over.bounty_recipients().is_empty());

    // The new dispute lapses, and the target leaves. Its withdrawal pays
    // nobody: not the new dispute's voter, and not the convicting voters
    // again.
    h.finish_epoch_with_witnesses().await;
    let mut everyone_before = vec![];
    for client_idx in 0..h.clients.len() {
        everyone_before.push(h.holdings_of(client_idx).await);
    }
    let remaining = h.participant_of(target).await.bond_amount;
    assert_eq!(remaining, BOND - SLASHING_RATE);
    h.request_withdraw(target, remaining).await;
    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY)
        .await
        .unwrap();
    // No recipient and no reporter: nothing it owes is paid here.
    h.try_finalize_withdraw(target).await.unwrap();
    for (client_idx, before) in everyone_before.into_iter().enumerate() {
        assert_eq!(
            h.holdings_of(client_idx).await,
            before,
            "client {client_idx} (new voter {new_voter}) gained or lost at the target's withdrawal"
        );
    }
    assert_eq!(h.participant_of(target).await.bond_amount, 0);
    for (voter, before) in voters.iter().zip(voters_before) {
        assert_eq!(h.holdings_of(*voter).await - before, share);
    }
    assert_bonds_add_up(&mut h).await;
}

/// The issue's second case. A target cleared on appeal also carries a
/// `run_slash` from the run authority. That slash's bounty goes to the
/// reporter the withdrawal names, and the losing verifiers get none of it.
#[tokio::test]
pub async fn a_cleared_target_pays_an_unrelated_slash_to_its_reporter_not_its_losers(
) {
    let mut h = setup("Leviathan bounty cleared", 313).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let losers = h.quorum_voters();
    let bench = h.bench();

    h.convict_and_overturn().await;
    let main_authority = h.main_authority.insecure_clone();
    let (wallet, p2p) = (
        h.clients[target].insecure_clone(),
        h.p2ps[target].insecure_clone(),
    );
    let (run, coordinator_account, run_id, target_index) = (
        h.run,
        h.coordinator_account,
        h.run_id.clone(),
        h.target_index,
    );
    slash_on_signed_payload(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &run,
        &coordinator_account,
        &run_id,
        target_index,
        &wallet,
        &p2p,
    )
    .await
    .unwrap();
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);

    // An overturn owes its bench, never its losers, and there is no
    // conviction of the target to charge.
    let overturned = h.verdict_of(&target_key).await;
    let bench_keys: Vec<Pubkey> =
        bench.iter().map(|i| h.clients[*i].pubkey()).collect();
    assert_eq!(overturned.bounty_recipients(), bench_keys.as_slice());
    assert_eq!(
        refusal_code(h.charge_conviction_of(&target_key).await),
        treasurer_refusal(ProgramError::ConvictionNotChargeable),
    );
    let settled = settle(&mut h).await;
    assert_eq!(settled.losers_charged, losers.len());
    assert_eq!(settled.seats_credited, bench.len());
    let owed_to_bench = h.verdict_of(&target_key).await.bounty_owed;

    let reporter = (0..h.clients.len())
        .find(|i| *i != target && !losers.contains(i) && !bench.contains(i))
        .expect("a client that is neither the target, a loser nor the bench");
    let mut before = vec![];
    for client_idx in losers.iter().chain(&bench) {
        before.push(h.holdings_of(*client_idx).await);
    }
    let reporter_before = h.holdings_of(reporter).await;

    h.request_withdraw(target, BOND).await;
    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY)
        .await
        .unwrap();
    // The slash no verdict accounts for owes a reporter, which the withdrawal
    // has to name.
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target).await),
        treasurer_refusal(ProgramError::MissingReporter),
    );
    let wallet_before = h.collateral_of(target).await;
    h.try_finalize_withdraw_naming(target, reporter)
        .await
        .unwrap();
    assert_eq!(
        h.collateral_of(target).await - wallet_before,
        BOND - SLASHING_RATE
    );
    assert_eq!(h.holdings_of(reporter).await - reporter_before, bounty());
    for (client_idx, before) in losers.iter().chain(&bench).zip(before) {
        assert_eq!(
            h.holdings_of(*client_idx).await,
            before,
            "client {client_idx} was paid out of a slash it had nothing to do with"
        );
    }
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Overturned);
    assert_eq!(verdict.bounty_owed, owed_to_bench);
    assert_bonds_add_up(&mut h).await;
}

/// The issue's third case. A committee whose quorum is larger than one
/// transaction can name convicts the target. The target still withdraws, even
/// with a convicting voter's token account closed, and every voter is credited
/// its share by seat.
#[tokio::test]
pub async fn a_conviction_by_more_voters_than_a_transaction_holds_settles() {
    let mut h = setup_sized(
        "Leviathan bounty large",
        314,
        RunSize {
            clients: 46,
            verification_percent: 90,
            unbonded: 0,
        },
    )
    .await;
    // A legacy transaction holds about two dozen recipient accounts beside a
    // withdrawal's own.
    assert!(h.verifier_quorum > 24, "quorum {}", h.verifier_quorum);
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let voters = h.quorum_voters();
    convict_and_finalize(&mut h).await;
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);

    // A voter closes its emptied collateral account; nothing waits on it.
    let closer = voters[0];
    let closer_wallet = h.clients[closer].insecure_clone();
    let closer_collateral = h.clients_collateral[closer];
    assert_eq!(h.collateral_of(closer).await, 0);
    let close = spl_token::instruction::close_account(
        &spl_token::ID,
        &closer_collateral,
        &closer_wallet.pubkey(),
        &closer_wallet.pubkey(),
        &[],
    )
    .unwrap();
    let payer = h.payer.insecure_clone();
    h.endpoint
        .process_instruction_with_signers(&payer, close, &[&closer_wallet])
        .await
        .unwrap();
    assert!(h
        .endpoint
        .get_account(&closer_collateral)
        .await
        .unwrap()
        .is_none());

    let mut bonds_before = vec![];
    for voter in &voters {
        bonds_before.push(h.participant_of(*voter).await.bond_amount);
    }
    h.request_withdraw(target, BOND).await;
    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY)
        .await
        .unwrap();
    let wallet_before = h.collateral_of(target).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(
        h.collateral_of(target).await - wallet_before,
        BOND - SLASHING_RATE
    );
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.charge, ConvictionCharge::Charged);
    assert_eq!(verdict.bounty_owed, bounty());
    assert_eq!(verdict.bounty_recipients().len(), voters.len());

    h.pay_every_seat_of(&target_key).await;
    let share = bounty() / voters.len() as u64;
    for (voter, before) in voters.iter().zip(bonds_before) {
        assert_eq!(h.participant_of(*voter).await.bond_amount - before, share);
    }
    assert!(h.verdict_of(&target_key).await.is_settled());
    assert_bonds_add_up(&mut h).await;
}

/// A conviction that never reached the ledger is charged at the target's
/// withdrawal (wienerlabs/leviathan-net#36), and what that owes its voters is
/// recorded on the verdict like any other conviction's: the verdict stays
/// unsettled until every voter is credited.
#[tokio::test]
pub async fn a_conviction_settled_at_withdrawal_owes_its_voters_until_credited()
{
    let mut h = setup("Leviathan bounty unlanded", 315).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let voters = h.quorum_voters();
    h.request_withdraw(target, BOND).await;
    h.reach_verifier_quorum().await;
    h.finish_epoch_with_witnesses().await;
    h.endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY)
        .await
        .unwrap();
    assert_eq!(h.slashed_of(&target_key).await, 0);
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);

    let wallet_before = h.collateral_of(target).await;
    h.try_finalize_withdraw(target).await.unwrap();
    assert_eq!(
        h.collateral_of(target).await - wallet_before,
        BOND - SLASHING_RATE
    );
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.charge, ConvictionCharge::Charged);
    assert_eq!(verdict.bounty_owed, bounty());

    // The votes are released, but the bounty is still owed.
    for voter in &voters {
        let target_key = h.target_key;
        h.release_vote(None, &target_key, *voter).await.unwrap();
    }
    assert!(!h.verdict_of(&target_key).await.is_settled());
    let mut before = vec![];
    for voter in &voters {
        before.push(h.holdings_of(*voter).await);
    }
    h.pay_every_seat_of(&target_key).await;
    for (voter, before) in voters.iter().zip(before) {
        assert_eq!(
            h.holdings_of(*voter).await - before,
            bounty() / voters.len() as u64
        );
    }
    assert!(h.verdict_of(&target_key).await.is_settled());
    assert_bonds_add_up(&mut h).await;
}

/// A run whose join authority is not the treasurer admits a client that never
/// created a participant account. Convicted, it has no bond to charge: its
/// conviction is charged nothing, owes nothing, and settles, so the client can
/// be disputed again when it comes back. Requiring the account left the
/// verdict open for good, and the client out of every later dispute's reach.
#[tokio::test]
pub async fn a_target_with_no_bond_is_charged_nothing_and_can_be_disputed_again(
) {
    let mut h = setup_sized(
        "Leviathan bounty unbonded",
        316,
        RunSize {
            clients: 7,
            verification_percent: 50,
            unbonded: 1,
        },
    )
    .await;
    let target = h.clients.len() - 1;
    let target_key = h.clients[target].pubkey();
    assert!(get_participant(
        &mut h.endpoint,
        &find_participant(&h.run, &target_key)
    )
    .await
    .unwrap()
    .is_none());

    let target_index = h.advance_until_trains(target).await;
    let committee = h.committee_now().await;
    let voters: Vec<usize> = committee
        .verifiers
        .iter()
        .take(committee.verifier_quorum)
        .map(|(voter, _)| *voter)
        .collect();
    let commitment = h.signed_payload_of(target).await;
    for voter in &voters {
        h.convict_on(*voter, target, target_index, &commitment)
            .await
            .unwrap();
    }
    h.endpoint
        .forward_clock_unix_timestamp(CHALLENGE_WINDOW as u64)
        .await
        .unwrap();
    assert!(settle_of(&mut h, &target_key).await.conviction_finalized);
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Upheld);
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);

    let bonded_before = get_run(&mut h.endpoint, &h.run)
        .await
        .unwrap()
        .unwrap()
        .total_bonded_amount;
    h.charge_conviction_of(&target_key).await.unwrap();
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.charge, ConvictionCharge::Charged);
    assert_eq!(verdict.bounty_owed, 0);
    assert!(verdict.is_settled());
    assert_eq!(
        get_run(&mut h.endpoint, &h.run)
            .await
            .unwrap()
            .unwrap()
            .total_bonded_amount,
        bonded_before
    );
    assert!(
        get_participant(
            &mut h.endpoint,
            &find_participant(&h.run, &target_key)
        )
        .await
        .unwrap()
        .is_none(),
        "charging touches no account the target never made"
    );

    // It comes back, and a later dispute can take its verdict over.
    h.rejoin_all_and_start_epoch().await;
    let target_index = h.advance_until_trains(target).await;
    let committee = h.committee_now().await;
    let voter = committee.verifiers[0].0;
    let commitment = h.signed_payload_of(target).await;
    h.convict_on(voter, target, target_index, &commitment)
        .await
        .unwrap();
    let taken_over = h.verdict_of(&target_key).await;
    assert_eq!(taken_over.status, VerdictStatus::Voting);
    assert_eq!(taken_over.voters, vec![h.clients[voter].pubkey()]);
    assert_bonds_add_up(&mut h).await;
}

/// The bench earns a conviction's bounty only by ruling on it. An appeal that
/// runs out its window with fewer tie-breakers than its quorum is finalised as
/// the verifiers reached it, and the bounty is theirs: one tie-breaker voting
/// before the deadline used to take all of it.
#[tokio::test]
pub async fn an_appeal_that_runs_out_its_window_pays_the_verifiers() {
    let mut h = setup("Leviathan bounty appeal timeout", 317).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let voters = h.quorum_voters();
    assert!(h.tie_breaker_quorum > 1);
    let lone = h.bench()[0];

    h.reach_verifier_quorum().await;
    let wallet = h.clients[target].insecure_clone();
    let run = h.run;
    process_treasurer_run_open_challenge(
        &mut h.endpoint,
        &h.payer,
        &wallet,
        &run,
    )
    .await
    .unwrap();
    h.cast_appeal(lone, true).await.unwrap();
    assert_eq!(h.verdict_status().await, VerdictStatus::Challenged);
    h.endpoint
        .forward_clock_unix_timestamp(APPEAL_WINDOW as u64)
        .await
        .unwrap();
    assert!(settle(&mut h).await.conviction_finalized);
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert!(!verdict.appeal_upheld);
    assert_eq!(verdict.appeal_voters, vec![h.clients[lone].pubkey()]);
    let voter_keys: Vec<Pubkey> = voters
        .iter()
        .map(|voter| h.clients[*voter].pubkey())
        .collect();
    assert_eq!(verdict.bounty_recipients(), voter_keys.as_slice());

    h.finish_epoch_with_witnesses().await;
    let mut before = vec![];
    for voter in &voters {
        before.push(h.holdings_of(*voter).await);
    }
    let lone_before = h.holdings_of(lone).await;
    let settled = settle(&mut h).await;
    assert!(settled.conviction_charged);
    assert_eq!(settled.seats_credited, voters.len());
    for (voter, before) in voters.iter().zip(before) {
        assert_eq!(
            h.holdings_of(*voter).await - before,
            bounty() / voters.len() as u64
        );
    }
    assert_eq!(h.holdings_of(lone).await, lone_before);
    assert_bonds_add_up(&mut h).await;
}

/// Nobody disputes the target again, and nobody withdraws. The verifier
/// daemon's sweep finds the conviction in the run all the same: it finalises
/// it once its window closes, charges it once its epoch is charged, and
/// credits every voter, after which the run holds no open dispute.
#[tokio::test]
pub async fn a_sweep_pays_a_conviction_nobody_disputes_again() {
    let mut h = setup("Leviathan bounty sweep", 318).await;
    let target_key = h.target_key;
    let voters = h.quorum_voters();
    let payer = h.payer.insecure_clone();
    let (run, coordinator_account) = (h.run, h.coordinator_account);

    h.reach_verifier_quorum().await;
    let open = unsettled_disputes(&mut h.endpoint, &run).await.unwrap();
    assert_eq!(open.len(), 1);
    assert_eq!(open[0].target, target_key);
    h.endpoint
        .forward_clock_unix_timestamp(CHALLENGE_WINDOW as u64)
        .await
        .unwrap();
    let sweep = settle_open_disputes(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert_eq!(sweep.open, 1);
    assert_eq!(sweep.convictions_finalized, 1);
    assert_eq!(sweep.votes_freed, voters.len());
    assert_eq!(sweep.convictions_charged, 0, "its epoch is still running");
    assert!(sweep.failures.is_empty());

    h.finish_epoch_with_witnesses().await;
    let mut before = vec![];
    for voter in &voters {
        before.push(h.holdings_of(*voter).await);
    }
    let sweep = settle_open_disputes(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert_eq!(sweep.open, 1);
    assert_eq!(sweep.convictions_charged, 1);
    assert_eq!(sweep.seats_credited, voters.len());
    assert!(sweep.failures.is_empty());
    for (voter, before) in voters.iter().zip(before) {
        assert_eq!(
            h.holdings_of(*voter).await - before,
            bounty() / voters.len() as u64
        );
    }
    assert!(unsettled_disputes(&mut h.endpoint, &run)
        .await
        .unwrap()
        .is_empty());
    let sweep = settle_open_disputes(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert_eq!(sweep.open, 0);
    assert!(sweep.is_quiet());
    assert_bonds_add_up(&mut h).await;
}
