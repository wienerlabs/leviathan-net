use leviathan_indexer::live::read_run;
use leviathan_indexer::metrics::IndexerMetrics;
use leviathan_indexer::verdicts::status_label;
use leviathan_indexer::verdicts::VerdictLedger;
use leviathan_indexer::verdicts::VerdictTally;
use leviathan_indexer::verdicts::VERDICT_STATUSES;
use leviathan_telemetry::exposition::render;
use prometheus_client::registry::Registry;
use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_finalize_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_challenge_config;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use anchor_lang::error::ErrorCode as AnchorErrorCode;
use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use psyche_solana_treasurer::accounts::RunOpenChallengeAccounts;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::instruction::RunOpenChallenge;
use psyche_solana_treasurer::logic::RunOpenChallengeParams;
use psyche_solana_treasurer::state::ConvictionCharge;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::state::WithdrawalHold;
use psyche_solana_treasurer::state::WithdrawalStanding;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::instruction::Instruction;

use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

use super::signed_payload::replayed_digest;

use super::appeals_harness::*;

#[tokio::test]
pub async fn upheld_appeal_slashes_the_target() {
    let mut h = setup("Leviathan appeals uphold", 91).await;
    let target_key = h.target_key;

    h.reach_verifier_quorum().await;

    let target = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();

    let quorum = h.tie_breaker_quorum as usize;
    let appellates: Vec<usize> = h.tie_breakers.iter().take(quorum).map(|(i, _)| *i).collect();
    for client_idx in appellates {
        h.cast_appeal(client_idx, false).await.unwrap();
    }

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected),
        "an upheld appeal finalises the slash against the target",
    );

    h.settle_epoch().await;

    assert_eq!(
        h.slashed_of(&target_key).await,
        SLASHING_RATE,
        "the confirmed cheater forfeits its bond",
    );
    for (client_idx, _) in h.verifiers.clone().iter().take(h.verifier_quorum as usize) {
        let key = h.clients[*client_idx].pubkey();
        assert_eq!(
            h.slashed_of(&key).await,
            0,
            "verifiers on the winning side keep their bonds",
        );
    }

    let bench: Vec<usize> = h
        .tie_breakers
        .iter()
        .take(h.tie_breaker_quorum as usize)
        .map(|(client_idx, _)| *client_idx)
        .collect();
    process_treasurer_participant_bond_request_withdraw(
        &mut h.endpoint,
        &h.payer,
        &target,
        &h.run,
        BOND,
    )
    .await
    .unwrap();
    h.endpoint
        .forward_clock_unix_timestamp(100)
        .await
        .unwrap();
    h.finalize_target_withdrawal().await.unwrap();
    // The withdrawal charged the conviction and recorded what it owes the
    // bench that upheld it; each tie-breaker is credited by seat.
    let appeal_bounty = (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64;
    let appeal_share = appeal_bounty / bench.len() as u64;
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.bounty_owed, appeal_bounty);
    let bench_keys: Vec<Pubkey> = bench
        .iter()
        .map(|client_idx| h.clients[*client_idx].pubkey())
        .collect();
    assert_eq!(verdict.bounty_recipients(), bench_keys.as_slice());
    h.pay_every_seat_of(&target_key).await;
    for tie_breaker in &bench {
        assert_eq!(
            h.holdings_of(*tie_breaker).await,
            BOND + appeal_share,
            "an upheld appeal pays the tie-breakers out of the target's forfeited bond",
        );
    }
}

#[tokio::test]
pub async fn unchallenged_verdict_finalises_after_the_window() {
    let mut h = setup("Leviathan appeals finalise", 92).await;
    let target_key = h.target_key;
    let target_index = h.target_index;

    h.reach_verifier_quorum().await;

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy),
        "a pending slash does not eject before finalisation",
    );

    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .expect_err("finalising before the window elapses must be rejected");

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
        target_index,
    )
    .await
    .unwrap();

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected),
        "an unchallenged verdict finalises into a slash once the window closes",
    );

    h.settle_epoch().await;

    assert_eq!(
        h.slashed_of(&target_key).await,
        SLASHING_RATE,
        "the unchallenged target forfeits its bond",
    );
}

/// A verdict that turns out wrong is paid for out of the verifier's bond after
/// the overturn, so a bond already on its way out cannot back a vote
/// (wienerlabs/leviathan-net#34).
#[tokio::test]
pub async fn a_verifier_leaving_its_bond_cannot_vote() {
    let mut h = setup("Leviathan appeals leaving voter", 93).await;
    let (leaving, _) = h.verifiers[0];
    let (staying, _) = h.verifiers[1];

    h.request_withdraw(leaving, 1).await;
    assert_eq!(
        refusal_code(h.cast_verdict(leaving).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a verifier with a withdrawal pending cast a verdict its bond may not be there to answer for",
    );
    h.cast_verdict(staying).await.unwrap();
}

/// A tie-breaker sits on a bonded bench for the same reason, and is held to the
/// same bond (wienerlabs/leviathan-net#34).
#[tokio::test]
pub async fn a_tie_breaker_leaving_its_bond_cannot_judge_the_appeal() {
    let mut h = setup("Leviathan appeals leaving bench", 94).await;
    h.reach_verifier_quorum().await;
    let target = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();

    let (leaving, _) = h.tie_breakers[0];
    let (staying, _) = h.tie_breakers[1];
    h.request_withdraw(leaving, 1).await;
    assert_eq!(
        refusal_code(h.cast_appeal(leaving, true).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a tie-breaker with a withdrawal pending judged an appeal",
    );
    h.cast_appeal(staying, true).await.unwrap();
}

/// The target defends a conviction rather than taking on new work, and its bond
/// is still in the vault while the withdrawal waits, so a pending withdrawal
/// does not cost it the appeal (wienerlabs/leviathan-net#34).
#[tokio::test]
pub async fn a_target_with_a_pending_withdrawal_can_still_appeal() {
    let mut h = setup("Leviathan appeals pending target", 95).await;
    let target_idx = h.target_client_idx;
    h.request_withdraw(target_idx, 1).await;
    h.reach_verifier_quorum().await;

    let target = h.clients[target_idx].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
    let (tie_breaker, _) = h.tie_breakers[0];
    h.cast_appeal(tie_breaker, true).await.unwrap();
}

/// The challenger's slashed points are read from the run's own coordinator: a
/// challenge that names another run's coordinator account is refused.
#[tokio::test]
pub async fn a_challenge_against_another_runs_coordinator_is_refused() {
    let mut h = setup("Leviathan appeals foreign ledger", 96).await;
    h.reach_verifier_quorum().await;

    let other_account = h
        .endpoint
        .process_system_new_exempt(
            &h.payer,
            CoordinatorAccount::space_with_discriminator(),
            &psyche_solana_coordinator::ID,
        )
        .await
        .unwrap();
    process_treasurer_run_create(
        &mut h.endpoint,
        &h.payer,
        &h.collateral_mint,
        &other_account,
        RunCreateParams {
            index: 97,
            run_id: "Leviathan appeals other ledger".to_string(),
            main_authority: Keypair::new().pubkey(),
            join_authority: Keypair::new().pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    let target = h.clients[h.target_client_idx].insecure_clone();
    let instruction = Instruction {
        program_id: psyche_solana_treasurer::ID,
        accounts: RunOpenChallengeAccounts {
            challenger: target.pubkey(),
            challenger_participant: find_participant(&h.run, &target.pubkey()),
            run: h.run,
            coordinator_account: other_account,
            audit_verdict: find_audit_verdict(&h.run, &target.pubkey()),
        }
        .to_account_metas(None),
        data: RunOpenChallenge {
            params: RunOpenChallengeParams {},
        }
        .data(),
    };
    assert_eq!(
        refusal_code(
            h.endpoint
                .process_instruction_with_signers(&h.payer, instruction, &[&target])
                .await
        ),
        u32::from(AnchorErrorCode::ConstraintRaw),
    );
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
}


/// While a conviction is inside its challenge window it can still be
/// overturned or finalised, so the target's withdrawal is held, however long
/// ago its delay elapsed. Past the window, in an epoch that will be charged, it
/// waits for that epoch, where the conviction lands through the ledger and
/// settlement forfeits it (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_target_cannot_leave_while_its_conviction_is_pending() {
    let mut h = setup("Leviathan appeals pending leave", 98).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target_idx, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    h.reach_verifier_quorum().await;
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);

    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::ConvictionPending)
    );
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::ConvictionPending),
        "a target left with its whole bond while its conviction was pending"
    );

    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled)
    );
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::EpochNotSettled),
    );

    let target_index = h.roster_index_of(&target_key).await;
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .unwrap();
    h.settle_epoch().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: false }
    );
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
}

/// A target that walks away from its conviction (its epoch ends before anyone
/// finalises it, and it never comes back) cannot be ejected into any epoch the
/// ledger will charge. Its withdrawal settles the conviction against the bond
/// itself, records the verdict as upheld, so it is neither lost nor allowed to
/// hold the bond forever (wienerlabs/leviathan-net#36), and records what it
/// owes the convicting verifiers, who are credited by seat
/// (wienerlabs/leviathan-net#64).
#[tokio::test]
pub async fn a_conviction_the_target_walks_away_from_is_settled_at_withdrawal() {
    let mut h = setup("Leviathan appeals walked away", 102).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    h.settle_epoch().await;
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    assert_eq!(h.slashed_of(&target_key).await, 0);
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: true }
    );

    let voters = h.quorum_voters();
    let mut voters_before = vec![];
    for voter in &voters {
        voters_before.push(h.holdings_of(*voter).await);
    }
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
    let bounty = (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64;
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.charge, ConvictionCharge::Charged);
    assert_eq!(verdict.bounty_owed, bounty);
    h.pay_every_seat_of(&target_key).await;
    let share = bounty / voters.len() as u64;
    for (voter, before) in voters.iter().zip(voters_before) {
        assert_eq!(h.holdings_of(*voter).await - before, share);
    }

    // Settled once: the verdict cannot be finalised against the target again.
    assert_eq!(
        refusal_code(
            process_treasurer_run_finalize_slash(
                &mut h.endpoint,
                &h.payer,
                &h.run,
                &h.coordinator_account,
                &h.run_id,
                &target_key,
                0,
            )
            .await
        ),
        treasurer_refusal(ProgramError::VerdictNotPending),
    );
}

/// An appeal holds the bond for as long as the bench may still rule, across
/// the epoch boundary too. Once its window has passed with no ruling, the
/// conviction the verifiers reached stands, and lands through the ledger in
/// the epoch the target comes back for (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_challenged_target_cannot_leave_until_the_appeal_resolves() {
    let mut h = setup("Leviathan appeals open leave", 99).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    let target = h.clients[target_idx].insecure_clone();
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();

    h.settle_epoch().await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::ConvictionPending)
    );
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::ConvictionPending),
        "a target left with its whole bond while its appeal was open"
    );

    h.endpoint
        .forward_clock_unix_timestamp(APPEAL_WINDOW as u64)
        .await
        .unwrap();
    h.rejoin_all_and_start_epoch().await;
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled)
    );
    let target_index = h.roster_index_of(&target_key).await;
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .unwrap();
    assert_eq!(h.client_state(&target_key).await, Some(ClientState::Ejected));
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::EpochNotSettled),
    );
    h.settle_epoch().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
}

/// An innocent target whose appeal overturns the conviction is not held at
/// all: it leaves with its whole bond (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn an_overturned_target_leaves_with_its_whole_bond() {
    let mut h = setup("Leviathan appeals cleared leave", 100).await;
    let target_idx = h.target_client_idx;
    let target = h.clients[target_idx].insecure_clone();
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
    let quorum = h.tie_breaker_quorum as usize;
    let bench: Vec<usize> = h.tie_breakers.iter().take(quorum).map(|(i, _)| *i).collect();
    for client_idx in bench {
        h.cast_appeal(client_idx, true).await.unwrap();
    }

    h.settle_epoch().await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: false }
    );
    let before = h.collateral_of(target_idx).await;
    h.try_finalize_withdraw(target_idx).await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND);
}

/// A conviction finalised between epochs used to be ejected onto the ended
/// epoch's roster and never charged. It is recorded as due instead, and the
/// target's withdrawal settles it against the bond (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_conviction_finalised_after_its_epoch_still_reaches_the_bond() {
    let mut h = setup("Leviathan appeals late finalise", 101).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.client_state(&target_key).await, Some(ClientState::Healthy));
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
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy),
        "the target was ejected onto a roster nobody will charge"
    );
    // Nothing more can land between epochs. Sent by another payer, so it is a
    // transaction of its own and not a replay of the first.
    let keeper = Keypair::new();
    h.endpoint
        .request_airdrop(&keeper.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    assert_eq!(
        refusal_code(
            process_treasurer_run_finalize_slash(
                &mut h.endpoint,
                &keeper,
                &h.run,
                &h.coordinator_account,
                &h.run_id,
                &target_key,
                h.target_index,
            )
            .await
        ),
        treasurer_refusal(ProgramError::TargetNotInChargingEpoch),
    );
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: true }
    );

    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
}

/// The bench sits whenever its seats are drawn, between epochs too, so an
/// innocent target's appeal window is not spent waiting for an epoch: a bench
/// that overturns between epochs clears it, and it leaves with its whole bond
/// (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn the_bench_can_overturn_between_epochs() {
    let mut h = setup("Leviathan appeals late overturn", 103).await;
    h.advance_to_cooldown_with_witnesses().await;
    let committee = h.committee_now().await;
    let (target, target_index) = committee.trainers[0];
    h.request_withdraw(target, BOND).await;
    for (voter, _) in committee.verifiers.iter().take(committee.verifier_quorum) {
        h.convict_by(*voter, target, target_index).await.unwrap();
    }
    let target_wallet = h.clients[target].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target_wallet, &h.run)
        .await
        .unwrap();
    h.finish_epoch_with_witnesses().await;

    for (appellate, _) in committee.tie_breakers.iter().take(committee.tie_breaker_quorum) {
        h.rule_by(*appellate, target, target_index, true).await.unwrap();
    }
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Overturned);

    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    let before = h.collateral_of(target).await;
    h.try_finalize_withdraw(target).await.unwrap();
    assert_eq!(h.collateral_of(target).await - before, BOND);
}

/// A bench that upholds between epochs cannot eject the target into an epoch
/// that will charge it, so the conviction is recorded as due rather than lost
/// on the ended epoch's roster, and the target's withdrawal settles it
/// (wienerlabs/leviathan-net#36), owing the tie-breakers who ruled, who are
/// credited by seat (wienerlabs/leviathan-net#64).
#[tokio::test]
pub async fn an_appeal_upheld_between_epochs_settles_at_withdrawal() {
    let mut h = setup("Leviathan appeals late uphold", 106).await;
    h.advance_to_cooldown_with_witnesses().await;
    let committee = h.committee_now().await;
    let (target, target_index) = committee.trainers[0];
    let target_key = h.clients[target].pubkey();
    h.request_withdraw(target, BOND).await;
    for (voter, _) in committee.verifiers.iter().take(committee.verifier_quorum) {
        h.convict_by(*voter, target, target_index).await.unwrap();
    }
    let target_wallet = h.clients[target].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target_wallet, &h.run)
        .await
        .unwrap();
    h.finish_epoch_with_witnesses().await;

    let bench: Vec<usize> = committee
        .tie_breakers
        .iter()
        .take(committee.tie_breaker_quorum)
        .map(|(i, _)| *i)
        .collect();
    for appellate in &bench {
        h.rule_by(*appellate, target, target_index, false).await.unwrap();
    }
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Due);
    assert_eq!(h.client_state(&target_key).await, Some(ClientState::Healthy));

    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        h.observed_standing(target).await,
        WithdrawalStanding::Free { settle_conviction: true }
    );
    let mut bench_before = vec![];
    for appellate in &bench {
        bench_before.push(h.holdings_of(*appellate).await);
    }
    let before = h.collateral_of(target).await;
    h.try_finalize_withdraw(target).await.unwrap();
    assert_eq!(h.collateral_of(target).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Upheld);
    h.pay_every_seat_of(&target_key).await;
    let share =
        (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64 / bench.len() as u64;
    for (appellate, before) in bench.iter().zip(bench_before) {
        assert_eq!(h.holdings_of(*appellate).await - before, share);
    }
}

/// With no challenge window a quorum convicts at once. Reached between epochs
/// it cannot eject the target into a charged epoch, so it is recorded as due
/// and settled at the target's withdrawal (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_verdict_quorum_reached_between_epochs_is_due() {
    let mut h = setup("Leviathan appeals late quorum", 107).await;
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_set_challenge_config(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        0,
        0,
        0,
    )
    .await
    .unwrap();
    h.finish_epoch_with_witnesses().await;
    let committee = h.committee_now().await;
    let (target, target_index) = committee.trainers[0];
    let target_key = h.clients[target].pubkey();
    h.request_withdraw(target, BOND).await;
    let voters: Vec<usize> = committee
        .verifiers
        .iter()
        .take(committee.verifier_quorum)
        .map(|(i, _)| *i)
        .collect();
    for voter in &voters {
        h.convict_by(*voter, target, target_index).await.unwrap();
    }
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Due);
    assert_eq!(h.client_state(&target_key).await, Some(ClientState::Healthy));

    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    let mut voters_before = vec![];
    for voter in &voters {
        voters_before.push(h.holdings_of(*voter).await);
    }
    let before = h.collateral_of(target).await;
    h.try_finalize_withdraw(target).await.unwrap();
    assert_eq!(h.collateral_of(target).await - before, BOND - SLASHING_RATE);
    h.pay_every_seat_of(&target_key).await;
    let share = (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64
        / voters.len() as u64;
    for (voter, before) in voters.iter().zip(voters_before) {
        assert_eq!(h.holdings_of(*voter).await - before, share);
    }
}

/// A conviction finalised in a warmup is not ejected, because pausing a warmup
/// skips the charge, so a pause there cannot shed it: it stays due and the
/// target's withdrawal from the paused run settles it
/// (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_paused_warmup_cannot_shed_a_conviction() {
    let mut h = setup("Leviathan appeals paused warmup", 108).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    h.settle_epoch().await;
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();

    h.rejoin_all().await;
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled)
    );
    let target_index = h.roster_index_of(&target_key).await;
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .unwrap();
    assert_eq!(h.verdict_status().await, VerdictStatus::Due);
    assert_eq!(h.client_state(&target_key).await, Some(ClientState::Healthy));

    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_update(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        &h.coordinator_instance,
        &h.coordinator_account,
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
    assert_eq!(h.run_state().await, RunState::Paused);
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: true }
    );
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
}

/// With appeals on, an appeal must have a deadline: one without could hold a
/// target's bond for as long as no bench rules, and once the target or the
/// run is gone none ever can (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_challenge_window_needs_an_appeal_deadline() {
    let mut h = setup("Leviathan appeals no deadline", 105).await;
    let main_authority = h.main_authority.insecure_clone();
    assert_eq!(
        refusal_code(
            process_treasurer_run_set_challenge_config(
                &mut h.endpoint,
                &h.payer,
                &main_authority,
                &h.run,
                CHALLENGE_WINDOW,
                TIE_BREAKER_SIZE,
                0,
            )
            .await
        ),
        treasurer_refusal(ProgramError::AppealWindowRequired),
    );
    process_treasurer_run_set_challenge_config(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        0,
        0,
        0,
    )
    .await
    .unwrap();
}

/// A seat is read from a voter's index in the roster now, and drawn over the
/// roster as it stood when the round began. In a warmup the roster is already
/// the next epoch's while the round is still the last one, so the two do not
/// agree and no vote is taken (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn no_vote_is_taken_in_a_warmup() {
    let mut h = setup("Leviathan appeals warmup vote", 109).await;
    h.settle_epoch().await;
    h.rejoin_all().await;
    let (voter, _) = h.verifiers[0];
    assert_eq!(
        refusal_code(h.cast_verdict(voter).await),
        treasurer_refusal(ProgramError::CommitteeNotDrawnForRoster),
    );
}

/// The end of `Cooldown` compacts the roster, so once someone has left since
/// the last round was drawn, every index after theirs points at a different
/// client than the seat it was drawn for, and no vote is taken until a round
/// draws a committee over the roster again (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn no_vote_is_taken_once_the_roster_has_moved_under_the_committee() {
    let mut h = setup("Leviathan appeals moved roster", 110).await;
    h.advance_to_cooldown_with_witnesses().await;
    let committee = h.committee_now().await;
    let (leaver, leaver_index) = committee.trainers[0];
    let (target, _) = committee.trainers[1];
    // Both signed for the batches they trained in the round just ended.
    let leaver_commitment = h.signed_payload_of(leaver).await;
    let target_commitment = h.signed_payload_of(target).await;
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_slash(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        leaver_index,
        &leaver_commitment,
        replayed_digest(),
    )
    .await
    .unwrap();
    h.finish_epoch_with_witnesses().await;
    let leaver_key = h.clients[leaver].pubkey();
    assert_eq!(h.client_state(&leaver_key).await, None, "the roster was not compacted");

    let target_index = h.roster_index_of(&h.clients[target].pubkey()).await;
    let (voter, _) = committee.verifiers[0];
    assert_eq!(
        refusal_code(
            h.convict_on(voter, target, target_index, &target_commitment)
                .await
        ),
        treasurer_refusal(ProgramError::CommitteeNotDrawnForRoster),
    );
}

/// What `leviathan-indexer` sees of one verdict account, and what it exports.
async fn indexer_look(
    h: &mut AppealsHarness,
    ledger: &mut VerdictLedger,
    metrics: &IndexerMetrics,
) -> VerdictTally {
    let run_id = h.run_id.clone();
    let observation = read_run(&mut h.endpoint, None, &run_id)
        .await
        .unwrap()
        .observe(ledger);
    metrics.observe(&run_id, &observation);
    observation
        .treasurer
        .expect("the appeals run is treasurer-managed")
        .verdicts
}

/// The indexer follows a dispute from its first vote to its overturn and
/// counts the conviction and the overturn once each, however many polls see
/// them: the overturn is the chain's record of an innocent target convicted,
/// which the honest-false-positive kill switch fires on
/// (wienerlabs/leviathan-net#57).
#[tokio::test]
pub async fn the_indexer_counts_a_conviction_and_its_overturn_once() {
    let mut h = setup("Leviathan indexer appeal", 111).await;
    let mut ledger = VerdictLedger::default();
    let mut registry = Registry::default();
    let metrics = IndexerMetrics::register(&mut registry);
    let status = |name: &str| {
        VERDICT_STATUSES
            .iter()
            .position(|s| status_label(*s) == name)
            .unwrap()
    };

    let quiet = indexer_look(&mut h, &mut ledger, &metrics).await;
    assert_eq!(quiet, VerdictTally::default());

    // One vote opens the dispute; it is open, not lapsed, in its own round.
    let voters: Vec<usize> = h
        .verifiers
        .iter()
        .take(h.verifier_quorum as usize)
        .map(|(i, _)| *i)
        .collect();
    h.cast_verdict(voters[0]).await.unwrap();
    let open = indexer_look(&mut h, &mut ledger, &metrics).await;
    assert_eq!(open.by_status[status("voting")], 1);
    assert_eq!((open.new_convictions, open.new_lapsed), (0, 0));

    // The rest of the quorum convicts.
    for voter in &voters[1..] {
        h.cast_verdict(*voter).await.unwrap();
    }
    let convicted = indexer_look(&mut h, &mut ledger, &metrics).await;
    assert_eq!(convicted.by_status[status("slash_pending")], 1);
    assert_eq!(convicted.new_convictions, 1);
    assert_eq!(
        indexer_look(&mut h, &mut ledger, &metrics)
            .await
            .new_convictions,
        0,
        "a conviction standing over two polls is one conviction"
    );

    // The target challenges, and the tie-breakers overturn.
    let target = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(
        &mut h.endpoint,
        &h.payer,
        &target,
        &h.run,
    )
    .await
    .unwrap();
    let challenged = indexer_look(&mut h, &mut ledger, &metrics).await;
    assert_eq!(challenged.by_status[status("challenged")], 1);
    assert_eq!(
        (challenged.new_convictions, challenged.new_overturned),
        (0, 0)
    );
    let appellates: Vec<usize> = h
        .tie_breakers
        .iter()
        .take(h.tie_breaker_quorum as usize)
        .map(|(i, _)| *i)
        .collect();
    for appellate in appellates {
        h.cast_appeal(appellate, true).await.unwrap();
    }
    let overturned = indexer_look(&mut h, &mut ledger, &metrics).await;
    assert_eq!(overturned.by_status[status("overturned")], 1);
    assert_eq!(
        (overturned.new_convictions, overturned.new_overturned),
        (0, 1)
    );
    assert_eq!(
        indexer_look(&mut h, &mut ledger, &metrics)
            .await
            .new_overturned,
        0
    );

    let scrape = render(&registry).unwrap();
    for line in [
        r#"leviathan_run_convictions_total{run_id="Leviathan indexer appeal"} 1"#,
        r#"leviathan_run_overturned_convictions_total{run_id="Leviathan indexer appeal"} 1"#,
        r#"leviathan_run_lapsed_disputes_total{run_id="Leviathan indexer appeal"} 0"#,
        r#"leviathan_run_verdicts{run_id="Leviathan indexer appeal",status="overturned"} 1.0"#,
    ] {
        assert!(scrape.contains(line), "missing {line} in\n{scrape}");
    }
}

/// Runs the swarm on, round by round and into later epochs, until `seated`
/// finds what it looks for in the committee drawn for the round in force.
async fn advance_until_drawn<T>(
    h: &mut AppealsHarness,
    seated: impl Fn(&CommitteeNow) -> Option<T>,
) -> T {
    for _ in 0..120 {
        h.advance_to_next_round_across_epochs().await;
        if let Some(found) = seated(&h.committee_now().await) {
            return found;
        }
    }
    panic!("no round in 120 drew the committee looked for");
}

/// The bench is drawn afresh every round and an appeal stays open over many of
/// them, so the target of an appeal can be drawn onto its own bench. Seated,
/// it would vote to clear itself: with a one-seat bench its overturn alone is
/// the quorum, the verifiers who convicted it would forfeit their bonds to the
/// crank, and the bench's share of those forfeits would be credited to the
/// target's own bond. A target never judges its own appeal, and the bench
/// drawn after it still rules.
#[tokio::test]
pub async fn a_target_drawn_onto_its_own_bench_cannot_judge_its_appeal() {
    let mut h = setup("Leviathan appeals self judge", 167).await;
    // A one-seat bench, so one overturn is the whole quorum.
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_set_challenge_config(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        CHALLENGE_WINDOW,
        1,
        APPEAL_WINDOW,
    )
    .await
    .unwrap();
    // The round under way keeps the bench its clients drew; the next one
    // seats the new count.
    h.advance_to_next_round_across_epochs().await;

    let committee = h.committee_now().await;
    assert_eq!(committee.tie_breaker_quorum, 1);
    let (target, target_index) = committee.trainers[0];
    let target_key = h.clients[target].pubkey();
    let voters: Vec<usize> = committee
        .verifiers
        .iter()
        .take(committee.verifier_quorum)
        .map(|(i, _)| *i)
        .collect();
    for voter in &voters {
        h.convict_by(*voter, target, target_index).await.unwrap();
    }
    assert_eq!(
        h.verdict_status_of(target).await,
        VerdictStatus::SlashPending
    );
    let target_wallet = h.clients[target].insecure_clone();
    process_treasurer_run_open_challenge(
        &mut h.endpoint,
        &h.payer,
        &target_wallet,
        &h.run,
    )
    .await
    .unwrap();

    // A later round seats the target on the bench of its own appeal.
    let target_index = advance_until_drawn(&mut h, |committee| {
        committee
            .tie_breakers
            .iter()
            .any(|(client_idx, _)| *client_idx == target)
            .then(|| {
                committee
                    .tie_breakers
                    .iter()
                    .chain(&committee.verifiers)
                    .chain(&committee.trainers)
                    .find(|(client_idx, _)| *client_idx == target)
                    .map(|(_, index)| *index)
                    .unwrap()
            })
    })
    .await;
    assert_eq!(target_index, h.roster_index_of(&target_key).await);
    assert_eq!(
        refusal_code(h.rule_by(target, target, target_index, true).await),
        treasurer_refusal(ProgramError::AppellateIsTarget),
    );
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Challenged);
    assert!(verdict.appeal_voters.is_empty());
    assert_eq!((verdict.overturn_count, verdict.uphold_count), (0, 0));

    // The appeal is not stuck on it: the next neutral bench rules.
    let appellate = advance_until_drawn(&mut h, |committee| {
        committee
            .tie_breakers
            .iter()
            .map(|(client_idx, _)| *client_idx)
            .find(|client_idx| {
                *client_idx != target && !voters.contains(client_idx)
            })
    })
    .await;
    let target_index = h.roster_index_of(&target_key).await;
    h.rule_by(appellate, target, target_index, false)
        .await
        .unwrap();
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.status, VerdictStatus::Upheld);
    assert_eq!(verdict.appeal_voters, vec![h.clients[appellate].pubkey()]);
}
