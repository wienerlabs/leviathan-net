//! A verifier's vote can cost it its bond until the verdict it voted on can no
//! longer be overturned, and once it is overturned, until the crank has
//! charged it (wienerlabs/leviathan-net#39).

use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_finalize_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_challenge_config;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_tooling::verdict_votes::run_pay_bench_bounty_instruction;
use psyche_solana_tooling::verdict_votes::settle_previous_dispute;
use psyche_solana_tooling::verdict_votes::DisputeSettlement;
use psyche_solana_treasurer::accounts::RunSlashLosingVerifierAccounts;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::instruction::RunSlashLosingVerifier;
use psyche_solana_treasurer::logic::RunSlashLosingVerifierParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::state::WithdrawalHold;
use psyche_solana_treasurer::state::WithdrawalStanding;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::instruction::Instruction;
use solana_sdk::signer::Signer;

use super::appeals_harness::*;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

/// What the bench earns from one losing verifier's forfeit, per tie-breaker.
fn bench_share(bench: usize) -> u64 {
    (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64 / bench as u64
}

/// The bench overturns, and the crank charges each losing verifier against its
/// bond, recording what each forfeit owes the bench. The losers keep their
/// seats: the charge never went through the coordinator's ledger. Once every
/// loser is charged, each tie-breaker's bond is credited its share, once.
#[tokio::test]
pub async fn an_overturn_charges_each_losing_verifier_and_credits_the_bench() {
    let mut h = setup("Leviathan liability overturn", 122).await;
    let target_key = h.target_key;
    h.convict_and_overturn().await;
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );

    let losers = h.quorum_voters();
    let bench = h.bench();
    let mut bench_before = vec![];
    for tie_breaker in &bench {
        bench_before.push(h.participant_of(*tie_breaker).await.bond_amount);
    }
    let bonded_before = get_run(&mut h.endpoint, &h.run)
        .await
        .unwrap()
        .unwrap()
        .total_bonded_amount;

    for (position, loser) in losers.iter().enumerate() {
        assert_eq!(h.participant_of(*loser).await.open_verdict_votes, 1);
        h.crank_loser(position as u16).await.unwrap();
        let participant = h.participant_of(*loser).await;
        assert_eq!(participant.bond_amount, BOND - SLASHING_RATE);
        assert_eq!(participant.open_verdict_votes, 0);
        // A charge the ledger never sees is not a settled point.
        assert_eq!(participant.bond_settled_slashed_points, 0);
        let loser_key = h.clients[*loser].pubkey();
        assert_eq!(
            h.client_state(&loser_key).await,
            Some(ClientState::Healthy)
        );
        assert_eq!(h.slashed_of(&loser_key).await, 0);
    }
    assert!(
        h.crank_loser(losers.len() as u16).await.is_err(),
        "cranking past the last loser must be refused",
    );

    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(
        verdict.bench_bounty_owed,
        bench_share(1) * losers.len() as u64,
        "each forfeit's bounty is owed to the bench",
    );
    assert!(
        !verdict.is_settled(),
        "an overturn whose bench is unpaid is not taken over"
    );
    for (seat, (tie_breaker, before)) in
        bench.iter().zip(bench_before).enumerate()
    {
        h.pay_bench_seat(seat as u16).await.unwrap();
        assert_eq!(
            h.participant_of(*tie_breaker).await.bond_amount - before,
            bench_share(bench.len()) * losers.len() as u64,
            "each tie-breaker's bond is credited its share of every forfeit",
        );
    }
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    assert_eq!(
        refusal_code(h.pay_bench_seat(0).await),
        treasurer_refusal(ProgramError::BenchSeatPaid),
    );
    // The forfeits left the bonded total, and the bench's shares came back
    // into it as the tie-breakers' bonds.
    let credited =
        bench_share(bench.len()) * losers.len() as u64 * bench.len() as u64;
    assert_eq!(
        get_run(&mut h.endpoint, &h.run)
            .await
            .unwrap()
            .unwrap()
            .total_bonded_amount,
        bonded_before - SLASHING_RATE * losers.len() as u64 + credited,
    );

    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.settled_count as usize, losers.len());
    assert!(
        verdict.is_settled(),
        "a fully cranked overturn may be taken over"
    );

    // The loser's withdrawal is no longer held by its vote, only by the epoch
    // it still trains in.
    h.request_withdraw(losers[0], BOND - SLASHING_RATE).await;
    assert_eq!(
        h.observed_standing(losers[0]).await,
        WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled),
    );
}

/// The bench overturns between epochs, where nothing the loser does is charged
/// any more. The losing verifier asks for its whole bond and tries to leave
/// before the crank reaches it, and is refused. The crank then settles it, the
/// bench is paid, and the loser leaves with what is left.
#[tokio::test]
pub async fn a_losing_verifier_cannot_leave_before_the_crank() {
    let mut h = setup("Leviathan liability loser", 120).await;
    h.convict_and_overturn().await;
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.run_state().await, RunState::WaitingForMembers);

    let loser = h.quorum_voters()[0];
    h.request_withdraw(loser, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        h.observed_standing(loser).await,
        WithdrawalStanding::Held(WithdrawalHold::VotesAtStake),
    );
    let before = h.collateral_of(loser).await;
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(loser).await),
        treasurer_refusal(ProgramError::VerdictVotesOpen),
    );
    assert_eq!(h.collateral_of(loser).await, before);

    // The bench is paid once every loser is charged, and not before.
    assert_eq!(
        refusal_code(h.pay_bench_seat(0).await),
        treasurer_refusal(ProgramError::LosersNotAllSettled),
    );
    let losers = h.quorum_voters();
    for position in 0..losers.len() {
        h.crank_loser(position as u16).await.unwrap();
    }
    let bench = h.bench();
    for (seat, tie_breaker) in bench.iter().enumerate() {
        let before = h.participant_of(*tie_breaker).await.bond_amount;
        h.pay_bench_seat(seat as u16).await.unwrap();
        assert_eq!(
            h.participant_of(*tie_breaker).await.bond_amount - before,
            bench_share(bench.len()) * losers.len() as u64,
        );
    }
    // The request for the whole bond is cut to what the forfeit left.
    assert_eq!(
        h.participant_of(loser).await.bond_withdraw_pending_amount,
        BOND - SLASHING_RATE,
    );

    h.try_finalize_withdraw(loser).await.unwrap();
    assert_eq!(h.collateral_of(loser).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.participant_of(loser).await.bond_amount, 0);
}

/// A verifier whose verdict is inside its challenge window, or under appeal,
/// has not lost yet but may: it cannot leave, and its vote cannot be released.
#[tokio::test]
pub async fn a_verifier_cannot_leave_while_its_verdict_is_under_appeal() {
    let mut h = setup("Leviathan liability appeal", 121).await;
    let target = h.clients[h.target_client_idx].insecure_clone();
    let target_key = h.target_key;
    h.reach_verifier_quorum().await;
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);
    let voter = h.quorum_voters()[0];
    assert_eq!(
        refusal_code(h.release_vote(Some(voter), &target_key, voter).await),
        treasurer_refusal(ProgramError::VoteStillAtStake),
    );

    process_treasurer_run_open_challenge(
        &mut h.endpoint,
        &h.payer,
        &target,
        &h.run,
    )
    .await
    .unwrap();
    h.finish_epoch_with_witnesses().await;
    assert_eq!(
        refusal_code(h.release_vote(Some(voter), &target_key, voter).await),
        treasurer_refusal(ProgramError::VoteStillAtStake),
    );

    h.request_withdraw(voter, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    let before = h.collateral_of(voter).await;
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(voter).await),
        treasurer_refusal(ProgramError::VerdictVotesOpen),
    );
    assert_eq!(h.collateral_of(voter).await, before);
}

/// An overturned vote cannot be released: only the crank closes it, and it
/// charges the bond as it does.
#[tokio::test]
pub async fn a_losing_vote_is_closed_only_by_the_crank() {
    let mut h = setup("Leviathan liability no release", 123).await;
    let target_key = h.target_key;
    h.convict_and_overturn().await;
    let loser = h.quorum_voters()[0];
    assert_eq!(
        refusal_code(h.release_vote(Some(loser), &target_key, loser).await),
        treasurer_refusal(ProgramError::VoteStillAtStake),
    );
    h.crank_loser(0).await.unwrap();
    assert_eq!(
        refusal_code(h.release_vote(Some(loser), &target_key, loser).await),
        treasurer_refusal(ProgramError::VoteAlreadyReleased),
    );
}

/// A run that halts with an overturn uncranked does not strand the loser: the
/// crank no longer needs it on a roster that will be charged.
#[tokio::test]
pub async fn a_loser_in_a_halted_run_is_still_settled() {
    let mut h = setup("Leviathan liability halted", 124).await;
    h.convict_and_overturn().await;
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
    h.settle_epoch().await;
    assert_eq!(h.run_state().await, RunState::Paused);

    let loser = h.quorum_voters()[0];
    h.request_withdraw(loser, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(loser).await),
        treasurer_refusal(ProgramError::VerdictVotesOpen),
    );
    h.crank_loser(0).await.unwrap();
    let before = h.collateral_of(loser).await;
    h.try_finalize_withdraw(loser).await.unwrap();
    assert_eq!(h.collateral_of(loser).await - before, BOND - SLASHING_RATE);
}

/// A verifier whose conviction stands was right, but its vote holds its bond
/// until it is released. Anyone may release it, once.
#[tokio::test]
pub async fn an_honest_voter_is_held_until_its_vote_is_released() {
    let mut h = setup("Leviathan liability upheld", 125).await;
    let target_key = h.target_key;
    let target_index = h.target_index;
    h.reach_verifier_quorum().await;
    h.endpoint
        .forward_clock_unix_timestamp(CHALLENGE_WINDOW as u64)
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
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
    h.finish_epoch_with_witnesses().await;

    let voter = h.quorum_voters()[0];
    h.request_withdraw(voter, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(voter).await),
        treasurer_refusal(ProgramError::VerdictVotesOpen),
    );

    // A key that did not vote has nothing to release.
    let stranger = h.target_client_idx;
    assert_eq!(
        refusal_code(h.release_vote(None, &target_key, stranger).await),
        treasurer_refusal(ProgramError::NotAVoter),
    );
    h.release_vote(None, &target_key, voter).await.unwrap();
    assert_eq!(h.participant_of(voter).await.open_verdict_votes, 0);
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    assert_eq!(
        refusal_code(h.release_vote(None, &target_key, voter).await),
        treasurer_refusal(ProgramError::VoteAlreadyReleased),
    );

    let before = h.collateral_of(voter).await;
    h.try_finalize_withdraw(voter).await.unwrap();
    assert_eq!(h.collateral_of(voter).await - before, BOND);
}

/// A vote on a dispute that has not reached quorum may be withdrawn by its
/// voter at any time, and by anyone once the round is over, but not by a
/// stranger while the round is under way. A withdrawn vote leaves the count.
#[tokio::test]
pub async fn a_vote_before_quorum_is_withdrawn_by_its_voter_or_after_its_round()
{
    let mut h = setup("Leviathan liability lapse", 126).await;
    let target_key = h.target_key;
    let voter = h.quorum_voters()[0];
    h.cast_verdict(voter).await.unwrap();
    assert_eq!(h.verdict_status().await, VerdictStatus::Voting);
    assert_eq!(h.participant_of(voter).await.open_verdict_votes, 1);

    assert_eq!(
        refusal_code(h.release_vote(None, &target_key, voter).await),
        treasurer_refusal(ProgramError::VoteStillAtStake),
    );
    h.release_vote(Some(voter), &target_key, voter)
        .await
        .unwrap();
    let verdict = h.verdict_of(&target_key).await;
    assert!(verdict.voters.is_empty());
    assert_eq!(verdict.verdict_count, 0);
    assert_eq!(h.participant_of(voter).await.open_verdict_votes, 0);

    // A withdrawn vote no longer counts toward the quorum: with it gone, the
    // quorum still needs as many votes as before.
    let voters = h.quorum_voters();
    for other in voters.iter().skip(1) {
        h.cast_verdict(*other).await.unwrap();
    }
    assert_eq!(
        h.verdict_status().await,
        VerdictStatus::Voting,
        "the withdrawn vote was counted toward the quorum",
    );
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.verdict_count as usize, voters.len() - 1);

    // It may vote again, and once the round is over anyone may withdraw it.
    // The clock moves so the vote is not the same transaction as the first.
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    h.cast_verdict(voter).await.unwrap();
    assert_eq!(
        h.verdict_status().await,
        VerdictStatus::SlashPending,
        "its vote counts again once cast again",
    );
    // Quorum was reached, so this vote now waits for the challenge window.
    h.advance_to_next_round().await;
    assert_eq!(
        refusal_code(h.release_vote(None, &target_key, voter).await),
        treasurer_refusal(ProgramError::VoteStillAtStake),
    );
}

/// A vote on a dispute whose round ended without a quorum is withdrawn by
/// anyone, and comes off the verdict.
#[tokio::test]
pub async fn a_lapsed_vote_is_withdrawn_by_anyone() {
    let mut h = setup("Leviathan liability lapsed", 132).await;
    let target_key = h.target_key;
    let voter = h.quorum_voters()[0];
    h.cast_verdict(voter).await.unwrap();
    h.advance_to_next_round().await;
    h.release_vote(None, &target_key, voter).await.unwrap();
    assert_eq!(h.participant_of(voter).await.open_verdict_votes, 0);
    assert!(h.verdict_of(&target_key).await.voters.is_empty());
}

/// One verdict account holds one dispute against a target. A dispute that
/// lapsed with votes on it is taken over only once those votes are withdrawn,
/// which anyone may do: the verifier daemon does it before it votes.
#[tokio::test]
pub async fn a_lapsed_dispute_is_taken_over_once_its_votes_are_withdrawn() {
    let mut h = setup("Leviathan liability takeover", 127).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    let first_voter = h.quorum_voters()[0];
    h.cast_verdict(first_voter).await.unwrap();
    let round = h.verdict_of(&target_key).await.round_height;

    // A later round in which the target trains and a verifier can judge it.
    h.advance_to_next_round().await;
    let (voter, target_index) = loop {
        let committee = h.committee_now().await;
        let trains = committee
            .trainers
            .iter()
            .find(|(client_idx, _)| *client_idx == target)
            .map(|(_, index)| *index);
        match (trains, committee.verifiers.first()) {
            (Some(index), Some((voter, _))) => break (*voter, index),
            _ => h.advance_to_next_round().await,
        }
    };
    let commitment = h.signed_payload_of(target).await;

    assert_eq!(
        refusal_code(
            h.convict_on(voter, target, target_index, &commitment).await
        ),
        treasurer_refusal(ProgramError::PreviousVerdictUnsettled),
    );
    let payer = h.payer.insecure_clone();
    let settled = settle_previous_dispute(
        &mut h.endpoint,
        &payer,
        &h.run,
        &h.coordinator_account,
        &target_key,
    )
    .await
    .unwrap();
    assert_eq!(
        settled,
        DisputeSettlement {
            votes_freed: 1,
            ..DisputeSettlement::default()
        }
    );
    assert_eq!(h.participant_of(first_voter).await.open_verdict_votes, 0);

    h.convict_on(voter, target, target_index, &commitment)
        .await
        .unwrap();
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.voters, vec![h.clients[voter].pubkey()]);
    assert_ne!(verdict.round_height, round);
}

/// The crank charges the loser the verdict records at its position, and the
/// bench's share is credited only to the tie-breaker the verdict records at its
/// seat: neither can be redirected.
#[tokio::test]
pub async fn the_bench_is_credited_only_to_its_recorded_seats() {
    let mut h = setup("Leviathan liability recipients", 128).await;
    let target_key = h.target_key;
    h.convict_and_overturn().await;
    let losers = h.quorum_voters();
    let loser = losers[0];
    let bench = h.bench();
    let payer = h.payer.insecure_clone();

    // Naming another verifier for the position is refused.
    let other = h.clients[bench[0]].pubkey();
    let wrong_loser = Instruction {
        accounts: RunSlashLosingVerifierAccounts {
            payer: payer.pubkey(),
            run: h.run,
            coordinator_account: h.coordinator_account,
            audit_verdict: find_audit_verdict(&h.run, &target_key),
            loser_participant: find_participant(&h.run, &other),
        }
        .to_account_metas(None),
        data: RunSlashLosingVerifier {
            params: RunSlashLosingVerifierParams {
                target: target_key,
                voter_position: 0,
                loser: other,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    assert_eq!(
        refusal_code(h.endpoint.process_instruction(&payer, wrong_loser).await),
        treasurer_refusal(ProgramError::InvalidParameter),
    );
    assert_eq!(h.participant_of(loser).await.bond_amount, BOND);

    for position in 0..losers.len() {
        h.crank_loser(position as u16).await.unwrap();
    }
    assert_eq!(
        h.participant_of(loser).await.bond_amount,
        BOND - SLASHING_RATE
    );

    // The first seat's share, credited to the loser instead.
    let loser_key = h.clients[loser].pubkey();
    let diverted = run_pay_bench_bounty_instruction(
        &payer.pubkey(),
        &h.run,
        &target_key,
        0,
        &loser_key,
    );
    assert_eq!(
        refusal_code(h.endpoint.process_instruction(&payer, diverted).await),
        treasurer_refusal(ProgramError::InvalidParameter),
    );
    // A seat past the bench names nobody.
    let past_bench = run_pay_bench_bounty_instruction(
        &payer.pubkey(),
        &h.run,
        &target_key,
        bench.len() as u16,
        &other,
    );
    assert_eq!(
        refusal_code(h.endpoint.process_instruction(&payer, past_bench).await),
        treasurer_refusal(ProgramError::InvalidParameter),
    );
    assert_eq!(
        h.participant_of(loser).await.bond_amount,
        BOND - SLASHING_RATE
    );

    let before = h.participant_of(bench[0]).await.bond_amount;
    h.pay_bench_seat(0).await.unwrap();
    assert_eq!(
        h.participant_of(bench[0]).await.bond_amount - before,
        bench_share(bench.len()) * losers.len() as u64,
    );
}

/// A run that switches appeals off while one is open does not park it: there
/// is no deadline left to wait for, so the conviction finalises and its votes
/// are released.
#[tokio::test]
pub async fn switching_appeals_off_ends_an_open_appeal() {
    let mut h = setup("Leviathan liability appeals off", 130).await;
    let target = h.clients[h.target_client_idx].insecure_clone();
    let target_key = h.target_key;
    let target_index = h.target_index;
    h.reach_verifier_quorum().await;
    process_treasurer_run_open_challenge(
        &mut h.endpoint,
        &h.payer,
        &target,
        &h.run,
    )
    .await
    .unwrap();
    assert_eq!(
        refusal_code(
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
        ),
        treasurer_refusal(ProgramError::AppealWindowOpen),
    );
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
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
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
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
    for voter in h.quorum_voters() {
        h.release_vote(None, &target_key, voter).await.unwrap();
        assert_eq!(h.participant_of(voter).await.open_verdict_votes, 0);
    }
}

/// A conviction left inside nobody's hands after its challenge window closed
/// is finalised by the settlement a new dispute sends first, and its votes are
/// released, so the account can be taken over.
#[tokio::test]
pub async fn a_closed_conviction_is_finalised_and_released_before_a_takeover() {
    let mut h = setup("Leviathan liability closed", 131).await;
    let target_key = h.target_key;
    h.reach_verifier_quorum().await;
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);
    h.endpoint
        .forward_clock_unix_timestamp(CHALLENGE_WINDOW as u64)
        .await
        .unwrap();
    let payer = h.payer.insecure_clone();
    let settled = settle_previous_dispute(
        &mut h.endpoint,
        &payer,
        &h.run,
        &h.coordinator_account,
        &target_key,
    )
    .await
    .unwrap();
    assert_eq!(
        settled,
        DisputeSettlement {
            conviction_finalized: true,
            votes_freed: h.quorum_voters().len(),
            ..DisputeSettlement::default()
        }
    );
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
    assert!(h.verdict_of(&target_key).await.is_settled());
    for voter in h.quorum_voters() {
        assert_eq!(h.participant_of(voter).await.open_verdict_votes, 0);
    }
}

/// The verifier daemon settles an overturned dispute before it opens a new one
/// against the same target: it charges the losers and pays the bench, so the
/// account can be taken over.
#[tokio::test]
pub async fn an_overturned_dispute_is_settled_in_full_before_a_takeover() {
    let mut h = setup("Leviathan liability settle all", 129).await;
    let target_key = h.target_key;
    h.convict_and_overturn().await;
    let losers = h.quorum_voters();
    let bench = h.bench();
    let payer = h.payer.insecure_clone();
    let settled = settle_previous_dispute(
        &mut h.endpoint,
        &payer,
        &h.run,
        &h.coordinator_account,
        &target_key,
    )
    .await
    .unwrap();
    assert_eq!(
        settled,
        DisputeSettlement {
            conviction_finalized: false,
            votes_freed: 0,
            losers_charged: losers.len(),
            bench_paid: bench.len(),
        }
    );
    let verdict = h.verdict_of(&target_key).await;
    assert!(verdict.is_settled());
    for loser in losers {
        assert_eq!(h.participant_of(loser).await.open_verdict_votes, 0);
    }
    // Nothing is left to do a second time.
    h.endpoint.forward_clock_unix_timestamp(1).await.unwrap();
    assert_eq!(
        settle_previous_dispute(
            &mut h.endpoint,
            &payer,
            &h.run,
            &h.coordinator_account,
            &target_key,
        )
        .await
        .unwrap(),
        DisputeSettlement::default()
    );
}
