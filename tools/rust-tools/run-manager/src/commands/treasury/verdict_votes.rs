//! A verifier's votes hold its bond until each can no longer cost it
//! (wienerlabs/leviathan-net#39). Before a withdrawal is finalized, the
//! wallet's votes are freed where they can be:
//!
//! - a standing conviction's votes are released, and a lapsed dispute's
//!   withdrawn;
//! - a conviction whose windows have closed is finalized first;
//! - an overturned verdict's losers are charged up to and including this
//!   wallet.
//!
//! A share the wallet is owed out of a verdict's losing side is credited to its
//! bond: a tie-breaker's share of an overturn's forfeits, or a voter's, or an
//! upholding tie-breaker's, share of a convicted target's forfeit, charging
//! the target first if nobody has (wienerlabs/leviathan-net#64). A due
//! conviction is landed first where it can be, and charged against the
//! target's bond where it never can (wienerlabs/leviathan-net#74).

use anchor_client::solana_sdk::instruction::Instruction;
use anchor_client::solana_sdk::pubkey::Pubkey;
use anyhow::Result;
use psyche_coordinator::Coordinator;
use psyche_solana_treasurer::state::{
    appeal_window_open, conviction_landing, on_charging_roster, AuditVerdict, Landing, Run,
    VerdictStatus,
};

use crate::{instructions, SolanaBackend};

/// Planning runs twice: a finalized conviction's votes, and a share owed once
/// the losers or the target are charged, are only reachable after the first
/// pass.
const PASSES: usize = 2;

/// What to do about one verdict that holds this wallet's bond or owes it a
/// share of a bounty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoteStep {
    /// Release or withdraw the wallet's vote now.
    Release { target: Pubkey },
    /// Finalize the conviction: its windows have closed, or it is due and can
    /// land now. Its votes are released on the next pass.
    Finalize { target: Pubkey, target_index: u64 },
    /// The verdict was overturned: charge its losers in order, from the next
    /// one the crank has not reached through `through_position`.
    SettleLosers {
        target: Pubkey,
        from_position: u16,
        through_position: u16,
    },
    /// Charge the convicted target's bond for the conviction that ejected it,
    /// now that the ledger has charged the ejection, so its bounty is owed.
    ChargeConviction { target: Pubkey },
    /// Charge the convicted target's bond for the conviction its verdict
    /// records as due, now that the target is on no roster an epoch will still
    /// charge, so its bounty is owed.
    ChargeDueConviction { target: Pubkey },
    /// Credit the wallet's seat its share of the verdict's bounty.
    CreditSeat { target: Pubkey, seat: u16 },
    /// The vote can still cost this wallet and nothing can be done yet.
    Held {
        target: Pubkey,
        reason: &'static str,
    },
}

/// The chain state a plan is made from.
pub struct PlanContext<'a> {
    pub coordinator: &'a Coordinator,
    pub run: &'a Run,
    pub now: i64,
}

/// Where `target` sits on the roster, or `None` off it.
fn roster_index(coordinator: &Coordinator, target: &Pubkey) -> Option<u64> {
    coordinator
        .epoch_state
        .clients
        .iter()
        .position(|client| *client.id.signer() == target.to_bytes())
        .map(|index| index as u64)
}

/// What to do about a verdict holding an open vote of `user`'s, at `position`.
fn vote_step(
    verdict: &AuditVerdict,
    position: usize,
    context: &PlanContext,
    user: &Pubkey,
    withdraw_live_votes: bool,
) -> VoteStep {
    let target = verdict.target;
    let consenting = withdraw_live_votes.then_some(user);
    let round_over = verdict.round_over(context.coordinator);
    if verdict
        .releasable_votes(consenting, round_over)
        .contains(user)
    {
        return VoteStep::Release { target };
    }
    // Off the roster the index is not read: the conviction is recorded due.
    let finalize = VoteStep::Finalize {
        target,
        target_index: roster_index(context.coordinator, &target).unwrap_or_default(),
    };
    match verdict.status {
        VerdictStatus::Overturned => VoteStep::SettleLosers {
            target,
            from_position: verdict.settled_count,
            through_position: position as u16,
        },
        VerdictStatus::Voting => VoteStep::Held {
            target,
            reason: "the dispute's round is still under way; it lapses when the round ends, or pass --withdraw-live-votes to withdraw the vote now",
        },
        VerdictStatus::SlashPending
            if context.now < verdict.pending_since_unix + context.run.challenge_window_seconds =>
        {
            VoteStep::Held {
                target,
                reason: "the conviction is inside its challenge window",
            }
        },
        VerdictStatus::Challenged
            if appeal_window_open(
                context.run.appeal_window_seconds,
                verdict.challenged_since_unix,
                context.now,
            ) =>
        {
            VoteStep::Held {
                target,
                reason: "the conviction is under appeal",
            }
        },
        VerdictStatus::SlashPending | VerdictStatus::Challenged => finalize,
        // A conviction that stands is releasable; reaching here means the
        // release itself would be refused, which the chain will say.
        VerdictStatus::Upheld | VerdictStatus::Due => VoteStep::Held {
            target,
            reason: "the vote could not be released",
        },
    }
}

/// What a verdict owes `user`, the recipient at `seat`: the losers charged or
/// the target charged first, if that is what stands between it and its
/// share, and then the share. Nothing while the ejection that would be charged
/// waits for its epoch to end; that never holds the wallet's bond.
fn seat_step(verdict: &AuditVerdict, seat: usize, context: &PlanContext) -> Option<VoteStep> {
    if verdict.bounty_seat_paid(seat) {
        return None;
    }
    let target = verdict.target;
    if verdict.status == VerdictStatus::Overturned && !verdict.all_losers_settled() {
        return Some(VoteStep::SettleLosers {
            target,
            from_position: verdict.settled_count,
            through_position: verdict.voters.len().saturating_sub(1) as u16,
        });
    }
    if verdict.awaits_charge() {
        return verdict
            .ejection_charged(context.coordinator)
            .then_some(VoteStep::ChargeConviction { target });
    }
    if verdict.status == VerdictStatus::Due {
        return due_step(verdict, context);
    }
    (verdict.bounty_final().is_ok() && verdict.bounty_share() > 0).then_some(VoteStep::CreditSeat {
        target,
        seat: seat as u16,
    })
}

/// What a due conviction calls for before its recipients can be credited. On
/// the roster of an epoch the ledger will still charge, it lands there as an
/// ejection if it can, and otherwise waits for that epoch's charge. Off every
/// such roster it could land only if the target came back, so the target's
/// bond is charged for it directly.
fn due_step(verdict: &AuditVerdict, context: &PlanContext) -> Option<VoteStep> {
    let target = verdict.target;
    if !on_charging_roster(context.coordinator, &target) {
        return Some(VoteStep::ChargeDueConviction { target });
    }
    let target_index = roster_index(context.coordinator, &target)?;
    matches!(
        conviction_landing(context.coordinator, &target, target_index),
        Ok(Landing::Eject(_))
    )
    .then_some(VoteStep::Finalize {
        target,
        target_index,
    })
}

/// What each verdict in `verdicts` calls for: one holding an open vote of
/// `user`'s, or one owing `user` a share of its bounty.
///
/// A vote on a verdict still `Voting` is withdrawn only once its round is over,
/// unless `withdraw_live_votes`: the wallet then signs its own withdrawal.
pub fn plan_vote_steps(
    verdicts: &[AuditVerdict],
    context: &PlanContext,
    user: &Pubkey,
    withdraw_live_votes: bool,
) -> Vec<VoteStep> {
    let mut steps = vec![];
    for verdict in verdicts {
        if let Some(position) = verdict.voters.iter().position(|voter| voter == user) {
            if !verdict.vote_released(position) {
                steps.push(vote_step(
                    verdict,
                    position,
                    context,
                    user,
                    withdraw_live_votes,
                ));
            }
        }
        if let Some(seat) = verdict
            .bounty_recipients()
            .iter()
            .position(|recipient| recipient == user)
        {
            if let Some(step) = seat_step(verdict, seat, context) {
                if !steps.contains(&step) {
                    steps.push(step);
                }
            }
        }
    }
    steps
}

/// The instructions that carry out `step`, signed by `user`.
fn step_instructions(
    step: &VoteStep,
    verdicts: &[AuditVerdict],
    treasurer_index: u64,
    run_state: &Run,
    user: &Pubkey,
) -> Vec<Instruction> {
    match step {
        VoteStep::Release { target } => {
            vec![instructions::treasurer_participant_release_verdict_vote(
                treasurer_index,
                &run_state.coordinator_account,
                user,
                target,
                user,
            )]
        }
        VoteStep::Finalize {
            target,
            target_index,
        } => vec![instructions::treasurer_run_finalize_slash(
            treasurer_index,
            &run_state.coordinator_instance,
            &run_state.coordinator_account,
            user,
            target,
            *target_index,
        )],
        VoteStep::SettleLosers {
            target,
            from_position,
            through_position,
        } => {
            let Some(verdict) = verdicts.iter().find(|verdict| verdict.target == *target) else {
                return vec![];
            };
            (*from_position..=*through_position)
                .filter_map(|position| {
                    let loser = verdict.voters.get(position as usize)?;
                    Some(instructions::treasurer_run_slash_losing_verifier(
                        treasurer_index,
                        &run_state.coordinator_account,
                        user,
                        target,
                        position,
                        loser,
                    ))
                })
                .collect()
        }
        VoteStep::ChargeConviction { target } => {
            vec![instructions::treasurer_run_charge_conviction(
                treasurer_index,
                &run_state.coordinator_account,
                user,
                target,
            )]
        }
        VoteStep::ChargeDueConviction { target } => {
            vec![instructions::treasurer_run_charge_due_conviction(
                treasurer_index,
                &run_state.coordinator_account,
                user,
                target,
            )]
        }
        VoteStep::CreditSeat { target, seat } => {
            vec![instructions::treasurer_run_pay_verdict_bounty(
                treasurer_index,
                user,
                target,
                *seat,
                user,
            )]
        }
        VoteStep::Held { .. } => vec![],
    }
}

/// Frees what it can of `user`'s open votes in the run, and credits the bounty
/// shares it is owed. Returns the votes that still hold its bond, with why.
///
/// Each instruction goes on its own, so one that fails, for instance because
/// someone else just took the same step, does not stop the others; the next
/// pass reads the chain again.
pub async fn free_wallet_votes(
    backend: &SolanaBackend,
    treasurer_index: u64,
    run_address: &Pubkey,
    run_state: &Run,
    user: &Pubkey,
    withdraw_live_votes: bool,
) -> Result<Vec<(Pubkey, &'static str)>> {
    let mut held = vec![];
    for _ in 0..PASSES {
        let verdicts: Vec<AuditVerdict> = backend
            .get_treasurer_audit_verdicts(run_address)
            .await?
            .into_iter()
            .map(|(_, verdict)| verdict)
            .collect();
        let coordinator = backend
            .get_coordinator_account(&run_state.coordinator_account)
            .await?
            .state
            .coordinator;
        let now = backend.get_clock().await?.unix_timestamp;
        let context = PlanContext {
            coordinator: &coordinator,
            run: run_state,
            now,
        };
        let steps = plan_vote_steps(&verdicts, &context, user, withdraw_live_votes);
        held = steps
            .iter()
            .filter_map(|step| match step {
                VoteStep::Held { target, reason } => Some((*target, *reason)),
                _ => None,
            })
            .collect();
        let mut sent = 0;
        for step in &steps {
            for instruction in step_instructions(step, &verdicts, treasurer_index, run_state, user)
            {
                sent += 1;
                match backend
                    .send_and_retry("Free verdict votes", &[instruction], &[])
                    .await
                {
                    Ok(signature) => println!("{step:?} in transaction: {signature}"),
                    Err(error) => println!("{step:?} was not taken: {error:#}"),
                }
            }
        }
        if sent == 0 {
            break;
        }
    }
    Ok(held)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use psyche_coordinator::RunState;
    use psyche_solana_treasurer::state::{CommitmentEvidence, EjectionCharge};

    fn verdict(status: VerdictStatus, voters: &[Pubkey], round_height: u32) -> AuditVerdict {
        AuditVerdict {
            bump: 0,
            run: Pubkey::new_unique(),
            target: Pubkey::new_unique(),
            epoch: 0,
            round_height,
            status,
            verdict_count: voters.len() as u16,
            committed_hash: [0u8; 32],
            replayed_hash: [0u8; 32],
            target_index: 0,
            batch_start: 0,
            batch_end: 0,
            evidence: CommitmentEvidence::default(),
            pending_since_unix: 100,
            challenged_since_unix: 100,
            challenger: Pubkey::default(),
            overturn_count: 0,
            uphold_count: 0,
            settled_count: 0,
            released_voters: 0,
            bounty_owed: 0,
            bounty_paid: 0,
            charge: Default::default(),
            charge_epoch_start: 0,
            charge_rate: 0,
            appeal_upheld: false,
            voters: voters.to_vec(),
            appeal_voters: vec![Pubkey::new_unique()],
            vote_commitments: vec![[0u8; 32]; voters.len()],
        }
    }

    /// A coordinator in epoch 0 whose current round is `height`.
    fn coordinator_in_round(height: u32) -> Coordinator {
        let mut coordinator = Coordinator::zeroed();
        let head = coordinator.epoch_state.rounds_head as usize;
        coordinator.epoch_state.rounds[head].height = height;
        coordinator
    }

    /// A run whose challenge window is 50 seconds and appeal window 100.
    fn run() -> Run {
        Run {
            bump: 0,
            index: 0,
            main_authority: Pubkey::default(),
            join_authority: Pubkey::default(),
            coordinator_account: Pubkey::default(),
            coordinator_instance: Pubkey::default(),
            collateral_mint: Pubkey::default(),
            total_claimed_collateral_amount: 0,
            total_claimed_earned_points: 0,
            total_bonded_amount: 0,
            bond_minimum_amount: 0,
            bond_withdraw_delay_seconds: 0,
            slash_bounty_bps: 0,
            challenge_window_seconds: 50,
            tie_breaker_committee_size: 0,
            appeal_window_seconds: 100,
        }
    }

    fn plan(
        verdicts: &[AuditVerdict],
        coordinator: &Coordinator,
        now: i64,
        user: &Pubkey,
        withdraw_live_votes: bool,
    ) -> Vec<VoteStep> {
        let run = run();
        let context = PlanContext {
            coordinator,
            run: &run,
            now,
        };
        plan_vote_steps(verdicts, &context, user, withdraw_live_votes)
    }

    #[test]
    fn votes_the_wallet_did_not_cast_are_ignored() {
        let user = Pubkey::new_unique();
        let other = verdict(VerdictStatus::Upheld, &[Pubkey::new_unique()], 0);
        assert!(plan(&[other], &coordinator_in_round(0), 0, &user, false).is_empty());
    }

    /// A vote for another payload than the one the quorum convicted on was not
    /// part of it, so the wallet releases it at once, inside the challenge
    /// window or the appeal, or after an overturn, where a vote that convicted
    /// is held.
    #[test]
    fn a_dissenting_vote_is_released_at_once() {
        let user = Pubkey::new_unique();
        let other = Pubkey::new_unique();
        for status in [
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
            VerdictStatus::Overturned,
        ] {
            let mut dissented = verdict(status, &[user, other], 0);
            dissented.vote_commitments[0] = [9u8; 32];
            let convicted = verdict(status, &[user, other], 0);
            let steps = plan(
                &[dissented.clone(), convicted.clone()],
                &coordinator_in_round(0),
                0,
                &user,
                false,
            );
            assert_eq!(
                steps[0],
                VoteStep::Release {
                    target: dissented.target
                },
                "{status:?}"
            );
            assert!(
                !matches!(steps[1], VoteStep::Release { .. }),
                "{status:?}: a vote that convicted is not released yet"
            );
        }
    }

    #[test]
    fn a_standing_conviction_is_released_and_a_released_vote_skipped() {
        let user = Pubkey::new_unique();
        let upheld = verdict(VerdictStatus::Upheld, &[user], 0);
        let due = verdict(VerdictStatus::Due, &[user], 0);
        let mut released = verdict(VerdictStatus::Due, &[user], 0);
        released.released_voters = 1;
        let steps = plan(
            &[upheld.clone(), due.clone(), released.clone()],
            &coordinator_in_round(0),
            0,
            &user,
            false,
        );
        // The wallet is owed a share of each due conviction, whose target is
        // on no roster an epoch will still charge, so each is charged too
        // (wienerlabs/leviathan-net#74).
        assert_eq!(
            steps,
            vec![
                VoteStep::Release {
                    target: upheld.target
                },
                VoteStep::Release { target: due.target },
                VoteStep::ChargeDueConviction { target: due.target },
                VoteStep::ChargeDueConviction {
                    target: released.target
                },
            ]
        );
    }

    #[test]
    fn a_live_vote_is_withdrawn_only_when_asked_and_a_lapsed_one_always() {
        let user = Pubkey::new_unique();
        let live = verdict(VerdictStatus::Voting, &[user], 3);
        let lapsed = verdict(VerdictStatus::Voting, &[user], 2);
        let coordinator = coordinator_in_round(3);
        let steps = plan(
            &[live.clone(), lapsed.clone()],
            &coordinator,
            0,
            &user,
            false,
        );
        assert!(matches!(steps[0], VoteStep::Held { target, .. } if target == live.target));
        assert_eq!(
            steps[1],
            VoteStep::Release {
                target: lapsed.target
            }
        );
        let steps = plan(std::slice::from_ref(&live), &coordinator, 0, &user, true);
        assert_eq!(
            steps,
            vec![VoteStep::Release {
                target: live.target
            }]
        );
    }

    #[test]
    fn a_vote_inside_its_windows_is_held_and_its_conviction_finalized_after() {
        let user = Pubkey::new_unique();
        let coordinator = coordinator_in_round(0);
        // Opened at 100: the challenge window is 50, the appeal window 100.
        for (status, closes) in [
            (VerdictStatus::SlashPending, 150),
            (VerdictStatus::Challenged, 200),
        ] {
            let open = verdict(status, &[user], 0);
            let steps = plan(
                std::slice::from_ref(&open),
                &coordinator,
                closes - 1,
                &user,
                true,
            );
            assert!(matches!(steps[0], VoteStep::Held { .. }), "{status:?}");
            let steps = plan(
                std::slice::from_ref(&open),
                &coordinator,
                closes,
                &user,
                true,
            );
            assert_eq!(
                steps,
                vec![VoteStep::Finalize {
                    target: open.target,
                    target_index: 0
                }],
                "{status:?}"
            );
        }
    }

    #[test]
    fn an_overturned_verdict_is_cranked_through_the_wallet() {
        let user = Pubkey::new_unique();
        let first = Pubkey::new_unique();
        let mut overturned = verdict(VerdictStatus::Overturned, &[first, user], 0);
        let coordinator = coordinator_in_round(0);
        let steps = plan(
            std::slice::from_ref(&overturned),
            &coordinator,
            0,
            &user,
            true,
        );
        assert_eq!(
            steps,
            vec![VoteStep::SettleLosers {
                target: overturned.target,
                from_position: 0,
                through_position: 1,
            }]
        );
        overturned.settled_count = 1;
        overturned.released_voters = 1;
        let steps = plan(
            std::slice::from_ref(&overturned),
            &coordinator,
            0,
            &user,
            true,
        );
        assert_eq!(
            steps,
            vec![VoteStep::SettleLosers {
                target: overturned.target,
                from_position: 1,
                through_position: 1,
            }]
        );
    }

    #[test]
    fn a_tie_breaker_is_credited_once_every_loser_is_charged() {
        let user = Pubkey::new_unique();
        let losers = [Pubkey::new_unique(), Pubkey::new_unique()];
        let mut overturned = verdict(VerdictStatus::Overturned, &losers, 0);
        overturned.appeal_voters = vec![Pubkey::new_unique(), user];
        let coordinator = coordinator_in_round(0);
        let steps = plan(
            std::slice::from_ref(&overturned),
            &coordinator,
            0,
            &user,
            false,
        );
        assert_eq!(
            steps,
            vec![VoteStep::SettleLosers {
                target: overturned.target,
                from_position: 0,
                through_position: 1,
            }]
        );
        overturned.settled_count = 2;
        overturned.released_voters = 0b11;
        overturned.bounty_owed = 100;
        let steps = plan(
            std::slice::from_ref(&overturned),
            &coordinator,
            0,
            &user,
            false,
        );
        assert_eq!(
            steps,
            vec![VoteStep::CreditSeat {
                target: overturned.target,
                seat: 1,
            }]
        );
        // Credited already, or owed nothing: no step.
        overturned.bounty_paid = 0b10;
        assert!(plan(
            std::slice::from_ref(&overturned),
            &coordinator,
            0,
            &user,
            false
        )
        .is_empty());
        overturned.bounty_paid = 0;
        overturned.bounty_owed = 1;
        assert!(plan(
            std::slice::from_ref(&overturned),
            &coordinator,
            0,
            &user,
            false
        )
        .is_empty());
    }

    /// An upheld conviction, reached by `voters` with no appeal, that ejected
    /// its target in the epoch that started at 1000.
    fn ejected(voters: &[Pubkey]) -> AuditVerdict {
        let mut upheld = verdict(VerdictStatus::Voting, voters, 0);
        upheld.appeal_voters = vec![];
        upheld.land_ejection(EjectionCharge {
            epoch_start: 1_000,
            rate: 200,
        });
        upheld
    }

    #[test]
    fn a_voter_charges_the_target_once_its_epoch_is_charged_and_is_credited() {
        let user = Pubkey::new_unique();
        let other = Pubkey::new_unique();
        let mut upheld = ejected(&[other, user]);
        upheld.released_voters = 0b11;
        // The epoch it was ejected in is still running: nothing to do yet,
        // and nothing holds the bond.
        let mut running = coordinator_in_round(0);
        running.run_state = RunState::RoundTrain;
        running.epoch_state.start_timestamp = 1_000;
        assert!(plan(std::slice::from_ref(&upheld), &running, 0, &user, false).is_empty());
        // Charged: the target is charged first.
        let ended = coordinator_in_round(0);
        assert_eq!(
            plan(std::slice::from_ref(&upheld), &ended, 0, &user, false),
            vec![VoteStep::ChargeConviction {
                target: upheld.target
            }]
        );
        // Then the wallet's seat, the second voter, is credited.
        upheld.record_charge(100);
        assert_eq!(
            plan(std::slice::from_ref(&upheld), &ended, 0, &user, false),
            vec![VoteStep::CreditSeat {
                target: upheld.target,
                seat: 1,
            }]
        );
        upheld.bounty_paid = 0b10;
        assert!(plan(std::slice::from_ref(&upheld), &ended, 0, &user, false).is_empty());
    }

    /// A conviction reached by `voters` with no appeal, recorded as due.
    fn due(voters: &[Pubkey]) -> AuditVerdict {
        let mut due = verdict(VerdictStatus::Due, voters, 0);
        due.appeal_voters = vec![];
        due
    }

    /// `coordinator` in `run_state`, with `target` on its roster in `state`.
    fn with_target(
        mut coordinator: Coordinator,
        run_state: RunState,
        target: &Pubkey,
        state: psyche_coordinator::ClientState,
    ) -> Coordinator {
        coordinator.run_state = run_state;
        coordinator.epoch_state.clients =
            psyche_core::FixedVec::from_iter([psyche_coordinator::Client {
                id: psyche_core::NodeIdentity::from_single_key(target.to_bytes()),
                state,
                exited_height: 0,
            }]);
        coordinator
    }

    #[test]
    fn a_due_conviction_whose_target_is_gone_is_charged_and_its_voter_credited() {
        let user = Pubkey::new_unique();
        let mut conviction = due(&[user]);
        conviction.released_voters = 1;
        // Between epochs: the last one has been charged, so nothing will ever
        // land the conviction unless the target comes back.
        let ended = coordinator_in_round(0);
        assert_eq!(
            plan(std::slice::from_ref(&conviction), &ended, 0, &user, false),
            vec![VoteStep::ChargeDueConviction {
                target: conviction.target
            }]
        );
        // Charged: the wallet's seat is credited.
        conviction.settle_unlanded(100);
        assert_eq!(
            plan(std::slice::from_ref(&conviction), &ended, 0, &user, false),
            vec![VoteStep::CreditSeat {
                target: conviction.target,
                seat: 0,
            }]
        );
    }

    #[test]
    fn a_due_conviction_whose_target_is_back_lands_there_or_waits() {
        use psyche_coordinator::ClientState;
        let user = Pubkey::new_unique();
        let mut conviction = due(&[user]);
        conviction.released_voters = 1;
        let target = conviction.target;
        // Back on the roster of an epoch under way: it lands as an ejection.
        let training = with_target(
            coordinator_in_round(0),
            RunState::RoundTrain,
            &target,
            ClientState::Healthy,
        );
        assert_eq!(
            plan(
                std::slice::from_ref(&conviction),
                &training,
                0,
                &user,
                false
            ),
            vec![VoteStep::Finalize {
                target,
                target_index: 0,
            }]
        );
        // In a warmup, or ejected by another conviction already, it cannot
        // land, and the epoch's charge is still to come: nothing yet.
        for (run_state, state) in [
            (RunState::Warmup, ClientState::Healthy),
            (RunState::RoundTrain, ClientState::Ejected),
            (RunState::Cooldown, ClientState::Withdrawn),
        ] {
            let coordinator = with_target(coordinator_in_round(0), run_state, &target, state);
            assert!(
                plan(
                    std::slice::from_ref(&conviction),
                    &coordinator,
                    0,
                    &user,
                    false
                )
                .is_empty(),
                "{run_state} {state:?}"
            );
        }
    }

    #[test]
    fn a_voter_whose_vote_is_open_is_released_and_charges_the_target() {
        let user = Pubkey::new_unique();
        let upheld = ejected(&[user]);
        assert_eq!(
            plan(
                std::slice::from_ref(&upheld),
                &coordinator_in_round(0),
                0,
                &user,
                false
            ),
            vec![
                VoteStep::Release {
                    target: upheld.target
                },
                VoteStep::ChargeConviction {
                    target: upheld.target
                },
            ]
        );
    }

    #[test]
    fn an_upheld_appeal_credits_its_bench_and_not_the_verifiers() {
        let user = Pubkey::new_unique();
        let verifier = Pubkey::new_unique();
        let mut upheld = ejected(&[verifier]);
        upheld.appeal_voters = vec![user];
        upheld.appeal_upheld = true;
        upheld.released_voters = 1;
        upheld.record_charge(100);
        let coordinator = coordinator_in_round(0);
        assert_eq!(
            plan(std::slice::from_ref(&upheld), &coordinator, 0, &user, false),
            vec![VoteStep::CreditSeat {
                target: upheld.target,
                seat: 0,
            }]
        );
        assert!(plan(
            std::slice::from_ref(&upheld),
            &coordinator,
            0,
            &verifier,
            false
        )
        .is_empty());
    }

    #[test]
    fn an_appeal_that_ran_out_its_window_credits_the_verifiers() {
        let user = Pubkey::new_unique();
        let tie_breaker = Pubkey::new_unique();
        let mut upheld = ejected(&[user]);
        upheld.appeal_voters = vec![tie_breaker];
        upheld.released_voters = 1;
        upheld.record_charge(100);
        let coordinator = coordinator_in_round(0);
        assert_eq!(
            plan(std::slice::from_ref(&upheld), &coordinator, 0, &user, false),
            vec![VoteStep::CreditSeat {
                target: upheld.target,
                seat: 0,
            }]
        );
        assert!(plan(
            std::slice::from_ref(&upheld),
            &coordinator,
            0,
            &tie_breaker,
            false
        )
        .is_empty());
    }

    #[test]
    fn a_loser_of_an_overturn_is_never_credited() {
        let user = Pubkey::new_unique();
        let mut overturned = verdict(VerdictStatus::Overturned, &[user], 0);
        overturned.settled_count = 1;
        overturned.released_voters = 1;
        overturned.bounty_owed = 100;
        assert!(plan(
            std::slice::from_ref(&overturned),
            &coordinator_in_round(0),
            0,
            &user,
            false
        )
        .is_empty());
    }
}
