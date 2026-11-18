use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::on_charging_roster;
use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VerdictStatus;

/// Why a bond withdrawal may not be finalized yet.
///
/// A conviction reaches a bond in two steps: the coordinator charges slashed
/// points at the epoch boundary, and settlement forfeits them when the
/// participant finalizes a withdrawal. A bond that leaves before the first step
/// owes nothing, so the withdrawal waits for it (wienerlabs/leviathan-net#36).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WithdrawalHold {
    /// A conviction against the participant is still inside its challenge
    /// window or its appeal, so it can still be overturned or finalised.
    ConvictionPending,
    /// The participant trains in an epoch the coordinator has not charged yet,
    /// so a slash it earns there, or a conviction that lands there, is still
    /// to come.
    EpochNotSettled,
    /// The participant voted on a verdict that can still cost it: one not yet
    /// standing, or overturned and not yet charged by the losing-verifier
    /// crank. A vote on a conviction that stands, or on a dispute that lapsed,
    /// has to be released first (wienerlabs/leviathan-net#39).
    VotesAtStake,
    /// A slash no verdict accounts for ejected the participant, and its
    /// report has not been charged against the bond yet. Anyone can charge it
    /// once its epoch is charged; until then the bond waits, so the slash is
    /// paid and its reporter credited before the participant leaves
    /// (wienerlabs/leviathan-net#72).
    SlashReportsUncharged,
}

/// What finalizing a withdrawal does right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WithdrawalStanding {
    Held(WithdrawalHold),
    /// The withdrawal may settle. With `settle_conviction`, the participant's
    /// own conviction has run out its windows without reaching the ledger, and
    /// can no longer reach it, because the participant is not in an epoch that
    /// will be charged. Settlement then charges that conviction against the
    /// bond itself, so it is neither lost nor allowed to hold the bond forever.
    Free {
        settle_conviction: bool,
    },
}

/// Whether an appeal opened at `challenged_since` may still be ruled on at
/// `now`, given the run's appeal window as it stands. A run with appeals on
/// always has a window (wienerlabs/leviathan-net#36), so a zero window means
/// appeals were switched off, and an appeal still open has no deadline left to
/// wait for.
pub fn appeal_window_open(
    appeal_window_seconds: i64,
    challenged_since: i64,
    now: i64,
) -> bool {
    appeal_window_seconds > 0 && now < challenged_since + appeal_window_seconds
}

/// How `user`'s withdrawal stands, read from chain state: the run, the run's
/// coordinator account, `user`'s participant account, its own audit verdict if
/// it has one, and the cluster time.
pub fn withdrawal_standing(
    run: &Run,
    coordinator_account: &CoordinatorAccount,
    participant: &Participant,
    own_verdict: Option<&AuditVerdict>,
    user: &Pubkey,
    now: i64,
) -> WithdrawalStanding {
    // A vote that is still counted can still cost this bond, and only the
    // target's verdict records it, so nothing settled at this withdrawal could
    // reach it: the bond waits until each such vote is released or charged.
    if participant.open_verdict_votes > 0 {
        return WithdrawalStanding::Held(WithdrawalHold::VotesAtStake);
    }
    // A slash the roster earns is charged when the epoch ends; until then it
    // is still to come.
    let in_charging_epoch =
        on_charging_roster(&coordinator_account.state.coordinator, user);

    let pending = own_verdict.filter(|verdict| {
        matches!(
            verdict.status,
            VerdictStatus::SlashPending
                | VerdictStatus::Challenged
                | VerdictStatus::Due
        )
    });
    if let Some(verdict) = pending {
        let window_open = match verdict.status {
            VerdictStatus::SlashPending => {
                now < verdict.pending_since_unix + run.challenge_window_seconds
            },
            VerdictStatus::Challenged => appeal_window_open(
                run.appeal_window_seconds,
                verdict.challenged_since_unix,
                now,
            ),
            _ => false,
        };
        if window_open {
            return WithdrawalStanding::Held(WithdrawalHold::ConvictionPending);
        }
        // The conviction is due. In an epoch that will be charged it can
        // still land through `run_finalize_slash`, so the withdrawal waits for
        // that epoch's charge, and settles it then if nobody finalised it.
        if in_charging_epoch {
            return WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled);
        }
        return unless_slash_reports_uncharged(
            participant,
            WithdrawalStanding::Free {
                settle_conviction: true,
            },
        );
    }

    if in_charging_epoch {
        return WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled);
    }
    unless_slash_reports_uncharged(
        participant,
        WithdrawalStanding::Free {
            settle_conviction: false,
        },
    )
}

/// `free`, unless a slash that ejected the participant still waits to be
/// charged against its bond. That comes after every other hold: while its
/// epoch runs, the epoch holds the bond, and the report cannot be charged yet.
fn unless_slash_reports_uncharged(
    participant: &Participant,
    free: WithdrawalStanding,
) -> WithdrawalStanding {
    match participant.uncharged_slash_reports {
        0 => free,
        _ => WithdrawalStanding::Held(WithdrawalHold::SlashReportsUncharged),
    }
}

/// What a withdrawal settled now forfeits to points the ledger charged that no
/// verdict accounts for. A `run_slash` or an equivocation is charged through
/// its own report before the withdrawal can settle, and its reporter credited
/// from that (wienerlabs/leviathan-net#72), so what is left here is what the
/// ledger charged beyond any report: points charged while the participant had
/// no account, or beyond a bond a report's charge could not cover. Its bounty
/// stays in the vault: nobody recorded earning it.
///
/// Settlement first takes the participant's own conviction's charge, if a
/// conviction ejected it and is not charged yet, out of the same points and
/// the same bond; what the ledger charged beyond that, up to what the bond
/// still holds, is this (wienerlabs/leviathan-net#64).
pub fn unattributed_ledger_forfeit(
    participant: &Participant,
    coordinator_slashed_points: u64,
    own_verdict: Option<&AuditVerdict>,
) -> u64 {
    let conviction = own_verdict
        .filter(|verdict| verdict.awaits_charge())
        .map(|verdict| {
            participant.ejected_conviction_forfeit(
                coordinator_slashed_points,
                verdict.charge_rate,
            )
        })
        .unwrap_or(0);
    participant
        .unsettled_slashed_points(coordinator_slashed_points)
        .saturating_sub(conviction)
        .min(participant.bond_amount - conviction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::CommitmentEvidence;
    use crate::state::ConvictionCharge;
    use crate::state::EjectionCharge;

    fn participant(bond: u64, settled_points: u64) -> Participant {
        Participant {
            bump: 0,
            claimed_collateral_amount: 0,
            claimed_earned_points: 0,
            bond_amount: bond,
            bond_withdraw_pending_amount: bond,
            bond_withdraw_requested_at: 0,
            bond_settled_slashed_points: settled_points,
            bond_withdraw_delay_snapshot: 0,
            open_verdict_votes: 0,
            uncharged_slash_reports: 0,
        }
    }

    fn verdict(charge: ConvictionCharge, rate: u64) -> AuditVerdict {
        let mut verdict = AuditVerdict {
            bump: 0,
            run: Pubkey::new_unique(),
            target: Pubkey::new_unique(),
            epoch: 0,
            round_height: 0,
            status: VerdictStatus::Voting,
            verdict_count: 0,
            committed_hash: [0; 32],
            replayed_hash: [0; 32],
            target_index: 0,
            batch_start: 0,
            batch_end: 0,
            evidence: CommitmentEvidence::default(),
            pending_since_unix: 0,
            challenged_since_unix: 0,
            challenger: Pubkey::default(),
            overturn_count: 0,
            uphold_count: 0,
            settled_count: 0,
            released_voters: 0,
            bounty_owed: 0,
            bounty_paid: 0,
            charge: ConvictionCharge::Unlanded,
            charge_epoch_start: 0,
            charge_rate: 0,
            appeal_upheld: false,
            voters: vec![],
            appeal_voters: vec![],
            vote_commitments: vec![],
        };
        verdict.land_ejection(EjectionCharge {
            epoch_start: 1_000,
            rate,
        });
        if charge == ConvictionCharge::Charged {
            verdict.record_charge(0);
        }
        verdict
    }

    /// What settlement itself does: the conviction's charge, then the rest of
    /// the ledger.
    fn settle(
        mut participant: Participant,
        slashed: u64,
        verdict: Option<&AuditVerdict>,
    ) -> u64 {
        if let Some(verdict) = verdict.filter(|verdict| verdict.awaits_charge())
        {
            let forfeit = participant
                .ejected_conviction_forfeit(slashed, verdict.charge_rate);
            participant.settle_ledger_points(forfeit);
        }
        participant
            .unsettled_slashed_points(slashed)
            .min(participant.bond_amount)
    }

    #[test]
    fn an_uncharged_slash_report_holds_a_bond_that_is_otherwise_free() {
        let mut p = participant(500, 0);
        for free in [
            WithdrawalStanding::Free {
                settle_conviction: false,
            },
            WithdrawalStanding::Free {
                settle_conviction: true,
            },
        ] {
            p.uncharged_slash_reports = 0;
            assert_eq!(unless_slash_reports_uncharged(&p, free), free);
            for count in [1, 3] {
                p.uncharged_slash_reports = count;
                assert_eq!(
                    unless_slash_reports_uncharged(&p, free),
                    WithdrawalStanding::Held(
                        WithdrawalHold::SlashReportsUncharged
                    ),
                );
            }
        }
    }

    #[test]
    fn a_slash_with_no_verdict_is_unattributed() {
        let p = participant(500, 0);
        assert_eq!(unattributed_ledger_forfeit(&p, 200, None), 200);
        assert_eq!(settle(p, 200, None), 200);
    }

    #[test]
    fn an_ejected_conviction_owes_its_own_charge_to_its_voters() {
        let ejected = verdict(ConvictionCharge::Ejected, 200);
        // The conviction is the only charge: nothing is left for a reporter.
        let p = participant(500, 0);
        assert_eq!(unattributed_ledger_forfeit(&p, 200, Some(&ejected)), 0);
        // A `run_slash` in another epoch is still unattributed.
        assert_eq!(unattributed_ledger_forfeit(&p, 400, Some(&ejected)), 200);
        // A bond short of both pays the conviction first.
        let short = participant(300, 0);
        assert_eq!(
            unattributed_ledger_forfeit(&short, 400, Some(&ejected)),
            100
        );
        for (p, slashed) in [
            (participant(500, 0), 200),
            (participant(500, 0), 400),
            (participant(300, 0), 400),
        ] {
            assert_eq!(
                unattributed_ledger_forfeit(&p, slashed, Some(&ejected)),
                settle(p, slashed, Some(&ejected)),
            );
        }
    }

    #[test]
    fn a_charged_or_cleared_verdict_accounts_for_nothing_more() {
        // Already charged: its points are settled, and anything new the
        // ledger charged is someone else's.
        let charged = verdict(ConvictionCharge::Charged, 200);
        let p = participant(300, 200);
        assert_eq!(unattributed_ledger_forfeit(&p, 400, Some(&charged)), 200);
        // Overturned: the target was cleared, so every point is unattributed.
        let mut overturned = verdict(ConvictionCharge::Unlanded, 0);
        overturned.status = VerdictStatus::Overturned;
        overturned.charge = ConvictionCharge::Unlanded;
        let p = participant(500, 0);
        assert_eq!(
            unattributed_ledger_forfeit(&p, 200, Some(&overturned)),
            200
        );
    }
}
