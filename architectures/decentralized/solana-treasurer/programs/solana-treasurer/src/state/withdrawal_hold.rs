use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

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
    let coordinator = &coordinator_account.state.coordinator;
    let signer = user.to_bytes();
    // A slash the roster earns is charged when the epoch ends; until then it
    // is still to come. Between epochs the last one has been charged, even
    // though its roster is still on the account.
    let in_charging_epoch = coordinator.charges_pending()
        && coordinator
            .epoch_state
            .clients
            .iter()
            .chain(coordinator.epoch_state.exited_clients.iter())
            .any(|client| *client.id.signer() == signer);

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
        return WithdrawalStanding::Free {
            settle_conviction: true,
        };
    }

    if in_charging_epoch {
        return WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled);
    }
    WithdrawalStanding::Free {
        settle_conviction: false,
    }
}
