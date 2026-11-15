use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::AuditVerdict;
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

/// How `user`'s withdrawal stands, read from chain state: the run, the run's
/// coordinator account, `user`'s own audit verdict if it has one, and the
/// cluster time.
pub fn withdrawal_standing(
    run: &Run,
    coordinator_account: &CoordinatorAccount,
    own_verdict: Option<&AuditVerdict>,
    user: &Pubkey,
    now: i64,
) -> WithdrawalStanding {
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
            VerdictStatus::Challenged => {
                run.appeal_window_seconds == 0
                    || now
                        < verdict.challenged_since_unix
                            + run.appeal_window_seconds
            },
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
