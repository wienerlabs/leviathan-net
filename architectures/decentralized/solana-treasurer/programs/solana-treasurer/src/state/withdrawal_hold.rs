use anchor_lang::prelude::*;
use psyche_coordinator::RunState;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::AuditVerdict;
use crate::state::VerdictStatus;

/// Why a bond withdrawal may not be finalized yet.
///
/// A conviction reaches the bond in two steps: the coordinator charges slashed
/// points at the epoch boundary, and settlement forfeits them when the
/// participant finalizes a withdrawal. A bond that leaves before the first step
/// owes nothing, so the withdrawal waits for it (wienerlabs/leviathan-net#36).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WithdrawalHold {
    /// A conviction against the participant is waiting out its challenge
    /// window or its appeal, so it has not reached the ledger yet.
    ConvictionPending,
    /// The participant trains in an epoch the coordinator has not charged yet,
    /// so a slash it earns there is still to come.
    EpochNotSettled,
}

/// Whether `user`'s withdrawal is held, read from chain state: the run's
/// coordinator account and `user`'s own audit verdict, if one exists.
pub fn withdrawal_hold(
    coordinator_account: &CoordinatorAccount,
    own_verdict: Option<&AuditVerdict>,
    user: &Pubkey,
) -> Option<WithdrawalHold> {
    if let Some(verdict) = own_verdict {
        if matches!(
            verdict.status,
            VerdictStatus::SlashPending | VerdictStatus::Challenged
        ) {
            return Some(WithdrawalHold::ConvictionPending);
        }
    }

    // Slashed points are charged when the coordinator leaves `Cooldown` for
    // the next epoch (or ends a warmup that never filled), so in these states
    // the roster's charges are still to come. Outside them the last epoch has
    // been charged, even though its roster is still on the account.
    let coordinator = &coordinator_account.state.coordinator;
    let charges_pending = matches!(
        coordinator.run_state,
        RunState::Warmup
            | RunState::RoundTrain
            | RunState::RoundWitness
            | RunState::Cooldown
    );
    if charges_pending {
        let signer = user.to_bytes();
        let in_epoch = coordinator
            .epoch_state
            .clients
            .iter()
            .chain(coordinator.epoch_state.exited_clients.iter())
            .any(|client| *client.id.signer() == signer);
        if in_epoch {
            return Some(WithdrawalHold::EpochNotSettled);
        }
    }

    None
}
