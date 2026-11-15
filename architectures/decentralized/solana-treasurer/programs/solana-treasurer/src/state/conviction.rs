use anchor_lang::prelude::*;
use psyche_coordinator::Coordinator;

use crate::ProgramError;

/// Where a conviction against a target lands right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Landing {
    /// Eject the target at this roster index: the coordinator is certain to
    /// charge it at the end of this epoch.
    Eject(u64),
    /// The conviction stands but cannot reach the ledger now, because the
    /// target is not on the roster or no ejection here would be charged. The
    /// verdict records it as `Due` (wienerlabs/leviathan-net#36).
    Due,
}

/// Decides where a conviction against `target` lands, given the roster index
/// the caller names for it.
///
/// A target on the roster must be named at its own index: a wrong one would
/// otherwise let the caller, the target included, turn an ejection it could
/// not avoid into one put off to a later epoch.
pub fn conviction_landing(
    coordinator: &Coordinator,
    target: &Pubkey,
    target_index: u64,
) -> Result<Landing> {
    let signer = target.to_bytes();
    let position = coordinator
        .epoch_state
        .clients
        .iter()
        .position(|client| *client.id.signer() == signer);
    match position {
        Some(found) if found as u64 != target_index => {
            err!(ProgramError::TargetMismatch)
        },
        Some(found) if coordinator.ejections_charged() => {
            Ok(Landing::Eject(found as u64))
        },
        _ => Ok(Landing::Due),
    }
}
