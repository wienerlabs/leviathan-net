use anchor_lang::prelude::*;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_coordinator::RunState;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::AuditVerdict;
use crate::state::EjectionCharge;
use crate::ProgramError;

/// What a participant's own conviction, `own_verdict`, will still take from
/// its bond that the ledger in `account` has not charged
/// (`AuditVerdict::unledgered_charge`).
///
/// The bond checks read what the ledger charged. A conviction that stands but
/// has not reached the ledger used to count for nothing there, so its target
/// joined, claimed and voted on a bond that conviction was about to take from.
pub fn unledgered_conviction_charge(
    account: &CoordinatorAccount,
    own_verdict: Option<&AuditVerdict>,
) -> u64 {
    own_verdict.map_or(0, |verdict| {
        verdict.unledgered_charge(
            &account.state.coordinator,
            ejection_charge(account).rate,
        )
    })
}

/// The verdict at a participant's own verdict PDA, read from the account
/// passed for it. The address is pinned by the caller's seeds, so an empty
/// account there means the participant was never disputed.
pub fn read_own_verdict(account: &AccountInfo) -> Result<Option<AuditVerdict>> {
    let data = account.try_borrow_data()?;
    AuditVerdict::from_pda_account(account.owner, &data)
}

/// What the ledger will charge an ejection landed now, in the epoch the
/// coordinator in `account` is in.
pub fn ejection_charge(account: &CoordinatorAccount) -> EjectionCharge {
    EjectionCharge {
        epoch_start: account.state.coordinator.epoch_state.start_timestamp,
        rate: account
            .state
            .clients_state
            .current_epoch_rates
            .slashing_rate_per_client,
    }
}

/// Where a conviction against a target lands right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Landing {
    /// Eject the target at this roster index: the coordinator is certain to
    /// charge it at the end of this epoch.
    Eject(u64),
    /// The conviction stands but cannot reach the ledger now, because the
    /// target is not on the roster or no ejection here would be charged. The
    /// verdict records it as `Due` (wienerlabs/leviathan-net#36). So does a
    /// target the coordinator can no longer eject: withdrawn when the run
    /// paused or lost its witnesses, or already ejected by another conviction.
    /// Ejecting it would be refused and undo the vote that reached the
    /// conviction (wienerlabs/leviathan-net#40). A target already ejected pays
    /// for that other conviction through the ledger and for this one at its
    /// withdrawal: two convictions are charged once each, as they are when the
    /// first ejection has already left the roster (`docs/DISPUTE_DESIGN.md`).
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
        Some(found)
            if coordinator.ejections_charged()
                && matches!(
                    coordinator.epoch_state.clients[found].state,
                    ClientState::Healthy | ClientState::Dropped
                ) =>
        {
            Ok(Landing::Eject(found as u64))
        },
        _ => Ok(Landing::Due),
    }
}

/// Whether the committee in force was drawn over the roster as it stands.
///
/// Seats are drawn per round, over the roster as it stood when the round
/// began, and a voter's seat is read from its index in the roster now. Within
/// a round those agree: clients change state in place and leave only at the
/// round's end. The round that ends an epoch keeps its roster through
/// `Cooldown`, so they agree there too, and its committee can convict a fraud
/// in it until the epoch is charged (wienerlabs/leviathan-net#40). After
/// that they may not. The end of `Cooldown` compacts the roster, shifting
/// indices, and a warmup replaces it with the next epoch's while the round is
/// still the last epoch's. A seat read through a shifted index belongs to
/// someone else, and the old seed is public, so a vote is only taken while the
/// two still agree: not in a warmup, and only if nobody has left since the
/// round was drawn, which is when the lengths still match
/// (wienerlabs/leviathan-net#36).
///
/// A seat belongs to whoever the round drew for it, whatever has happened to
/// it since: a verifier dropped from training, or withdrawn when the run lost
/// its witnesses, still holds its seat, and its bond answers for its vote
/// (wienerlabs/leviathan-net#39).
pub fn committee_matches_roster(coordinator: &Coordinator) -> bool {
    if coordinator.run_state == RunState::Warmup {
        return false;
    }
    coordinator.current_round().is_some_and(|round| {
        round.clients_len as usize == coordinator.epoch_state.clients.len()
    })
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use psyche_coordinator::Client;
    use psyche_core::FixedVec;
    use psyche_core::NodeIdentity;

    use super::*;

    fn target() -> Pubkey {
        Pubkey::new_from_array([9; 32])
    }

    /// A coordinator in `run_state` whose roster is someone else, then the
    /// target in `state`.
    fn coordinator_with(
        run_state: RunState,
        state: ClientState,
    ) -> Box<Coordinator> {
        let mut coordinator = Box::new(Coordinator::zeroed());
        coordinator.run_state = run_state;
        coordinator.epoch_state.clients = FixedVec::from_iter([
            Client {
                id: NodeIdentity::from_single_key([1; 32]),
                state: ClientState::Healthy,
                exited_height: 0,
            },
            Client {
                id: NodeIdentity::from_single_key(target().to_bytes()),
                state,
                exited_height: 0,
            },
        ]);
        coordinator
    }

    #[test]
    fn a_target_that_can_be_ejected_is_ejected_at_its_index() {
        for state in [ClientState::Healthy, ClientState::Dropped] {
            let coordinator = coordinator_with(RunState::Cooldown, state);
            assert_eq!(
                conviction_landing(&coordinator, &target(), 1).unwrap(),
                Landing::Eject(1),
                "{state:?}"
            );
        }
    }

    #[test]
    fn a_target_on_the_roster_that_can_no_longer_be_ejected_is_due() {
        // Withdrawn when the run paused or lost its witnesses, or already
        // ejected by another conviction: the coordinator refuses an ejection,
        // and a conviction that tried one would undo the vote that reached it
        // (wienerlabs/leviathan-net#40).
        for state in [ClientState::Withdrawn, ClientState::Ejected] {
            let coordinator = coordinator_with(RunState::Cooldown, state);
            assert_eq!(
                conviction_landing(&coordinator, &target(), 1).unwrap(),
                Landing::Due,
                "{state:?}"
            );
        }
    }

    #[test]
    fn a_target_named_at_another_index_is_refused() {
        let coordinator =
            coordinator_with(RunState::RoundTrain, ClientState::Healthy);
        assert_eq!(
            conviction_landing(&coordinator, &target(), 0).unwrap_err(),
            error!(ProgramError::TargetMismatch),
        );
    }

    #[test]
    fn nothing_lands_where_no_ejection_would_be_charged() {
        let coordinator =
            coordinator_with(RunState::WaitingForMembers, ClientState::Healthy);
        assert_eq!(
            conviction_landing(&coordinator, &target(), 1).unwrap(),
            Landing::Due,
        );
        let coordinator =
            coordinator_with(RunState::RoundTrain, ClientState::Healthy);
        assert_eq!(
            conviction_landing(&coordinator, &Pubkey::new_unique(), 1).unwrap(),
            Landing::Due,
        );
    }
}
