use anchor_lang::prelude::*;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_coordinator::RunState;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::AuditVerdict;
use crate::state::EjectionCharge;
use crate::state::Run;
use crate::ProgramError;

/// Whether `user` was ejected in the epoch under way, by a conviction, an
/// authority slash or an equivocation: the ledger charges that ejection only
/// when the epoch ends. An epoch's roster is cleared when the next one starts,
/// and between the two no charge is pending, so an ejection already charged
/// is never read here.
pub fn ejection_pending_charge(coordinator: &Coordinator, user: &Pubkey) -> bool {
    let signer = user.to_bytes();
    coordinator.charges_pending()
        && coordinator
            .epoch_state
            .clients
            .iter()
            .chain(coordinator.epoch_state.exited_clients.iter())
            .any(|client| {
                *client.id.signer() == signer
                    && client.state == ClientState::Ejected
            })
}

/// What `user`'s bond still owes to slashes the ledger in `account` has not
/// charged yet, at `now` in `run`: an ejection in the epoch under way, and its
/// own conviction, `own_verdict`, if that cannot be overturned any more but
/// has not landed (`AuditVerdict::unledgered_charge`).
///
/// The bond checks read what the ledger charged. Neither of these reached it
/// yet, so each used to count for nothing there, and its target joined,
/// claimed and voted on a bond it was about to lose. An ejection is charged at
/// the epoch's rate. A conviction that lands later is charged at the rate of
/// the epoch it lands in, or at the withdrawal at the rate then, so it counts
/// at the higher of this epoch's rate and the next one's.
pub fn unledgered_conviction_charge(
    run: &Run,
    account: &CoordinatorAccount,
    user: &Pubkey,
    own_verdict: Option<&AuditVerdict>,
    now: i64,
) -> u64 {
    let rates = &account.state.clients_state;
    let epoch_rate = rates.current_epoch_rates.slashing_rate_per_client;
    let ejection = if ejection_pending_charge(&account.state.coordinator, user)
    {
        epoch_rate
    } else {
        0
    };
    let landing_rate =
        epoch_rate.max(rates.future_epoch_rates.slashing_rate_per_client);
    let conviction = own_verdict.map_or(0, |verdict| {
        verdict.unledgered_charge(run, now, landing_rate)
    });
    ejection.saturating_add(conviction)
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

/// Whether the ledger has charged an ejection landed in the epoch that
/// started at `epoch_start`: that epoch is over. Every way out of an epoch
/// that can eject passes through its charge (`Coordinator::ejections_charged`),
/// so an epoch that no longer has charges pending, or a later one, has made it.
pub fn epoch_charged(coordinator: &Coordinator, epoch_start: u64) -> bool {
    !(coordinator.charges_pending()
        && coordinator.epoch_state.start_timestamp == epoch_start)
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

    /// An ejection is owed from when it lands until its epoch is charged:
    /// on the roster or among the clients that left this epoch, through
    /// `Cooldown`. Once the epoch is over nothing is pending, and the next
    /// epoch starts with a fresh roster.
    #[test]
    fn an_ejection_is_owed_until_its_epoch_is_charged() {
        for run_state in [
            RunState::RoundTrain,
            RunState::RoundWitness,
            RunState::Cooldown,
        ] {
            let coordinator = coordinator_with(run_state, ClientState::Ejected);
            assert!(
                ejection_pending_charge(&coordinator, &target()),
                "{run_state}"
            );
        }

        // Moved off the roster at the round's end, still this epoch's.
        let mut left = coordinator_with(RunState::Cooldown, ClientState::Healthy);
        let mut ejected = left.epoch_state.clients[1];
        ejected.state = ClientState::Ejected;
        left.epoch_state.exited_clients = FixedVec::from_iter([ejected]);
        left.epoch_state.clients = FixedVec::from_iter([left.epoch_state.clients[0]]);
        assert!(ejection_pending_charge(&left, &target()));

        // Charged: the epoch is over.
        for run_state in [RunState::WaitingForMembers, RunState::Paused] {
            let coordinator = coordinator_with(run_state, ClientState::Ejected);
            assert!(
                !ejection_pending_charge(&coordinator, &target()),
                "{run_state}"
            );
        }

        // Not ejected, or someone else.
        for state in [ClientState::Healthy, ClientState::Dropped, ClientState::Withdrawn] {
            let coordinator = coordinator_with(RunState::RoundTrain, state);
            assert!(
                !ejection_pending_charge(&coordinator, &target()),
                "{state:?}"
            );
        }
        let coordinator = coordinator_with(RunState::RoundTrain, ClientState::Ejected);
        assert!(!ejection_pending_charge(
            &coordinator,
            &Pubkey::new_from_array([1; 32])
        ));
    }
}
