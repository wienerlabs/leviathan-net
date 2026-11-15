//! An ejection only counts if the ledger charges it, and the ledger charges an
//! epoch's ejections when that epoch ends (wienerlabs/leviathan-net#36).

use bytemuck::Zeroable;
use psyche_coordinator::{Client, ClientState, Coordinator, CoordinatorError, RunState};
use psyche_core::{FixedVec, NodeIdentity};

fn coordinator_in(run_state: RunState) -> Box<Coordinator> {
    let mut coordinator = Box::new(Coordinator::zeroed());
    coordinator.run_state = run_state;
    coordinator.epoch_state.clients = FixedVec::from_iter([Client {
        id: NodeIdentity::from_single_key([7; 32]),
        state: ClientState::Healthy,
        exited_height: 0,
    }]);
    coordinator
}

#[test]
fn an_ejection_lands_while_the_epoch_is_still_to_be_charged() {
    for run_state in [
        RunState::Warmup,
        RunState::RoundTrain,
        RunState::RoundWitness,
        RunState::Cooldown,
    ] {
        let mut coordinator = coordinator_in(run_state);
        assert!(coordinator.charges_pending(), "{run_state}");
        coordinator.eject(0).unwrap();
        assert_eq!(
            coordinator.epoch_state.clients[0].state,
            ClientState::Ejected
        );
    }
}

#[test]
fn an_ejection_between_epochs_is_refused() {
    // The roster of the epoch that just ended is still on the account here, so
    // an ejection would find its client and never be charged.
    for run_state in [
        RunState::WaitingForMembers,
        RunState::Paused,
        RunState::Finished,
        RunState::Uninitialized,
    ] {
        let mut coordinator = coordinator_in(run_state);
        assert!(!coordinator.charges_pending(), "{run_state}");
        assert!(matches!(
            coordinator.eject(0),
            Err(CoordinatorError::SlashOutsideEpoch)
        ));
        assert_eq!(
            coordinator.epoch_state.clients[0].state,
            ClientState::Healthy
        );
    }
}
