//! Records a landmine found in the second pass of the internal review
//! (wienerlabs/leviathan#15).
//!
//! `assign_data_for_state` asserts that no round ever reserves tie-breaker
//! tasks. The assert is dormant because `start_round_train` is called with
//! `tie_breaker_tasks = 0` at every one of its call sites, so
//! `CommitteeSelection::from_coordinator` never produces a TieBreaker seat.
//!
//! That matters because the recommended fix for the daemon/chain committee
//! mismatch (finding 4) is to write the run's `tie_breaker_committee_size` into
//! `round.tie_breaker_tasks`, so that both sides derive the same committee from
//! one field. Doing that walks straight into this assert. The fix has to teach
//! `assign_data_for_state` to skip tie-breakers first.

use psyche_coordinator::Client;
use psyche_coordinator::ClientState;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::Coordinator;
use psyche_coordinator::assign_data_for_state;
use psyche_core::FixedVec;
use psyche_core::NodeIdentity;

use bytemuck::Zeroable;

fn coordinator_with(tie_breaker_tasks: u16) -> Coordinator {
    let clients: Vec<Client> = (0..8u8)
        .map(|i| {
            let mut key = [0u8; 32];
            key[0] = i;
            Client {
                id: NodeIdentity::from_single_key(key),
                state: ClientState::Healthy,
                exited_height: 0,
            }
        })
        .collect();

    let mut coordinator = Coordinator::zeroed();
    coordinator.config.total_steps = 10;
    coordinator.config.global_batch_size_start = 64;
    coordinator.config.global_batch_size_end = 64;
    coordinator.epoch_state.clients = FixedVec::from_iter(clients);
    {
        let round = coordinator.current_round_mut().unwrap();
        round.clients_len = 8;
        round.tie_breaker_tasks = tie_breaker_tasks;
        round.random_seed = 4242;
    }
    coordinator
}

/// Today's configuration: no tie-breaker seats, data assignment works.
#[test]
fn no_tie_breakers_assigns_data_normally() {
    let coordinator = coordinator_with(0);
    let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
    let assignments = assign_data_for_state(&coordinator, &selection);
    assert!(
        !assignments.is_empty(),
        "with no tie-breakers every client is a trainer or verifier"
    );
}

/// Reserve even one tie-breaker seat and the same call aborts.
#[test]
#[should_panic(expected = "assertion")]
fn reserving_a_tie_breaker_seat_trips_the_assert() {
    let coordinator = coordinator_with(2);
    let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
    let _ = assign_data_for_state(&coordinator, &selection);
}
