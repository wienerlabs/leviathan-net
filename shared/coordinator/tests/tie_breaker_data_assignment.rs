//! Records a landmine found in the second pass of the internal review
//! (wienerlabs/leviathan#15).
//!
//! `assign_data_for_state` used to assert that no round ever reserves
//! tie-breaker tasks. The assert was dormant only because `start_round_train`
//! was called with `tie_breaker_tasks = 0` at every one of its call sites.
//!
//! It had to go before finding 4 could be fixed: that fix writes the run's
//! `tie_breaker_committee_size` into `round.tie_breaker_tasks` so the daemon and
//! the chain derive one committee from one field, which would have walked
//! straight into the assert on the first round. Tie-breakers are now skipped
//! the same way verifiers always were - they are simply not trainers.

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

/// Reserving tie-breaker seats is now an ordinary configuration: the seats are
/// skipped, the trainers that remain still get the whole batch, and nothing
/// aborts.
#[test]
fn reserving_tie_breaker_seats_just_leaves_them_out_of_the_data() {
    let coordinator = coordinator_with(2);
    let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
    let assignments = assign_data_for_state(&coordinator, &selection);

    assert!(!assignments.is_empty(), "the trainers still get data");
    assert_eq!(
        assignments.len(),
        selection.get_num_trainer_nodes() as usize,
        "one batch per trainer, and none for the reserved seats"
    );

    // The batch is covered exactly once, with no hole where a tie-breaker was.
    let total: u64 = assignments
        .keys()
        .map(|batch| batch.0.end - batch.0.start + 1)
        .sum();
    assert_eq!(total, 64, "the whole global batch is still assigned");
}

/// A program running an appeals court seats its bench in each round as the
/// round starts, and never more seats than the round's roster: a draw with
/// more seats than clients cannot be made, and every witness would be refused.
#[test]
fn a_round_seats_its_tie_breakers_up_to_its_roster() {
    let mut coordinator = coordinator_with(0);
    coordinator.seat_tie_breakers(3);
    assert_eq!(coordinator.current_round().unwrap().tie_breaker_tasks, 3);
    coordinator.seat_tie_breakers(20);
    assert_eq!(coordinator.current_round().unwrap().tie_breaker_tasks, 8);
    assert!(CommitteeSelection::from_coordinator(&coordinator, 0).is_ok());
}

/// The audit lottery draws the round the way its clients draw it, with the
/// seats the round reserves: every audit names a batch its target was
/// assigned, and no tie-breaker seat audits or is audited.
#[test]
fn the_audit_lottery_draws_the_round_its_clients_train() {
    let mut coordinator = coordinator_with(0);
    coordinator.config.verification_percent = 50;
    coordinator.seat_tie_breakers(2);
    let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
    let trained = assign_data_for_state(&coordinator, &selection);
    let audits = psyche_coordinator::select_audits_for_current_round(&coordinator).unwrap();
    assert!(!audits.is_empty());
    for audit in &audits {
        assert_eq!(trained.get(&audit.batch_id), Some(&audit.target));
        assert_eq!(
            selection.get_committee(audit.verifier_index).committee,
            psyche_coordinator::Committee::Verifier
        );
    }
}
