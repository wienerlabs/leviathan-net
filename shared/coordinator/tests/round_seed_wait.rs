//! A round starts only on the seed it is given: a tick, or a last warmup
//! witness, that would start a round without one leaves the run exactly where
//! it was, and the next one with a seed starts it on that seed
//! (wienerlabs/leviathan-net#62).

use bytemuck::Zeroable;
use psyche_coordinator::model::Model;
use psyche_coordinator::Client;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::RunState;
use psyche_coordinator::TickResult;
use psyche_coordinator::Witness;
use psyche_coordinator::WitnessProof;
use psyche_core::FixedVec;
use psyche_core::NodeIdentity;

#[derive(serde::Deserialize)]
struct RunConfig {
    config: CoordinatorConfig,
    model: Model,
}

const SEED: u64 = 0x5eed_0fa7_0a1d;
const CLIENTS: u8 = 3;

fn client(seed: u8) -> Client {
    Client {
        id: NodeIdentity::from_single_key([seed; 32]),
        state: ClientState::Healthy,
        exited_height: 0,
    }
}

fn witness(index: u64) -> Witness {
    Witness {
        proof: WitnessProof {
            position: index,
            index,
            witness: true.into(),
        },
        participant_bloom: Default::default(),
        broadcast_bloom: Default::default(),
        broadcast_merkle: Default::default(),
    }
}

/// A run on the genesis rehearsal config, `state` since time zero, with three
/// healthy clients on its roster.
fn run_in(state: RunState) -> Coordinator {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/leviathan/genesis-rehearsal.toml"
    );
    let run: RunConfig = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut coordinator = Coordinator::zeroed();
    coordinator.config = run.config;
    coordinator.model = run.model;
    coordinator.run_state = state;
    coordinator.progress.step = 5;
    coordinator.epoch_state.clients = FixedVec::from_iter((1..=CLIENTS).map(client));
    coordinator
}

fn snapshot(coordinator: &Coordinator) -> Vec<u8> {
    bytemuck::bytes_of(coordinator).to_vec()
}

/// A round under way, every witness in, its witness phase over.
fn witnessed_round() -> Coordinator {
    let mut coordinator = run_in(RunState::RoundWitness);
    coordinator.epoch_state.first_round = false.into();
    coordinator.epoch_state.rounds[0].height = 3;
    coordinator.epoch_state.rounds[0].clients_len = CLIENTS as u16;
    for index in 0..CLIENTS as u64 {
        coordinator.epoch_state.rounds[0]
            .witnesses
            .push(witness(index))
            .unwrap();
    }
    coordinator
}

#[test]
fn a_witness_phase_that_would_start_a_round_waits_for_its_seed() {
    let mut coordinator = witnessed_round();
    let after_timeout = coordinator.config.round_witness_time + 1;
    let before = snapshot(&coordinator);

    assert_eq!(
        coordinator
            .tick(None::<std::iter::Empty<&NodeIdentity>>, after_timeout, None)
            .unwrap(),
        TickResult::NeedsSeed
    );
    assert_eq!(
        snapshot(&coordinator),
        before,
        "a tick without a seed changed the run"
    );

    coordinator
        .tick(
            None::<std::iter::Empty<&NodeIdentity>>,
            after_timeout,
            Some(SEED),
        )
        .unwrap();
    assert_eq!(coordinator.run_state, RunState::RoundTrain);
    // The end of the round ran once.
    assert_eq!(coordinator.progress.step, 6);
    let round = coordinator.current_round().unwrap();
    assert_eq!(round.height, 4);
    assert_eq!(round.random_seed, SEED);
}

#[test]
fn a_warmup_that_would_start_the_first_round_waits_for_its_seed() {
    // Timed out.
    let mut coordinator = run_in(RunState::Warmup);
    let after_timeout = coordinator.config.warmup_time + 1;
    let before = snapshot(&coordinator);
    coordinator
        .tick(None::<std::iter::Empty<&NodeIdentity>>, after_timeout, None)
        .unwrap();
    assert_eq!(snapshot(&coordinator), before);
    coordinator
        .tick(
            None::<std::iter::Empty<&NodeIdentity>>,
            after_timeout,
            Some(SEED),
        )
        .unwrap();
    assert_eq!(coordinator.run_state, RunState::RoundTrain);
    assert_eq!(coordinator.current_round().unwrap().random_seed, SEED);
}

#[test]
fn the_last_warmup_witness_without_a_seed_leaves_the_first_round_to_the_next_tick() {
    let mut coordinator = run_in(RunState::Warmup);
    for (seed, index) in (1..=CLIENTS).zip(0..) {
        coordinator
            .warmup_witness(&client(seed).id, witness(index), 1, None)
            .unwrap();
    }
    // Every witness is in and recorded, and the round has not started.
    assert_eq!(coordinator.run_state, RunState::Warmup);
    assert_eq!(
        coordinator.current_round().unwrap().witnesses.len(),
        CLIENTS as usize
    );
    // The warmup's next tick with a seed starts it, timed out or not.
    coordinator
        .tick(None::<std::iter::Empty<&NodeIdentity>>, 2, None)
        .unwrap();
    assert_eq!(coordinator.run_state, RunState::Warmup);
    coordinator
        .tick(None::<std::iter::Empty<&NodeIdentity>>, 2, Some(SEED))
        .unwrap();
    assert_eq!(coordinator.run_state, RunState::RoundTrain);
    assert_eq!(coordinator.current_round().unwrap().random_seed, SEED);
}

#[test]
fn a_last_warmup_witness_with_a_seed_starts_the_first_round_on_it() {
    let mut coordinator = run_in(RunState::Warmup);
    for (seed, index) in (1..=CLIENTS).zip(0..) {
        coordinator
            .warmup_witness(&client(seed).id, witness(index), 1, Some(SEED))
            .unwrap();
    }
    assert_eq!(coordinator.run_state, RunState::RoundTrain);
    assert_eq!(coordinator.current_round().unwrap().random_seed, SEED);
}

#[test]
fn ticks_that_start_no_round_need_no_seed() {
    let mut coordinator = run_in(RunState::RoundTrain);
    coordinator.epoch_state.rounds[0].clients_len = CLIENTS as u16;
    let after_timeout = coordinator.config.max_round_train_time + 1;
    coordinator
        .tick(None::<std::iter::Empty<&NodeIdentity>>, after_timeout, None)
        .unwrap();
    assert_eq!(coordinator.run_state, RunState::RoundWitness);
}

/// The centralized server withdraws a client that disconnects during the
/// warmup, and the warmup's next tick drops it from the roster. The warmup
/// witnesses already in move with their clients: the one that left no longer
/// counts towards the first round, and the client whose position it took can
/// still witness for itself.
#[test]
fn warmup_witnesses_follow_their_clients_when_the_roster_shrinks() {
    let no_clients = || None::<std::iter::Empty<&NodeIdentity>>;
    let mut coordinator = run_in(RunState::Warmup);
    // Every client is a witness, and one is enough to run.
    coordinator.config.witness_nodes = 0;
    coordinator.config.min_clients = 1;
    for (seed, index) in [(1, 0), (2, 1)] {
        coordinator
            .warmup_witness(&client(seed).id, witness(index), 1, None)
            .unwrap();
    }
    coordinator.withdraw(0).unwrap();

    // Not timed out, so the tick only drops the client that left.
    assert_eq!(
        coordinator.tick(no_clients(), 2, None).unwrap(),
        TickResult::Ticked
    );
    assert_eq!(coordinator.run_state, RunState::Warmup);
    let roster: Vec<NodeIdentity> = coordinator
        .epoch_state
        .clients
        .iter()
        .map(|client| client.id)
        .collect();
    assert_eq!(roster, vec![client(2).id, client(3).id]);
    let witnesses = &coordinator.current_round().unwrap().witnesses;
    assert_eq!(
        witnesses.len(),
        1,
        "the witness of the client that left counts"
    );
    assert_eq!(witnesses[0].proof.index, 0, "client 2 sits first now");

    // Client 3 sits where client 2 did, and its witness is its own.
    coordinator
        .warmup_witness(&client(3).id, witness(1), 3, None)
        .unwrap();
    // Every client left has witnessed: the first round waits for its seed
    // alone.
    assert_eq!(
        coordinator.tick(no_clients(), 3, None).unwrap(),
        TickResult::NeedsSeed
    );
    coordinator.tick(no_clients(), 3, Some(SEED)).unwrap();
    assert_eq!(coordinator.run_state, RunState::RoundTrain);
    assert_eq!(coordinator.current_round().unwrap().random_seed, SEED);
}
