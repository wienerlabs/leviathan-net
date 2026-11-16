//! The round that ends an epoch keeps its roster as it was drawn until the
//! cooldown ends, so its committee can still convict a fraud in it through the
//! cooldown. The cooldown's end compacts the roster before the epoch is charged,
//! as it always has (wienerlabs/leviathan-net#40).

use bytemuck::Zeroable;
use psyche_coordinator::model::Model;
use psyche_coordinator::Client;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::RunState;
use psyche_coordinator::Witness;
use psyche_coordinator::WitnessProof;
use psyche_core::FixedVec;
use psyche_core::NodeIdentity;

#[derive(serde::Deserialize)]
struct RunConfig {
    config: CoordinatorConfig,
    model: Model,
}

const SEED: u64 = 0x5eed;
const HEIGHT: u32 = 3;
const STEP: u32 = 5;

fn client(seed: u8, state: ClientState) -> Client {
    Client {
        id: NodeIdentity::from_single_key([seed; 32]),
        state,
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

/// A round of the genesis rehearsal config whose witness phase is over, drawn
/// over `roster`, with a witness from each healthy client.
fn witnessed_round(roster: &[ClientState]) -> Box<Coordinator> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/leviathan/genesis-rehearsal.toml"
    );
    let run: RunConfig = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut coordinator = Box::new(Coordinator::zeroed());
    coordinator.config = run.config;
    coordinator.model = run.model;
    coordinator.run_state = RunState::RoundWitness;
    coordinator.progress.step = STEP;
    coordinator.epoch_state.first_round = false.into();
    coordinator.epoch_state.clients = FixedVec::from_iter(
        roster
            .iter()
            .enumerate()
            .map(|(index, state)| client(index as u8 + 1, *state)),
    );
    let round = &mut coordinator.epoch_state.rounds[0];
    round.height = HEIGHT;
    round.clients_len = roster.len() as u16;
    for (index, state) in roster.iter().enumerate() {
        if *state == ClientState::Healthy {
            round.witnesses.push(witness(index as u64)).unwrap();
        }
    }
    coordinator
}

fn tick(coordinator: &mut Coordinator, unix_timestamp: u64) {
    coordinator
        .tick(
            None::<std::iter::Empty<&NodeIdentity>>,
            unix_timestamp,
            Some(SEED),
        )
        .unwrap();
}

fn states(coordinator: &Coordinator) -> Vec<ClientState> {
    coordinator
        .epoch_state
        .clients
        .iter()
        .map(|client| client.state)
        .collect()
}

const ROSTER: [ClientState; 4] = [
    ClientState::Healthy,
    ClientState::Dropped,
    ClientState::Healthy,
    ClientState::Healthy,
];

#[test]
fn the_round_that_ends_the_epoch_keeps_its_roster_through_the_cooldown() {
    let mut coordinator = witnessed_round(&ROSTER);
    // This round's step is the epoch's last.
    coordinator.epoch_state.last_step = STEP + 1;
    let witness_over = coordinator.config.round_witness_time + 1;
    tick(&mut coordinator, witness_over);

    assert_eq!(coordinator.run_state, RunState::Cooldown);
    // The roster the round was drawn over, the dropped client included, so
    // a seat read by roster index is still the seat the round drew.
    assert_eq!(states(&coordinator), ROSTER);
    assert_eq!(
        coordinator.current_round().unwrap().clients_len as usize,
        coordinator.epoch_state.clients.len()
    );
    assert!(coordinator.epoch_state.exited_clients.is_empty());

    // The cooldown's end compacts it, at the round's own height, before the
    // epoch is charged.
    let cooldown_over = witness_over + coordinator.config.cooldown_time + 1;
    tick(&mut coordinator, cooldown_over);
    assert_eq!(coordinator.run_state, RunState::WaitingForMembers);
    assert_eq!(states(&coordinator), [ClientState::Healthy; 3]);
    let exited = &coordinator.epoch_state.exited_clients;
    assert_eq!(exited.len(), 1);
    assert_eq!(exited[0].state, ClientState::Dropped);
    assert_eq!(exited[0].exited_height, HEIGHT);
}

#[test]
fn a_round_too_short_of_healthy_clients_ends_the_epoch_on_its_roster() {
    // One healthy client of four, under the rehearsal's min_clients: too few
    // to go on, so this round ends the epoch though its step is not the last.
    let roster = [
        ClientState::Healthy,
        ClientState::Dropped,
        ClientState::Withdrawn,
        ClientState::Dropped,
    ];
    let mut coordinator = witnessed_round(&roster);
    assert!(coordinator.config.min_clients >= 2);
    let witness_over = coordinator.config.round_witness_time + 1;
    tick(&mut coordinator, witness_over);

    // One healthy client is too few to go on, counted as the compaction
    // would count it, and the roster stays as drawn through the cooldown.
    assert_eq!(coordinator.run_state, RunState::Cooldown);
    assert_eq!(states(&coordinator), roster);
}

#[test]
fn a_round_that_goes_on_compacts_its_roster_as_before() {
    let mut coordinator = witnessed_round(&ROSTER);
    let witness_over = coordinator.config.round_witness_time + 1;
    tick(&mut coordinator, witness_over);

    assert_eq!(coordinator.run_state, RunState::RoundTrain);
    assert_eq!(states(&coordinator), [ClientState::Healthy; 3]);
    let exited = &coordinator.epoch_state.exited_clients;
    assert_eq!(exited.len(), 1);
    assert_eq!(exited[0].exited_height, HEIGHT);
    // The next round is drawn over the compacted roster.
    assert_eq!(coordinator.current_round().unwrap().clients_len, 3);
}
