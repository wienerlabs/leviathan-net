//! A health check names a client by its seat in the previous round's draw,
//! and the committee proof it carries is read off that draw. Every round
//! starts by compacting the roster, so once someone has left since then,
//! every seat after theirs holds a different client than the one the draw
//! seated there. Judged by its seat's old role, an honest verifier read as a
//! trainer that trained nothing, and a health check dropped it.

use bytemuck::Zeroable;
use psyche_coordinator::model::Model;
use psyche_coordinator::Client;
use psyche_coordinator::ClientState;
use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::Coordinator;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::CoordinatorError;
use psyche_coordinator::RunState;
use psyche_coordinator::Witness;
use psyche_coordinator::WitnessBloom;
use psyche_coordinator::WitnessProof;
use psyche_core::sha256;
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
const CLIENTS: usize = 10;

fn client(seed: u8, state: ClientState) -> Client {
    Client {
        id: NodeIdentity::from_single_key([seed; 32]),
        state,
        exited_height: 0,
    }
}

/// A round of the genesis rehearsal config, with half the seats verifying,
/// drawn from `round_seed` over `roster`, whose witness phase is over. Its one
/// witness heard every healthy client the round seated as a trainer.
fn witnessed_round(roster: &[ClientState], round_seed: u64) -> Box<Coordinator> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../config/leviathan/genesis-rehearsal.toml"
    );
    let run: RunConfig = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut coordinator = Box::new(Coordinator::zeroed());
    coordinator.config = run.config;
    coordinator.config.verification_percent = 50;
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

    let draw = CommitteeSelection::new(
        0,
        coordinator.config.witness_nodes as usize,
        coordinator.config.verification_percent,
        roster.len(),
        round_seed,
    )
    .unwrap();
    let mut participants = participant_bloom();
    for (index, client) in coordinator.epoch_state.clients.iter().enumerate() {
        let trained = draw.get_committee(index as u64).committee == Committee::Trainer;
        if trained && client.state == ClientState::Healthy {
            participants.add(&sha256(client.id.signer()));
        }
    }
    let round = &mut coordinator.epoch_state.rounds[0];
    round.height = HEIGHT;
    round.clients_len = roster.len() as u16;
    round.random_seed = round_seed;
    round
        .witnesses
        .push(Witness {
            proof: WitnessProof {
                position: 0,
                index: 0,
                witness: true.into(),
            },
            participant_bloom: participants,
            broadcast_bloom: Default::default(),
            broadcast_merkle: Default::default(),
        })
        .unwrap();
    coordinator
}

/// The witness phase ends and the next round starts, compacting the roster.
fn next_round(coordinator: &mut Coordinator) {
    let witness_over = coordinator.config.round_witness_time + 1;
    coordinator
        .tick(
            None::<std::iter::Empty<&NodeIdentity>>,
            witness_over,
            Some(SEED),
        )
        .unwrap();
    assert_eq!(coordinator.run_state, RunState::RoundTrain);
    assert_eq!(coordinator.current_round().unwrap().height, HEIGHT + 1);
}

/// A participant bloom with fixed keys, so what it holds is the same every run.
fn participant_bloom() -> WitnessBloom {
    WitnessBloom::new(16 * 64, &[3, 5, 7, 11, 13, 17, 19, 23])
}

fn healthy_roster() -> [ClientState; CLIENTS] {
    [ClientState::Healthy; CLIENTS]
}

/// The first seat left during the round, so every client after it moves down
/// one seat when the next round starts.
fn roster_with_a_leaver() -> [ClientState; CLIENTS] {
    let mut roster = healthy_roster();
    roster[0] = ClientState::Dropped;
    roster
}

/// A round seed whose draw seats a verifier right after a trainer, past the
/// leaver's seat: once the roster moves down a seat, that verifier sits where
/// the draw put a trainer.
fn seed_moving_a_verifier_onto_a_trainer_seat() -> (u64, usize) {
    for round_seed in 1..1_000u64 {
        let draw = CommitteeSelection::new(0, 1, 50, CLIENTS, round_seed).unwrap();
        let role = |seat: usize| draw.get_committee(seat as u64).committee;
        if let Some(seat) = (1..CLIENTS - 1)
            .find(|&seat| role(seat) == Committee::Trainer && role(seat + 1) == Committee::Verifier)
        {
            return (round_seed, seat);
        }
    }
    panic!("no seed in a thousand draws a verifier after a trainer");
}

/// What every honest witness's client sends: each client on the roster now,
/// with the proof the previous round's draw gives its seat now.
fn proof_at(
    coordinator: &Coordinator,
    seat: usize,
) -> (NodeIdentity, psyche_coordinator::CommitteeProof) {
    let draw = CommitteeSelection::from_coordinator(coordinator, -1).unwrap();
    (
        coordinator.epoch_state.clients[seat].id,
        draw.get_committee(seat as u64),
    )
}

#[test]
fn a_verifier_whose_seat_moved_is_not_health_checked_out() {
    let (round_seed, trainer_seat) = seed_moving_a_verifier_onto_a_trainer_seat();
    let mut coordinator = witnessed_round(&roster_with_a_leaver(), round_seed);
    // The verifier the draw seated after that trainer.
    let verifier = coordinator.epoch_state.clients[trainer_seat + 1].id;
    next_round(&mut coordinator);
    assert_eq!(coordinator.epoch_state.clients.len(), CLIENTS - 1);
    assert_eq!(
        coordinator.epoch_state.clients[trainer_seat].id, verifier,
        "the verifier moved down onto the trainer's seat"
    );

    let (id, proof) = proof_at(&coordinator, trainer_seat);
    assert_eq!(
        proof.committee,
        Committee::Trainer,
        "its new seat was drawn a trainer"
    );
    let judged = coordinator.healthy(&id, &proof);
    assert!(
        matches!(judged, Err(CoordinatorError::RosterMoved)),
        "a seat on a moved roster names another client than the draw did, judged {judged:?}"
    );
    let checked = coordinator.health_check(&id, vec![(id, proof)]);
    assert!(
        matches!(checked, Err(CoordinatorError::RosterMoved)),
        "the health check was taken: {checked:?}"
    );
    assert_eq!(
        coordinator.epoch_state.clients[trainer_seat].state,
        ClientState::Healthy,
        "the honest verifier was dropped"
    );
}

/// Nobody left, so every seat holds the client the draw put there, and a
/// health check judges each by the role it was drawn: a verifier is healthy,
/// a trainer the witness heard is healthy, and one it did not is dropped.
#[test]
fn a_roster_that_did_not_move_is_health_checked_as_drawn() {
    let (round_seed, trainer_seat) = seed_moving_a_verifier_onto_a_trainer_seat();
    // A trainer that trained nothing: still on the roster, but no witness
    // heard from it.
    let mut coordinator = witnessed_round(&healthy_roster(), round_seed);
    let silent = trainer_seat;
    let hash = sha256(coordinator.epoch_state.clients[silent].id.signer());
    let bloom = &mut coordinator.epoch_state.rounds[0].witnesses[0].participant_bloom;
    let mut without = participant_bloom();
    for client in coordinator.epoch_state.clients.iter() {
        let signer_hash = sha256(client.id.signer());
        if signer_hash != hash && bloom.contains(&signer_hash) {
            without.add(&signer_hash);
        }
    }
    *bloom = without;
    next_round(&mut coordinator);
    assert_eq!(coordinator.epoch_state.clients.len(), CLIENTS);

    let (verifier, verifier_proof) = proof_at(&coordinator, trainer_seat + 1);
    assert_eq!(verifier_proof.committee, Committee::Verifier);
    assert!(matches!(
        coordinator.healthy(&verifier, &verifier_proof),
        Ok(true)
    ));

    let heard = (0..CLIENTS)
        .find(|&seat| {
            seat != silent && proof_at(&coordinator, seat).1.committee == Committee::Trainer
        })
        .expect("another trainer");
    let (trainer, trainer_proof) = proof_at(&coordinator, heard);
    assert!(matches!(
        coordinator.healthy(&trainer, &trainer_proof),
        Ok(true)
    ));

    let (id, proof) = proof_at(&coordinator, silent);
    assert!(matches!(coordinator.healthy(&id, &proof), Ok(false)));
    assert!(matches!(
        coordinator.health_check(&id, vec![(id, proof)]),
        Ok(1)
    ));
    assert_eq!(
        coordinator.epoch_state.clients[silent].state,
        ClientState::Dropped
    );
}
