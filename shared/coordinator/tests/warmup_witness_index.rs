//! Holds the fix for the unvalidated warmup witness index
//! (wienerlabs/leviathan#15, finding 15).
//!
//! `Coordinator::witness` verifies the caller's proof before storing it, so
//! `proof.index` is known to be in range. `Coordinator::warmup_witness`
//! deliberately does not check committee membership - everyone may witness
//! during warmup - but it used to skip the index entirely, and the duplicate
//! check that runs on the *next* warmup witness reads the stored index back and
//! indexes `epoch_state.clients` with it.
//!
//! `FixedVec`'s `Index` impl is `self.get(index).expect("Index out of bounds")`,
//! so one client could store an index nobody had looked at and abort every
//! other client's warmup witness for the rest of the round. Naming somebody
//! else's index was just as bad in a quieter way: the duplicate check would then
//! read that client as having already witnessed.
//!
//! Both are closed by the same check - the index has to name the sender - which
//! is not a committee check, so warmup stays open to everyone.

use psyche_coordinator::Client;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_coordinator::RunState;
use psyche_coordinator::Witness;
use psyche_coordinator::WitnessProof;
use psyche_core::FixedVec;
use psyche_core::NodeIdentity;

use bytemuck::Zeroable;

fn client(seed: u8) -> Client {
    let mut key = [0u8; 32];
    key[0] = seed;
    Client {
        id: NodeIdentity::from_single_key(key),
        state: ClientState::Healthy,
        exited_height: 0,
    }
}

fn warming_up() -> Coordinator {
    let mut coordinator = Coordinator::zeroed();
    coordinator.run_state = RunState::Warmup;
    coordinator.epoch_state.clients =
        FixedVec::from_iter([client(1), client(2), client(3)]);
    coordinator
}

fn witness_with_index(index: u64) -> Witness {
    Witness {
        proof: WitnessProof {
            position: 0,
            index,
            witness: true.into(),
        },
        participant_bloom: Default::default(),
        broadcast_bloom: Default::default(),
        broadcast_merkle: Default::default(),
    }
}

/// An index that resolves to nothing is refused, so it is never stored and can
/// never be read back.
#[test]
fn an_out_of_range_warmup_witness_is_refused() {
    let mut coordinator = warming_up();
    assert!(
        coordinator
            .warmup_witness(&client(1).id, witness_with_index(u64::MAX), 1_000, 7)
            .is_err(),
        "an index nobody can resolve is not a witness"
    );
    assert_eq!(
        coordinator.current_round().unwrap().witnesses.len(),
        0,
        "and nothing is stored for the next caller to trip over"
    );
}

/// So the honest client that witnesses next is unaffected. This is the case
/// that used to abort.
#[test]
fn a_later_warmup_witness_is_unaffected() {
    let mut coordinator = warming_up();
    let _ = coordinator.warmup_witness(&client(1).id, witness_with_index(u64::MAX), 1_000, 7);

    coordinator
        .warmup_witness(&client(2).id, witness_with_index(1), 1_001, 7)
        .expect("an honest witness with its own index still goes through");
    assert_eq!(coordinator.current_round().unwrap().witnesses.len(), 1);
}

/// Claiming another client's index is refused too. It is in range, so a bounds
/// check alone would have let it through - and the duplicate check would then
/// have counted that client as having witnessed.
#[test]
fn a_warmup_witness_cannot_claim_another_clients_index() {
    let mut coordinator = warming_up();
    assert!(
        coordinator
            .warmup_witness(&client(1).id, witness_with_index(2), 1_000, 7)
            .is_err(),
        "client 1 sits at index 0, not index 2"
    );

    // And the client whose index was borrowed can still witness for itself.
    coordinator
        .warmup_witness(&client(3).id, witness_with_index(2), 1_001, 7)
        .expect("client 3 does sit at index 2");
}

/// The honest path is unchanged: every client witnesses once, from its own
/// index, and a second attempt is the duplicate it always was.
#[test]
fn every_client_may_witness_once_from_its_own_index() {
    let mut coordinator = warming_up();
    for (index, seed) in [(0u64, 1u8), (1, 2)] {
        coordinator
            .warmup_witness(&client(seed).id, witness_with_index(index), 1_000, 7)
            .expect("warmup is open to everyone");
    }
    assert_eq!(coordinator.current_round().unwrap().witnesses.len(), 2);

    assert!(
        coordinator
            .warmup_witness(&client(1).id, witness_with_index(0), 1_002, 7)
            .is_err(),
        "twice is still a duplicate"
    );
}

/// The round-witness path was always checked, and still is.
#[test]
fn the_round_witness_path_rejects_the_same_input() {
    let mut coordinator = warming_up();
    coordinator.run_state = RunState::RoundTrain;
    let result = coordinator.witness(&client(1).id, witness_with_index(u64::MAX), 1_000);
    assert!(
        result.is_err(),
        "the checked path must refuse an index it cannot resolve"
    );
    assert_eq!(
        coordinator.current_round().unwrap().witnesses.len(),
        0,
        "nothing is stored, so nothing can poison a later caller"
    );
}
