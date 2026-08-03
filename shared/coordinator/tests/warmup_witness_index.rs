//! Reproduces the unvalidated warmup witness index found in the internal review
//! of the on-chain programs (wienerlabs/leviathan#15).
//!
//! `Coordinator::witness` verifies the caller's proof before storing it, so
//! `proof.index` is known to be in range. `Coordinator::warmup_witness`
//! deliberately does not - "everyone can send a witness in the warmup phase so
//! we don't need to check for the committee" - but the duplicate check that runs
//! on the *next* warmup witness reads the stored index back and uses it to index
//! `epoch_state.clients` directly.
//!
//! `FixedVec`'s `Index` impl is `self.get(index).expect("Index out of bounds")`,
//! so an out-of-range index stored by one client aborts the transaction of every
//! client that witnesses after it, for as long as the round's witness list
//! holds it.

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

/// A witness naming a client index that does not exist is accepted, because the
/// warmup path stores the proof without looking at it.
#[test]
fn an_out_of_range_warmup_witness_is_accepted() {
    let mut coordinator = warming_up();
    coordinator
        .warmup_witness(&client(1).id, witness_with_index(u64::MAX), 1_000, 7)
        .expect("BUG: the warmup path stores an index it never checked");
    assert_eq!(
        coordinator.current_round().unwrap().witnesses.len(),
        1,
        "the poisoned witness is now in the round"
    );
}

/// And the next client to witness in that round pays for it: the duplicate check
/// reads the stored index back and panics out of bounds, taking the whole
/// transaction with it.
#[test]
#[should_panic(expected = "Index out of bounds")]
fn a_later_warmup_witness_panics_on_the_stored_index() {
    let mut coordinator = warming_up();
    coordinator
        .warmup_witness(&client(1).id, witness_with_index(u64::MAX), 1_000, 7)
        .unwrap();

    // An honest second client, with a perfectly valid proof of its own.
    let _ = coordinator.warmup_witness(&client(2).id, witness_with_index(1), 1_001, 7);
}

/// The same shape of input on the round-witness path is rejected instead, which
/// is what the warmup path should be doing: `verify_witness_for_client` bounds
/// checks the index before anything stores it.
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
