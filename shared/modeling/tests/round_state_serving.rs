//! A node keeps the state each round trains from and serves it by its digest,
//! and a verifier fetches it (wienerlabs/leviathan-net#89,
//! `docs/ROUND_AUDIT.md`). The states are a real trainer's, on the genesis
//! rehearsal's model, and the node and the verifier are real network
//! connections on this machine, as a client starts them.

mod common;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use common::{Recipe, distro, genesis_recipe, load, round_trainer, train_round};
use psyche_metrics::ClientMetrics;
use psyche_modeling::{round_state_file, round_state_len};
use psyche_network::allowlist::{AllowAll, AllowDynamic, Allowlist};
use psyche_network::{
    DiscoveryMode, FetchRoundStateError, NetworkConnection, RelayKind, RoundStates,
    RoundStatesError, SecretKey, TransmittableDistroResult,
};

type Connection = NetworkConnection<TransmittableDistroResult, TransmittableDistroResult>;

/// How often the stores in these tests collect what no tag keeps.
const GC_INTERVAL: Duration = Duration::from_millis(100);

/// How many collection intervals a forgotten state may still be served for,
/// before a test takes it as never dropped.
const COLLECTIONS_TO_WAIT: u32 = 100;

/// A network connection as a client starts one, on this machine, without
/// relays, serving `round_states` if it keeps any.
async fn connect<A: Allowlist + Send + Sync + 'static>(
    secret_key: SecretKey,
    allowlist: A,
    round_states: Option<RoundStates>,
) -> Connection {
    Connection::init(
        "round-state-serving",
        None,
        None,
        DiscoveryMode::Local,
        RelayKind::Disabled,
        vec![],
        Some(secret_key),
        allowlist,
        Arc::new(ClientMetrics::new(None, None)),
        None,
        round_states,
    )
    .await
    .unwrap()
}

fn secret_key() -> SecretKey {
    SecretKey::generate(&mut rand::rng())
}

/// Two rounds of a real trainer that keeps its states where `states` keeps
/// them: the digests the rounds commit to.
fn two_rounds(recipe: &Recipe, states: &RoundStates) -> ([u8; 32], [u8; 32]) {
    let trainer = round_trainer(recipe, Some(states.rounds_dir().to_path_buf()));
    let first = train_round(recipe, trainer, 1, 1, vec![]);
    let sent = first.distro_results.clone().unwrap();
    let second = train_round(recipe, first.trainer, 2, 2, vec![sent]);
    (
        first.state_digest.expect("the trainer commits"),
        second.state_digest.expect("the trainer commits"),
    )
}

/// The length of a state of the run's model, as a verifier works it out from
/// its own copy of the model.
fn state_len(recipe: &Recipe) -> u64 {
    let model = load(recipe);
    round_state_len(&*model, &distro(recipe, &*model)).unwrap()
}

/// What a verifier fetched to `dest`.
fn fetched(dest: &Path) -> Vec<u8> {
    std::fs::read(dest).unwrap()
}

/// Whether anything was left at `dest`, whole or partial.
fn nothing_at(dest: &Path) -> bool {
    let dir = dest.parent().unwrap();
    !dest.exists() && std::fs::read_dir(dir).unwrap().next().is_none()
}

/// A node serves the state a payload commits to under its digest, and a
/// verifier fetches exactly that state: the bytes the trainer wrote, of the
/// length the run's model gives, hashing to the digest.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_serves_the_state_a_payload_commits_to() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, second) = two_rounds(&recipe, &states);
    states.keep(1, first).await.unwrap();
    states.keep(1, second).await.unwrap();
    assert!(states.is_kept(&first).await.unwrap());
    assert!(states.is_kept(&second).await.unwrap());

    let node = connect(secret_key(), AllowAll, Some(states.clone())).await;
    let verifier = connect(secret_key(), AllowAll, None).await;
    let node_addr = node.router().endpoint().addr();
    let len = state_len(&recipe);

    let out = tempfile::tempdir().unwrap();
    for digest in [first, second] {
        let dest = out.path().join(format!(
            "{}.state",
            blake3::Hash::from_bytes(digest).to_hex()
        ));
        verifier
            .fetch_round_state(node_addr.clone(), digest, len, &dest)
            .await
            .unwrap();
        let bytes = fetched(&dest);
        let written = std::fs::read(round_state_file(states.rounds_dir(), &digest)).unwrap();
        assert_eq!(bytes, written);
        assert_eq!(bytes.len() as u64, len);
        assert_eq!(*blake3::hash(&bytes).as_bytes(), digest);
    }
}

/// A verifier refuses a state of another length than the run's model gives,
/// before downloading it, and keeps nothing of it.
#[tokio::test(flavor = "multi_thread")]
async fn a_verifier_refuses_a_state_of_another_length() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, _) = two_rounds(&recipe, &states);
    states.keep(1, first).await.unwrap();

    let node = connect(secret_key(), AllowAll, Some(states)).await;
    let verifier = connect(secret_key(), AllowAll, None).await;
    let len = state_len(&recipe);

    let out = tempfile::tempdir().unwrap();
    let dest = out.path().join("first.state");
    let err = verifier
        .fetch_round_state(node.router().endpoint().addr(), first, len - 1, &dest)
        .await
        .unwrap_err();
    assert!(
        matches!(err, FetchRoundStateError::Size { expected, served } if expected == len - 1 && served == len),
        "{err}"
    );
    assert!(nothing_at(&dest));
}

/// A node serves nothing under a digest it does not keep: a state it trained
/// from but never kept, so no verifier can be handed one it was not.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_serves_nothing_under_a_digest_it_does_not_keep() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, second) = two_rounds(&recipe, &states);
    states.keep(1, first).await.unwrap();
    assert!(!states.is_kept(&second).await.unwrap());

    let node = connect(secret_key(), AllowAll, Some(states)).await;
    let verifier = connect(secret_key(), AllowAll, None).await;

    let out = tempfile::tempdir().unwrap();
    let dest = out.path().join("second.state");
    let err = verifier
        .fetch_round_state(
            node.router().endpoint().addr(),
            second,
            state_len(&recipe),
            &dest,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, FetchRoundStateError::Transfer(_)), "{err}");
    assert!(nothing_at(&dest));
}

/// Only a peer on the node's roster can fetch its states: the node refuses
/// one it does not allow, and serves it once it does.
#[tokio::test(flavor = "multi_thread")]
async fn only_a_peer_on_the_roster_can_fetch_a_state() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, _) = two_rounds(&recipe, &states);
    states.keep(1, first).await.unwrap();

    let verifier_key = secret_key();
    let roster = AllowDynamic::with_nodes([secret_key().public()]);
    let node = connect(secret_key(), roster.clone(), Some(states)).await;
    let verifier = connect(verifier_key.clone(), AllowAll, None).await;
    let node_addr = node.router().endpoint().addr();
    let len = state_len(&recipe);

    let out = tempfile::tempdir().unwrap();
    let dest = out.path().join("first.state");
    let err = verifier
        .fetch_round_state(node_addr.clone(), first, len, &dest)
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            FetchRoundStateError::Connect(_) | FetchRoundStateError::Transfer(_)
        ),
        "{err}"
    );
    assert!(nothing_at(&dest));

    roster.add(verifier_key.public());
    verifier
        .fetch_round_state(node_addr, first, len, &dest)
        .await
        .unwrap();
    assert_eq!(*blake3::hash(&fetched(&dest)).as_bytes(), first);
}

/// A state is kept until its epoch ends. Keeping the first state of a later
/// epoch forgets every state of an earlier one: its file is gone and it is no
/// longer kept at once, and once the store's collector has run, no verifier
/// can fetch it, while the later one is served. A state as small as this
/// model's is held inside the store's database rather than in its file, so
/// the collector, not the file's removal, is what stops it being served.
#[tokio::test(flavor = "multi_thread")]
async fn a_state_is_forgotten_once_its_epoch_ends() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, second) = two_rounds(&recipe, &states);
    states.keep(1, first).await.unwrap();
    assert!(states.is_kept(&first).await.unwrap());

    states.keep(2, second).await.unwrap();
    assert!(!states.is_kept(&first).await.unwrap());
    assert!(!round_state_file(states.rounds_dir(), &first).exists());
    assert!(states.is_kept(&second).await.unwrap());

    let node = connect(secret_key(), AllowAll, Some(states)).await;
    let verifier = connect(secret_key(), AllowAll, None).await;
    let node_addr = node.router().endpoint().addr();
    let len = state_len(&recipe);
    let out = tempfile::tempdir().unwrap();

    let gone = out.path().join("first.state");
    let mut collections = 0;
    while verifier
        .fetch_round_state(node_addr.clone(), first, len, &gone)
        .await
        .is_ok()
    {
        std::fs::remove_file(&gone).unwrap();
        collections += 1;
        assert!(
            collections < COLLECTIONS_TO_WAIT,
            "a forgotten state was still served after {collections} collection intervals"
        );
        tokio::time::sleep(GC_INTERVAL).await;
    }
    assert!(!gone.exists());

    let served = out.path().join("second.state");
    verifier
        .fetch_round_state(node_addr, second, len, &served)
        .await
        .unwrap();
    assert_eq!(*blake3::hash(&fetched(&served)).as_bytes(), second);
}

/// A file that is not the state it is named by, as a disk would leave one it
/// damaged, is refused: it is not kept, and it is removed.
#[tokio::test(flavor = "multi_thread")]
async fn a_damaged_state_is_refused() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, _) = two_rounds(&recipe, &states);
    let file = round_state_file(states.rounds_dir(), &first);
    let mut bytes = std::fs::read(&file).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&file, &bytes).unwrap();

    let err = states.keep(1, first).await.unwrap_err();
    let found = *blake3::hash(&bytes).as_bytes();
    assert!(
        matches!(
            &err,
            RoundStatesError::Mismatch { expected, found: hash }
                if expected.as_bytes() == &first && hash.as_bytes() == &found
        ),
        "{err}"
    );
    assert!(!states.is_kept(&first).await.unwrap());
    assert!(!file.exists());
}

/// The states a node kept survive its restart, so it can still reveal them
/// for the epoch under way. A file it wrote and never kept, or never made
/// whole, names no state anyone can fetch, and is removed when it opens its
/// states again.
#[tokio::test(flavor = "multi_thread")]
async fn kept_states_survive_a_restart_and_strays_are_removed() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, second) = two_rounds(&recipe, &states);
    states.keep(1, first).await.unwrap();
    // The second state was written, and the node stopped before keeping it.
    let unkept = round_state_file(states.rounds_dir(), &second);
    assert!(unkept.exists());
    let partial = states.rounds_dir().join(".1-0.partial");
    std::fs::copy(&unkept, &partial).unwrap();
    states.shutdown().await.unwrap();
    drop(states);

    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    assert!(states.is_kept(&first).await.unwrap());
    assert!(round_state_file(states.rounds_dir(), &first).exists());
    assert!(!unkept.exists());
    assert!(!partial.exists());
    assert!(!states.is_kept(&second).await.unwrap());

    let node = connect(secret_key(), AllowAll, Some(states)).await;
    let verifier = connect(secret_key(), AllowAll, None).await;
    let out = tempfile::tempdir().unwrap();
    let dest = out.path().join("first.state");
    verifier
        .fetch_round_state(
            node.router().endpoint().addr(),
            first,
            state_len(&recipe),
            &dest,
        )
        .await
        .unwrap();
    assert_eq!(*blake3::hash(&fetched(&dest)).as_bytes(), first);
}

/// A node that keeps no states, started without a state directory, serves
/// none: its connection does not take the protocol, even for a state it
/// trained from and has on disk.
#[tokio::test(flavor = "multi_thread")]
async fn a_node_keeping_no_states_serves_none() {
    let recipe = genesis_recipe();
    let node_dir = tempfile::tempdir().unwrap();
    let states = RoundStates::open(node_dir.path(), GC_INTERVAL)
        .await
        .unwrap();
    let (first, _) = two_rounds(&recipe, &states);
    states.keep(1, first).await.unwrap();

    let node = connect(secret_key(), AllowAll, None).await;
    let verifier = connect(secret_key(), AllowAll, None).await;
    let out = tempfile::tempdir().unwrap();
    let dest = out.path().join("first.state");
    let err = verifier
        .fetch_round_state(
            node.router().endpoint().addr(),
            first,
            state_len(&recipe),
            &dest,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, FetchRoundStateError::Connect(_)), "{err}");
    assert!(nothing_at(&dest));
}
