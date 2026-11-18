//! The states a node trained its rounds from, kept on disk and served by
//! their digest, so a verifier can fetch the one a payload commits to
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! The trainer writes each state into `rounds/`, in a file named by its
//! digest (`psyche_modeling::keep_round_state`). The node then imports the
//! file, in place, into an iroh blob store of its own under `store/`, on disk
//! rather than in the memory store the run's results go through: a state is
//! about twice the model, every round. iroh names a blob by the BLAKE3 hash of
//! its bytes, which is the digest, so the store serves a state under the
//! digest its payload signs, and a verifier checks every piece it fetches
//! against it.
//!
//! A state is kept until its epoch ends. The treasurer convicts only on a
//! commitment from the epoch under way (`CommitmentEvidence::
//! is_from_current_epoch`), so once a later epoch starts, no audit can name a
//! state of an earlier one.

use std::collections::HashSet;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::time::Duration;

use bao_tree::io::BaoContentItem;
use futures_util::StreamExt;
use iroh::{Endpoint, EndpointAddr};
use iroh_blobs::api::blobs::{AddPathOptions, ImportMode};
use iroh_blobs::api::proto::TagInfo;
use iroh_blobs::get::request::{GetBlobItem, get_blob, get_verified_size};
use iroh_blobs::store::GcConfig;
use iroh_blobs::store::fs::FsStore;
use iroh_blobs::store::fs::options::Options;
use iroh_blobs::{BlobFormat, BlobsProtocol, Hash};
use psyche_modeling::{ROUND_STATE_EXTENSION, round_state_file};
use thiserror::Error;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tracing::{debug, warn};

/// The protocol a node serves the states it kept under. It is iroh's blob
/// protocol, over the node's own store of states.
pub const ROUND_STATE_ALPN: &[u8] = b"/leviathan/round-state/1";

/// What the tag of every kept state starts with. The epoch and the digest
/// follow (`state_tag`).
const TAG_PREFIX: &str = "round-state/";

/// The directory, under a node's state directory, the trainer writes states
/// into.
const ROUNDS_DIR: &str = "rounds";

/// The directory, under a node's state directory, of the blob store that
/// serves them.
const STORE_DIR: &str = "store";

/// The extension a fetched state is written under before it is whole.
const PARTIAL_FETCH_EXTENSION: &str = "partial";

fn epoch_prefix(epoch_start: u64) -> String {
    format!("{TAG_PREFIX}{epoch_start}/")
}

fn state_tag(epoch_start: u64, digest: &[u8; 32]) -> String {
    format!(
        "{}{}",
        epoch_prefix(epoch_start),
        Hash::from_bytes(*digest).to_hex()
    )
}

#[derive(Debug, Error)]
pub enum RoundStatesError {
    #[error("a round state's file could not be read or written: {0}")]
    Io(#[from] std::io::Error),
    #[error("the round state store failed: {0:#}")]
    Store(anyhow::Error),
    #[error("the file kept for state {expected} holds state {found}")]
    Mismatch { expected: Hash, found: Hash },
}

fn store_error(err: impl Into<anyhow::Error>) -> RoundStatesError {
    RoundStatesError::Store(err.into())
}

/// The states a node kept of the rounds it trained, and the store that
/// serves them.
#[derive(Debug, Clone)]
pub struct RoundStates {
    rounds: PathBuf,
    store: FsStore,
}

impl RoundStates {
    /// Opens the states kept under `dir`, creating it if need be. The store
    /// drops a state whose tag is gone at its next collection, every
    /// `gc_interval`. A file in `rounds/` the store does not keep, written by
    /// a node that stopped before importing it or never made whole, names no
    /// state anyone can fetch, and is removed.
    pub async fn open(dir: &Path, gc_interval: Duration) -> Result<Self, RoundStatesError> {
        let rounds = dir.join(ROUNDS_DIR);
        let store_dir = dir.join(STORE_DIR);
        tokio::fs::create_dir_all(&rounds).await?;
        tokio::fs::create_dir_all(&store_dir).await?;
        let mut options = Options::new(&store_dir);
        options.gc = Some(GcConfig {
            interval: gc_interval,
            add_protected: None,
        });
        let store = FsStore::load_with_opts(store_dir.join("blobs.db"), options)
            .await
            .map_err(store_error)?;
        let states = Self { rounds, store };
        let swept = states.sweep().await?;
        if swept > 0 {
            debug!(swept, "Removed round state files the store did not keep");
        }
        Ok(states)
    }

    /// The directory the trainer writes the states it commits to into
    /// (`psyche_modeling::LocalTrainer::keeping_round_states_in`).
    pub fn rounds_dir(&self) -> &Path {
        &self.rounds
    }

    /// The protocol the node serves its kept states with, under
    /// `ROUND_STATE_ALPN`.
    pub fn protocol(&self) -> BlobsProtocol {
        BlobsProtocol::new(&self.store, None)
    }

    /// Keeps the state named by `digest`, which the trainer wrote into the
    /// rounds directory, for the epoch that started at `epoch_start`, and
    /// serves it from then on. The store hashes the file as it imports it, so
    /// a file that is not the state it is named by is refused, and removed.
    /// Every state kept for another epoch is then forgotten.
    pub async fn keep(&self, epoch_start: u64, digest: [u8; 32]) -> Result<(), RoundStatesError> {
        let path = round_state_file(&self.rounds, &digest);
        let tag = state_tag(epoch_start, &digest);
        let imported = self
            .store
            .blobs()
            .add_path_with_opts(AddPathOptions {
                path: path.clone(),
                format: BlobFormat::Raw,
                mode: ImportMode::TryReference,
            })
            .with_named_tag(&tag)
            .await
            .map_err(store_error)?;
        let expected = Hash::from_bytes(digest);
        if imported.hash != expected {
            self.store.tags().delete(&tag).await.map_err(store_error)?;
            remove_state_file(&path).await?;
            return Err(RoundStatesError::Mismatch {
                expected,
                found: imported.hash,
            });
        }
        let forgotten = self.forget_other_epochs(epoch_start).await?;
        if forgotten > 0 {
            debug!(
                epoch_start,
                forgotten, "Forgot the states of earlier epochs"
            );
        }
        Ok(())
    }

    /// Whether the state named by `digest` is kept, whole, and served.
    pub async fn is_kept(&self, digest: &[u8; 32]) -> Result<bool, RoundStatesError> {
        let tagged = self
            .tags()
            .await?
            .iter()
            .any(|tag| tag.hash.as_bytes() == digest);
        if !tagged {
            return Ok(false);
        }
        self.store
            .blobs()
            .has(Hash::from_bytes(*digest))
            .await
            .map_err(store_error)
    }

    /// Forgets every state kept for an epoch other than the one that started
    /// at `epoch_start`: its tag, so the store's collector drops it, and its
    /// file. A state kept for this epoch too stays. Returns how many states
    /// were forgotten.
    pub async fn forget_other_epochs(&self, epoch_start: u64) -> Result<usize, RoundStatesError> {
        let current = epoch_prefix(epoch_start);
        let (this_epoch, others): (Vec<_>, Vec<_>) = self
            .tags()
            .await?
            .into_iter()
            .partition(|tag| tag.name.0.starts_with(current.as_bytes()));
        let still_kept: HashSet<Hash> = this_epoch.iter().map(|tag| tag.hash).collect();
        let mut forgotten = 0;
        for tag in others {
            self.store
                .tags()
                .delete(&tag.name)
                .await
                .map_err(store_error)?;
            if !still_kept.contains(&tag.hash) {
                remove_state_file(&round_state_file(&self.rounds, tag.hash.as_bytes())).await?;
                forgotten += 1;
            }
        }
        Ok(forgotten)
    }

    /// Every kept state's tag.
    async fn tags(&self) -> Result<Vec<TagInfo>, RoundStatesError> {
        let tags = self
            .store
            .tags()
            .list_prefix(TAG_PREFIX)
            .await
            .map_err(store_error)?;
        tags.map(|tag| tag.map_err(store_error))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect()
    }

    /// Removes every file in the rounds directory that holds no kept state,
    /// and returns how many it removed.
    async fn sweep(&self) -> Result<usize, RoundStatesError> {
        let kept: HashSet<PathBuf> = self
            .tags()
            .await?
            .iter()
            .map(|tag| round_state_file(&self.rounds, tag.hash.as_bytes()))
            .collect();
        let mut entries = tokio::fs::read_dir(&self.rounds).await?;
        let mut removed = 0;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let is_state =
                path.extension().and_then(|ext| ext.to_str()) == Some(ROUND_STATE_EXTENSION);
            if is_state && kept.contains(&path) {
                continue;
            }
            if entry.file_type().await?.is_file() {
                remove_state_file(&path).await?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Stops the store, once nothing serves from it any more.
    pub async fn shutdown(&self) -> Result<(), RoundStatesError> {
        self.store.shutdown().await.map_err(store_error)
    }
}

/// Removes a state's file. One already gone is fine.
async fn remove_state_file(path: &Path) -> std::io::Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

/// Why a trainer's state was not revealed to a verifier.
#[derive(Debug, Error)]
pub enum RevealError {
    /// The trainer serves something under the digest that cannot be the
    /// state: one of another length. The digest names it, so asking again
    /// would only serve it again.
    #[error("the trainer serves no state of the run's model under its digest: {0}")]
    Refused(FetchRoundStateError),
    /// The trainer did not serve the state before the time to reveal it ran
    /// out.
    #[error("the trainer did not serve the state in time; the last attempt failed: {0}")]
    NotRevealed(FetchRoundStateError),
    /// The verifier could not write what it fetched: its own trouble, not the
    /// trainer's.
    #[error("the revealed state could not be written: {0}")]
    Local(FetchRoundStateError),
}

/// Fetches from `node` the state named by `digest`, as `fetch_round_state`
/// does, asking again every `retry_every` until `time_to_reveal` has gone by.
///
/// A trainer keeps the state beside its payload's broadcast, not before it,
/// so a verifier that asks first finds it missing, and asks again. A state of
/// another length is refused at once. Returns how many times it asked.
pub async fn reveal_round_state(
    endpoint: &Endpoint,
    node: impl Into<EndpointAddr>,
    digest: [u8; 32],
    expected_len: u64,
    dest: &Path,
    retry_every: Duration,
    time_to_reveal: Duration,
) -> Result<u32, RevealError> {
    let node = node.into();
    let started = tokio::time::Instant::now();
    let mut attempts = 0;
    loop {
        attempts += 1;
        let err = match fetch_round_state(endpoint, node.clone(), digest, expected_len, dest).await
        {
            Ok(()) => return Ok(attempts),
            Err(err @ FetchRoundStateError::Size { .. }) => return Err(RevealError::Refused(err)),
            Err(err @ FetchRoundStateError::Io(_)) => return Err(RevealError::Local(err)),
            Err(err) => err,
        };
        if started.elapsed() + retry_every > time_to_reveal {
            return Err(RevealError::NotRevealed(err));
        }
        tokio::time::sleep(retry_every).await;
    }
}

#[derive(Debug, Error)]
pub enum FetchRoundStateError {
    #[error("the node could not be reached: {0:#}")]
    Connect(anyhow::Error),
    #[error("the node serves a state of {served} bytes, where the run's model has {expected}")]
    Size { expected: u64, served: u64 },
    #[error("the state could not be fetched: {0:#}")]
    Transfer(anyhow::Error),
    #[error("the fetched state could not be written: {0}")]
    Io(#[from] std::io::Error),
}

/// Fetches from `node` the state named by `digest`, which has to be
/// `expected_len` bytes long (`psyche_modeling::round_state_len`), into
/// `dest`. Its size is checked first, with iroh's proof of it, so a node
/// cannot make a verifier download more than the state of the run's model.
/// Every piece is then checked against the digest as it arrives, so a node
/// can serve nothing under it but the state itself. The state lands at `dest`
/// only whole.
pub async fn fetch_round_state(
    endpoint: &Endpoint,
    node: impl Into<EndpointAddr>,
    digest: [u8; 32],
    expected_len: u64,
    dest: &Path,
) -> Result<(), FetchRoundStateError> {
    let connection = endpoint
        .connect(node, ROUND_STATE_ALPN)
        .await
        .map_err(|err| FetchRoundStateError::Connect(err.into()))?;
    let fetched = fetch_over(&connection, Hash::from_bytes(digest), expected_len, dest).await;
    connection.close(0u32.into(), b"done");
    fetched
}

async fn fetch_over(
    connection: &iroh::endpoint::Connection,
    hash: Hash,
    expected_len: u64,
    dest: &Path,
) -> Result<(), FetchRoundStateError> {
    let (served, _) = get_verified_size(connection, &hash)
        .await
        .map_err(|err| FetchRoundStateError::Transfer(err.into()))?;
    if served != expected_len {
        return Err(FetchRoundStateError::Size {
            expected: expected_len,
            served,
        });
    }
    let mut partial = dest.as_os_str().to_owned();
    partial.push(".");
    partial.push(PARTIAL_FETCH_EXTENSION);
    let partial = PathBuf::from(partial);
    match write_fetched(connection, hash, served, &partial).await {
        Ok(()) => {
            tokio::fs::rename(&partial, dest).await?;
            Ok(())
        }
        Err(err) => {
            if let Err(cleanup) = remove_state_file(&partial).await {
                warn!(
                    path = %partial.display(),
                    "Could not remove a partly fetched state: {cleanup}"
                );
            }
            Err(err)
        }
    }
}

/// Writes the blob `hash`, `len` bytes long, from `connection` to `path`, as
/// its verified pieces arrive.
async fn write_fetched(
    connection: &iroh::endpoint::Connection,
    hash: Hash,
    len: u64,
    path: &Path,
) -> Result<(), FetchRoundStateError> {
    let mut file = tokio::fs::File::create(path).await?;
    let mut items = get_blob(connection.clone(), hash);
    let mut received = 0u64;
    while let Some(item) = items.next().await {
        match item {
            GetBlobItem::Item(BaoContentItem::Leaf(leaf)) => {
                file.seek(SeekFrom::Start(leaf.offset)).await?;
                file.write_all(&leaf.data).await?;
                received += leaf.data.len() as u64;
            }
            GetBlobItem::Item(BaoContentItem::Parent(_)) | GetBlobItem::Done(_) => {}
            GetBlobItem::Error(err) => return Err(FetchRoundStateError::Transfer(err.into())),
        }
    }
    if received != len {
        return Err(FetchRoundStateError::Transfer(anyhow::anyhow!(
            "the node sent {received} of the state's {len} bytes"
        )));
    }
    file.flush().await?;
    Ok(())
}
