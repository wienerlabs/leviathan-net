//! Checks that real clients keep and serve the state each payload they send
//! commits to (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! Each client kept the commitment record of every payload a peer sent it
//! (`--write-gradients-dir`), and the record names the state the payload was
//! trained from. For each record, this finds the client that trained the
//! payload and checks its `--round-state-dir`:
//!
//! - a payload of the run's last epoch has its state kept, whole, named by
//!   its digest and of the length the run's model gives; served from the
//!   client's store over the network, the state hashes to the digest;
//! - a payload of an earlier epoch has its state forgotten: no audit can name
//!   it any more.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use leviathan_verifier::fidelity::read_client_dumps;
use leviathan_verifier::round_states::load_node_state;
use leviathan_verifier::{parse_device, parse_dtype, read_run_model};
use psyche_metrics::ClientMetrics;
use psyche_modeling::{round_state_file, round_state_len};
use psyche_network::allowlist::AllowAll;
use psyche_network::{
    blobs_gc_interval, commitment_record_path, CommitmentRecord, DiscoveryMode, NetworkConnection,
    RelayKind, RoundStates, TransmittableDistroResult,
};

type Connection = NetworkConnection<TransmittableDistroResult, TransmittableDistroResult>;

#[derive(Parser, Debug)]
#[command(
    name = "check-round-states",
    about = "Check that real clients keep and serve the state each payload commits to (wienerlabs/leviathan-net#89)"
)]
struct Args {
    /// The configuration the clients' run was started with.
    #[arg(long)]
    run_config: PathBuf,
    /// Each client's `--write-gradients-dir`, in the order of `--states`.
    #[arg(long, required = true, num_args = 1..)]
    dumps: Vec<PathBuf>,
    /// Each client's `--round-state-dir`, in the order of `--dumps`.
    #[arg(long, required = true, num_args = 1..)]
    states: Vec<PathBuf>,
    /// The device and dtype the clients trained in. The client loads its model
    /// in bf16.
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "bf16")]
    dtype: String,
}

/// A payload a peer kept the commitment record of: who trained it, when, and
/// the state it commits to.
struct Committed {
    signer: String,
    epoch_start: u64,
    step: u32,
    digest: [u8; 32],
}

/// The signer of the client whose dumps are in `dir`: the one whose payloads
/// it kept without a commitment record, which are its own.
fn own_signer(dir: &Path) -> Result<String> {
    let mut signers: Vec<String> = read_client_dumps(dir)?
        .into_iter()
        .filter(|dump| !commitment_record_path(&dump.path).exists())
        .map(|dump| dump.signer)
        .collect();
    signers.sort();
    signers.dedup();
    match signers.as_slice() {
        [signer] => Ok(signer.clone()),
        other => bail!(
            "{} holds the unrecorded payloads of {} signers, where a client's own are one's",
            dir.display(),
            other.len()
        ),
    }
}

/// Every payload a peer sent the client whose dumps are in `dir`, by the
/// commitment records it kept.
fn committed(dir: &Path) -> Result<Vec<Committed>> {
    let mut committed = vec![];
    for dump in read_client_dumps(dir)? {
        let record_path = commitment_record_path(&dump.path);
        if !record_path.exists() {
            continue;
        }
        let bytes = std::fs::read(&record_path)
            .with_context(|| format!("reading {}", record_path.display()))?;
        let record = CommitmentRecord::from_bytes(&bytes)
            .map_err(|err| anyhow!("{}: {err}", record_path.display()))?;
        let digest = record
            .state_digest
            .ok_or_else(|| anyhow!("the payload {} commits to no state", dump.path.display()))?;
        committed.push(Committed {
            signer: dump.signer,
            epoch_start: record.epoch_start,
            step: record.step,
            digest,
        });
    }
    Ok(committed)
}

/// Checks the state `digest` names is kept in `states`, whole, `len` bytes
/// long and hashing to the digest.
async fn check_kept(states: &RoundStates, digest: &[u8; 32], len: u64) -> Result<()> {
    if !states.is_kept(digest).await? {
        bail!("not kept");
    }
    let bytes = std::fs::read(round_state_file(states.rounds_dir(), digest))?;
    if bytes.len() as u64 != len {
        bail!("{} bytes, where the run's model gives {len}", bytes.len());
    }
    if blake3::hash(&bytes).as_bytes() != digest {
        bail!("its file holds another state");
    }
    Ok(())
}

/// Checks the state `digest` names is forgotten by `states`.
async fn check_forgotten(states: &RoundStates, digest: &[u8; 32]) -> Result<()> {
    if states.is_kept(digest).await? {
        bail!("still kept");
    }
    if round_state_file(states.rounds_dir(), digest).exists() {
        bail!("its file is still there");
    }
    Ok(())
}

/// A network connection on this machine, without relays, serving `states` if
/// given.
async fn connect(states: Option<RoundStates>) -> Result<Connection> {
    Connection::init(
        "check-round-states",
        None,
        None,
        DiscoveryMode::Local,
        RelayKind::Disabled,
        vec![],
        None,
        AllowAll,
        Arc::new(ClientMetrics::new(None, None)),
        None,
        states,
    )
    .await
}

/// Serves `states` from a connection of its own, and fetches every state in
/// `digests` from it through another, as a verifier would.
async fn check_served(states: RoundStates, digests: &[[u8; 32]], len: u64) -> Result<()> {
    let node = connect(Some(states)).await?;
    let verifier = connect(None).await?;
    let node_addr = node.router().endpoint().addr();
    let out = tempfile::tempdir()?;
    for digest in digests {
        let dest = out.path().join("fetched.state");
        verifier
            .fetch_round_state(node_addr.clone(), *digest, len, &dest)
            .await
            .with_context(|| format!("fetching state {}", hex(digest)))?;
        if blake3::hash(&std::fs::read(&dest)?).as_bytes() != digest {
            bail!("the state served under {} is another", hex(digest));
        }
        std::fs::remove_file(&dest)?;
    }
    node.shutdown().await?;
    verifier.shutdown().await?;
    Ok(())
}

fn hex(digest: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*digest).to_hex().to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.dumps.len() != args.states.len() {
        bail!(
            "{} dump directories for {} state directories",
            args.dumps.len(),
            args.states.len()
        );
    }
    let llm = read_run_model(&args.run_config)?;
    let (model, distro) =
        load_node_state(&llm, parse_device(&args.device)?, parse_dtype(&args.dtype)?)?;
    let len = round_state_len(&*model, &distro)?;

    let mut states_of = BTreeMap::new();
    let mut committed_all = vec![];
    for (dumps, states) in args.dumps.iter().zip(&args.states) {
        let states = RoundStates::open(states, blobs_gc_interval())
            .await
            .with_context(|| format!("opening {}", states.display()))?;
        states_of.insert(own_signer(dumps)?, states);
        committed_all.extend(committed(dumps)?);
    }
    let last_epoch = committed_all
        .iter()
        .map(|payload| payload.epoch_start)
        .max()
        .ok_or_else(|| anyhow!("no client kept a peer's payload"))?;

    let mut kept: BTreeMap<String, Vec<[u8; 32]>> = BTreeMap::new();
    let mut kept_steps = vec![];
    let mut forgotten = 0;
    for payload in &committed_all {
        let states = states_of.get(&payload.signer).ok_or_else(|| {
            anyhow!(
                "a payload of {}, whom no client here is, was kept",
                payload.signer
            )
        })?;
        let what = format!(
            "the state {}'s payload of step {} (epoch {}) commits to",
            payload.signer, payload.step, payload.epoch_start
        );
        if payload.epoch_start == last_epoch {
            check_kept(states, &payload.digest, len)
                .await
                .with_context(|| what.clone())?;
            kept.entry(payload.signer.clone())
                .or_default()
                .push(payload.digest);
            kept_steps.push(payload.step);
        } else {
            check_forgotten(states, &payload.digest)
                .await
                .with_context(|| what.clone())?;
            forgotten += 1;
        }
    }
    kept_steps.sort();
    kept_steps.dedup();
    if kept.len() != states_of.len() {
        bail!(
            "{} of {} clients kept a state of the last epoch",
            kept.len(),
            states_of.len()
        );
    }
    // How many rounds the last epoch holds depends on how fast the machine
    // trains them, and may be one. What the check needs is a last epoch that
    // is not the run's first: a node carries its error feedback across
    // epochs, so its states there hold a buffer the run filled, and the
    // earlier epochs' states have something to have been forgotten from.
    if forgotten == 0 {
        bail!("no payload is of an earlier epoch; the check needs the run to cover two");
    }

    let kept_count: usize = kept.values().map(Vec::len).sum();
    for (signer, digests) in kept {
        let states = states_of
            .remove(&signer)
            .ok_or_else(|| anyhow!("{signer} has no states"))?;
        check_served(states, &digests, len)
            .await
            .with_context(|| format!("serving {signer}'s states"))?;
    }
    println!(
        "{kept_count} states of the last epoch kept, whole, and served by digest ({len} bytes each, steps {kept_steps:?}); {forgotten} states of earlier epochs forgotten"
    );
    Ok(())
}
