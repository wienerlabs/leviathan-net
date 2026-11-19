//! Audits real clients' payloads past a run's first step, each replayed from
//! the state its trainer revealed (wienerlabs/leviathan-net#89,
//! `docs/ROUND_AUDIT.md`).
//!
//! For every payload a verifier kept with its revealed state beside it, the
//! round is replayed from that state, with the values the trainer sent the
//! round before read off it, on the batch the payload names, and the payload
//! is judged against the replay by the band. Given the dumps of a client that
//! sent forged results, its payloads have to be convicted and every other one
//! cleared; without, every one has to be cleared.
//!
//! With `--continuity`, each round whose trainer trained the next one too is
//! also checked to leave the state that next round committed to: replayed
//! from its revealed state, with what the verifier's node applied at the
//! start of the next round applied, it has to hold that state to the bit.
//! Every one checked has to hold it, the clients training the verifier's
//! class of hardware.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use leviathan_verifier::fidelity::{read_client_dumps, ClientDump};
use leviathan_verifier::replay::ReplayWork;
use leviathan_verifier::revealed::{next_round, revealed_round, Continuity, RevealedRound};
use leviathan_verifier::run_data::RunData;
use leviathan_verifier::{
    build_replay_trainer, checkpoint_repo_files, judge_payload, parse_device, parse_dtype,
    payload_band, published_checkpoint, read_run_model, ReplayAssignment, ReplayTrainerConfig,
    TrainerReplayEngine,
};
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::BatchDataCPU;
use psyche_network::commitment_record_path;

#[derive(Parser, Debug)]
#[command(
    name = "audit-revealed-rounds",
    about = "Replay real clients' payloads from the states their trainers revealed, and judge them (wienerlabs/leviathan-net#89)"
)]
struct Args {
    /// The configuration the clients' run was started with.
    #[arg(long)]
    run_config: PathBuf,
    /// Each client's `--write-gradients-dir`.
    #[arg(long, required = true, num_args = 1..)]
    dumps: Vec<PathBuf>,
    /// The `--write-gradients-dir` of a client that sent forged results: its
    /// own payloads are the ones it kept without a commitment record.
    #[arg(long)]
    forged_by: Option<PathBuf>,
    /// The band a payload is held to, as the verifier daemon takes it.
    #[arg(long)]
    band: Option<f32>,
    #[arg(long)]
    churn_band: Option<f32>,
    #[arg(long)]
    flip_band: Option<f32>,
    /// The device and dtype the clients trained in. The client loads its model
    /// in bf16.
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    /// Also check that each round leaves the state its trainer's next round
    /// commits to.
    #[arg(long)]
    continuity: bool,
}

/// The signer whose own payloads the client with dumps in `dir` kept: those
/// without a commitment record beside them.
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
            "{} holds the unrecorded payloads of {} signers",
            dir.display(),
            other.len()
        ),
    }
}

/// The first bytes of `digest`, in hex, for a line a human reads.
fn short(digest: &[u8; 32]) -> String {
    digest[..4]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// What `--continuity` found.
#[derive(Default)]
struct ContinuityTally {
    held: usize,
    /// Of those held, the rounds whose next one applied results first.
    held_applying: usize,
    broken: Vec<String>,
    pending: usize,
    unchecked: Vec<String>,
}

impl ContinuityTally {
    fn check(
        &mut self,
        engine: &TrainerReplayEngine,
        dump: &ClientDump,
        round: &RevealedRound,
    ) -> Result<()> {
        let next = match next_round(&dump.path) {
            Continuity::Ready(next) => next,
            Continuity::Pending => {
                self.pending += 1;
                return Ok(());
            }
            Continuity::Unchecked(why) => {
                self.unchecked
                    .push(format!("{} at step {}: {why}", dump.signer, dump.key.step));
                return Ok(());
            }
        };
        let left = engine
            .replay_continuity(&dump.key, round, &next)
            .map_err(|err| {
                anyhow!(
                    "replaying {} on to the next round: {err}",
                    dump.path.display()
                )
            })?;
        match left == next.committed {
            true => {
                self.held += 1;
                self.held_applying += usize::from(next.applied.is_some());
            }
            false => self.broken.push(format!(
                "{} at step {}: left {}, its step {} commits to {}",
                dump.signer,
                dump.key.step,
                short(&left),
                next.step,
                short(&next.committed),
            )),
        }
        Ok(())
    }
}

#[derive(Default)]
struct Tally {
    cleared: usize,
    convicted: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let llm = read_run_model(&args.run_config)?;
    if llm.cold_start_warmup_steps != 0 {
        bail!("the run warms up from a cold start, whose bounds this audit cannot rebuild from its dumps");
    }
    let band = payload_band(args.band, args.churn_band, args.flip_band)?;
    let device = parse_device(&args.device)?;
    let forger = args.forged_by.as_deref().map(own_signer).transpose()?;

    let mut provider = RunData::open(&llm, None).await?;
    let mut revealed = vec![];
    for dir in &args.dumps {
        for dump in read_client_dumps(dir)? {
            let Some(round) = revealed_round(&dump.path)
                .with_context(|| format!("{} cannot be replayed", dump.path.display()))?
            else {
                continue;
            };
            let data: Vec<BatchDataCPU> = provider
                .get_samples(dump.key.batch_id)
                .await?
                .into_iter()
                .map(|sample| BatchDataCPU {
                    input_ids: sample.input_ids,
                    labels: sample.labels,
                    position_ids: sample.position_ids,
                    sequence_lengths: sample.sequence_lengths,
                })
                .collect();
            let assignment = ReplayAssignment {
                epoch_start: dump.key.epoch_start,
                step: dump.key.step,
                warmup_lr_between: None,
                batch_id: dump.key.batch_id,
                data,
            };
            revealed.push((dump, round, assignment));
        }
    }
    if revealed.is_empty() {
        bail!("no payload here has its state revealed beside it");
    }

    let config = ReplayTrainerConfig::for_run(
        &llm,
        checkpoint_repo_files(&published_checkpoint(&llm)?)?,
        device,
        parse_dtype(&args.dtype)?,
    );
    let quantize_1bit = config.quantize_1bit();
    let trainer = build_replay_trainer(&config)?;
    let engine = TrainerReplayEngine::new(
        trainer,
        revealed
            .iter()
            .map(|(_, _, assignment)| assignment.clone())
            .collect(),
        device,
        quantize_1bit,
    );

    let mut tally: BTreeMap<bool, Tally> = BTreeMap::new();
    let mut wrong = vec![];
    let mut continuity = ContinuityTally::default();
    for (dump, round, _) in &revealed {
        let replayed = engine
            .replay_revealed_work(&dump.key, round)
            .map_err(|err| anyhow!("replaying {}: {err}", dump.path.display()))?;
        let judged = judge_payload(&dump.path, &replayed, band, device)?;
        let forged = forger.as_deref() == Some(dump.signer.as_str());
        let counts = tally.entry(forged).or_default();
        match judged.verdict.fraud {
            true => counts.convicted += 1,
            false => counts.cleared += 1,
        }
        if judged.verdict.fraud != forged {
            wrong.push(format!(
                "{} at step {}: {} (distance {:.4}{})",
                dump.signer,
                dump.key.step,
                if judged.verdict.fraud {
                    "convicted, though its trainer is honest"
                } else {
                    "cleared, though its trainer forged it"
                },
                judged.verdict.distance,
                judged
                    .signs
                    .map(|drift| format!(", churn {:.4}, flips {:.6}", drift.churn, drift.flips))
                    .unwrap_or_default(),
            ));
        }
        if args.continuity {
            continuity.check(&engine, dump, round)?;
        }
    }
    let honest = tally.remove(&false).unwrap_or_default();
    let forged = tally.remove(&true).unwrap_or_default();
    println!(
        "{} payloads replayed from their revealed states: of the honest, {} cleared and {} convicted; of the forged, {} convicted and {} cleared",
        revealed.len(),
        honest.cleared,
        honest.convicted,
        forged.convicted,
        forged.cleared,
    );
    if args.continuity {
        println!(
            "continuity: {} rounds left the state their trainer's next round committed to ({} of them once the results that round applied first were applied), {} did not; {} had no next payload of their trainer, {} could not be checked here{}",
            continuity.held,
            continuity.held_applying,
            continuity.broken.len(),
            continuity.pending,
            continuity.unchecked.len(),
            match continuity.unchecked.is_empty() {
                true => String::new(),
                false => format!(" ({})", continuity.unchecked.join("; ")),
            },
        );
    }
    if !wrong.is_empty() {
        bail!("misjudged:\n{}", wrong.join("\n"));
    }
    if args.continuity {
        if !continuity.broken.is_empty() {
            bail!(
                "rounds that did not leave the state their trainer's next round committed to:\n{}",
                continuity.broken.join("\n")
            );
        }
        if continuity.held == 0 {
            bail!("no round could be checked for continuity");
        }
        if continuity.held_applying == 0 {
            bail!("no round checked for continuity was followed by one that applied results");
        }
    }
    if honest.cleared == 0 {
        bail!("no honest payload was replayed, so nothing was cleared");
    }
    if forger.is_some() && forged.convicted == 0 {
        bail!("no forged payload had its state revealed, so nothing was convicted");
    }
    Ok(())
}
