//! Measures what DisTrO's error feedback costs a replay audit
//! (wienerlabs/leviathan-net#48), under a real run configuration:
//!
//! - the convergence a reset at a public boundary costs, as the nodes' loss on
//!   data they have not yet seen, under each reset policy;
//! - the drift a verifier sees when it replays a node from public data since
//!   the last reset, on its own device and dtype;
//! - the size of the buffer a reveal on audit would send.
//!
//! See `leviathan_verifier::error_feedback` for what is simulated, and
//! `check-error-feedback-fidelity` for the check that it trains a run the way
//! real clients do.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::device_label;
use leviathan_verifier::error_feedback::check_cohort;
use leviathan_verifier::error_feedback::error_feedback_bytes;
use leviathan_verifier::error_feedback::read_rounds;
use leviathan_verifier::error_feedback::schedule;
use leviathan_verifier::error_feedback::share_size;
use leviathan_verifier::error_feedback::simulate;
use leviathan_verifier::error_feedback::ShadowStart;
use leviathan_verifier::error_feedback::Simulation;
use leviathan_verifier::error_feedback::StepRecord;
use leviathan_verifier::parse_device;
use leviathan_verifier::parse_dtype;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::read_run_global_batch;
use leviathan_verifier::read_run_model;
use leviathan_verifier::read_run_verification_percent;
use leviathan_verifier::run_data::RunData;
use leviathan_verifier::ReplayTrainerConfig;
use psyche_coordinator::CoordinatorProgress;

#[derive(Parser, Debug)]
#[command(
    name = "measure-error-feedback",
    about = "Measure what DisTrO's error feedback costs a replay audit (wienerlabs/leviathan-net#48)"
)]
struct Args {
    /// The run configuration to measure (`config/leviathan/*.toml`): its
    /// checkpoint, data, DisTrO settings, learning-rate schedule, sequence
    /// length and global batch.
    #[arg(long)]
    run_config: PathBuf,
    /// A local copy of the run's data, read in the run's layout. Without it the
    /// run's own HTTP location is read, as a node reads it.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// How many nodes train the run together. Each keeps its own error
    /// feedback and trains an even share of every round's global batch.
    #[arg(long, default_value_t = 2)]
    nodes: usize,
    /// How many training rounds to train under each reset policy.
    #[arg(long, default_value_t = 1000)]
    steps: u32,
    /// How many rounds each epoch trains before the two that close it. A run's
    /// configuration does not fix it: rounds run until the epoch's time is up,
    /// so it is the epoch's training time over the time a round takes on the
    /// swarm's hardware. 0 trains one epoch that never closes.
    #[arg(long)]
    epoch_rounds: u32,
    /// The reset policies to compare, comma separated: 0 keeps the error
    /// feedback for good, as the client does; N zeroes it every N training
    /// rounds, as a reset at a public boundary would.
    #[arg(long, value_delimiter = ',', default_value = "0")]
    reset_every: Vec<u32>,
    /// The device and dtype the nodes train in. The client trains in bf16.
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    /// A verifier that replays node 0 from public data, on this device. Leave
    /// it out to measure convergence alone.
    #[arg(long)]
    shadow_device: Option<String>,
    #[arg(long, default_value = "fp32")]
    shadow_dtype: String,
    /// Where the shadow starts each round from: `own`, its own weights since
    /// the run's start and its own buffer since the last reset; `resync`, the
    /// same but with node 0's weights as a checkpoint at every reset, as every
    /// node would take a canonical checkpoint at a public boundary; `reveal`,
    /// node 0's weights, buffer and previous round's values at every round, as
    /// a node would reveal them for an audit of that round.
    #[arg(long, value_enum, default_value_t = Start::Own, requires = "shadow_device")]
    shadow_start: Start,
    /// The last this many training rounds of each policy are averaged for its
    /// final loss.
    #[arg(long, default_value_t = 100)]
    window: usize,
    /// Every training round of every policy, as CSV.
    #[arg(long)]
    csv_out: Option<PathBuf>,
}

/// Where the shadow starts each round from, as `--shadow-start` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Start {
    Own,
    Resync,
    Reveal,
}

/// A directory under the system's temporary directory that no other process
/// holds, removed when it is dropped.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn create() -> Result<Self> {
        let mut attempt = 0u32;
        loop {
            let dir = std::env::temp_dir().join(format!(
                "leviathan-error-feedback-{}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(Self(dir)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => attempt += 1,
                Err(error) => {
                    return Err(error).with_context(|| format!("creating {}", dir.display()))
                }
            }
        }
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!(
                "[error-feedback] could not remove {}: {error}",
                self.0.display()
            );
        }
    }
}

/// The mean of `values`, or None for none.
fn mean(values: impl Iterator<Item = f32>) -> Option<f32> {
    let (sum, count) = values.fold((0.0f64, 0u64), |(sum, count), value| {
        (sum + value as f64, count + 1)
    });
    (count > 0).then(|| (sum / count as f64) as f32)
}

/// What one policy's records say, for the report.
fn summarize(reset_every: u32, records: &[StepRecord], window: usize) -> String {
    let mut out = String::new();
    let tail = &records[records.len().saturating_sub(window)..];
    let _ = writeln!(
        out,
        "policy reset_every={reset_every}: final loss {:.4} (mean of the last {} training rounds)",
        mean(tail.iter().map(|r| r.loss)).unwrap_or(f32::NAN),
        tail.len()
    );
    let drifts: Vec<_> = records
        .iter()
        .filter_map(|r| r.shadow.map(|d| (r.since_reset, d)))
        .collect();
    if drifts.is_empty() {
        return out;
    }
    let max_distance = drifts
        .iter()
        .map(|(_, d)| d.distance)
        .fold(0.0f32, f32::max);
    for (label, first) in [("first rounds of a window", true), ("later rounds", false)] {
        let part: Vec<_> = drifts
            .iter()
            .filter(|(since, _)| (*since == 1) == first)
            .map(|(_, d)| *d)
            .collect();
        let _ = writeln!(
            out,
            "  shadow, {label}: distance mean {:.4}, churn mean {:.4}, flips mean {:.6}",
            mean(part.iter().map(|d| d.distance)).unwrap_or(f32::NAN),
            mean(part.iter().map(|d| d.churn)).unwrap_or(f32::NAN),
            mean(part.iter().map(|d| d.flips)).unwrap_or(f32::NAN),
        );
    }
    let _ = writeln!(out, "  shadow: distance max {max_distance:.4}");
    out
}

/// The CSV header and one row per training round of every policy.
fn csv(policies: &[(u32, Vec<StepRecord>)]) -> String {
    let mut out = String::from("reset_every,step,since_reset,loss,distance,churn,flips\n");
    for (reset_every, records) in policies {
        for r in records {
            let drift = match r.shadow {
                Some(d) => format!("{},{},{}", d.distance, d.churn, d.flips),
                None => ",,".to_string(),
            };
            let _ = writeln!(
                out,
                "{reset_every},{},{},{},{drift}",
                r.step, r.since_reset, r.loss
            );
        }
    }
    out
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.steps == 0 || args.reset_every.is_empty() || args.window == 0 {
        return Err(anyhow!(
            "--steps and --window must be above 0, and --reset-every must name a policy"
        ));
    }
    let llm = read_run_model(&args.run_config)?;
    let global_batch = read_run_global_batch(&args.run_config)?;
    share_size(global_batch, args.nodes)?;
    check_cohort(
        args.nodes,
        read_run_verification_percent(&args.run_config)?,
        llm.cold_start_warmup_steps,
    )?;
    let first_step = CoordinatorProgress::default().step;
    let repo_files = checkpoint_repo_files(&published_checkpoint(&llm)?)?;
    let device = parse_device(&args.device)?;
    let kind = parse_dtype(&args.dtype)?;
    let nodes = ReplayTrainerConfig::for_run(&llm, repo_files.clone(), device, kind);
    let shadow = match &args.shadow_device {
        Some(shadow_device) => Some(ReplayTrainerConfig::for_run(
            &llm,
            repo_files,
            parse_device(shadow_device)?,
            parse_dtype(&args.shadow_dtype)?,
        )),
        None => None,
    };

    eprintln!(
        "[error-feedback] {} on {} {kind:?}, {} nodes, global batch {global_batch}, {} training rounds, {} per epoch, policies {:?}{}{}",
        llm.checkpoint,
        device_label(device),
        args.nodes,
        args.steps,
        args.epoch_rounds,
        args.reset_every,
        match &shadow {
            Some(shadow) => format!(
                ", shadow on {} {:?}",
                device_label(shadow.device),
                shadow.kind
            ),
            None => String::new(),
        },
        match (&shadow, args.shadow_start) {
            (None, _) | (Some(_), Start::Own) => "",
            (Some(_), Start::Resync) => ", resynced to node 0 at every reset",
            (Some(_), Start::Reveal) => ", starting each round from what node 0 reveals",
        }
    );
    let slots = schedule(first_step, args.steps, args.epoch_rounds);
    let mut data = RunData::open(&llm, args.data_dir.as_deref()).await?;
    let rounds = read_rounds(&mut data, &slots, first_step, global_batch, args.nodes).await?;
    // The weights hold one value per parameter in the training dtype, as the
    // buffer does, so a reveal of both is twice the buffer. The previous
    // round's values come on top, printed once a round has trained.
    let buffer = error_feedback_bytes(&nodes)?;
    println!(
        "error feedback per node: {buffer} bytes ({:.1} MiB) in {kind:?}; a reveal of the weights and the buffer sends {:.1} MiB",
        buffer as f64 / (1 << 20) as f64,
        2.0 * buffer as f64 / (1 << 20) as f64,
    );

    // Node 0's checkpoint at each reset, in a directory of this run's own.
    let scratch = match args.shadow_start {
        Start::Resync => Some(ScratchDir::create()?),
        Start::Own | Start::Reveal => None,
    };
    let shadow_start = match (args.shadow_start, &scratch) {
        (Start::Resync, Some(dir)) => ShadowStart::Resync(dir.0.join("checkpoint")),
        (Start::Reveal, _) => ShadowStart::Reveal,
        _ => ShadowStart::Own,
    };
    let mut policies = Vec::with_capacity(args.reset_every.len());
    for reset_every in &args.reset_every {
        let simulation = Simulation {
            nodes: nodes.clone(),
            node_count: args.nodes,
            shadow: shadow.clone(),
            reset_every: *reset_every,
            shadow_start: shadow_start.clone(),
        };
        let mut sent_values = None;
        let records = simulate(&simulation, rounds.clone(), &mut |observed| {
            // What a round's result holds besides its public indices and
            // signs: one value per coordinate sent, in the training dtype.
            if let (None, Some(sent)) = (sent_values, observed.sent.first()) {
                let values: i64 = sent
                    .iter()
                    .map(|result| result.sparse_val.numel() as i64)
                    .sum();
                sent_values = Some(values as u64 * kind.elt_size_in_bytes() as u64);
            }
            let record = observed.record;
            eprintln!(
                "[error-feedback] reset_every={reset_every} step {} since_reset {} loss {:.4}{}",
                record.step,
                record.since_reset,
                record.loss,
                match record.shadow {
                    Some(d) => format!(
                        " distance {:.4} churn {:.4} flips {:.6}",
                        d.distance, d.churn, d.flips
                    ),
                    None => String::new(),
                }
            )
        })?;
        if let (Some(bytes), true) = (sent_values, policies.is_empty()) {
            println!(
                "a round's result values per node: {bytes} bytes ({:.1} MiB) in {kind:?}",
                bytes as f64 / (1 << 20) as f64
            );
        }
        print!("{}", summarize(*reset_every, &records, args.window));
        policies.push((*reset_every, records));
        if let Some(path) = &args.csv_out {
            std::fs::write(path, csv(&policies))
                .with_context(|| format!("writing {}", path.display()))?;
        }
    }
    drop(scratch);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use leviathan_verifier::error_feedback::ShadowDrift;

    fn record(
        step: u32,
        since_reset: u32,
        loss: f32,
        shadow: Option<(f32, f32, f32)>,
    ) -> StepRecord {
        StepRecord {
            step,
            since_reset,
            loss,
            shadow: shadow.map(|(distance, churn, flips)| ShadowDrift {
                distance,
                churn,
                flips,
            }),
        }
    }

    #[test]
    fn the_final_loss_is_the_mean_of_the_window() {
        let records = vec![
            record(1, 1, 10.0, None),
            record(2, 2, 4.0, None),
            record(3, 3, 2.0, None),
        ];
        let report = summarize(0, &records, 2);
        assert!(
            report.contains("final loss 3.0000 (mean of the last 2 training rounds)"),
            "{report}"
        );
        assert!(!report.contains("shadow"));
        let report = summarize(0, &records, 10);
        assert!(
            report.contains("final loss 5.3333 (mean of the last 3 training rounds)"),
            "{report}"
        );
    }

    #[test]
    fn shadow_drift_is_split_into_first_and_later_rounds_of_a_window() {
        let records = vec![
            record(1, 1, 3.0, Some((0.0, 0.0, 0.0))),
            record(2, 2, 3.0, Some((0.5, 0.25, 0.002))),
            record(3, 1, 3.0, Some((0.1, 0.05, 0.0))),
            record(4, 2, 3.0, Some((0.3, 0.15, 0.004))),
        ];
        let report = summarize(2, &records, 10);
        assert!(
            report.contains(
                "first rounds of a window: distance mean 0.0500, churn mean 0.0250, flips mean 0.000000"
            ),
            "{report}"
        );
        assert!(
            report.contains(
                "later rounds: distance mean 0.4000, churn mean 0.2000, flips mean 0.003000"
            ),
            "{report}"
        );
        assert!(report.contains("distance max 0.5000"), "{report}");
    }

    #[test]
    fn every_training_round_of_every_policy_is_a_csv_row() {
        let policies = vec![
            (0, vec![record(1, 1, 2.5, None)]),
            (3, vec![record(1, 1, 2.5, Some((0.25, 0.5, 0.125)))]),
        ];
        assert_eq!(
            csv(&policies),
            "reset_every,step,since_reset,loss,distance,churn,flips\n0,1,1,2.5,,,\n3,1,1,2.5,0.25,0.5,0.125\n"
        );
    }

    #[test]
    fn a_scratch_directory_is_its_own_and_goes_when_dropped() {
        let first = ScratchDir::create().unwrap();
        let second = ScratchDir::create().unwrap();
        assert_ne!(first.0, second.0);
        assert!(first.0.is_dir() && second.0.is_dir());
        std::fs::write(first.0.join("model.safetensors"), b"weights").unwrap();
        let (first_path, second_path) = (first.0.clone(), second.0.clone());
        drop(first);
        drop(second);
        assert!(!first_path.exists());
        assert!(!second_path.exists());
    }
}
