use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::audit_dirs;
use leviathan_verifier::audit_with_replay;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::cold_start_warmup_bounds;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::index_dir;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::replay_epoch;
use leviathan_verifier::DumpKey;
use leviathan_verifier::read_run_model;
use leviathan_verifier::replay_data_layout;
use leviathan_verifier::ContributionOutcome;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::TrainerReplayEngine;
use psyche_coordinator::CoordinatorProgress;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::BatchDataCPU;
use psyche_verifier::DEFAULT_BAND;

#[derive(Parser, Debug)]
#[command(
    name = "leviathan-verifier",
    about = "Replay-audit DisTrO gradient dumps against a recomputed reference"
)]
struct Args {
    #[arg(long)]
    submitted: PathBuf,

    /// A directory of honest dumps to compare against. Omit it and pass
    /// --run-config instead to have this verifier recompute the reference
    /// itself rather than trusting someone else's dumps.
    #[arg(long)]
    reference: Option<PathBuf>,

    /// The run configuration the dumps were trained under
    /// (`config/leviathan/*.toml`). The replay trains exactly as it says:
    /// checkpoint, learning-rate schedule, optimizer, sequence length and
    /// data layout.
    #[arg(long)]
    run_config: Option<PathBuf>,
    /// A local copy of the run's data shards.
    #[arg(long)]
    replay_data_dir: Option<PathBuf>,
    /// The run's first step, as the coordinator reports it
    /// (`Coordinator::replayable_step`): a run's first step unless it says
    /// otherwise. It is the only step a replay can reproduce. Dumps from any
    /// other step are left alone.
    #[arg(long)]
    first_step: Option<u32>,
    /// The epoch whose first step is replayed, by when it started (the
    /// coordinator's `epoch_state.start_timestamp`). Needed only when the dumps
    /// of the first step come from more than one epoch, as they do once a run
    /// restarts from its first step (wienerlabs/leviathan-net#67).
    #[arg(long)]
    epoch_start: Option<u64>,
    /// Whether the run's first epoch was flagged a cold start, which is when
    /// it trains within the run's cold-start warmup bounds. A run that was
    /// paused before its first epoch is; one started directly is not.
    #[arg(long, default_value_t = false)]
    cold_start_epoch: bool,

    #[arg(long, default_value_t = DEFAULT_BAND)]
    band: f32,
    #[arg(long, default_value_t = false)]
    cuda: bool,
    #[arg(long)]
    json_out: Option<PathBuf>,
}

fn hex8(bytes: &[u8; 32]) -> String {
    bytes[..4].iter().map(|b| format!("{b:02x}")).collect()
}

fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cuda {
        tch::Device::cuda_if_available()
    } else {
        tch::Device::Cpu
    };

    let summary = match (&args.reference, &args.run_config) {
        (Some(reference), None) => audit_dirs(&args.submitted, reference, args.band, device)?,
        (None, Some(run_config)) => {
            let data_dir = args.replay_data_dir.as_ref().ok_or_else(|| {
                anyhow!("--replay-data-dir is required to recompute a reference")
            })?;
            let first_step = args
                .first_step
                .unwrap_or(CoordinatorProgress::default().step);
            let llm = read_run_model(run_config)?;
            let config = ReplayTrainerConfig::for_run(
                &llm,
                checkpoint_repo_files(&published_checkpoint(&llm)?)?,
                device,
                tch::Kind::Float,
            );
            let trainer = build_replay_trainer(&config)?;
            let (token_size, shuffle) = replay_data_layout(&llm)?;
            let mut provider = LocalDataProvider::new_from_directory(
                data_dir,
                token_size,
                config.sequence_length,
                shuffle,
            )?;

            let submitted = index_dir(&args.submitted)?;
            let dumps: Vec<DumpKey> = submitted
                .keys()
                .filter_map(|key| parse_assignment_key(key))
                .collect();
            let epoch = replay_epoch(&dumps, first_step, args.epoch_start)?;
            let mut assignments = Vec::new();
            for (target_index, key) in submitted.keys().enumerate() {
                let Some(DumpKey {
                    epoch_start,
                    step,
                    batch_id,
                }) = parse_assignment_key(key)
                else {
                    println!(
                        "skip  {key}: cannot read an epoch, step and batch out of this dump name"
                    );
                    continue;
                };
                if step != first_step {
                    println!(
                        "skip  {key}: only the run's first step, {first_step}, can be replayed"
                    );
                    continue;
                }
                if Some(epoch_start) != epoch {
                    println!("skip  {key}: not of the epoch being replayed");
                    continue;
                }
                let data: Vec<BatchDataCPU> = provider
                    .get_samples(batch_id)
                    .await?
                    .into_iter()
                    .map(|x| BatchDataCPU {
                        input_ids: x.input_ids,
                        labels: x.labels,
                        position_ids: x.position_ids,
                        sequence_lengths: x.sequence_lengths,
                    })
                    .collect();
                assignments.push(ReplayAssignment {
                    target_index: target_index as u64,
                    step,
                    warmup_lr_between: args
                        .cold_start_epoch
                        .then(|| cold_start_warmup_bounds(&llm, first_step))
                        .flatten(),
                    batch_id,
                    data,
                });
            }

            let engine =
                TrainerReplayEngine::new(trainer, assignments, device, config.quantize_1bit());
            audit_with_replay(&args.submitted, &engine, args.band, device)?
        }
        (Some(_), Some(_)) => {
            return Err(anyhow!(
                "pass either --reference or --run-config, not both: they are two different sources of truth"
            ));
        }
        (None, None) => {
            return Err(anyhow!(
                "pass --reference <dir> to compare against honest dumps, or --run-config <toml> to recompute the reference here"
            ));
        }
    };

    for outcome in &summary.outcomes {
        match outcome {
            ContributionOutcome::Ok { key, distance } => {
                println!("ok    {key}: distance {distance:.4} within band {:.4}", args.band)
            }
            ContributionOutcome::Fraud { key, proof } => println!(
                "FRAUD {key}: distance {:.4} exceeds band {:.4} (committed {} replayed {})",
                proof.distance,
                proof.band,
                hex8(&proof.committed_hash),
                hex8(&proof.replayed_hash)
            ),
            ContributionOutcome::LengthMismatch { key, submitted, reference } => println!(
                "FRAUD {key}: length mismatch ({submitted} vs {reference}) is itself a fraud signal"
            ),
            ContributionOutcome::NoReference { key } => {
                println!("skip  {key}: no reference contribution to replay against")
            }
        }
    }

    println!();
    println!(
        "audited {} contributions, {} fraud verdicts, band {:.4}",
        summary.audited(),
        summary.fraud(),
        args.band
    );

    if let Some(path) = &args.json_out {
        let items: Vec<String> = summary
            .proofs()
            .iter()
            .map(|proof| {
                format!(
                    "{{\"index\":{},\"committed_hash\":\"{}\",\"replayed_hash\":\"{}\",\"distance\":{},\"band\":{}}}",
                    proof.target_index,
                    hex32(&proof.committed_hash),
                    hex32(&proof.replayed_hash),
                    proof.distance,
                    proof.band
                )
            })
            .collect();
        std::fs::write(path, format!("[{}]\n", items.join(",")))?;
    }

    if summary.fraud() > 0 {
        std::process::exit(2);
    }
    Ok(())
}
