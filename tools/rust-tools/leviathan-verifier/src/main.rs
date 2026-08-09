use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::audit_dirs;
use leviathan_verifier::audit_with_replay;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::index_dir;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::ContributionOutcome;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::TrainerReplayEngine;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_data_provider::download_model_repo_sync;
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
    /// --replay-model instead to have this verifier recompute the reference
    /// itself rather than trusting someone else's dumps.
    #[arg(long)]
    reference: Option<PathBuf>,

    #[arg(long)]
    replay_model: Option<String>,
    #[arg(long)]
    replay_data_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    replay_sequence_length: usize,
    #[arg(long, default_value_t = 4.0e-4)]
    replay_lr: f64,
    #[arg(long, default_value_t = 0.999)]
    replay_compression_decay: f32,
    #[arg(long, default_value_t = 8)]
    replay_compression_topk: u16,
    #[arg(long, default_value_t = 64)]
    replay_compression_chunk: u16,

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

    let summary = match (&args.reference, &args.replay_model) {
        (Some(reference), None) => audit_dirs(&args.submitted, reference, args.band, device)?,
        (None, Some(model)) => {
            let data_dir = args.replay_data_dir.as_ref().ok_or_else(|| {
                anyhow!("--replay-data-dir is required to recompute a reference")
            })?;
            let repo_files = download_model_repo_sync(&model.to_string(), None, None, None, true)?;
            let trainer = build_replay_trainer(&ReplayTrainerConfig {
                repo_files,
                sequence_length: args.replay_sequence_length,
                micro_batch_size: 1,
                learning_rate: args.replay_lr,
                compression_decay: args.replay_compression_decay,
                compression_topk: args.replay_compression_topk,
                compression_chunk: args.replay_compression_chunk,
                clip_grad_norm: Some(1.0),
                quantize_1bit: false,
                device,
                kind: tch::Kind::Float,
            })?;

            let mut provider = LocalDataProvider::new_from_directory(
                data_dir,
                TokenSize::TwoBytes,
                args.replay_sequence_length,
                Shuffle::DontShuffle,
            )?;

            let mut assignments = Vec::new();
            for (target_index, (key, _)) in index_dir(&args.submitted)?.iter().enumerate() {
                let Some((step, batch_id)) = parse_assignment_key(key) else {
                    println!("skip  {key}: cannot read a step and batch out of this dump name");
                    continue;
                };
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
                    batch_id,
                    data,
                });
            }

            let engine = TrainerReplayEngine::new(trainer, assignments, device);
            audit_with_replay(&args.submitted, &engine, args.band, device)?
        }
        (Some(_), Some(_)) => {
            return Err(anyhow!(
                "pass either --reference or --replay-model, not both: they are two different sources of truth"
            ));
        }
        (None, None) => {
            return Err(anyhow!(
                "pass --reference <dir> to compare against honest dumps, or --replay-model <repo> to recompute the reference here"
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
