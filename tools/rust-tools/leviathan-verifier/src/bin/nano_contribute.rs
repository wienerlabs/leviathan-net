use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::cold_start_warmup_bounds;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::read_run_model;
use leviathan_verifier::replay_data_layout;
use leviathan_verifier::write_results_dump;
use leviathan_verifier::ReplayTrainerConfig;
use psyche_coordinator::CoordinatorProgress;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::Trainer;
use tokio_util::sync::CancellationToken;

#[derive(Parser, Debug)]
#[command(
    name = "nano-contribute",
    about = "Train one batch and write the gradient dump a node would submit, so the replay verifier can be exercised without a full swarm"
)]
struct Args {
    /// The run configuration the contribution is trained under
    /// (`config/leviathan/*.toml`): checkpoint, learning-rate schedule,
    /// optimizer, sequence length and data layout, as a node trains with them.
    #[arg(long)]
    run_config: PathBuf,
    #[arg(long)]
    data_dir: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value = "node0")]
    committer: String,
    /// The step trained. Defaults to a run's first step.
    #[arg(long)]
    step: Option<u32>,
    /// Whether the epoch is flagged a cold start, which is when a node trains
    /// within the run's cold-start warmup bounds.
    #[arg(long, default_value_t = false)]
    cold_start_epoch: bool,
    #[arg(long, default_value_t = 0)]
    batch_start: u64,
    #[arg(long, default_value_t = 0)]
    batch_end: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir)?;

    let llm = read_run_model(&args.run_config)?;
    let step = args.step.unwrap_or(CoordinatorProgress::default().step);
    let config = ReplayTrainerConfig::for_run(
        &llm,
        checkpoint_repo_files(&published_checkpoint(&llm)?)?,
        tch::Device::Cpu,
        tch::Kind::Float,
    );
    let trainer = build_replay_trainer(&config)?;
    let (token_size, shuffle) = replay_data_layout(&llm)?;

    let batch_id = BatchId(ClosedInterval::new(args.batch_start, args.batch_end));
    let mut provider = LocalDataProvider::new_from_directory(
        &args.data_dir,
        token_size,
        config.sequence_length,
        shuffle,
    )?;
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

    // Trained as a node trains the run's first step: with a fresh trainer, so
    // from zeroed error feedback, and within the warmup bounds if the epoch is
    // flagged a cold start.
    let output = trainer
        .train(
            step,
            Batch {
                id: batch_id,
                data: BatchData::CPU(data),
            },
            args.cold_start_epoch
                .then(|| cold_start_warmup_bounds(&llm, step))
                .flatten(),
            true,
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        )
        .map_err(|err| anyhow!("training failed: {err}"))?;

    let results = output
        .distro_results
        .ok_or_else(|| anyhow!("this optimizer produced no DisTrO results"))?;
    // Sent as the run sends it.
    let results = match config.quantize_1bit() {
        true => Trainer::quantize_results(&results),
        false => results,
    };

    let name = format!(
        "result-{}-step{}-batch{}.vec-postcard",
        args.committer, step, batch_id
    );
    let path = args.out_dir.join(&name);
    write_results_dump(&path, &results)?;

    println!(
        "loss {:.4}, wrote {} tensors to {}",
        output.loss,
        results.len(),
        path.display()
    );
    Ok(())
}
