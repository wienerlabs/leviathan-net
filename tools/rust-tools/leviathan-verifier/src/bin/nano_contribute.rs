use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::write_results_dump;
use leviathan_verifier::ReplayTrainerConfig;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_data_provider::download_model_repo_sync;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use tokio_util::sync::CancellationToken;

#[derive(Parser, Debug)]
#[command(
    name = "nano-contribute",
    about = "Train one batch and write the gradient dump a node would submit, so the replay verifier can be exercised without a full swarm"
)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    data_dir: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value = "node0")]
    committer: String,
    #[arg(long, default_value_t = 1)]
    step: u32,
    #[arg(long, default_value_t = 0)]
    batch_start: u64,
    #[arg(long, default_value_t = 0)]
    batch_end: u64,
    #[arg(long, default_value_t = 64)]
    sequence_length: usize,
    #[arg(long, default_value_t = 4.0e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0.999)]
    compression_decay: f32,
    #[arg(long, default_value_t = 8)]
    compression_topk: u16,
    #[arg(long, default_value_t = 64)]
    compression_chunk: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir)?;

    let repo_files = download_model_repo_sync(&args.model, None, None, None, true)?;
    let trainer = build_replay_trainer(&ReplayTrainerConfig {
        repo_files,
        sequence_length: args.sequence_length,
        micro_batch_size: 1,
        learning_rate: args.lr,
        compression_decay: args.compression_decay,
        compression_topk: args.compression_topk,
        compression_chunk: args.compression_chunk,
        clip_grad_norm: Some(1.0),
        quantize_1bit: false,
        device: tch::Device::Cpu,
        kind: tch::Kind::Float,
    })?;

    let batch_id = BatchId(ClosedInterval::new(args.batch_start, args.batch_end));
    let mut provider = LocalDataProvider::new_from_directory(
        &args.data_dir,
        TokenSize::TwoBytes,
        args.sequence_length,
        Shuffle::DontShuffle,
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

    let output = trainer
        .train(
            args.step,
            Batch {
                id: batch_id,
                data: BatchData::CPU(data),
            },
            None,
            true,
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        )
        .map_err(|err| anyhow!("training failed: {err}"))?;

    let results = output
        .distro_results
        .ok_or_else(|| anyhow!("this optimizer produced no DisTrO results"))?;

    let name = format!(
        "result-{}-step{}-batchB[{}, {}].vec-postcard",
        args.committer, args.step, args.batch_start, args.batch_end
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
