//! What committing to a round's state costs a node of a run
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! Each round a node copies its weights and its DisTrO error-feedback buffer
//! off the device, encodes them, and hashes the encoding. This loads the run's
//! published checkpoint on the device and in the dtype a node trains in,
//! fills the buffer with a real step on the run's model, and times that
//! commitment over a number of rounds.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Parser;
use leviathan_verifier::{
    checkpoint_repo_files, device_label, parse_device, parse_dtype, published_checkpoint,
    read_run_model,
};
use psyche_core::OptimizerDefinition;
use psyche_modeling::{
    auto_model_for_causal_lm_from_pretrained, round_state_digest, write_round_state, Distro,
};
use tch::Tensor;

#[derive(Parser)]
struct Args {
    /// The run's config, whose model and optimizer are measured.
    #[arg(long)]
    run_config: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    /// How many commitments to time.
    #[arg(long, default_value_t = 10)]
    rounds: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let llm = read_run_model(&args.run_config)?;
    let OptimizerDefinition::Distro {
        compression_decay,
        compression_chunk,
        compression_topk,
        quantize_1bit,
        robust_aggregation,
        weight_decay,
        ..
    } = llm.optimizer
    else {
        return Err(anyhow!("the run does not train with DisTrO"));
    };
    let device = parse_device(&args.device)?;
    let kind = parse_dtype(&args.dtype)?;
    let files = checkpoint_repo_files(&published_checkpoint(&llm)?)?;
    let model = auto_model_for_causal_lm_from_pretrained(
        files,
        Some(kind),
        None,
        Some(device),
        None,
        Some(llm.max_seq_len as usize),
    )?;
    model.prepare_for_training();
    let mut distro = Distro::new(
        &*model,
        compression_decay as f64,
        compression_chunk as i64,
        compression_topk as i64,
        weight_decay.unwrap_or(0.0) as f64,
        robust_aggregation,
        quantize_1bit,
    );
    // A real step fills the buffer, as a node's is after its first round.
    let tokens: Vec<i64> = (0..llm.max_seq_len as i64).collect();
    let input = Tensor::from_slice(&tokens)
        .reshape([1, tokens.len() as i64])
        .to_device(device);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.ok_or_else(|| anyhow!("a forward pass with labels has a loss"))?
        .backward();
    distro.generate(&*model, &[], 0.0, 1e-4, false);

    let mut bytes = vec![];
    write_round_state(&mut bytes, &*model, &distro)?;
    let mut times = vec![];
    for _ in 0..args.rounds {
        let started = Instant::now();
        round_state_digest(&*model, &distro)?;
        times.push(started.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{} on {} {kind:?}: a round's state is {} bytes ({:.1} MiB); committing to it takes {:.3} s mean, {:.3} s median, {:.3} s max over {} rounds",
        llm.checkpoint,
        device_label(device),
        bytes.len(),
        bytes.len() as f64 / (1 << 20) as f64,
        mean,
        times[times.len() / 2],
        times[times.len() - 1],
        args.rounds,
    );
    Ok(())
}
