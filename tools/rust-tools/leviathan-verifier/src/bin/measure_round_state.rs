//! What committing to a round's state costs a node of a run
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! Each round a node copies its weights and its DisTrO error-feedback buffer
//! off the device, encodes them, and hashes the encoding. A node that keeps
//! its states writes the encoding to disk in the same pass, on the trainer
//! thread, and its client then imports the file into the store it serves
//! states from. This loads the run's published checkpoint on the device and
//! in the dtype a node trains in, fills the buffer with a real step on the
//! run's model, and times the commitment over a number of rounds, and, given
//! `--keep-in`, keeping the state and importing it as well.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Parser;
use leviathan_verifier::round_states::load_node_state;
use leviathan_verifier::{device_label, parse_device, parse_dtype, read_run_model};
use psyche_modeling::{keep_round_state, round_state_digest, write_round_state, CausalLM, Distro};
use psyche_network::{blobs_gc_interval, RoundStates};
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
    /// A directory to keep the states in, as a node started with
    /// `--round-state-dir` does, to time keeping them too. It has to be empty
    /// or absent, and is left as the node would leave it.
    #[arg(long)]
    keep_in: Option<PathBuf>,
}

/// The mean, median and largest of `times`, in seconds.
fn summary(mut times: Vec<f64>) -> (f64, f64, f64) {
    times.sort_by(|a, b| a.total_cmp(b));
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    (mean, times[times.len() / 2], times[times.len() - 1])
}

/// Times keeping the state `model` and `distro` train from in `dir`, each
/// round on the trainer thread, and the client's import of the first one into
/// the store it serves states from.
fn time_keeping(
    dir: &Path,
    model: &dyn CausalLM,
    distro: &Distro,
    rounds: u32,
) -> Result<(Vec<f64>, f64)> {
    let runtime = tokio::runtime::Runtime::new()?;
    let states = runtime.block_on(RoundStates::open(dir, blobs_gc_interval()))?;
    let mut times = vec![];
    let mut digest = None;
    for _ in 0..rounds {
        let started = Instant::now();
        let kept = keep_round_state(states.rounds_dir(), model, distro)?;
        times.push(started.elapsed().as_secs_f64());
        kept.file?;
        digest = Some(kept.digest);
    }
    let digest = digest.ok_or_else(|| anyhow!("no round was kept"))?;
    let started = Instant::now();
    runtime.block_on(states.keep(1, digest))?;
    let import = started.elapsed().as_secs_f64();
    runtime.block_on(states.shutdown())?;
    Ok((times, import))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let llm = read_run_model(&args.run_config)?;
    let device = parse_device(&args.device)?;
    let kind = parse_dtype(&args.dtype)?;
    let (model, mut distro) = load_node_state(&llm, device, kind)?;
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
    let (mean, median, max) = summary(times);
    println!(
        "{} on {} {kind:?}: a round's state is {} bytes ({:.1} MiB); committing to it takes {mean:.3} s mean, {median:.3} s median, {max:.3} s max over {} rounds",
        llm.checkpoint,
        device_label(device),
        bytes.len(),
        bytes.len() as f64 / (1 << 20) as f64,
        args.rounds,
    );
    if let Some(dir) = &args.keep_in {
        let (times, import) = time_keeping(dir, &*model, &distro, args.rounds)?;
        let (mean, median, max) = summary(times);
        println!(
            "keeping it as well, on the trainer thread: {mean:.3} s mean, {median:.3} s median, {max:.3} s max over {} rounds; the client's import into the store: {import:.3} s",
            args.rounds,
        );
    }
    Ok(())
}
