use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use leviathan_verifier::write_token_dataset;

#[derive(Parser, Debug)]
#[command(
    name = "nano-dataset",
    about = "Write a tokenized dataset sized for a small model, so a replay verifier can recompute what a node trained"
)]
struct Args {
    #[arg(long)]
    out: PathBuf,
    /// Must match the model's vocabulary. A corpus tokenized for a larger
    /// vocabulary indexes past a small model's embedding table.
    #[arg(long)]
    vocab_size: u16,
    #[arg(long, default_value_t = 64)]
    sequence_length: usize,
    #[arg(long, default_value_t = 256)]
    sequences: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_token_dataset(
        &args.out,
        args.vocab_size,
        args.sequence_length,
        args.sequences,
        args.seed,
    )?;
    println!(
        "wrote {} sequences of {} tokens over a vocabulary of {} to {} (seed {})",
        args.sequences,
        args.sequence_length,
        args.vocab_size,
        args.out.display(),
        args.seed
    );
    Ok(())
}
