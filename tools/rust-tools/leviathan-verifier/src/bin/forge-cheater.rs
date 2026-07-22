use std::io::Cursor;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use psyche_modeling::DistroResult;
use psyche_network::distro_results_from_reader;
use psyche_network::distro_results_to_bytes;
use psyche_network::SerializedDistroResult;

#[derive(Parser, Debug)]
#[command(
    name = "forge-cheater",
    about = "Apply the sign-flip cheat transform to a real gradient dump so the replay verifier can be exercised against live data"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.input)?;
    let mut tampered: Vec<SerializedDistroResult> = Vec::new();
    for serialized in distro_results_from_reader(Cursor::new(bytes)) {
        let mut result: DistroResult = (&serialized?).try_into()?;
        result.sparse_val = result.sparse_val * -5.0;
        tampered.push((&result).try_into()?);
    }
    let out_bytes = distro_results_to_bytes(&tampered)?;
    std::fs::write(&args.output, out_bytes)?;
    println!(
        "forged cheater dump {} -> {}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}
