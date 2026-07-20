use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use leviathan_verifier::audit_dirs;
use leviathan_verifier::ContributionOutcome;
use psyche_verifier::DEFAULT_BAND;

#[derive(Parser, Debug)]
#[command(
    name = "leviathan-verifier",
    about = "Replay-audit DisTrO gradient dumps against a recomputed reference"
)]
struct Args {
    #[arg(long)]
    submitted: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long, default_value_t = DEFAULT_BAND)]
    band: f32,
    #[arg(long, default_value_t = false)]
    cuda: bool,
}

fn hex8(bytes: &[u8; 32]) -> String {
    bytes[..4].iter().map(|b| format!("{b:02x}")).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cuda {
        tch::Device::cuda_if_available()
    } else {
        tch::Device::Cpu
    };

    let summary = audit_dirs(&args.submitted, &args.reference, args.band, device)?;

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

    if summary.fraud() > 0 {
        std::process::exit(2);
    }
    Ok(())
}
