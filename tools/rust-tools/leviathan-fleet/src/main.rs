//! Collects what nodes say about their hardware and folds it into one summary.
//!
//! Two steps on purpose. `report` runs on a node and prints that node's own
//! inventory; `summarize` runs on the operator's machine and folds the reports
//! into the aggregate the dashboard publishes. The per-node reports never leave
//! the operator, because a report tied to a node is a targeting hint: the
//! committee lottery is seeded from `Round.random_seed`, which is public, so
//! anyone can already work out which identities hold the verifier seats for a
//! round. What they cannot do, and must not be handed, is a map from those
//! identities to the machines behind them.

use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use clap::Subcommand;
use psyche_metrics::hardware::FleetSummary;
use psyche_metrics::hardware::HardwareReport;
use psyche_metrics::hardware::local_report;
use psyche_metrics::hardware::summarize;

#[derive(Parser)]
#[command(
    name = "leviathan-fleet",
    about = "Report local hardware, and summarize a fleet without attributing it"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print this machine's own inventory as JSON.
    Report {
        /// Write to a file instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Fold node reports into the aggregate the dashboard reads.
    Summarize {
        /// The report files to fold, as written by `report`.
        #[arg(required = true)]
        reports: Vec<PathBuf>,

        /// Write to a file instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn write(value: &str, out: Option<PathBuf>) -> Result<()> {
    match out {
        Some(path) => {
            fs::write(&path, format!("{value}\n"))
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        },
        None => println!("{value}"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Report { out } => {
            let report = local_report();
            if report.gpus.is_empty() {
                eprintln!(
                    "no NVIDIA devices visible, reporting a CPU-only node"
                );
            }
            write(&serde_json::to_string_pretty(&report)?, out)
        },
        Command::Summarize { reports, out } => {
            let mut parsed: Vec<HardwareReport> = Vec::new();
            for path in &reports {
                let text = fs::read_to_string(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                let report: HardwareReport = serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", path.display()))?;
                parsed.push(report);
            }

            let summary: FleetSummary = summarize(&parsed);
            eprintln!(
                "folded {} reports into {} GPUs across {} models",
                summary.nodes_reporting,
                summary.total_gpus,
                summary.gpus.len()
            );
            write(&serde_json::to_string_pretty(&summary)?, out)
        },
    }
}
