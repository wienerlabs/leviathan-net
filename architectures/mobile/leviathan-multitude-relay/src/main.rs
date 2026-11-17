use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use leviathan_multitude::model::default_device;
use leviathan_multitude::RunConfig;
use leviathan_multitude_relay::http::serve;
use leviathan_multitude_relay::{Options, Relay};

const SHAKESPEARE: &str =
    include_str!("../../../../shared/client/src/state/prompt_texts/index.json");

#[derive(Parser, Debug)]
#[command(
    name = "multitude-relay",
    about = "Host a Leviathan Multitude run that phones can join"
)]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080", help = "Address to listen on")]
    listen: String,
    #[arg(long, help = "Run config as JSON; the defaults otherwise")]
    config: Option<PathBuf>,
    #[arg(
        long,
        help = "Train on this file instead of the bundled Shakespeare prompts"
    )]
    corpus: Option<PathBuf>,
    #[arg(long, default_value_t = 1651, help = "Seed of the genesis weights")]
    seed: u64,
    #[arg(long, help = "Override the config's round length")]
    round_seconds: Option<f32>,
    #[arg(long, help = "Override the config's audit rate, in percent")]
    verification_percent: Option<u8>,
    #[arg(
        long,
        default_value_t = 30,
        help = "Rounds an audited phone has to reveal its buffer"
    )]
    reveal_rounds: u32,
    #[arg(long, default_value_t = 64, help = "Most phones the run admits")]
    max_phones: usize,
    #[arg(
        long,
        help = "Convict on the replay band too; leave off until the band is measured on real phones"
    )]
    enforce_band: bool,
}

fn shakespeare() -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Prompt {
        text: String,
    }
    #[derive(serde::Deserialize)]
    struct Prompts {
        prompts: Vec<Prompt>,
    }
    let prompts: Prompts =
        serde_json::from_str(SHAKESPEARE).context("parsing the bundled prompts")?;
    Ok(prompts
        .prompts
        .into_iter()
        .map(|p| p.text)
        .collect::<Vec<_>>()
        .join("\n\n"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut cfg = match &args.config {
        Some(path) => serde_json::from_str::<RunConfig>(
            &std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?,
        None => RunConfig::default(),
    };
    if let Some(seconds) = args.round_seconds {
        cfg.round_seconds = seconds;
    }
    if let Some(percent) = args.verification_percent {
        cfg.audit.verification_percent = percent;
    }
    let corpus = match &args.corpus {
        Some(path) => std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        None => shakespeare()?.into_bytes(),
    };
    let opts = Options {
        reveal_rounds: args.reveal_rounds,
        max_phones: args.max_phones,
        enforce_band: args.enforce_band,
        ..Options::default()
    };
    let relay = Relay::new(cfg, corpus, args.seed, default_device(), opts)?;
    let info = relay.info();
    println!(
        "Leviathan Multitude relay: run {} with {} parameters, {}s rounds, audits at {}%, band {:.4} ({})",
        info.cfg.run_id,
        info.cfg.model.num_params(),
        info.cfg.round_seconds,
        info.cfg.audit.verification_percent,
        info.cfg.audit.band,
        if args.enforce_band { "enforced" } else { "dry run" },
    );
    let listener = tokio::net::TcpListener::bind(&args.listen)
        .await
        .with_context(|| format!("listening on {}", args.listen))?;
    println!("Phones join at http://{}", listener.local_addr()?);
    serve(listener, relay).await?;
    Ok(())
}
