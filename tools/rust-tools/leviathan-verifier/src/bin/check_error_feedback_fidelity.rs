//! Checks that `measure-error-feedback` trains a run the way real clients do
//! (wienerlabs/leviathan-net#48). It rebuilds a run's rounds from the payloads
//! one client dumped with `--write-gradients-dir`, trains them in the
//! simulation from the same checkpoint and data, and compares every payload
//! byte for byte. See `leviathan_verifier::fidelity`.

use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::error_feedback::check_cohort;
use leviathan_verifier::error_feedback::ShadowStart;
use leviathan_verifier::error_feedback::Simulation;
use leviathan_verifier::fidelity::check_fidelity;
use leviathan_verifier::fidelity::plan_rounds;
use leviathan_verifier::fidelity::read_client_dumps;
use leviathan_verifier::fidelity::read_planned;
use leviathan_verifier::parse_device;
use leviathan_verifier::parse_dtype;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::read_run_model;
use leviathan_verifier::read_run_verification_percent;
use leviathan_verifier::run_data::RunData;
use leviathan_verifier::ReplayTrainerConfig;

#[derive(Parser, Debug)]
#[command(
    name = "check-error-feedback-fidelity",
    about = "Check the error-feedback simulation against the payloads real clients sent (wienerlabs/leviathan-net#48)"
)]
struct Args {
    /// The configuration the clients' run was started with.
    #[arg(long)]
    run_config: PathBuf,
    /// One client's `--write-gradients-dir`: every node's payload of every
    /// round.
    #[arg(long)]
    dumps: PathBuf,
    /// A local copy of the run's data. Without it the run's own HTTP location
    /// is read, as the clients read it.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// The device and dtype the clients trained in. The client loads its model
    /// in bf16.
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    /// The fewest epochs the run has to cover: the rounds that close an epoch
    /// and the reset of the next one's start are what the check exists for.
    #[arg(long, default_value_t = 2)]
    min_epochs: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let llm = read_run_model(&args.run_config)?;
    let dumps = read_client_dumps(&args.dumps)?;
    let (planned, signers) = plan_rounds(&dumps)?;
    check_cohort(
        signers.len(),
        read_run_verification_percent(&args.run_config)?,
        llm.cold_start_warmup_steps,
    )?;
    let epochs = planned.iter().filter(|round| round.height == 0).count();
    let training = planned
        .iter()
        .filter(|round| round.shares.is_some())
        .count();
    let closing = planned.len() - training;
    println!(
        "{} payloads from {} nodes: {epochs} epochs, {training} training rounds, steps {} to {}",
        dumps.len(),
        signers.len(),
        planned.first().map_or(0, |round| round.step),
        planned.last().map_or(0, |round| round.step),
    );
    if epochs < args.min_epochs || closing < 2 * args.min_epochs.saturating_sub(1) {
        return Err(anyhow!(
            "the run covers {epochs} epochs and {closing} closing rounds; the check needs {} epochs and the rounds that close them",
            args.min_epochs
        ));
    }

    let mut data = RunData::open(&llm, args.data_dir.as_deref()).await?;
    let rounds = read_planned(&mut data, &planned).await?;
    let repo_files = checkpoint_repo_files(&published_checkpoint(&llm)?)?;
    let simulation = Simulation {
        nodes: ReplayTrainerConfig::for_run(
            &llm,
            repo_files,
            parse_device(&args.device)?,
            parse_dtype(&args.dtype)?,
        ),
        node_count: signers.len(),
        shadow: None,
        reset_every: 0,
        shadow_start: ShadowStart::Own,
    };
    let report = check_fidelity(&simulation, rounds, &dumps, &signers)?;
    for (step, signer) in &report.mismatched {
        println!("step {step}: {signer}'s payload differs");
    }
    match report.mismatched.is_empty() {
        true => {
            println!("all {} payloads match byte for byte", report.matched);
            Ok(())
        }
        false => Err(anyhow!(
            "{} of {} payloads differ",
            report.mismatched.len(),
            report.matched + report.mismatched.len()
        )),
    }
}
