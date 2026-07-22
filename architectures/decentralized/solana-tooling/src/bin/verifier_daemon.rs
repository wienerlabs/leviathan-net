use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_verifier::DEFAULT_BAND;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::read_keypair_file;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

#[derive(Parser, Debug)]
#[command(
    name = "leviathan-verifier-daemon",
    about = "Always-on replay verifier: watches a live run, replay-audits committed contributions, and slashes fraud on chain"
)]
struct Args {
    #[arg(long)]
    run_id: String,
    #[arg(long)]
    coordinator_account: String,
    #[arg(long)]
    run: Option<String>,
    #[arg(long)]
    authority: PathBuf,
    #[arg(long)]
    submitted_dir: PathBuf,
    #[arg(long)]
    reference_dir: PathBuf,
    #[arg(long, default_value_t = DEFAULT_BAND)]
    band: f32,
    #[arg(long, default_value_t = 8)]
    poll_secs: u64,
    #[arg(long, default_value_t = false)]
    once: bool,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    #[arg(long, default_value_t = false)]
    audit_assigned: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let authority = read_keypair_file(&args.authority).map_err(|err| {
        anyhow!(
            "cannot read authority keypair {}: {}",
            args.authority.display(),
            err
        )
    })?;
    let coordinator_account = Pubkey::from_str(&args.coordinator_account)?;
    let run = match &args.run {
        Some(value) => Pubkey::from_str(value)?,
        None => {
            let index = u64::from_le_bytes(
                solana_sdk::hash::hash(args.run_id.as_bytes()).to_bytes()[0..8]
                    .try_into()
                    .unwrap(),
            );
            psyche_solana_treasurer::find_run(index)
        }
    };
    let mut endpoint = ToolboxEndpoint::new_devnet().await;

    let config = AuditConfig {
        run_id: args.run_id.clone(),
        submitted_dir: args.submitted_dir.clone(),
        reference_dir: args.reference_dir.clone(),
        band: args.band,
        audit_assigned: args.audit_assigned,
        dry_run: args.dry_run,
    };

    println!(
        "[verifier-daemon] run_id={} coordinator={} run={} authority={} band={:.4} mode={} dry_run={}",
        config.run_id,
        coordinator_account,
        run,
        authority.pubkey(),
        config.band,
        if config.audit_assigned {
            "audit-assigned"
        } else {
            "audit-all"
        },
        config.dry_run
    );

    let mut convicted: HashSet<String> = HashSet::new();
    loop {
        match audit_pass(
            &mut endpoint,
            &authority,
            &coordinator_account,
            &run,
            &config,
            &mut convicted,
        )
        .await
        {
            Ok(new_convictions) => {
                if args.once {
                    println!(
                        "[verifier-daemon] single pass complete, {} new conviction(s)",
                        new_convictions
                    );
                    break;
                }
            }
            Err(err) => eprintln!("[verifier-daemon] pass error: {err:#}"),
        }
        tokio::time::sleep(Duration::from_secs(args.poll_secs)).await;
    }

    Ok(())
}
