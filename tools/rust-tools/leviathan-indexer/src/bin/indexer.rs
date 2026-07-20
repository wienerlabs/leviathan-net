use std::str::FromStr;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_indexer::assess_security;
use leviathan_indexer::compute_telemetry;
use leviathan_indexer::RunEconomics;
use leviathan_indexer::DEFAULT_LEADERBOARD_SIZE;
use psyche_solana_coordinator::Client;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::pubkey::Pubkey;
use solana_toolbox_endpoint::ToolboxEndpoint;

#[derive(Parser, Debug)]
#[command(
    name = "leviathan-indexer",
    about = "Emit honest run telemetry from a coordinator account"
)]
struct Args {
    #[arg(long)]
    coordinator_account: String,
    #[arg(long, default_value = "unknown")]
    run_id: String,
    #[arg(long, default_value = "devnet")]
    rpc: String,
    #[arg(long, default_value_t = DEFAULT_LEADERBOARD_SIZE)]
    leaderboard: usize,
    #[arg(long)]
    reward_per_round: Option<f64>,
    #[arg(long)]
    bond: Option<f64>,
    #[arg(long)]
    slash_when_caught: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut endpoint = ToolboxEndpoint::new_rpc_with_url_or_moniker_and_commitment(
        &args.rpc,
        CommitmentConfig::confirmed(),
    );
    let coordinator_account = Pubkey::from_str(&args.coordinator_account)
        .map_err(|e| anyhow!("invalid coordinator account: {e}"))?;
    let state = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await?
        .ok_or_else(|| anyhow!("coordinator account {coordinator_account} not found"))?;
    let clients: Vec<Client> = state.clients_state.clients.iter().copied().collect();
    let mut telemetry =
        compute_telemetry(&state.coordinator, &clients, &args.run_id, args.leaderboard);
    if let (Some(reward_per_round), Some(bond), Some(slash_when_caught)) =
        (args.reward_per_round, args.bond, args.slash_when_caught)
    {
        telemetry.security = Some(assess_security(
            telemetry.audit_probability,
            &RunEconomics {
                reward_per_round,
                bond,
                slash_when_caught,
            },
        ));
    }
    println!("{}", serde_json::to_string_pretty(&telemetry)?);
    Ok(())
}
