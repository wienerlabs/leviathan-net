use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_indexer::compute_telemetry;
use leviathan_indexer::live::export_forever;
use leviathan_indexer::live::read_run;
use leviathan_indexer::metrics::IndexerMetrics;
use leviathan_indexer::observe::treasurer_security;
use leviathan_indexer::DEFAULT_LEADERBOARD_SIZE;
use leviathan_telemetry::server::serve;
use prometheus_client::registry::Registry;
use psyche_solana_coordinator::Client;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_toolbox_endpoint::ToolboxEndpoint;

#[derive(Parser, Debug)]
#[command(
    name = "leviathan-indexer",
    about = "Read a run off chain: print its telemetry as JSON, or serve it to Prometheus"
)]
struct Args {
    /// The run to read. Its coordinator account, and its treasurer run if one
    /// manages it, are derived from the run id on chain. With
    /// --metrics-listen, repeat it to export several runs.
    #[arg(long = "run-id", required = true)]
    run_ids: Vec<String>,
    #[arg(long, env = "RPC", default_value = "devnet")]
    rpc: String,
    #[arg(long, default_value_t = DEFAULT_LEADERBOARD_SIZE)]
    leaderboard: usize,
    /// Serve the runs' metrics at http://<address>/metrics for Prometheus,
    /// polling the chain every --poll-secs, instead of printing JSON once.
    #[arg(long)]
    metrics_listen: Option<SocketAddr>,
    #[arg(long, default_value_t = 10)]
    poll_secs: u64,
}

fn connect(rpc: &str, commitment: CommitmentConfig) -> ToolboxEndpoint {
    ToolboxEndpoint::new_rpc_with_url_or_moniker_and_commitment(rpc, commitment)
}

/// One run's telemetry, with its economic security verdict when a treasurer
/// run manages it: the reward, the slash and the bond are the run's own.
async fn print_json(args: &Args) -> Result<()> {
    let [run_id] = args.run_ids.as_slice() else {
        return Err(anyhow!(
            "printing JSON reads one run; pass --metrics-listen to export several"
        ));
    };
    let mut endpoint = connect(&args.rpc, CommitmentConfig::confirmed());
    let snapshot = read_run(&mut endpoint, None, run_id).await?;
    let state = snapshot.state();
    let clients: Vec<Client> = state.clients_state.clients.iter().copied().collect();
    let mut telemetry = compute_telemetry(&state.coordinator, &clients, run_id, args.leaderboard);
    if let Some(treasurer) = &snapshot.treasurer {
        telemetry.security = Some(treasurer_security(state, &treasurer.run));
    }
    println!("{}", serde_json::to_string_pretty(&telemetry)?);
    Ok(())
}

async fn export(args: &Args, listen: SocketAddr) -> Result<()> {
    if args.poll_secs == 0 {
        return Err(anyhow!("--poll-secs must be at least 1"));
    }
    let mut registry = Registry::default();
    let metrics = IndexerMetrics::register(&mut registry);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    eprintln!(
        "[leviathan-indexer] serving {} run(s) at http://{}/metrics, polling every {}s",
        args.run_ids.len(),
        listener.local_addr()?,
        args.poll_secs
    );
    let server = tokio::spawn(serve(listener, Arc::new(registry)));
    let mut endpoint = connect(&args.rpc, CommitmentConfig::confirmed());
    let mut tip = connect(&args.rpc, CommitmentConfig::processed());
    tokio::select! {
        served = server => Err(match served {
            Ok(Ok(())) => anyhow!("the metrics server stopped"),
            Ok(Err(err)) => err.context("the metrics server failed"),
            Err(err) => anyhow!("the metrics server panicked: {err}"),
        }),
        never = export_forever(
            &mut endpoint,
            &mut tip,
            &args.run_ids,
            &metrics,
            Duration::from_secs(args.poll_secs),
        ) => never,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    match args.metrics_listen {
        Some(listen) => export(&args, listen).await,
        None => print_json(&args).await,
    }
}
