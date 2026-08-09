use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::index_dir;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::TrainerReplayEngine;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_data_provider::download_model_repo_sync;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::BatchDataCPU;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::parse_committer;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
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
    #[arg(long, env = "SOLANA_RPC_URL")]
    rpc_url: Option<String>,
    #[arg(long)]
    run: Option<String>,
    #[arg(long)]
    authority: PathBuf,
    #[arg(long)]
    submitted_dir: PathBuf,

    /// Honest dumps to audit against. Omit it and pass --replay-model to have
    /// this verifier recompute the reference itself.
    #[arg(long)]
    reference_dir: Option<PathBuf>,

    #[arg(long)]
    replay_model: Option<String>,
    #[arg(long)]
    replay_data_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    replay_sequence_length: usize,
    #[arg(long, default_value_t = 4.0e-4)]
    replay_lr: f64,
    #[arg(long, default_value_t = 0.999)]
    replay_compression_decay: f32,
    #[arg(long, default_value_t = 8)]
    replay_compression_topk: u16,
    #[arg(long, default_value_t = 64)]
    replay_compression_chunk: u16,
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
    #[arg(long, default_value_t = false)]
    verdict: bool,
}

async fn build_replay_engine(
    args: &Args,
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
) -> Result<TrainerReplayEngine> {
    let model = args
        .replay_model
        .as_ref()
        .ok_or_else(|| anyhow!("a replay model is required to recompute references"))?;
    let data_dir = args
        .replay_data_dir
        .as_ref()
        .ok_or_else(|| anyhow!("--replay-data-dir is required alongside --replay-model"))?;

    let state = get_coordinator_account_state(endpoint, coordinator_account)
        .await?
        .ok_or_else(|| anyhow!("coordinator account {} not found", coordinator_account))?;

    let repo_files = download_model_repo_sync(&model.to_string(), None, None, None, true)?;
    let trainer = build_replay_trainer(&ReplayTrainerConfig {
        repo_files,
        sequence_length: args.replay_sequence_length,
        micro_batch_size: 1,
        learning_rate: args.replay_lr,
        compression_decay: args.replay_compression_decay,
        compression_topk: args.replay_compression_topk,
        compression_chunk: args.replay_compression_chunk,
        clip_grad_norm: Some(1.0),
        quantize_1bit: false,
        device: tch::Device::Cpu,
        // What the daemon has always replayed in. The network trains in
        // BFloat16 and that gap is the largest source of honest drift, but
        // narrowing it is a decision for the calibration harness to inform
        // rather than a side effect of making the dtype configurable: moving
        // the verifier's dtype moves what counts as a forgery.
        kind: tch::Kind::Float,
    })?;

    let mut provider = LocalDataProvider::new_from_directory(
        data_dir,
        TokenSize::TwoBytes,
        args.replay_sequence_length,
        Shuffle::DontShuffle,
    )?;

    let mut assignments = Vec::new();
    for (key, path) in index_dir(&args.submitted_dir)? {
        let Some(committer) = parse_committer(&path) else {
            continue;
        };
        let Some(index) = state
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| format!("{}", client.id) == committer)
        else {
            println!("[verifier-daemon] {committer} is not in the epoch roster, nothing to replay");
            continue;
        };
        let Some((step, batch_id)) = parse_assignment_key(&key) else {
            continue;
        };
        let data: Vec<BatchDataCPU> = provider
            .get_samples(batch_id)
            .await?
            .into_iter()
            .map(|x| BatchDataCPU {
                input_ids: x.input_ids,
                labels: x.labels,
                position_ids: x.position_ids,
                sequence_lengths: x.sequence_lengths,
            })
            .collect();
        assignments.push(ReplayAssignment {
            target_index: index as u64,
            step,
            batch_id,
            data,
        });
    }

    println!(
        "[verifier-daemon] replay engine ready over {} assignment(s)",
        assignments.len()
    );
    Ok(TrainerReplayEngine::new(
        trainer,
        assignments,
        tch::Device::Cpu,
    ))
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
    let mut endpoint = match &args.rpc_url {
        Some(url) => ToolboxEndpoint::new_rpc_with_url_or_moniker_and_commitment(
            url,
            solana_sdk::commitment_config::CommitmentConfig::confirmed(),
        ),
        None => ToolboxEndpoint::new_devnet().await,
    };

    let config = AuditConfig {
        run_id: args.run_id.clone(),
        submitted_dir: args.submitted_dir.clone(),
        reference_dir: args.reference_dir.clone(),
        band: args.band,
        audit_assigned: args.audit_assigned,
        dry_run: args.dry_run,
        verdict_mode: args.verdict,
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
    println!(
        "[verifier-daemon] action={}",
        if config.verdict_mode {
            "submit-verdict (bonded committee vote)"
        } else {
            "slash (single authority)"
        }
    );

    let replay = match args.replay_model {
        Some(_) => Some(build_replay_engine(&args, &mut endpoint, &coordinator_account).await?),
        None => None,
    };

    let mut convicted: HashSet<String> = HashSet::new();
    loop {
        match audit_pass(
            &mut endpoint,
            &authority,
            &coordinator_account,
            &run,
            &config,
            replay.as_ref().map(|e| e as &(dyn psyche_verifier::ReplayEngine + Sync)),
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
