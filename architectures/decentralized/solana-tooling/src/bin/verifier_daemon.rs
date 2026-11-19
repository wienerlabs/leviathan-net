use std::collections::HashMap;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use std::time::Instant;

use tch::Tensor;

use anyhow::anyhow;
use anyhow::Result;
use clap::Parser;
use leviathan_telemetry::server::serve;
use leviathan_telemetry::verifier::SlowestPass;
use leviathan_telemetry::verifier::VerifierMetrics;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::index_dir;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::parse_dtype;
use leviathan_verifier::payload_band;
use leviathan_verifier::DumpKey;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::replay_data_layout;
use leviathan_verifier::revealed::revealed_round;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::ReplayWork;
use leviathan_verifier::TrainerReplayEngine;
use prometheus_client::registry::Registry;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::Trainer;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::parse_committer;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
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

    /// Honest dumps to audit against. Omit it and pass --replay-data-dir to
    /// have this verifier recompute the reference itself.
    #[arg(long)]
    reference_dir: Option<PathBuf>,

    /// A local copy of the run's data shards. With it, this verifier replays
    /// contributions itself, exactly as the run trains them: the checkpoint,
    /// learning-rate schedule, optimizer, sequence length and data layout all
    /// come from the run on chain, and so does the step a replay can
    /// reproduce, the first of a cold-start epoch (wienerlabs/leviathan-net#47).
    #[arg(long)]
    replay_data_dir: Option<PathBuf>,
    /// The band on a payload's relative L2 distance from the honest result.
    /// A run that sends 1-bit signs is judged on its churn and flips instead,
    /// and is refused this band unless the daemon only watches (--dry-run).
    #[arg(long)]
    band: Option<f32>,
    /// For a run that sends 1-bit signs, the share of a payload's coordinates
    /// a replay may pick differently, and of those both pick, the share it
    /// may sign differently. Calibrate both with `calibrate-band` for the
    /// hardware this daemon replays on (`docs/BAND_CALIBRATION.md`,
    /// wienerlabs/leviathan-net#49).
    #[arg(long)]
    churn_band: Option<f32>,
    #[arg(long)]
    flip_band: Option<f32>,
    /// The dtype this daemon replays in. The network trains in bf16. A replay
    /// in bf16 may churn less against it, which the measurements so far hint
    /// at without settling, but it runs far slower on a CPU. Its band has to
    /// be calibrated for it (`docs/BAND_CALIBRATION.md`).
    #[arg(long, default_value = "fp32")]
    replay_dtype: String,
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
    /// Also check that each round replayed from its revealed state leaves
    /// the state its trainer's next round committed to, and convict that
    /// next round's payload if it does not (wienerlabs/leviathan-net#89).
    /// A replay holds a state to the bit only in the class of hardware and
    /// the dtype the run trains in: set it only for a run known to train in
    /// this daemon's (`docs/ROUND_AUDIT.md`).
    #[arg(long, default_value_t = false)]
    continuity: bool,

    /// Serve what each pass judged at http://<address>/metrics for Prometheus:
    /// contributions cleared, fraud found, convictions submitted, and
    /// submissions the chain refused.
    #[arg(long)]
    metrics_listen: Option<SocketAddr>,
}

/// The daemon's `/metrics` endpoint: the counters it records into, and the
/// server task, watched so the daemon stops rather than audit on unseen.
struct Exporter {
    metrics: VerifierMetrics,
    server: tokio::task::JoinHandle<Result<()>>,
}

impl Exporter {
    async fn start(listen: SocketAddr) -> Result<Self> {
        let mut registry = Registry::default();
        let metrics = VerifierMetrics::register(&mut registry);
        let listener = tokio::net::TcpListener::bind(listen).await?;
        println!(
            "[verifier-daemon] serving metrics at http://{}/metrics",
            listener.local_addr()?
        );
        let server =
            tokio::spawn(serve(listener, std::sync::Arc::new(registry)));
        Ok(Self { metrics, server })
    }

    /// An error once the server has stopped: the daemon's monitoring is gone.
    async fn check(&mut self) -> Result<()> {
        if !self.server.is_finished() {
            return Ok(());
        }
        Err(match (&mut self.server).await {
            Ok(Ok(())) => anyhow!("the metrics server stopped"),
            Ok(Err(err)) => err.context("the metrics server failed"),
            Err(err) => anyhow!("the metrics server panicked: {err}"),
        })
    }
}

/// The run's published checkpoint, loaded with the run's own configuration,
/// and the run's data.
struct Loaded {
    checkpoint: HubRepo,
    /// Lent to each pass's engine and handed back after it.
    trainer: Option<Trainer>,
    /// The checkpoint's weights, which a replay of the run's first step
    /// trains from after one from a revealed state left the trainer holding
    /// that state (wienerlabs/leviathan-net#89). Copies, not the trainer's
    /// own tensors.
    weights: HashMap<String, Tensor>,
    provider: LocalDataProvider,
    quantize_1bit: bool,
}

/// The replay side of the daemon. Each pass gets an engine over the
/// contributions on disk at that moment, so results that arrive while the
/// daemon runs are judged as they land.
struct Replayer {
    data_dir: PathBuf,
    /// The dtype it replays in (`--replay-dtype`).
    kind: tch::Kind,
    loaded: Option<Loaded>,
}

impl Replayer {
    fn new(data_dir: PathBuf, kind: tch::Kind) -> Self {
        Self {
            data_dir,
            kind,
            loaded: None,
        }
    }

    /// The run as it stands, loaded again if it names another published
    /// checkpoint, or if a failed replay consumed the trainer.
    fn load(&mut self, llm: &LLM) -> Result<&mut Loaded> {
        let checkpoint = published_checkpoint(llm)?;
        let current = self.loaded.as_ref().is_some_and(|loaded| {
            loaded.checkpoint == checkpoint && loaded.trainer.is_some()
        });
        if !current {
            let config = ReplayTrainerConfig::for_run(
                llm,
                checkpoint_repo_files(&checkpoint)?,
                tch::Device::Cpu,
                // fp32 unless the operator chose otherwise. The network trains
                // in BFloat16. A bf16 replay may churn less against it, but is
                // emulated on a CPU and far slower, and moving the dtype moves
                // what counts as a forgery, so the band has to be calibrated
                // for the dtype chosen (wienerlabs/leviathan-net#49).
                self.kind,
            );
            let mut trainer = build_replay_trainer(&config)?;
            let weights = trainer
                .extract()?
                .into_iter()
                .map(|(name, tensor)| (name, tensor.copy()))
                .collect();
            let (token_size, shuffle) = replay_data_layout(llm)?;
            let provider = LocalDataProvider::new_from_directory(
                &self.data_dir,
                token_size,
                config.sequence_length,
                shuffle,
            )?;
            println!(
                "[verifier-daemon] replay engine loaded {} with the run's {:?} data order, quantize_1bit={}",
                checkpoint.repo_id,
                shuffle,
                config.quantize_1bit()
            );
            self.loaded = Some(Loaded {
                checkpoint,
                trainer: Some(trainer),
                weights,
                provider,
                quantize_1bit: config.quantize_1bit(),
            });
        }
        self.loaded
            .as_mut()
            .ok_or_else(|| anyhow!("the replay model did not load"))
    }

    async fn engine(
        &mut self,
        args: &Args,
        endpoint: &mut ToolboxEndpoint,
        coordinator_account: &Pubkey,
    ) -> Result<Option<TrainerReplayEngine>> {
        let state = get_coordinator_account_state(endpoint, coordinator_account)
            .await?
            .ok_or_else(|| anyhow!("coordinator account {} not found", coordinator_account))?;
        let coordinator = &state.coordinator;
        // What a replay can reproduce: the run's first step, trained by every
        // node from the published checkpoint with fresh state
        // (wienerlabs/leviathan-net#48), and any round of the epoch under way
        // whose payload has the state its trainer revealed beside it
        // (wienerlabs/leviathan-net#89). Without either there is nothing to
        // replay, and nothing worth loading.
        let replayable =
            |epoch_start: u64, step: u32, path: &std::path::Path| {
                coordinator.is_replayable(epoch_start, step)
                    || (epoch_start == coordinator.epoch_state.start_timestamp
                        && matches!(revealed_round(path), Ok(Some(_))))
            };
        let work: Vec<(DumpKey, PathBuf)> = index_dir(&args.submitted_dir)?
            .into_iter()
            .filter_map(|(key, path)| Some((parse_assignment_key(&key)?, path)))
            .filter(|(key, path)| replayable(key.epoch_start, key.step, path))
            .collect();
        if work.is_empty() {
            return Ok(None);
        }
        let Model::LLM(llm) = coordinator.model;
        let loaded = self.load(&llm)?;

        let mut assignments = Vec::new();
        for (
            DumpKey {
                epoch_start,
                step,
                batch_id,
            },
            path,
        ) in work
        {
            // Only a payload someone in the epoch signed is worth the data.
            // The dump names carry the full base58 signer (finding 20), so the
            // roster is matched on the same. Where it sits there is not kept:
            // the replay is of the work, and the pass that judges the payload
            // finds the work in the payload's own name.
            let Some(committer) = parse_committer(&path) else {
                continue;
            };
            if !coordinator.epoch_state.clients.iter().any(|client| {
                bs58::encode(client.id.signer()).into_string() == committer
            }) {
                continue;
            }
            let data: Vec<BatchDataCPU> = loaded
                .provider
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
                epoch_start,
                step,
                warmup_lr_between: coordinator
                    .get_cold_start_warmup_bounds(),
                batch_id,
                data,
            });
        }

        let trainer = loaded.trainer.take().ok_or_else(|| {
            anyhow!("the replay model was lost in an earlier pass")
        })?;
        let weights = loaded
            .weights
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.copy()))
            .collect();
        Ok(Some(
            TrainerReplayEngine::new(
                trainer,
                assignments,
                tch::Device::Cpu,
                loaded.quantize_1bit,
            )
            .with_checkpoint(weights),
        ))
    }

    /// Takes the trainer back after a pass. A replay that failed consumed it,
    /// and the next pass then loads the checkpoint again.
    fn reclaim(&mut self, engine: TrainerReplayEngine) {
        if let Some(loaded) = self.loaded.as_mut() {
            loaded.trainer = engine.into_trainer();
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let band = payload_band(args.band, args.churn_band, args.flip_band)?;
    let replay_kind = parse_dtype(&args.replay_dtype)?;
    // Without either, the daemon has nothing to judge a payload against. A
    // pass with neither judges only equivocations, which is what a replaying
    // daemon does once nothing can be replayed, so the mistake is caught here
    // rather than read as that.
    if args.reference_dir.is_none() && args.replay_data_dir.is_none() {
        return Err(anyhow!(
            "pass --reference-dir or --replay-data-dir: the daemon audits payloads against one or the other"
        ));
    }
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
        band,
        audit_assigned: args.audit_assigned,
        dry_run: args.dry_run,
        verdict_mode: args.verdict,
        continuity: args.continuity,
    };

    println!(
        "[verifier-daemon] run_id={} coordinator={} run={} authority={} {} mode={} dry_run={}",
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
        "[verifier-daemon] action={} continuity={}",
        if config.verdict_mode {
            "submit-verdict (bonded committee vote)"
        } else {
            "slash (single authority)"
        },
        config.continuity
    );

    let mut exporter = match args.metrics_listen {
        Some(listen) => Some(Exporter::start(listen).await?),
        None => None,
    };
    let mut replayer = args
        .replay_data_dir
        .clone()
        .map(|dir| Replayer::new(dir, replay_kind));

    let mut judged: HashSet<String> = HashSet::new();
    // At least a second between passes: with none the loop would spin, and
    // the slowest-pass window would fill with a pass every few milliseconds.
    let poll_interval = Duration::from_secs(args.poll_secs.max(1));
    // The slowest pass of the last hour, which is what bounds how late a
    // vote on a fraud can come (wienerlabs/leviathan-net#40).
    let mut slowest_pass = SlowestPass::new(Duration::from_secs(3_600));
    loop {
        if let Some(exporter) = exporter.as_mut() {
            exporter.check().await?;
        }
        // Timed from here, so the pass covers preparing the replay as well as
        // judging and submitting: all of it delays a vote on a fraud
        // (wienerlabs/leviathan-net#40).
        let pass_started = Instant::now();
        let engine = match replayer.as_mut() {
            Some(replayer) => match replayer.engine(&args, &mut endpoint, &coordinator_account).await {
                Ok(Some(engine)) => Some(engine),
                // Without a replay the pass still judges equivocation, which
                // needs none and is the one fraud provable past the first
                // epoch. Skipping the pass skipped it too, for the rest of
                // the run.
                Ok(None) => {
                    println!(
                        "[verifier-daemon] the run is past its first epoch, so no contribution can be replayed (wienerlabs/leviathan-net#48); judging equivocations only"
                    );
                    None
                }
                Err(err) => {
                    eprintln!(
                        "[verifier-daemon] cannot prepare this pass's replay, judging equivocations only: {err:#}"
                    );
                    if let Some(exporter) = &exporter {
                        exporter.metrics.pass_failed(&args.run_id);
                    }
                    None
                }
            },
            None => None,
        };
        let result = audit_pass(
            &mut endpoint,
            &authority,
            &coordinator_account,
            &run,
            &config,
            engine.as_ref().map(|e| e as &(dyn ReplayWork + Sync)),
            &mut judged,
        )
        .await;
        if let (Some(replayer), Some(engine)) = (replayer.as_mut(), engine) {
            replayer.reclaim(engine);
        }
        match result {
            Ok(report) => {
                if let Some(exporter) = &exporter {
                    exporter.metrics.record(&args.run_id, &report);
                }
                if args.once {
                    println!(
                        "[verifier-daemon] single pass complete, {} new conviction(s), {} submission(s) refused",
                        report.convictions(),
                        report.submission_failures
                    );
                    break;
                }
            }
            Err(err) => {
                eprintln!("[verifier-daemon] pass error: {err:#}");
                if let Some(exporter) = &exporter {
                    exporter.metrics.pass_failed(&args.run_id);
                }
                // A single pass is one pass, whether or not it worked.
                if args.once {
                    return Err(err);
                }
            }
        }
        note_pass(
            &mut slowest_pass,
            &exporter,
            &args.run_id,
            pass_started,
            poll_interval,
        );
        tokio::time::sleep(poll_interval).await;
    }

    Ok(())
}

/// Times the pass that began at `started`, whether it judged anything or
/// failed: a failed pass delays the next vote just as long. Logs it, and
/// exports the slowest pass of the last hour with the polling interval
/// (wienerlabs/leviathan-net#40).
fn note_pass(
    slowest_pass: &mut SlowestPass,
    exporter: &Option<Exporter>,
    run_id: &str,
    started: Instant,
    poll_interval: Duration,
) {
    let took = started.elapsed();
    let slowest = slowest_pass.record(Instant::now(), took);
    println!(
        "[verifier-daemon] pass took {:.3}s (slowest in the last hour {:.3}s), next in {}s",
        took.as_secs_f64(),
        slowest.as_secs_f64(),
        poll_interval.as_secs()
    );
    if let Some(exporter) = exporter {
        exporter.metrics.record_timing(run_id, slowest, poll_interval);
    }
}
