//! The Solana client sends a checkpoint in the background, and a client that
//! exits right after its run's last checkpoint used to drop the send with the
//! process (wienerlabs/leviathan-net#85). This runs the client's own backend,
//! `SolanaBackendRunner`, against a real `solana-test-validator` with the
//! coordinator and authorizer programs deployed, since the backend speaks RPC
//! and memnet has none.
//!
//! It is ignored by default, since it needs:
//! - `solana-test-validator` on `PATH`;
//! - the two programs built for SBF, `psyche_solana_coordinator.so` and
//!   `psyche_solana_authorizer.so`, in the directory `LEVIATHAN_SBF_DIR`
//!   names, or else where `anchor build` leaves each.
//!
//! Run it with
//! `cargo test -p psyche-solana-tooling --test mod -- --ignored validator_`.

use std::net::TcpListener;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anchor_client::Cluster;
use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::RunState;
use psyche_core::ConstantLR;
use psyche_core::FixedString;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::logic::InitCoordinatorParams;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_rpc::SolanaBackend;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_init;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_set_paused;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick_and_reveal;
use psyche_solana_tooling::process_coordinator_instructions::process_update;
use psyche_watcher::Backend;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::signed_payload::identity;

const RUN_ID: &str = "leviathan-checkpoint-flush";
/// Long enough that the run stays in its warmup, with the client on the
/// roster, for the whole test.
const WARMUP_TIME: u64 = 3_600;
const SETTLE: Duration = Duration::from_secs(120);

/// A `solana-test-validator` of this test's own, on ports nothing else
/// holds, with its ledger in a directory of its own. Stopped and removed
/// when dropped.
struct LocalValidator {
    child: Child,
    ledger: PathBuf,
    rpc_url: String,
    ws_url: String,
}

impl Drop for LocalValidator {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.ledger);
    }
}

/// The first of `count` consecutive local ports that are all free now, below
/// the ephemeral range the system hands out, from a start that differs from
/// one process to the next.
fn free_ports(count: u16) -> u16 {
    const LOW: u16 = 20_000;
    const HIGH: u16 = 45_000;
    let offset = (std::process::id() % u32::from(HIGH - LOW)) as u16;
    (LOW..HIGH - count)
        .step_by(count as usize)
        .map(|base| LOW + (base - LOW + offset) % (HIGH - LOW - count))
        .find(|base| {
            (*base..base + count)
                .all(|port| TcpListener::bind(("127.0.0.1", port)).is_ok())
        })
        .expect("a run of free local ports")
}

/// Where the program `name` was built for SBF: `LEVIATHAN_SBF_DIR`, or where
/// `anchor build` leaves it in the program's workspace.
fn sbf_program(name: &str, workspace: &str) -> PathBuf {
    let file = format!("{name}.so");
    let path = match std::env::var("LEVIATHAN_SBF_DIR") {
        Ok(dir) => Path::new(&dir).join(&file),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(workspace)
            .join("target/deploy")
            .join(&file),
    };
    assert!(
        path.is_file(),
        "{} is not built for SBF; build it with `cargo build-sbf` and name its \
         directory in LEVIATHAN_SBF_DIR",
        path.display()
    );
    path
}

fn start_validator() -> LocalValidator {
    let coordinator =
        sbf_program("psyche_solana_coordinator", "solana-coordinator");
    let authorizer =
        sbf_program("psyche_solana_authorizer", "solana-authorizer");
    let ports = free_ports(40);
    let (rpc, faucet, gossip) = (ports, ports + 2, ports + 3);
    let ledger = std::env::temp_dir().join(format!(
        "leviathan-checkpoint-flush-{}-{ports}",
        std::process::id()
    ));
    let child = Command::new("solana-test-validator")
        .arg("--reset")
        .arg("--quiet")
        .arg("--ledger")
        .arg(&ledger)
        .args(["--bind-address", "127.0.0.1"])
        .args(["--rpc-port", &rpc.to_string()])
        .args(["--faucet-port", &faucet.to_string()])
        .args(["--gossip-port", &gossip.to_string()])
        .args([
            "--dynamic-port-range",
            &format!("{}-{}", ports + 10, ports + 39),
        ])
        .arg("--bpf-program")
        .arg(psyche_solana_coordinator::ID.to_string())
        .arg(&coordinator)
        .arg("--bpf-program")
        .arg(psyche_solana_authorizer::ID.to_string())
        .arg(&authorizer)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("solana-test-validator runs; is it on PATH?");
    LocalValidator {
        child,
        ledger,
        rpc_url: format!("http://127.0.0.1:{rpc}"),
        ws_url: format!("ws://127.0.0.1:{}", rpc + 1),
    }
}

/// Waits until the validator answers and has funded `payer`.
async fn ready(endpoint: &mut ToolboxEndpoint, payer: &Pubkey) {
    let deadline = Instant::now() + SETTLE;
    loop {
        if endpoint.get_sysvar_clock().await.is_ok()
            && endpoint
                .request_airdrop(payer, 100_000_000_000)
                .await
                .is_ok()
        {
            return;
        }
        assert!(Instant::now() < deadline, "the validator never came up");
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn repo(repo_id: &str, revision: &str) -> HubRepo {
    HubRepo {
        repo_id: FixedString::from_str_truncated(repo_id),
        revision: Some(FixedString::from_str_truncated(revision)),
    }
}

/// The Hub repository the coordinator records as the run's checkpoint.
async fn recorded_checkpoint(
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
) -> HubRepo {
    let state = get_coordinator_account_state(endpoint, coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let Model::LLM(llm) = state.coordinator.model;
    match llm.checkpoint {
        Checkpoint::Hub(repo) => repo,
        other => panic!("the run's checkpoint is not on the Hub: {other:?}"),
    }
}

async fn run_state(
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
) -> RunState {
    get_coordinator_account_state(endpoint, coordinator_account)
        .await
        .unwrap()
        .unwrap()
        .coordinator
        .run_state
}

/// The client's backend, for `wallet`, on the validator.
fn backend(validator: &LocalValidator, wallet: &Keypair) -> SolanaBackend {
    SolanaBackend::new(
        Cluster::Custom(validator.rpc_url.clone(), validator.ws_url.clone()),
        vec![],
        Arc::new(wallet.insecure_clone()),
        CommitmentConfig::confirmed(),
    )
    .unwrap()
}

/// A run on the validator whose one client, `client`, is on the roster of
/// its warmup. Returns its instance, its account and its main authority.
async fn run_with_client_on_roster(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    client: &Keypair,
) -> (Pubkey, Pubkey, Keypair) {
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let ticker = Keypair::new();
    endpoint
        .request_airdrop(&client.pubkey(), 10_000_000_000)
        .await
        .unwrap();
    let coordinator_account = endpoint
        .process_system_new_exempt(
            payer,
            CoordinatorAccount::space_with_discriminator(),
            &psyche_solana_coordinator::ID,
        )
        .await
        .unwrap();
    let coordinator_instance = process_coordinator_init(
        endpoint,
        payer,
        &coordinator_account,
        InitCoordinatorParams {
            run_id: RUN_ID.to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: join_authority.pubkey(),
            client_version: "test".to_string(),
        },
    )
    .await
    .unwrap();
    process_update(
        endpoint,
        payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        None,
        Some(CoordinatorConfig {
            warmup_time: WARMUP_TIME,
            cooldown_time: 60,
            max_round_train_time: 60,
            round_witness_time: 10,
            min_clients: 1,
            init_min_clients: 1,
            global_batch_size_start: 1,
            global_batch_size_end: 1,
            global_batch_size_warmup_tokens: 0,
            verification_percent: 0,
            witness_nodes: 1,
            epoch_time: 2 * WARMUP_TIME,
            total_steps: 100,
            waiting_for_members_extra_time: 1,
        }),
        Some(Model::LLM(LLM {
            architecture: LLMArchitecture::HfLlama,
            checkpoint: Checkpoint::Hub(repo("leviathan/initial-model", "0")),
            max_seq_len: 4096,
            data_type: LLMTrainingDataType::Pretraining,
            data_location: LLMTrainingDataLocation::default(),
            lr_schedule: LearningRateSchedule::Constant(ConstantLR::default()),
            optimizer: OptimizerDefinition::Distro {
                clip_grad_norm: None,
                compression_decay: 1.0,
                compression_topk: 1,
                compression_chunk: 1,
                quantize_1bit: false,
                robust_aggregation: false,
                weight_decay: None,
            },
            cold_start_warmup_steps: 0,
        })),
        None,
    )
    .await
    .unwrap();
    let authorization = process_authorizer_authorization_create(
        endpoint,
        payer,
        &join_authority,
        &client.pubkey(),
        JOIN_RUN_AUTHORIZATION_SCOPE,
    )
    .await
    .unwrap();
    process_authorizer_authorization_grantor_update(
        endpoint,
        payer,
        &join_authority,
        &authorization,
        AuthorizationGrantorUpdateParams { active: true },
    )
    .await
    .unwrap();
    process_coordinator_set_paused(
        endpoint,
        payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        false,
    )
    .await
    .unwrap();
    process_coordinator_join_run(
        endpoint,
        payer,
        client,
        &authorization,
        &coordinator_instance,
        &coordinator_account,
        identity(client, &Keypair::new()),
    )
    .await
    .unwrap();
    // The run leaves `WaitingForMembers` once its extra time is up, on the
    // validator's own clock.
    let deadline = Instant::now() + SETTLE;
    while run_state(endpoint, &coordinator_account).await != RunState::Warmup {
        assert!(
            Instant::now() < deadline,
            "the run never reached its warmup"
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
        let _ = process_coordinator_tick_and_reveal(
            endpoint,
            payer,
            &ticker,
            &coordinator_instance,
            &coordinator_account,
        )
        .await;
    }
    (coordinator_instance, coordinator_account, main_authority)
}

#[tokio::test]
#[ignore = "needs solana-test-validator and the programs built for SBF"]
pub async fn validator_a_client_waits_for_its_checkpoint_to_be_recorded() {
    let validator = start_validator();
    let mut endpoint =
        ToolboxEndpoint::new_rpc_with_url_or_moniker_and_commitment(
            &validator.rpc_url,
            CommitmentConfig::confirmed(),
        );
    let payer = Keypair::new();
    ready(&mut endpoint, &payer.pubkey()).await;
    let client = Keypair::new();
    let (coordinator_instance, coordinator_account, main_authority) =
        run_with_client_on_roster(&mut endpoint, &payer, &client).await;
    let initial = repo("leviathan/initial-model", "0");
    assert_eq!(
        recorded_checkpoint(&mut endpoint, &coordinator_account).await,
        initial
    );

    // Without a flush, a client whose process ends right after handing the
    // checkpoint over loses it: the send never leaves the runtime that goes
    // away with the process.
    let lost = Checkpoint::Hub(repo("leviathan/lost-model", "1"));
    let exiting = backend(&validator, &client);
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let mut runner = exiting
                .start(RUN_ID.to_string(), coordinator_account)
                .await
                .unwrap();
            runner.send_checkpoint(lost).await.unwrap();
        });
    })
    .join()
    .unwrap();
    // Given time to land, it does not.
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(
        recorded_checkpoint(&mut endpoint, &coordinator_account).await,
        initial
    );

    // With a flush, the checkpoint is on chain once it returns.
    let uploaded = repo("leviathan/uploaded-model", "2");
    let mut runner = backend(&validator, &client)
        .start(RUN_ID.to_string(), coordinator_account)
        .await
        .unwrap();
    runner
        .send_checkpoint(Checkpoint::Hub(uploaded))
        .await
        .unwrap();
    runner.flush_checkpoints().await.unwrap();
    assert_eq!(
        recorded_checkpoint(&mut endpoint, &coordinator_account).await,
        uploaded
    );

    // A run that has halted, as one does once it finishes, still records the
    // checkpoint a client of its roster sends, and the flush still waits.
    process_coordinator_set_paused(
        &mut endpoint,
        &payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        true,
    )
    .await
    .unwrap();
    assert_eq!(
        run_state(&mut endpoint, &coordinator_account).await,
        RunState::Paused
    );
    let last = repo("leviathan/last-model", "3");
    runner.send_checkpoint(Checkpoint::Hub(last)).await.unwrap();
    runner.flush_checkpoints().await.unwrap();
    assert_eq!(
        recorded_checkpoint(&mut endpoint, &coordinator_account).await,
        last
    );

    // A checkpoint the coordinator refuses, from a wallet that is not one of
    // its clients, fails the flush, and is not recorded.
    let stranger = Keypair::new();
    endpoint
        .request_airdrop(&stranger.pubkey(), 10_000_000_000)
        .await
        .unwrap();
    let mut refused = backend(&validator, &stranger)
        .start(RUN_ID.to_string(), coordinator_account)
        .await
        .unwrap();
    refused
        .send_checkpoint(Checkpoint::Hub(repo("someone/else", "4")))
        .await
        .unwrap();
    let error = refused
        .flush_checkpoints()
        .await
        .expect_err("a refused checkpoint fails the flush");
    assert!(
        format!("{error:#}").contains("Checkpoint transaction failed"),
        "{error:#}"
    );
    assert_eq!(
        recorded_checkpoint(&mut endpoint, &coordinator_account).await,
        last
    );
    // Reported once: the next flush has nothing left to report.
    refused.flush_checkpoints().await.unwrap();
}
