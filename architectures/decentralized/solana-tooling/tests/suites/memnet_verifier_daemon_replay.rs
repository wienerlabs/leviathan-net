use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use leviathan_telemetry::verifier::PassReport;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::Coordinator;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::SOLANA_MAX_NUM_WITNESSES;
use psyche_coordinator::WAITING_FOR_MEMBERS_EXTRA_SECONDS;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::read_run_model;
use leviathan_verifier::write_results_dump;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::TrainerReplayEngine;
use psyche_core::BatchId;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::DistroResult;
use psyche_modeling::Trainer;
use tokio_util::sync::CancellationToken;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::instruction::Witness;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick_and_reveal;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_witness;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::signature::Keypair;
use psyche_solana_tooling::get_accounts::get_run;
use solana_sdk::signer::Signer;

use super::signed_payload::identity;
use super::signed_payload::keep_signed_payload;
use super::signed_payload::payload_header;
use super::chain_state::coordinator_state;

const RUN_ID: &str = "Leviathan daemon replay slash";
const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("levdaemonreplay-{}-{}", std::process::id(), tag));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The model of the genesis rehearsal, which this run trains: a 1-bit DisTrO
/// payload under a Cosine warmup, as every Leviathan configuration has.
fn genesis_model() -> LLM {
    read_run_model(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../config/leviathan/genesis-rehearsal.toml"
    )))
    .expect("the genesis rehearsal config is readable")
}

/// The samples of a batch, as the suite's dataset holds them.
fn batch_data(llm: &LLM) -> Vec<BatchDataCPU> {
    let input_ids: Vec<i32> =
        (0..llm.max_seq_len as i32).map(|i| i % 29).collect();
    vec![BatchDataCPU {
        input_ids,
        labels: None,
        position_ids: None,
        sequence_lengths: None,
    }]
}

/// A trainer built from the run's model, as the client and the daemon build
/// one.
fn trainer_for(llm: &LLM) -> (Trainer, ReplayTrainerConfig) {
    let config = ReplayTrainerConfig::for_run(
        llm,
        checkpoint_repo_files(&published_checkpoint(llm).unwrap()).unwrap(),
        tch::Device::Cpu,
        // What the daemon replays in.
        tch::Kind::Float,
    );
    (build_replay_trainer(&config).unwrap(), config)
}

/// What a node sends for `batch_id` at the run's first step: trained from a
/// fresh trainer under the run's model, within the warmup bounds the
/// coordinator gives, and quantized as the run says.
fn first_step_payload(
    llm: &LLM,
    coordinator: &Coordinator,
    step: u32,
    batch_id: BatchId,
) -> Vec<DistroResult> {
    let (trainer, config) = trainer_for(llm);
    let warmup = coordinator.get_cold_start_warmup_bounds();
    let output = trainer
        .train(
            step,
            Batch {
                id: batch_id,
                data: BatchData::CPU(batch_data(llm)),
            },
            warmup,
            warmup.is_some(),
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        )
        .expect("the run's model trains a batch");
    let results = output.distro_results.expect("the run trains with DisTrO");
    match config.quantize_1bit() {
        true => Trainer::quantize_results(&results),
        false => results,
    }
}

/// The same payload with every value's sign turned over.
fn flipped(payload: &[DistroResult]) -> Vec<DistroResult> {
    payload
        .iter()
        .map(|result| DistroResult {
            sparse_val: match result.sparse_val.kind() {
                tch::Kind::Bool => result.sparse_val.logical_not(),
                _ => result.sparse_val.neg(),
            },
            ..result.clone()
        })
        .collect()
}

#[tokio::test]
pub async fn run() {
    let mut endpoint = create_memnet_endpoint().await;

    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 5_000_000_000)
        .await
        .unwrap();

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    endpoint
        .request_airdrop(&main_authority.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    let join_authority = Keypair::new();
    let ticker = Keypair::new();
    let mut clients = vec![];
    for _ in 0..4 {
        clients.push(Keypair::new());
    }
    let p2ps: Vec<Keypair> = clients.iter().map(|_| Keypair::new()).collect();
    let cheater = 0usize;
    let warmup_time = 10;
    let round_witness_time = 10;
    let cooldown_time = 88;

    let collateral_mint = endpoint
        .process_spl_token_mint_new(&payer, &mint_authority.pubkey(), None, 0)
        .await
        .unwrap();

    let coordinator_account = endpoint
        .process_system_new_exempt(
            &payer,
            CoordinatorAccount::space_with_discriminator(),
            &psyche_solana_coordinator::ID,
        )
        .await
        .unwrap();

    let (run, coordinator_instance) = process_treasurer_run_create(
        &mut endpoint,
        &payer,
        &collateral_mint,
        &coordinator_account,
        RunCreateParams {
            index: 74,
            run_id: RUN_ID.to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: join_authority.pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    process_treasurer_run_bond_config_update(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        RunBondConfigUpdateParams {
            bond_minimum_amount: BOND,
            bond_withdraw_delay_seconds: 100,
        },
    )
    .await
    .unwrap();

    let mut clients_collateral = vec![];
    for client in &clients {
        let ata = endpoint
            .process_spl_associated_token_account_get_or_init(
                &payer,
                &client.pubkey(),
                &collateral_mint,
            )
            .await
            .unwrap();
        endpoint
            .process_spl_token_mint_to(&payer, &collateral_mint, &mint_authority, &ata, BOND)
            .await
            .unwrap();
        clients_collateral.push(ata);
    }

    for client in &clients {
        process_treasurer_participant_create(&mut endpoint, &payer, client, &run)
            .await
            .unwrap();
    }

    process_treasurer_participant_bond_deposit(
        &mut endpoint,
        &payer,
        &clients[cheater],
        &clients_collateral[cheater],
        &collateral_mint,
        &run,
        BOND,
    )
    .await
    .unwrap();

    process_treasurer_run_update(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        &coordinator_instance,
        &coordinator_account,
        RunUpdateParams {
            metadata: None,
            config: Some(CoordinatorConfig {
                warmup_time,
                cooldown_time,
                max_round_train_time: 15,
                round_witness_time,
                min_clients: 1,
                init_min_clients: 1,
                global_batch_size_start: 1,
                global_batch_size_end: clients.len() as u16,
                global_batch_size_warmup_tokens: 0,
                verification_percent: 0,
                witness_nodes: 0,
                epoch_time: 30,
                total_steps: 100,
                waiting_for_members_extra_time: WAITING_FOR_MEMBERS_EXTRA_SECONDS as u8,
            }),
            model: Some(Model::LLM(genesis_model())),
            progress: None,
            epoch_earning_rate_total_shared: Some(4_000),
            epoch_slashing_rate_per_client: Some(SLASHING_RATE),
            paused: Some(false),
            client_version: None,
        },
    )
    .await
    .unwrap();

    for (client, p2p) in clients.iter().zip(&p2ps) {
        let authorization = process_authorizer_authorization_create(
            &mut endpoint,
            &payer,
            &join_authority,
            &client.pubkey(),
            JOIN_RUN_AUTHORIZATION_SCOPE,
        )
        .await
        .unwrap();
        process_authorizer_authorization_grantor_update(
            &mut endpoint,
            &payer,
            &join_authority,
            &authorization,
            AuthorizationGrantorUpdateParams { active: true },
        )
        .await
        .unwrap();
        process_coordinator_join_run(
            &mut endpoint,
            &payer,
            client,
            &authorization,
            &coordinator_instance,
            &coordinator_account,
            identity(client, p2p),
        )
        .await
        .unwrap();
    }

    endpoint
        .forward_clock_unix_timestamp(WAITING_FOR_MEMBERS_EXTRA_SECONDS)
        .await
        .unwrap();
    process_coordinator_tick_and_reveal(
        &mut endpoint,
        &payer,
        &ticker,
        &coordinator_instance,
        &coordinator_account,
    )
    .await
    .unwrap();
    endpoint
        .forward_clock_unix_timestamp(warmup_time)
        .await
        .unwrap();
    process_coordinator_tick_and_reveal(
        &mut endpoint,
        &payer,
        &ticker,
        &coordinator_instance,
        &coordinator_account,
    )
    .await
    .unwrap();

    // The first round trains the run's first step from the published
    // checkpoint, the one step this replay engine reproduces. What the cheater
    // trained there, and signed, is what it is judged on.
    let first_round =
        coordinator_state(&mut endpoint, &coordinator_account).await;
    let tie_breakers = get_run(&mut endpoint, &run)
        .await
        .unwrap()
        .unwrap()
        .tie_breaker_committee_size;
    let header = payload_header(
        &first_round.coordinator,
        tie_breakers,
        &identity(&clients[cheater], &p2ps[cheater]),
    );
    // The run every commitment here is made in, as the coordinator names it.
    let run_id = String::from(&first_round.coordinator.run_id);
    let (step, _, batch_id) = header;
    assert_eq!(
        step, first_round.coordinator.epoch_state.start_step,
        "the first round trains the epoch's first step"
    );

    for _ in 0..4 {
        let state =
            coordinator_state(&mut endpoint, &coordinator_account).await;
        for client in &clients {
            let position = state
                .coordinator
                .epoch_state
                .clients
                .iter()
                .position(|c| *c.id.signer() == client.pubkey().to_bytes());
            let Some(position) = position else {
                continue;
            };
            let witness_proof = CommitteeSelection::from_coordinator(&state.coordinator, 0)
                .unwrap()
                .get_witness(position as u64);
            if witness_proof.position >= SOLANA_MAX_NUM_WITNESSES as u64 {
                continue;
            }
            process_coordinator_witness(
                &mut endpoint,
                &payer,
                client,
                &coordinator_instance,
                &coordinator_account,
                &Witness {
                    proof: witness_proof,
                    participant_bloom: Default::default(),
                    broadcast_bloom: Default::default(),
                    broadcast_merkle: Default::default(),
                    metadata: Default::default(),
                },
            )
            .await
            .unwrap();
        }
        endpoint
            .forward_clock_unix_timestamp(round_witness_time)
            .await
            .unwrap();
        process_coordinator_tick_and_reveal(
            &mut endpoint,
            &payer,
            &ticker,
            &coordinator_instance,
            &coordinator_account,
        )
        .await
        .unwrap();
    }

    let cheater_committer = clients[cheater].pubkey().to_string();
    let roster = coordinator_state(&mut endpoint, &coordinator_account).await;

    // The cheater and an honest node, each with the payload it sent for its
    // batch of the run's first step and the commitment it signed.
    let llm = genesis_model();
    let honest = (cheater + 1) % clients.len();
    let honest_header = payload_header(
        &first_round.coordinator,
        tie_breakers,
        &identity(&clients[honest], &p2ps[honest]),
    );
    let submitted_dir = scratch("submitted");
    let staging = scratch("staging");
    for (node, node_header, payload) in [
        (
            cheater,
            header,
            flipped(&first_step_payload(
                &llm,
                &first_round.coordinator,
                step,
                batch_id,
            )),
        ),
        (
            honest,
            honest_header,
            first_step_payload(
                &llm,
                &first_round.coordinator,
                honest_header.0,
                honest_header.2,
            ),
        ),
    ] {
        assert_eq!(
            node_header.0, step,
            "both nodes trained the run's first step"
        );
        let staged = staging.join(format!("{node}.vec-postcard"));
        write_results_dump(&staged, &payload).unwrap();
        keep_signed_payload(
            &submitted_dir,
            &staged,
            &clients[node],
            &p2ps[node],
            &run_id,
            node_header,
        );
    }

    // The daemon's engine, built from the run on chain: its model, the step
    // it can replay and the warmup bounds of that step.
    let replay_step = roster
        .coordinator
        .replayable_step()
        .expect("the run's first epoch has a replayable step");
    assert_eq!(replay_step, step);
    let roster_index = |node: usize| {
        roster
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| {
                *client.id.signer() == clients[node].pubkey().to_bytes()
            })
            .expect("the node is in the epoch roster") as u64
    };
    // Timed as the daemon times a pass: building the replay engine from the
    // run's model, replaying, judging and submitting. It is the figure the
    // final round's audit window is sized against
    // (wienerlabs/leviathan-net#40).
    let pass_started = std::time::Instant::now();
    let (trainer, replay_config) = trainer_for(&llm);
    let engine = TrainerReplayEngine::new(
        trainer,
        [(cheater, batch_id), (honest, honest_header.2)]
            .into_iter()
            .map(|(node, node_batch)| ReplayAssignment {
                target_index: roster_index(node),
                step: replay_step,
                warmup_lr_between: roster
                    .coordinator
                    .get_cold_start_warmup_bounds(),
                batch_id: node_batch,
                data: batch_data(&llm),
            })
            .collect(),
        tch::Device::Cpu,
        replay_config.quantize_1bit(),
    );
    assert!(
        replay_config.quantize_1bit(),
        "the genesis rehearsal sends 1-bit signs"
    );

    let config = AuditConfig {
        run_id: RUN_ID.to_string(),
        submitted_dir,
        reference_dir: None,
        band: 0.05,
        audit_assigned: false,
        dry_run: false,
        verdict_mode: false,
    };

    let mut convicted = HashSet::new();
    let report = audit_pass(
        &mut endpoint,
        &main_authority,
        &coordinator_account,
        &run,
        &config,
        Some(&engine),
        &mut convicted,
    )
    .await
    .unwrap();
    println!(
        "[audit-latency] engine and pass replaying 2 contributions of the genesis rehearsal model on {}: {:.3}s",
        std::env::consts::ARCH,
        pass_started.elapsed().as_secs_f64()
    );

    // The honest node's contribution replays within band, the cheater's does
    // not and is convicted.
    assert_eq!(
        report,
        PassReport {
            cleared: 1,
            fraud_found: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
    assert!(convicted.contains(&cheater_committer));
    assert!(
        !convicted.contains(&clients[honest].pubkey().to_string()),
        "the honest node was convicted"
    );

    endpoint
        .forward_clock_unix_timestamp(cooldown_time)
        .await
        .unwrap();
    process_coordinator_tick_and_reveal(
        &mut endpoint,
        &payer,
        &ticker,
        &coordinator_instance,
        &coordinator_account,
    )
    .await
    .unwrap();

    let state = coordinator_state(&mut endpoint, &coordinator_account).await;
    let cheater_state = state
        .clients_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == clients[cheater].pubkey().to_bytes())
        .unwrap();
    assert_eq!(cheater_state.slashed, SLASHING_RATE);
    assert_eq!(cheater_state.earned, 0);
}
