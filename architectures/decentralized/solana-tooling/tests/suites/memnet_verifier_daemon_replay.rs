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
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use psyche_solana_tooling::get_accounts::get_run;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::signed_payload::identity;
use super::signed_payload::keep_signed_payload;
use super::signed_payload::payload_header;
use super::signed_payload::slash_on_signed_payload;
use super::chain_state::coordinator_state;

const RUN_ID: &str = "Leviathan daemon replay slash";
const RUN_ID_MOVED: &str = "Leviathan replay moved seat";
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
    first_step_payload_on(llm, coordinator, step, batch_id, batch_data(llm))
}

/// The samples of `batch_id` in a dataset whose batches differ from one
/// another, so that one batch's honest result is not another's.
fn distinct_batch_data(llm: &LLM, batch_id: BatchId) -> Vec<BatchDataCPU> {
    let offset = batch_id.0.start as i32;
    let input_ids: Vec<i32> = (0..llm.max_seq_len as i32)
        .map(|i| (i * 7 + offset * 11 + 3) % 29)
        .collect();
    vec![BatchDataCPU {
        input_ids,
        labels: None,
        position_ids: None,
        sequence_lengths: None,
    }]
}

/// As `first_step_payload`, trained on `data`.
fn first_step_payload_on(
    llm: &LLM,
    coordinator: &Coordinator,
    step: u32,
    batch_id: BatchId,
    data: Vec<BatchDataCPU>,
) -> Vec<DistroResult> {
    let (trainer, config) = trainer_for(llm);
    let warmup = coordinator.get_cold_start_warmup_bounds();
    let output = trainer
        .train(
            step,
            Batch {
                id: batch_id,
                data: BatchData::CPU(data),
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

/// The pieces of a bonded run the replay suites drive.
struct ReplayRun {
    endpoint: ToolboxEndpoint,
    payer: Keypair,
    main_authority: Keypair,
    ticker: Keypair,
    run: Pubkey,
    coordinator_instance: Pubkey,
    coordinator_account: Pubkey,
    clients: Vec<Keypair>,
    p2ps: Vec<Keypair>,
}

const WARMUP_TIME: u64 = 10;
const ROUND_WITNESS_TIME: u64 = 10;
const COOLDOWN_TIME: u64 = 88;

/// A run of the genesis rehearsal model with four clients, `bonded` of them
/// bonded, ticked into its first round: the round that trains the run's first
/// step, the one step a replay reproduces.
async fn start_replay_run(
    run_index: u64,
    run_id: &str,
    bonded: &[usize],
) -> ReplayRun {
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
    let warmup_time = WARMUP_TIME;
    let round_witness_time = ROUND_WITNESS_TIME;
    let cooldown_time = COOLDOWN_TIME;

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
            index: run_index,
            run_id: run_id.to_string(),
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

    for node in bonded {
        process_treasurer_participant_bond_deposit(
            &mut endpoint,
            &payer,
            &clients[*node],
            &clients_collateral[*node],
            &collateral_mint,
            &run,
            BOND,
        )
        .await
        .unwrap();
    }

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

    ReplayRun {
        endpoint,
        payer,
        main_authority,
        ticker,
        run,
        coordinator_instance,
        coordinator_account,
        clients,
        p2ps,
    }
}

/// Every client the round draws as a witness reports it, and the run is ticked
/// on past its witness phase.
async fn witness_round(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    ticker: &Keypair,
    clients: &[Keypair],
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
) {
    let state = coordinator_state(endpoint, coordinator_account).await;
    for client in clients {
        let position = state
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|c| *c.id.signer() == client.pubkey().to_bytes());
        let Some(position) = position else {
            continue;
        };
        let witness_proof =
            CommitteeSelection::from_coordinator(&state.coordinator, 0)
                .unwrap()
                .get_witness(position as u64);
        if witness_proof.position >= SOLANA_MAX_NUM_WITNESSES as u64 {
            continue;
        }
        process_coordinator_witness(
            endpoint,
            payer,
            client,
            coordinator_instance,
            coordinator_account,
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
        .forward_clock_unix_timestamp(ROUND_WITNESS_TIME)
        .await
        .unwrap();
    process_coordinator_tick_and_reveal(
        endpoint,
        payer,
        ticker,
        coordinator_instance,
        coordinator_account,
    )
    .await
    .unwrap();
}

#[tokio::test]
pub async fn run() {
    let cheater = 0usize;
    let cooldown_time = COOLDOWN_TIME;
    let ReplayRun {
        mut endpoint,
        payer,
        main_authority,
        ticker,
        run,
        coordinator_instance,
        coordinator_account,
        clients,
        p2ps,
    } = start_replay_run(74, RUN_ID, &[cheater]).await;

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
    let (step, batch_id) = (header.step, header.batch_id);
    assert_eq!(
        step, first_round.coordinator.epoch_state.start_step,
        "the first round trains the epoch's first step"
    );

    for _ in 0..4 {
        witness_round(
            &mut endpoint,
            &payer,
            &ticker,
            &clients,
            &coordinator_instance,
            &coordinator_account,
        )
        .await;
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
                honest_header.step,
                honest_header.batch_id,
            ),
        ),
    ] {
        assert_eq!(
            node_header.step, step,
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
    // Timed as the daemon times a pass: building the replay engine from the
    // run's model, replaying, judging and submitting. It is the figure the
    // final round's audit window is sized against
    // (wienerlabs/leviathan-net#40).
    let pass_started = std::time::Instant::now();
    let (trainer, replay_config) = trainer_for(&llm);
    let engine = TrainerReplayEngine::new(
        trainer,
        [batch_id, honest_header.batch_id]
            .into_iter()
            .map(|node_batch| ReplayAssignment {
                epoch_start: header.epoch_start,
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

/// A pass builds its replay engine from one read of the run and judges each
/// payload against a later one, and the engine's first build loads the model,
/// which takes minutes. A client that leaves the epoch in between moves
/// everyone after it down a seat. The replay a payload is judged against is
/// found by the work the payload is signed for, so an honest node whose seat
/// moved is still judged against its own batch. Found by seat, it was judged
/// against the batch of whoever held that seat when the engine was built, and
/// convicted on its own genuine signature.
#[tokio::test]
pub async fn an_honest_node_whose_seat_moved_is_judged_against_its_own_batch() {
    let mut r = start_replay_run(75, RUN_ID_MOVED, &[]).await;
    let first_round =
        coordinator_state(&mut r.endpoint, &r.coordinator_account).await;
    let tie_breakers = get_run(&mut r.endpoint, &r.run)
        .await
        .unwrap()
        .unwrap()
        .tie_breaker_committee_size;
    let run_id = String::from(&first_round.coordinator.run_id);
    let node_at = |seat: usize| {
        let signer = *first_round.coordinator.epoch_state.clients[seat]
            .id
            .signer();
        r.clients
            .iter()
            .position(|key| key.pubkey().to_bytes() == signer)
            .expect("every seat is one of the suite's clients")
    };
    // The leaver sits ahead of the honest node, and the honest node ahead of
    // the cheater.
    let (leaver, honest, cheater) = (node_at(0), node_at(1), node_at(2));

    // Each trained its own batch of the run's first step, on that batch's
    // samples, and the cheater sent the opposite of what it computed.
    let llm = genesis_model();
    let submitted_dir = scratch("moved-submitted");
    let staging = scratch("moved-staging");
    let mut assignments = vec![];
    for node in [leaver, honest, cheater] {
        let header = payload_header(
            &first_round.coordinator,
            tie_breakers,
            &identity(&r.clients[node], &r.p2ps[node]),
        );
        let data = distinct_batch_data(&llm, header.batch_id);
        let trained = first_step_payload_on(
            &llm,
            &first_round.coordinator,
            header.step,
            header.batch_id,
            data.clone(),
        );
        let payload = match node == cheater {
            true => flipped(&trained),
            false => trained,
        };
        let staged = staging.join(format!("{node}.vec-postcard"));
        write_results_dump(&staged, &payload).unwrap();
        keep_signed_payload(
            &submitted_dir,
            &staged,
            &r.clients[node],
            &r.p2ps[node],
            &run_id,
            header,
        );
        assignments.push(ReplayAssignment {
            epoch_start: header.epoch_start,
            step: header.step,
            warmup_lr_between: first_round
                .coordinator
                .get_cold_start_warmup_bounds(),
            batch_id: header.batch_id,
            data,
        });
    }

    // The engine, built as a pass builds it, from the run as it stands.
    let (trainer, replay_config) = trainer_for(&llm);
    let engine = TrainerReplayEngine::new(
        trainer,
        assignments,
        tch::Device::Cpu,
        replay_config.quantize_1bit(),
    );

    // Then the leaver is convicted out of the epoch, and the next round
    // compacts the roster under the engine.
    slash_on_signed_payload(
        &mut r.endpoint,
        &r.payer,
        &r.main_authority,
        &r.run,
        &r.coordinator_account,
        RUN_ID_MOVED,
        0,
        &r.clients[leaver],
        &r.p2ps[leaver],
    )
    .await
    .unwrap();
    witness_round(
        &mut r.endpoint,
        &r.payer,
        &r.ticker,
        &r.clients,
        &r.coordinator_instance,
        &r.coordinator_account,
    )
    .await;
    let roster =
        coordinator_state(&mut r.endpoint, &r.coordinator_account).await;
    let seat_of = |node: usize| {
        roster
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| {
                *client.id.signer() == r.clients[node].pubkey().to_bytes()
            })
    };
    assert_eq!(seat_of(leaver), None, "the leaver left the roster");
    assert_eq!(
        (seat_of(honest), seat_of(cheater)),
        (Some(0), Some(1)),
        "everyone after the leaver moved down a seat"
    );
    assert_eq!(
        roster.coordinator.replayable_step(),
        Some(first_round.coordinator.progress.step),
        "the run is still in the epoch whose first step is replayed"
    );

    // The pass that judges the payloads reads the moved roster.
    let config = AuditConfig {
        run_id: RUN_ID_MOVED.to_string(),
        submitted_dir,
        reference_dir: None,
        band: 0.05,
        audit_assigned: false,
        dry_run: false,
        verdict_mode: false,
    };
    let mut convicted = HashSet::new();
    let report = audit_pass(
        &mut r.endpoint,
        &r.main_authority,
        &r.coordinator_account,
        &r.run,
        &config,
        Some(&engine),
        &mut convicted,
    )
    .await
    .unwrap();

    assert!(
        !convicted.contains(&r.clients[honest].pubkey().to_string()),
        "the honest node was convicted against another node's batch"
    );
    assert!(convicted.contains(&r.clients[cheater].pubkey().to_string()));
    assert_eq!(
        report,
        PassReport {
            cleared: 1,
            fraud_found: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
}
