use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::SOLANA_MAX_NUM_WITNESSES;
use psyche_coordinator::WAITING_FOR_MEMBERS_EXTRA_SECONDS;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::decompress_results;
use leviathan_verifier::write_dense_dump;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::TrainerReplayEngine;
use psyche_core::BatchId;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_data_provider::download_model_repo_sync;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::Trainer;
use tokio_util::sync::CancellationToken;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::instruction::Witness;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick;
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

const RUN_ID: &str = "Leviathan daemon replay slash";
const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("levdaemonreplay-{}-{}", std::process::id(), tag));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

const NANO_MODEL: &str = "pefontana/Nano-Llama";
const NANO_SEQUENCE_LENGTH: usize = 64;

fn nano_batch() -> Vec<BatchDataCPU> {
    let input_ids: Vec<i32> = (0..NANO_SEQUENCE_LENGTH as i32).map(|i| i % 29).collect();
    vec![BatchDataCPU {
        input_ids,
        labels: None,
        position_ids: None,
        sequence_lengths: None,
    }]
}

fn nano_trainer() -> Trainer {
    let repo_files = download_model_repo_sync(&NANO_MODEL.to_string(), None, None, None, true)
        .expect("the nano model must be in the local hub cache");
    build_replay_trainer(&ReplayTrainerConfig {
        repo_files,
        sequence_length: NANO_SEQUENCE_LENGTH,
        micro_batch_size: 1,
        learning_rate: 4.0e-4,
        compression_decay: 0.999,
        compression_topk: 8,
        compression_chunk: 64,
        clip_grad_norm: Some(1.0),
        quantize_1bit: false,
        device: tch::Device::Cpu,
        // Matches the daemon, so this suite keeps checking what the daemon
        // actually does rather than a dtype only the test uses.
        kind: tch::Kind::Float,
    })
    .expect("the replay trainer must build")
}

fn nano_train_once(
    trainer: Trainer,
    step: u32,
    batch_id: BatchId,
) -> (Trainer, Vec<f32>) {
    let output = trainer
        .train(
            step,
            Batch {
                id: batch_id,
                data: BatchData::CPU(nano_batch()),
            },
            None,
            true,
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        )
        .expect("the nano model must train a batch");
    let results = output
        .distro_results
        .clone()
        .expect("a DisTrO run must produce results");
    let dense = decompress_results(&results, tch::Device::Cpu).expect("results must decompress");
    (output.trainer, dense)
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
            model: Some(Model::LLM(LLM {
                architecture: LLMArchitecture::HfLlama,
                checkpoint: Checkpoint::Dummy(HubRepo::dummy()),
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
                    weight_decay: None,
                },
                cold_start_warmup_steps: 0,
            })),
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
    process_coordinator_tick(
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
    process_coordinator_tick(
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
        get_coordinator_account_state(&mut endpoint, &coordinator_account)
            .await
            .unwrap()
            .unwrap();
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
        let state = get_coordinator_account_state(&mut endpoint, &coordinator_account)
            .await
            .unwrap()
            .unwrap();
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
        process_coordinator_tick(
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
    let roster = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let cheater_index = roster
        .coordinator
        .epoch_state
        .clients
        .iter()
        .position(|client| *client.id.signer() == clients[cheater].pubkey().to_bytes())
        .expect("the cheater must be in the epoch roster") as u64;

    let (_, honest) = nano_train_once(nano_trainer(), step, batch_id);
    let forged: Vec<f32> = honest.iter().map(|v| -5.0 * v).collect();
    let forged_dump = scratch("forged").join("forged.vec-postcard");
    write_dense_dump(&forged_dump, &forged, 6).unwrap();
    let submitted_dir = scratch("submitted");
    keep_signed_payload(
        &submitted_dir,
        &forged_dump,
        &clients[cheater],
        &p2ps[cheater],
        &run_id,
        header,
    );

    let engine = TrainerReplayEngine::new(
        nano_trainer(),
        vec![ReplayAssignment {
            target_index: cheater_index,
            step,
            batch_id,
            data: nano_batch(),
        }],
        tch::Device::Cpu,
    );

    let config = AuditConfig {
        run_id: RUN_ID.to_string(),
        submitted_dir,
        reference_dir: None,
        band: 0.05,
        audit_assigned: false,
        dry_run: false,
        verdict_mode: false,
        replay_step: Some(step),
    };

    let mut convicted = HashSet::new();
    let convictions = audit_pass(
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

    assert_eq!(convictions, 1);
    assert!(convicted.contains(&cheater_committer));

    endpoint
        .forward_clock_unix_timestamp(cooldown_time)
        .await
        .unwrap();
    process_coordinator_tick(
        &mut endpoint,
        &payer,
        &ticker,
        &coordinator_instance,
        &coordinator_account,
    )
    .await
    .unwrap();

    let state = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let cheater_state = state
        .clients_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == clients[cheater].pubkey().to_bytes())
        .unwrap();
    assert_eq!(cheater_state.slashed, SLASHING_RATE);
    assert_eq!(cheater_state.earned, 0);
}
