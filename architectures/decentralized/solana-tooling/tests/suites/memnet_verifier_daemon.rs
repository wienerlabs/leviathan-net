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
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::NodeIdentity;
use psyche_core::OptimizerDefinition;
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
use solana_sdk::signer::Signer;

const RUN_ID: &str = "Leviathan verifier daemon slash";
const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("levdaemon-{}-{}", std::process::id(), tag));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn place_fixture(dir: &PathBuf, fixture: &str, committer: &str) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/daemon")
        .join(fixture);
    let name = format!("result-{committer}-step1-batchB[0, 1].vec-postcard");
    fs::copy(source, dir.join(name)).unwrap();
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

    for client in &clients {
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
            NodeIdentity::new(client.pubkey().to_bytes(), Default::default()),
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

    let cheater_committer = format!(
        "{}",
        NodeIdentity::new(clients[cheater].pubkey().to_bytes(), Default::default())
    );
    let submitted_dir = scratch("submitted");
    let reference_dir = scratch("reference");
    place_fixture(&reference_dir, "honest.vec-postcard", &cheater_committer);
    place_fixture(&submitted_dir, "fraud.vec-postcard", &cheater_committer);

    let config = AuditConfig {
        run_id: RUN_ID.to_string(),
        submitted_dir,
        reference_dir,
        band: 0.05,
        audit_assigned: false,
        dry_run: false,
    };

    let mut convicted = HashSet::new();
    let convictions = audit_pass(
        &mut endpoint,
        &main_authority,
        &coordinator_account,
        &run,
        &config,
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
