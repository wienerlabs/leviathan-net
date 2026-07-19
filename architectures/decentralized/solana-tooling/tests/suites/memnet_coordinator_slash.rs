use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::SOLANA_MAX_NUM_WITNESSES;
use psyche_coordinator::WAITING_FOR_MEMBERS_EXTRA_SECONDS;
use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLM;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::NodeIdentity;
use psyche_core::OptimizerDefinition;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::instruction::Witness;
use psyche_solana_coordinator::logic::InitCoordinatorParams;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordiantor_set_future_epoch_rates;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_init;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_set_paused;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_slash_client;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_witness;
use psyche_solana_tooling::process_coordinator_instructions::process_update;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

#[tokio::test]
pub async fn run() {
    let mut endpoint = create_memnet_endpoint().await;

    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 5_000_000_000)
        .await
        .unwrap();

    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let stranger = Keypair::new();
    let mut clients = vec![];
    for _ in 0..8 {
        clients.push(Keypair::new());
    }
    let ticker = Keypair::new();
    let warmup_time = 10;
    let round_witness_time = 10;
    let cooldown_time = 88;
    let epoch_time = 30;
    let earning_rate = 448_000;
    let slashing_rate = 1_000;
    let cheater = 0usize;

    let coordinator_account = endpoint
        .process_system_new_exempt(
            &payer,
            CoordinatorAccount::space_with_discriminator(),
            &psyche_solana_coordinator::ID,
        )
        .await
        .unwrap();

    let coordinator_instance = process_coordinator_init(
        &mut endpoint,
        &payer,
        &coordinator_account,
        InitCoordinatorParams {
            run_id: "Leviathan slash suite run".to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: join_authority.pubkey(),
            client_version: "test".to_string(),
        },
    )
    .await
    .unwrap();

    process_update(
        &mut endpoint,
        &payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        None,
        Some(CoordinatorConfig {
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
            epoch_time,
            waiting_for_members_extra_time: WAITING_FOR_MEMBERS_EXTRA_SECONDS
                as u8,
            total_steps: 100,
        }),
        Some(Model::LLM(LLM {
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
        None,
    )
    .await
    .unwrap();

    process_coordiantor_set_future_epoch_rates(
        &mut endpoint,
        &payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        Some(earning_rate),
        Some(slashing_rate),
    )
    .await
    .unwrap();

    process_coordinator_set_paused(
        &mut endpoint,
        &payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        false,
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

    // A stranger cannot slash: the verdict is authority gated.
    process_coordinator_slash_client(
        &mut endpoint,
        &payer,
        &stranger,
        &coordinator_instance,
        &coordinator_account,
        cheater as u64,
        [0x11; 32],
        [0x22; 32],
    )
    .await
    .unwrap_err();

    for _ in 0..4 {
        let coordinator_account_state =
            get_coordinator_account_state(&mut endpoint, &coordinator_account)
                .await
                .unwrap()
                .unwrap();
        for client in &clients {
            let position = coordinator_account_state
                .coordinator
                .epoch_state
                .clients
                .iter()
                .position(|c| *c.id.signer() == client.pubkey().to_bytes());
            let Some(position) = position else {
                continue;
            };
            let witness_proof = CommitteeSelection::from_coordinator(
                &coordinator_account_state.coordinator,
                0,
            )
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

    // The dispute authority convicts the cheater right before the epoch closes:
    // ejection carries into exited_clients where the slashing rate is applied.
    process_coordinator_slash_client(
        &mut endpoint,
        &payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        cheater as u64,
        [0x11; 32],
        [0x22; 32],
    )
    .await
    .unwrap();

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

    let coordinator_account_state =
        get_coordinator_account_state(&mut endpoint, &coordinator_account)
            .await
            .unwrap()
            .unwrap();
    for (i, client) in clients.iter().enumerate() {
        let client_state = coordinator_account_state
            .clients_state
            .clients
            .iter()
            .find(|c| *c.id.signer() == client.pubkey().to_bytes())
            .unwrap();
        if i == cheater {
            assert_eq!(
                client_state.slashed, slashing_rate,
                "cheater should carry the slashing rate",
            );
            assert_eq!(
                client_state.earned, 0,
                "cheater should earn nothing",
            );
        } else {
            assert_eq!(
                client_state.slashed, 0,
                "honest client should never be slashed",
            );
            assert!(
                client_state.earned > 0,
                "honest client should earn rewards",
            );
        }
    }
}
