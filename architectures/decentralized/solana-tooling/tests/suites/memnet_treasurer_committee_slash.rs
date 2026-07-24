use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::WAITING_FOR_MEMBERS_EXTRA_SECONDS;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::NodeIdentity;
use psyche_core::OptimizerDefinition;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw_with_voters;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_slash_bounty;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

const RUN_ID: &str = "Leviathan committee slash";
const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;
const BOUNTY_BPS: u16 = 5_000;
const COMMITTED: [u8; 32] = [0xAA; 32];
const REPLAYED: [u8; 32] = [0xBB; 32];

#[tokio::test]
pub async fn run() {
    let mut endpoint = create_memnet_endpoint().await;

    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 10_000_000_000)
        .await
        .unwrap();

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let ticker = Keypair::new();
    let clients: Vec<Keypair> = (0..6).map(|_| Keypair::new()).collect();
    let warmup_time = 10;

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
            index: 88,
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

    process_treasurer_run_set_slash_bounty(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        BOUNTY_BPS,
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
        process_treasurer_participant_create(&mut endpoint, &payer, client, &run)
            .await
            .unwrap();
        process_treasurer_participant_bond_deposit(
            &mut endpoint,
            &payer,
            client,
            &ata,
            &collateral_mint,
            &run,
            BOND,
        )
        .await
        .unwrap();
        clients_collateral.push(ata);
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
                cooldown_time: 88,
                max_round_train_time: 15,
                round_witness_time: 10,
                min_clients: clients.len() as u16,
                init_min_clients: clients.len() as u16,
                global_batch_size_start: 1,
                global_batch_size_end: clients.len() as u16,
                global_batch_size_warmup_tokens: 0,
                verification_percent: 50,
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

    let state = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let selection = CommitteeSelection::from_coordinator(&state.coordinator, 0).unwrap();
    let quorum = (2 * selection.get_num_verifier_nodes()).div_ceil(3).max(1);

    let mut verifiers: Vec<(&Keypair, u64)> = vec![];
    let mut target: Option<(Pubkey, u64)> = None;
    for (epoch_index, client) in state.coordinator.epoch_state.clients.iter().enumerate() {
        let signer = *client.id.signer();
        let keypair = clients.iter().find(|k| k.pubkey().to_bytes() == signer);
        match selection.get_committee(epoch_index as u64).committee {
            Committee::Verifier => {
                if let Some(keypair) = keypair {
                    verifiers.push((keypair, epoch_index as u64));
                }
            }
            Committee::Trainer => {
                if target.is_none() {
                    if let Some(keypair) = keypair {
                        target = Some((keypair.pubkey(), epoch_index as u64));
                    }
                }
            }
            Committee::TieBreaker => {}
        }
    }

    let (target_key, target_index) = target.expect("no trainer target found");
    assert!(
        verifiers.len() as u64 >= quorum,
        "not enough verifiers ({}) for quorum {}",
        verifiers.len(),
        quorum
    );

    for (verifier, _) in verifiers.iter().take((quorum - 1) as usize) {
        cast_verdict(&mut endpoint, &payer, verifier, &run, &coordinator_account, &target_key, target_index)
            .await
            .unwrap();
    }

    let mid = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let target_state_mid = mid
        .coordinator
        .epoch_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == target_key.to_bytes())
        .map(|c| c.state);
    assert_eq!(
        target_state_mid,
        Some(psyche_coordinator::ClientState::Healthy),
        "target must not be slashed before quorum"
    );

    let non_verifier = clients
        .iter()
        .find(|k| k.pubkey() == target_key)
        .expect("target is a trainer, use it as the non-verifier");
    cast_verdict(&mut endpoint, &payer, non_verifier, &run, &coordinator_account, &target_key, target_index)
        .await
        .expect_err("a non-verifier must be rejected");

    cast_verdict(&mut endpoint, &payer, verifiers[0].0, &run, &coordinator_account, &target_key, target_index)
        .await
        .expect_err("a duplicate verdict must be rejected");

    cast_verdict(
        &mut endpoint,
        &payer,
        verifiers[(quorum - 1) as usize].0,
        &run,
        &coordinator_account,
        &target_key,
        target_index,
    )
    .await
    .unwrap();

    let after = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let target_state_after = after
        .coordinator
        .epoch_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == target_key.to_bytes())
        .map(|c| c.state);
    assert_eq!(
        target_state_after,
        Some(psyche_coordinator::ClientState::Ejected),
        "target must be ejected once quorum is reached"
    );

    let mut target_slashed = 0;
    for _ in 0..12 {
        endpoint
            .forward_clock_unix_timestamp(60)
            .await
            .unwrap();
        let _ = process_coordinator_tick(
            &mut endpoint,
            &payer,
            &ticker,
            &coordinator_instance,
            &coordinator_account,
        )
        .await;
        let settled = get_coordinator_account_state(&mut endpoint, &coordinator_account)
            .await
            .unwrap()
            .unwrap();
        target_slashed = settled
            .clients_state
            .clients
            .iter()
            .find(|c| *c.id.signer() == target_key.to_bytes())
            .unwrap()
            .slashed;
        if target_slashed > 0 {
            break;
        }
    }
    assert_eq!(target_slashed, SLASHING_RATE);

    let target_position = clients
        .iter()
        .position(|k| k.pubkey() == target_key)
        .unwrap();
    let voter_collaterals: Vec<Pubkey> = verifiers
        .iter()
        .take(quorum as usize)
        .map(|(verifier, _)| {
            let position = clients
                .iter()
                .position(|k| k.pubkey() == verifier.pubkey())
                .unwrap();
            clients_collateral[position]
        })
        .collect();

    process_treasurer_participant_bond_request_withdraw(
        &mut endpoint,
        &payer,
        &clients[target_position],
        &run,
        BOND,
    )
    .await
    .unwrap();
    endpoint
        .forward_clock_unix_timestamp(100)
        .await
        .unwrap();
    process_treasurer_participant_bond_finalize_withdraw_with_voters(
        &mut endpoint,
        &payer,
        &clients[target_position],
        &clients_collateral[target_position],
        &collateral_mint,
        &run,
        &coordinator_account,
        &voter_collaterals,
    )
    .await
    .unwrap();

    let bounty = (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64;
    let share = bounty / voter_collaterals.len() as u64;
    for voter_collateral in &voter_collaterals {
        assert_amount(&mut endpoint, voter_collateral, share).await;
    }
    assert_amount(
        &mut endpoint,
        &clients_collateral[target_position],
        BOND - SLASHING_RATE,
    )
    .await;
}

async fn assert_amount(
    endpoint: &mut solana_toolbox_endpoint::ToolboxEndpoint,
    account: &Pubkey,
    expected_amount: u64,
) {
    assert_eq!(
        endpoint
            .get_spl_token_account(account)
            .await
            .unwrap()
            .unwrap()
            .amount,
        expected_amount,
    );
}

#[allow(clippy::too_many_arguments)]
async fn cast_verdict(
    endpoint: &mut solana_toolbox_endpoint::ToolboxEndpoint,
    payer: &Keypair,
    verifier: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
    target_index: u64,
) -> anyhow::Result<()> {
    process_treasurer_run_submit_audit_verdict(
        endpoint,
        payer,
        verifier,
        run,
        coordinator_account,
        RUN_ID,
        target,
        target_index,
        0,
        4,
        COMMITTED,
        REPLAYED,
    )
    .await
}
