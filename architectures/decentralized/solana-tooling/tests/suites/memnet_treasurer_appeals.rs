use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::ClientState;
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
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw_with_appeal;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw_with_voters;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_slash_bounty;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_finalize_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_challenge_config;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash_losing_verifier;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_appeal_verdict;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;
const BOUNTY_BPS: u16 = 5_000;
const CHALLENGE_WINDOW: i64 = 50;
const TIE_BREAKER_SIZE: u16 = 2;
const EPOCH_TIME: u64 = 300;
const COMMITTED: [u8; 32] = [0xAA; 32];
const REPLAYED: [u8; 32] = [0xBB; 32];

struct AppealsHarness {
    endpoint: ToolboxEndpoint,
    payer: Keypair,
    ticker: Keypair,
    run: Pubkey,
    run_id: String,
    coordinator_instance: Pubkey,
    coordinator_account: Pubkey,
    collateral_mint: Pubkey,
    clients: Vec<Keypair>,
    clients_collateral: Vec<Pubkey>,
    verifiers: Vec<(usize, u64)>,
    tie_breakers: Vec<(usize, u64)>,
    target_client_idx: usize,
    target_key: Pubkey,
    target_index: u64,
    verifier_quorum: u64,
    tie_breaker_quorum: u64,
}

async fn setup(run_id: &str, index: u64) -> AppealsHarness {
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
            index,
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

    process_treasurer_run_set_challenge_config(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        CHALLENGE_WINDOW,
        TIE_BREAKER_SIZE,
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
                epoch_time: EPOCH_TIME,
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
    let selection = CommitteeSelection::from_coordinator_with_tie_breakers(
        &state.coordinator,
        0,
        TIE_BREAKER_SIZE,
    )
    .unwrap();
    let verifier_quorum = (2 * selection.get_num_verifier_nodes()).div_ceil(3).max(1);
    let tie_breaker_quorum = (2 * selection.get_num_tie_breaker_nodes()).div_ceil(3).max(1);

    let mut verifiers: Vec<(usize, u64)> = vec![];
    let mut tie_breakers: Vec<(usize, u64)> = vec![];
    let mut target: Option<(usize, Pubkey, u64)> = None;
    for (epoch_index, client) in state.coordinator.epoch_state.clients.iter().enumerate() {
        let signer = *client.id.signer();
        let client_idx = clients.iter().position(|k| k.pubkey().to_bytes() == signer);
        let Some(client_idx) = client_idx else {
            continue;
        };
        match selection.get_committee(epoch_index as u64).committee {
            Committee::Verifier => verifiers.push((client_idx, epoch_index as u64)),
            Committee::TieBreaker => tie_breakers.push((client_idx, epoch_index as u64)),
            Committee::Trainer => {
                if target.is_none() {
                    target = Some((
                        client_idx,
                        clients[client_idx].pubkey(),
                        epoch_index as u64,
                    ));
                }
            }
        }
    }

    let (target_client_idx, target_key, target_index) =
        target.expect("no trainer target found");
    assert!(
        verifiers.len() as u64 >= verifier_quorum,
        "not enough verifiers ({}) for quorum {}",
        verifiers.len(),
        verifier_quorum
    );
    assert!(
        tie_breakers.len() as u64 >= tie_breaker_quorum,
        "not enough tie-breakers ({}) for quorum {}",
        tie_breakers.len(),
        tie_breaker_quorum
    );

    AppealsHarness {
        endpoint,
        payer,
        ticker,
        run,
        run_id: run_id.to_string(),
        coordinator_instance,
        coordinator_account,
        collateral_mint,
        clients,
        clients_collateral,
        verifiers,
        tie_breakers,
        target_client_idx,
        target_key,
        target_index,
        verifier_quorum,
        tie_breaker_quorum,
    }
}

impl AppealsHarness {
    async fn cast_verdict(&mut self, client_idx: usize) -> anyhow::Result<()> {
        let verifier = &self.clients[client_idx];
        process_treasurer_run_submit_audit_verdict(
            &mut self.endpoint,
            &self.payer,
            verifier,
            &self.run,
            &self.coordinator_account,
            &self.run_id,
            &self.target_key,
            self.target_index,
            0,
            4,
            COMMITTED,
            REPLAYED,
        )
        .await
    }

    async fn reach_verifier_quorum(&mut self) {
        let quorum = self.verifier_quorum as usize;
        let voters: Vec<usize> = self.verifiers.iter().take(quorum).map(|(i, _)| *i).collect();
        for client_idx in voters {
            self.cast_verdict(client_idx).await.unwrap();
        }
    }

    async fn cast_appeal(&mut self, client_idx: usize, overturn: bool) -> anyhow::Result<()> {
        let appellate = &self.clients[client_idx];
        process_treasurer_run_submit_appeal_verdict(
            &mut self.endpoint,
            &self.payer,
            appellate,
            &self.run,
            &self.coordinator_account,
            &self.run_id,
            &self.target_key,
            self.target_index,
            overturn,
        )
        .await
    }

    async fn client_state(&mut self, key: &Pubkey) -> Option<ClientState> {
        let state = get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
            .await
            .unwrap()
            .unwrap();
        let found = state
            .coordinator
            .epoch_state
            .clients
            .iter()
            .find(|c| *c.id.signer() == key.to_bytes())
            .map(|c| c.state);
        found
    }

    async fn settle_epoch(&mut self) {
        for _ in 0..15 {
            self.endpoint
                .forward_clock_unix_timestamp(60)
                .await
                .unwrap();
            let _ = process_coordinator_tick(
                &mut self.endpoint,
                &self.payer,
                &self.ticker,
                &self.coordinator_instance,
                &self.coordinator_account,
            )
            .await;
        }
    }

    async fn slashed_of(&mut self, key: &Pubkey) -> u64 {
        let state = get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
            .await
            .unwrap()
            .unwrap();
        let found = state
            .clients_state
            .clients
            .iter()
            .find(|c| *c.id.signer() == key.to_bytes())
            .map(|c| c.slashed)
            .unwrap_or(0);
        found
    }
}

#[tokio::test]
pub async fn overturn_penalises_the_verifiers() {
    let mut h = setup("Leviathan appeals overturn", 90).await;
    let target_key = h.target_key;

    h.reach_verifier_quorum().await;

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy),
        "the target must not be slashed while the challenge window is open",
    );

    let target = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();

    let quorum = h.tie_breaker_quorum as usize;
    let appellates: Vec<usize> = h.tie_breakers.iter().take(quorum).map(|(i, _)| *i).collect();
    for client_idx in appellates {
        h.cast_appeal(client_idx, true).await.unwrap();
    }

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy),
        "an overturned target is never slashed",
    );

    let verifier_keys: Vec<Pubkey> = h
        .verifiers
        .iter()
        .take(h.verifier_quorum as usize)
        .map(|(i, _)| h.clients[*i].pubkey())
        .collect();
    for position in 0..verifier_keys.len() {
        process_treasurer_run_slash_losing_verifier(
            &mut h.endpoint,
            &h.payer,
            &h.run,
            &h.coordinator_account,
            &h.run_id,
            &target_key,
            position as u16,
        )
        .await
        .unwrap();
    }

    process_treasurer_run_slash_losing_verifier(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        verifier_keys.len() as u16,
    )
    .await
    .expect_err("cranking past the last loser must be rejected");

    h.settle_epoch().await;

    for key in &verifier_keys {
        assert_eq!(
            h.slashed_of(key).await,
            SLASHING_RATE,
            "each convicting verifier forfeits its bond when overturned",
        );
    }
    assert_eq!(
        h.slashed_of(&target_key).await,
        0,
        "the wrongly accused target keeps its bond",
    );

    let loser_key = verifier_keys[0];
    let loser_client_idx = h
        .clients
        .iter()
        .position(|k| k.pubkey() == loser_key)
        .unwrap();
    let loser = h.clients[loser_client_idx].insecure_clone();
    let loser_ata = h.clients_collateral[loser_client_idx];
    let tie_breaker_collaterals: Vec<Pubkey> = h
        .tie_breakers
        .iter()
        .take(h.tie_breaker_quorum as usize)
        .map(|(client_idx, _)| h.clients_collateral[*client_idx])
        .collect();
    process_treasurer_participant_bond_request_withdraw(
        &mut h.endpoint,
        &h.payer,
        &loser,
        &h.run,
        BOND,
    )
    .await
    .unwrap();
    h.endpoint
        .forward_clock_unix_timestamp(100)
        .await
        .unwrap();
    process_treasurer_participant_bond_finalize_withdraw_with_appeal(
        &mut h.endpoint,
        &h.payer,
        &loser,
        &loser_ata,
        &h.collateral_mint,
        &h.run,
        &h.coordinator_account,
        &target_key,
        &tie_breaker_collaterals,
    )
    .await
    .unwrap();
    let appeal_bounty = (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64;
    let appeal_share = appeal_bounty / tie_breaker_collaterals.len() as u64;
    for tie_breaker_collateral in &tie_breaker_collaterals {
        assert_eq!(
            h.endpoint
                .get_spl_token_account(tie_breaker_collateral)
                .await
                .unwrap()
                .unwrap()
                .amount,
            appeal_share,
            "each tie-breaker earns a share of the overturned verifier's forfeited bond",
        );
    }

    let target_ata = h.clients_collateral[h.target_client_idx];
    process_treasurer_participant_bond_request_withdraw(
        &mut h.endpoint,
        &h.payer,
        &target,
        &h.run,
        BOND,
    )
    .await
    .unwrap();
    h.endpoint
        .forward_clock_unix_timestamp(100)
        .await
        .unwrap();
    process_treasurer_participant_bond_finalize_withdraw(
        &mut h.endpoint,
        &h.payer,
        &target,
        &target_ata,
        &h.collateral_mint,
        &h.run,
        &h.coordinator_account,
    )
    .await
    .unwrap();
    assert_eq!(
        h.endpoint
            .get_spl_token_account(&target_ata)
            .await
            .unwrap()
            .unwrap()
            .amount,
        BOND,
        "the cleared target withdraws its full bond",
    );
}

#[tokio::test]
pub async fn upheld_appeal_slashes_the_target() {
    let mut h = setup("Leviathan appeals uphold", 91).await;
    let target_key = h.target_key;

    h.reach_verifier_quorum().await;

    let target = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();

    let quorum = h.tie_breaker_quorum as usize;
    let appellates: Vec<usize> = h.tie_breakers.iter().take(quorum).map(|(i, _)| *i).collect();
    for client_idx in appellates {
        h.cast_appeal(client_idx, false).await.unwrap();
    }

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected),
        "an upheld appeal finalises the slash against the target",
    );

    h.settle_epoch().await;

    assert_eq!(
        h.slashed_of(&target_key).await,
        SLASHING_RATE,
        "the confirmed cheater forfeits its bond",
    );
    for (client_idx, _) in h.verifiers.clone().iter().take(h.verifier_quorum as usize) {
        let key = h.clients[*client_idx].pubkey();
        assert_eq!(
            h.slashed_of(&key).await,
            0,
            "verifiers on the winning side keep their bonds",
        );
    }

    let target_ata = h.clients_collateral[h.target_client_idx];
    let tie_breaker_collaterals: Vec<Pubkey> = h
        .tie_breakers
        .iter()
        .take(h.tie_breaker_quorum as usize)
        .map(|(client_idx, _)| h.clients_collateral[*client_idx])
        .collect();
    process_treasurer_participant_bond_request_withdraw(
        &mut h.endpoint,
        &h.payer,
        &target,
        &h.run,
        BOND,
    )
    .await
    .unwrap();
    h.endpoint
        .forward_clock_unix_timestamp(100)
        .await
        .unwrap();
    process_treasurer_participant_bond_finalize_withdraw_with_voters(
        &mut h.endpoint,
        &h.payer,
        &target,
        &target_ata,
        &h.collateral_mint,
        &h.run,
        &h.coordinator_account,
        &tie_breaker_collaterals,
    )
    .await
    .unwrap();
    let appeal_bounty = (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64;
    let appeal_share = appeal_bounty / tie_breaker_collaterals.len() as u64;
    for tie_breaker_collateral in &tie_breaker_collaterals {
        assert_eq!(
            h.endpoint
                .get_spl_token_account(tie_breaker_collateral)
                .await
                .unwrap()
                .unwrap()
                .amount,
            appeal_share,
            "an upheld appeal pays the tie-breakers out of the target's forfeited bond",
        );
    }
}

#[tokio::test]
pub async fn unchallenged_verdict_finalises_after_the_window() {
    let mut h = setup("Leviathan appeals finalise", 92).await;
    let target_key = h.target_key;

    h.reach_verifier_quorum().await;

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy),
        "a pending slash does not eject before finalisation",
    );

    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
    )
    .await
    .expect_err("finalising before the window elapses must be rejected");

    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();

    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
    )
    .await
    .unwrap();

    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected),
        "an unchallenged verdict finalises into a slash once the window closes",
    );

    h.settle_epoch().await;

    assert_eq!(
        h.slashed_of(&target_key).await,
        SLASHING_RATE,
        "the unchallenged target forfeits its bond",
    );
}
