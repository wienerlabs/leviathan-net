use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_coordinator::WitnessBloom;
use psyche_coordinator::BLOOM_FALSE_RATE;
use psyche_core::sha256;
use psyche_solana_coordinator::instruction::Witness;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_witness;
use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::WAITING_FOR_MEMBERS_EXTRA_SECONDS;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::NodeIdentity;
use psyche_core::OptimizerDefinition;
use psyche_solana_authorizer::find_authorization;
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
use anchor_lang::error::ErrorCode as AnchorErrorCode;
use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use psyche_solana_treasurer::accounts::RunOpenChallengeAccounts;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::instruction::RunOpenChallenge;
use psyche_solana_treasurer::logic::RunOpenChallengeParams;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_treasurer::state::withdrawal_standing;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::state::WithdrawalHold;
use psyche_solana_treasurer::state::WithdrawalStanding;
use psyche_solana_treasurer::ProgramError;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_coordinator::ProgramError as CoordinatorProgramError;
use solana_sdk::instruction::Instruction;

use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;
const BOUNTY_BPS: u16 = 5_000;
const CHALLENGE_WINDOW: i64 = 50;
const TIE_BREAKER_SIZE: u16 = 2;
/// Long enough that these tests never hit it: they exercise the appeal itself,
/// not the deadline that stops a challenge holding a conviction for ever
/// (wienerlabs/leviathan#15, finding 10).
const APPEAL_WINDOW: i64 = 10_000;
const EPOCH_TIME: u64 = 300;
const COMMITTED: [u8; 32] = [0xAA; 32];
const REPLAYED: [u8; 32] = [0xBB; 32];

struct AppealsHarness {
    endpoint: ToolboxEndpoint,
    payer: Keypair,
    ticker: Keypair,
    run: Pubkey,
    run_id: String,
    main_authority: Keypair,
    join_authority: Pubkey,
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
        APPEAL_WINDOW,
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
        main_authority,
        join_authority: join_authority.pubkey(),
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

    /// `client_idx` asks for `amount` of its bond back, which starts it on its
    /// way out of the vault.
    async fn request_withdraw(&mut self, client_idx: usize, amount: u64) {
        let client = self.clients[client_idx].insecure_clone();
        process_treasurer_participant_bond_request_withdraw(
            &mut self.endpoint,
            &self.payer,
            &client,
            &self.run,
            amount,
        )
        .await
        .unwrap();
    }

    /// `client_idx` finalizes its pending withdrawal, with no bounty recipients.
    async fn try_finalize_withdraw(&mut self, client_idx: usize) -> anyhow::Result<()> {
        let client = self.clients[client_idx].insecure_clone();
        let collateral = self.clients_collateral[client_idx];
        process_treasurer_participant_bond_finalize_withdraw(
            &mut self.endpoint,
            &self.payer,
            &client,
            &collateral,
            &self.collateral_mint,
            &self.run,
            &self.coordinator_account,
        )
        .await
    }

    /// Every client joins the next epoch with the authorization the run's join
    /// authority granted it, and the run is driven into its first round, paced
    /// by the run's own config.
    async fn rejoin_all_and_start_epoch(&mut self) {
        for client_idx in 0..self.clients.len() {
            let client = self.clients[client_idx].insecure_clone();
            let authorization = find_authorization(
                &self.join_authority,
                &client.pubkey(),
                JOIN_RUN_AUTHORIZATION_SCOPE,
            );
            process_coordinator_join_run(
                &mut self.endpoint,
                &self.payer,
                &client,
                &authorization,
                &self.coordinator_instance,
                &self.coordinator_account,
                NodeIdentity::new(client.pubkey().to_bytes(), Default::default()),
            )
            .await
            .unwrap();
        }
        let config = get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
            .await
            .unwrap()
            .unwrap()
            .coordinator
            .config;
        for seconds in [config.waiting_for_members_extra_time as u64, config.warmup_time] {
            self.endpoint
                .forward_clock_unix_timestamp(seconds)
                .await
                .unwrap();
            process_coordinator_tick(
                &mut self.endpoint,
                &self.payer,
                &self.ticker,
                &self.coordinator_instance,
                &self.coordinator_account,
            )
            .await
            .unwrap();
        }
        let state = get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(state.coordinator.run_state, RunState::RoundTrain);
    }

    async fn tick_after(&mut self, seconds: u64) {
        self.endpoint
            .forward_clock_unix_timestamp(seconds)
            .await
            .unwrap();
        process_coordinator_tick(
            &mut self.endpoint,
            &self.payer,
            &self.ticker,
            &self.coordinator_instance,
            &self.coordinator_account,
        )
        .await
        .unwrap();
    }

    /// Runs the epoch to its end the way a healthy swarm does: every drawn
    /// witness reports the roster each round, so the clients finish Healthy and
    /// the ended epoch's roster stays on the account into `WaitingForMembers`.
    async fn finish_epoch_with_witnesses(&mut self) {
        let start = get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
            .await
            .unwrap()
            .unwrap()
            .coordinator
            .progress
            .epoch;
        for _ in 0..1_000 {
            let state = get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
                .await
                .unwrap()
                .unwrap();
            let coordinator = state.coordinator;
            let config = coordinator.config;
            if coordinator.progress.epoch > start
                && coordinator.run_state == RunState::WaitingForMembers
            {
                return;
            }
            match coordinator.run_state {
                RunState::RoundTrain => {
                    let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
                    let roster: Vec<NodeIdentity> =
                        coordinator.epoch_state.clients.iter().map(|c| c.id).collect();
                    for (index, id) in roster.iter().enumerate() {
                        let proof = selection.get_witness(index as u64);
                        if !bool::from(proof.witness) {
                            continue;
                        }
                        let Some(client_idx) = self
                            .clients
                            .iter()
                            .position(|k| k.pubkey().to_bytes() == *id.signer())
                        else {
                            continue;
                        };
                        let mut participant_bloom =
                            WitnessBloom::random(roster.len(), BLOOM_FALSE_RATE);
                        for participant in &roster {
                            participant_bloom.add(&sha256(participant.signer()));
                        }
                        let witness = self.clients[client_idx].insecure_clone();
                        process_coordinator_witness(
                            &mut self.endpoint,
                            &self.payer,
                            &witness,
                            &self.coordinator_instance,
                            &self.coordinator_account,
                            &Witness {
                                proof,
                                participant_bloom,
                                broadcast_bloom: WitnessBloom::random(1, BLOOM_FALSE_RATE),
                                broadcast_merkle: Default::default(),
                                metadata: Default::default(),
                            },
                        )
                        .await
                        .unwrap();
                    }
                    let now = get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
                        .await
                        .unwrap()
                        .unwrap()
                        .coordinator
                        .run_state;
                    if now == RunState::RoundTrain {
                        self.tick_after(config.max_round_train_time).await;
                    }
                },
                RunState::RoundWitness => self.tick_after(config.round_witness_time).await,
                RunState::Cooldown => self.tick_after(config.cooldown_time).await,
                RunState::Warmup => self.tick_after(config.warmup_time).await,
                RunState::WaitingForMembers => {
                    self.tick_after(config.waiting_for_members_extra_time as u64)
                        .await
                },
                halted => panic!("the run halted in {halted}"),
            }
        }
        panic!("epoch {start} never ended");
    }

    /// Where `key` sits in the current epoch's roster.
    async fn roster_index_of(&mut self, key: &Pubkey) -> u64 {
        get_coordinator_account_state(&mut self.endpoint, &self.coordinator_account)
            .await
            .unwrap()
            .unwrap()
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|c| *c.id.signer() == key.to_bytes())
            .expect("the client is in this epoch's roster") as u64
    }

    /// The verifiers whose votes convicted the target, as bounty recipients.
    fn convicting_verifier_collaterals(&self) -> Vec<Pubkey> {
        self.verifiers
            .iter()
            .take(self.verifier_quorum as usize)
            .map(|(client_idx, _)| self.clients_collateral[*client_idx])
            .collect()
    }

    /// The standing run-manager reads before it sends a finalize: the run, its
    /// coordinator account, the client's own verdict account and the cluster
    /// clock sysvar, decoded the way the treasurer and the backend decode them.
    async fn observed_standing(&mut self, client_idx: usize) -> WithdrawalStanding {
        let user = self.clients[client_idx].pubkey();
        let run = get_run(&mut self.endpoint, &self.run).await.unwrap().unwrap();
        let coordinator = self
            .endpoint
            .get_account(&self.coordinator_account)
            .await
            .unwrap()
            .unwrap();
        let verdict = self
            .endpoint
            .get_account(&find_audit_verdict(&self.run, &user))
            .await
            .unwrap();
        let own_verdict = match verdict {
            Some(account) => AuditVerdict::from_pda_account(&account.owner, &account.data).unwrap(),
            None => None,
        };
        let now = self
            .endpoint
            .get_account(&solana_sdk::sysvar::clock::ID)
            .await
            .unwrap()
            .unwrap()
            .deserialize_data::<solana_sdk::clock::Clock>()
            .unwrap()
            .unix_timestamp;
        withdrawal_standing(
            &run,
            coordinator_account_from_bytes(&coordinator.data).unwrap(),
            own_verdict.as_ref(),
            &user,
            now,
        )
    }

    async fn verdict_status(&mut self) -> VerdictStatus {
        let account = self
            .endpoint
            .get_account(&find_audit_verdict(&self.run, &self.target_key))
            .await
            .unwrap()
            .unwrap();
        AuditVerdict::from_pda_account(&account.owner, &account.data)
            .unwrap()
            .unwrap()
            .status
    }

    /// The convicted target finalizes its withdrawal, paying the bounty to the
    /// verifiers whose quorum convicted it.
    async fn finalize_target_withdrawal(&mut self) -> anyhow::Result<()> {
        let target_idx = self.target_client_idx;
        let target = self.clients[target_idx].insecure_clone();
        let target_collateral = self.clients_collateral[target_idx];
        let voters = self.convicting_verifier_collaterals();
        process_treasurer_participant_bond_finalize_withdraw_with_voters(
            &mut self.endpoint,
            &self.payer,
            &target,
            &target_collateral,
            &self.collateral_mint,
            &self.run,
            &self.coordinator_account,
            &voters,
        )
        .await
    }

    async fn collateral_of(&mut self, client_idx: usize) -> u64 {
        let account = self.clients_collateral[client_idx];
        self.endpoint
            .get_spl_token_account(&account)
            .await
            .unwrap()
            .unwrap()
            .amount
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
    let target_index = h.target_index;

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
        target_index,
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
        target_index,
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

/// A verdict that turns out wrong is paid for out of the verifier's bond after
/// the overturn, so a bond already on its way out cannot back a vote
/// (wienerlabs/leviathan-net#34).
#[tokio::test]
pub async fn a_verifier_leaving_its_bond_cannot_vote() {
    let mut h = setup("Leviathan appeals leaving voter", 93).await;
    let (leaving, _) = h.verifiers[0];
    let (staying, _) = h.verifiers[1];

    h.request_withdraw(leaving, 1).await;
    assert_eq!(
        refusal_code(h.cast_verdict(leaving).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a verifier with a withdrawal pending cast a verdict its bond may not be there to answer for",
    );
    h.cast_verdict(staying).await.unwrap();
}

/// A tie-breaker sits on a bonded bench for the same reason, and is held to the
/// same bond (wienerlabs/leviathan-net#34).
#[tokio::test]
pub async fn a_tie_breaker_leaving_its_bond_cannot_judge_the_appeal() {
    let mut h = setup("Leviathan appeals leaving bench", 94).await;
    h.reach_verifier_quorum().await;
    let target = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();

    let (leaving, _) = h.tie_breakers[0];
    let (staying, _) = h.tie_breakers[1];
    h.request_withdraw(leaving, 1).await;
    assert_eq!(
        refusal_code(h.cast_appeal(leaving, true).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a tie-breaker with a withdrawal pending judged an appeal",
    );
    h.cast_appeal(staying, true).await.unwrap();
}

/// The target defends a conviction rather than taking on new work, and its bond
/// is still in the vault while the withdrawal waits, so a pending withdrawal
/// does not cost it the appeal (wienerlabs/leviathan-net#34).
#[tokio::test]
pub async fn a_target_with_a_pending_withdrawal_can_still_appeal() {
    let mut h = setup("Leviathan appeals pending target", 95).await;
    let target_idx = h.target_client_idx;
    h.request_withdraw(target_idx, 1).await;
    h.reach_verifier_quorum().await;

    let target = h.clients[target_idx].insecure_clone();
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
    let (tie_breaker, _) = h.tie_breakers[0];
    h.cast_appeal(tie_breaker, true).await.unwrap();
}

/// The challenger's slashed points are read from the run's own coordinator: a
/// challenge that names another run's coordinator account is refused.
#[tokio::test]
pub async fn a_challenge_against_another_runs_coordinator_is_refused() {
    let mut h = setup("Leviathan appeals foreign ledger", 96).await;
    h.reach_verifier_quorum().await;

    let other_account = h
        .endpoint
        .process_system_new_exempt(
            &h.payer,
            CoordinatorAccount::space_with_discriminator(),
            &psyche_solana_coordinator::ID,
        )
        .await
        .unwrap();
    process_treasurer_run_create(
        &mut h.endpoint,
        &h.payer,
        &h.collateral_mint,
        &other_account,
        RunCreateParams {
            index: 97,
            run_id: "Leviathan appeals other ledger".to_string(),
            main_authority: Keypair::new().pubkey(),
            join_authority: Keypair::new().pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    let target = h.clients[h.target_client_idx].insecure_clone();
    let instruction = Instruction {
        program_id: psyche_solana_treasurer::ID,
        accounts: RunOpenChallengeAccounts {
            challenger: target.pubkey(),
            challenger_participant: find_participant(&h.run, &target.pubkey()),
            run: h.run,
            coordinator_account: other_account,
            audit_verdict: find_audit_verdict(&h.run, &target.pubkey()),
        }
        .to_account_metas(None),
        data: RunOpenChallenge {
            params: RunOpenChallengeParams {},
        }
        .data(),
    };
    assert_eq!(
        refusal_code(
            h.endpoint
                .process_instruction_with_signers(&h.payer, instruction, &[&target])
                .await
        ),
        u32::from(AnchorErrorCode::ConstraintRaw),
    );
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
}


/// While a conviction is inside its challenge window it can still be
/// overturned or finalised, so the target's withdrawal is held, however long
/// ago its delay elapsed. Past the window, in an epoch that will be charged, it
/// waits for that epoch, where the conviction lands through the ledger and
/// settlement forfeits it (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_target_cannot_leave_while_its_conviction_is_pending() {
    let mut h = setup("Leviathan appeals pending leave", 98).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target_idx, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    h.reach_verifier_quorum().await;
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);

    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::ConvictionPending)
    );
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::ConvictionPending),
        "a target left with its whole bond while its conviction was pending"
    );

    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled)
    );
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::EpochNotSettled),
    );

    let target_index = h.roster_index_of(&target_key).await;
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .unwrap();
    h.settle_epoch().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: false }
    );
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
}

/// A target that walks away from its conviction (its epoch ends before anyone
/// finalises it, and it never comes back) cannot be ejected into any epoch the
/// ledger will charge. Its withdrawal settles the conviction against the bond
/// itself, pays the convicting verifiers, and records the verdict as upheld,
/// so it is neither lost nor allowed to hold the bond forever
/// (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_conviction_the_target_walks_away_from_is_settled_at_withdrawal() {
    let mut h = setup("Leviathan appeals walked away", 102).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    h.settle_epoch().await;
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    assert_eq!(h.slashed_of(&target_key).await, 0);
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: true }
    );

    let voter_collaterals = h.convicting_verifier_collaterals();
    let mut voters_before = vec![];
    for collateral in &voter_collaterals {
        voters_before.push(token_amount(&mut h.endpoint, collateral).await);
    }
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
    let share = (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64
        / voter_collaterals.len() as u64;
    for (collateral, before) in voter_collaterals.iter().zip(voters_before) {
        assert_eq!(token_amount(&mut h.endpoint, collateral).await - before, share);
    }

    // Settled once: the verdict cannot be finalised against the target again.
    assert_eq!(
        refusal_code(
            process_treasurer_run_finalize_slash(
                &mut h.endpoint,
                &h.payer,
                &h.run,
                &h.coordinator_account,
                &h.run_id,
                &target_key,
                0,
            )
            .await
        ),
        treasurer_refusal(ProgramError::VerdictNotPending),
    );
}

/// An appeal holds the bond for as long as the bench may still rule, across
/// the epoch boundary too. Once its window has passed with no ruling, the
/// conviction the verifiers reached stands, and lands through the ledger in
/// the epoch the target comes back for (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_challenged_target_cannot_leave_until_the_appeal_resolves() {
    let mut h = setup("Leviathan appeals open leave", 99).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    let target = h.clients[target_idx].insecure_clone();
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();

    h.settle_epoch().await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::ConvictionPending)
    );
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::ConvictionPending),
        "a target left with its whole bond while its appeal was open"
    );

    h.endpoint
        .forward_clock_unix_timestamp(APPEAL_WINDOW as u64)
        .await
        .unwrap();
    h.rejoin_all_and_start_epoch().await;
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled)
    );
    let target_index = h.roster_index_of(&target_key).await;
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .unwrap();
    assert_eq!(h.client_state(&target_key).await, Some(ClientState::Ejected));
    assert_eq!(
        refusal_code(h.try_finalize_withdraw(target_idx).await),
        treasurer_refusal(ProgramError::EpochNotSettled),
    );
    h.settle_epoch().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
}

/// An innocent target whose appeal overturns the conviction is not held at
/// all: it leaves with its whole bond (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn an_overturned_target_leaves_with_its_whole_bond() {
    let mut h = setup("Leviathan appeals cleared leave", 100).await;
    let target_idx = h.target_client_idx;
    let target = h.clients[target_idx].insecure_clone();
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
    let quorum = h.tie_breaker_quorum as usize;
    let bench: Vec<usize> = h.tie_breakers.iter().take(quorum).map(|(i, _)| *i).collect();
    for client_idx in bench {
        h.cast_appeal(client_idx, true).await.unwrap();
    }

    h.settle_epoch().await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    assert_eq!(
        h.observed_standing(target_idx).await,
        WithdrawalStanding::Free { settle_conviction: false }
    );
    let before = h.collateral_of(target_idx).await;
    h.try_finalize_withdraw(target_idx).await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND);
}

/// A conviction finalised between epochs would be ejected onto the ended
/// epoch's roster and never charged. The coordinator refuses that ejection, so
/// the conviction stays due, and the target's withdrawal settles it against the
/// bond instead (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_conviction_finalised_after_its_epoch_still_reaches_the_bond() {
    let mut h = setup("Leviathan appeals late finalise", 101).await;
    let target_idx = h.target_client_idx;
    let target_key = h.target_key;
    h.request_withdraw(target_idx, BOND).await;
    h.reach_verifier_quorum().await;
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.client_state(&target_key).await, Some(ClientState::Healthy));
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();

    assert_eq!(
        refusal_code(
            process_treasurer_run_finalize_slash(
                &mut h.endpoint,
                &h.payer,
                &h.run,
                &h.coordinator_account,
                &h.run_id,
                &target_key,
                h.target_index,
            )
            .await
        ),
        u32::from(CoordinatorProgramError::CoordinatorErrorSlashOutsideEpoch),
        "an ejection between epochs would never be charged"
    );
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);

    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    let before = h.collateral_of(target_idx).await;
    h.finalize_target_withdrawal().await.unwrap();
    assert_eq!(h.collateral_of(target_idx).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
}

/// A verdict is cast while an epoch is still to be charged: a quorum completed
/// between epochs could not eject anyone the ledger would ever see
/// (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn verdict_votes_wait_for_an_epoch() {
    let mut h = setup("Leviathan appeals late verdict", 103).await;
    let (first, _) = h.verifiers[0];
    let (second, _) = h.verifiers[1];
    h.cast_verdict(first).await.unwrap();
    h.finish_epoch_with_witnesses().await;
    assert_eq!(
        refusal_code(h.cast_verdict(second).await),
        treasurer_refusal(ProgramError::VotingOutsideEpoch),
    );
}

/// The bench is held to the same epoch, on both sides: between epochs it could
/// otherwise only ever clear, since upholding needs an ejection and
/// overturning does not (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn appeal_votes_wait_for_an_epoch() {
    let mut h = setup("Leviathan appeals late bench", 106).await;
    let target = h.clients[h.target_client_idx].insecure_clone();
    h.reach_verifier_quorum().await;
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
    h.finish_epoch_with_witnesses().await;

    let (tie_breaker, _) = h.tie_breakers[0];
    for overturn in [true, false] {
        assert_eq!(
            refusal_code(h.cast_appeal(tie_breaker, overturn).await),
            treasurer_refusal(ProgramError::VotingOutsideEpoch),
        );
    }
    assert_eq!(h.verdict_status().await, VerdictStatus::Challenged);
}

/// A losing verifier cannot be ejected between epochs either: the crank keeps
/// its place and settles it once the loser is in an epoch that will be charged
/// (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn the_losing_verifier_crank_waits_for_an_epoch() {
    let mut h = setup("Leviathan appeals crank waits", 104).await;
    let target = h.clients[h.target_client_idx].insecure_clone();
    let target_key = h.target_key;
    h.reach_verifier_quorum().await;
    process_treasurer_run_open_challenge(&mut h.endpoint, &h.payer, &target, &h.run)
        .await
        .unwrap();
    let quorum = h.tie_breaker_quorum as usize;
    let bench: Vec<usize> = h.tie_breakers.iter().take(quorum).map(|(i, _)| *i).collect();
    for client_idx in bench {
        h.cast_appeal(client_idx, true).await.unwrap();
    }
    assert_eq!(h.verdict_status().await, VerdictStatus::Overturned);
    h.finish_epoch_with_witnesses().await;

    assert_eq!(
        refusal_code(
            process_treasurer_run_slash_losing_verifier(
                &mut h.endpoint,
                &h.payer,
                &h.run,
                &h.coordinator_account,
                &h.run_id,
                &target_key,
                0,
            )
            .await
        ),
        treasurer_refusal(ProgramError::LoserNotSlashableYet),
    );

    h.rejoin_all_and_start_epoch().await;
    process_treasurer_run_slash_losing_verifier(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        0,
    )
    .await
    .unwrap();
    let (loser, _) = h.verifiers[0];
    let loser_key = h.clients[loser].pubkey();
    assert_eq!(h.client_state(&loser_key).await, Some(ClientState::Ejected));
}

/// With appeals on, an appeal must have a deadline: one without could hold a
/// target's bond for as long as no bench rules, and once the target or the
/// run is gone none ever can (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_challenge_window_needs_an_appeal_deadline() {
    let mut h = setup("Leviathan appeals no deadline", 105).await;
    let main_authority = h.main_authority.insecure_clone();
    assert_eq!(
        refusal_code(
            process_treasurer_run_set_challenge_config(
                &mut h.endpoint,
                &h.payer,
                &main_authority,
                &h.run,
                CHALLENGE_WINDOW,
                TIE_BREAKER_SIZE,
                0,
            )
            .await
        ),
        treasurer_refusal(ProgramError::AppealWindowRequired),
    );
    process_treasurer_run_set_challenge_config(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        0,
        0,
        0,
    )
    .await
    .unwrap();
}

async fn token_amount(endpoint: &mut ToolboxEndpoint, account: &Pubkey) -> u64 {
    endpoint
        .get_spl_token_account(account)
        .await
        .unwrap()
        .unwrap()
        .amount
}
