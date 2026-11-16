//! The run the appeals suites share: six bonded clients, a challenge window,
//! a two-seat tie-breaker bench and a bounty, with a trainer singled out as
//! the target and its signed payload to convict it on.

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
use psyche_coordinator::RunState;
use psyche_coordinator::WitnessBloom;
use psyche_coordinator::BLOOM_FALSE_RATE;
use psyche_coordinator::WAITING_FOR_MEMBERS_EXTRA_SECONDS;
use psyche_core::sha256;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::NodeIdentity;
use psyche_core::OptimizerDefinition;
use psyche_solana_authorizer::find_authorization;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_coordinator::instruction::Witness;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick_and_reveal;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_witness;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw_with_voters;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_challenge_config;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_slash_bounty;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_appeal_verdict;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_tooling::verdict_votes::get_audit_verdict;
use psyche_solana_tooling::verdict_votes::process_treasurer_participant_release_verdict_vote;
use psyche_solana_tooling::verdict_votes::process_treasurer_run_pay_bench_bounty;
use psyche_solana_tooling::verdict_votes::process_treasurer_run_slash_losing_verifier;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use psyche_solana_treasurer::state::withdrawal_standing;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::Participant;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::state::WithdrawalStanding;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::signed_payload::identity;
use super::signed_payload::replayed_digest;
use super::signed_payload::signed_payload;
use super::signed_payload::FRAUD_PAYLOAD;

pub(super) const BOND: u64 = 500;
pub(super) const SLASHING_RATE: u64 = 200;
pub(super) const BOUNTY_BPS: u16 = 5_000;
pub(super) const CHALLENGE_WINDOW: i64 = 50;
pub(super) const TIE_BREAKER_SIZE: u16 = 2;
/// Long enough that these tests never hit it: they exercise the appeal itself,
/// not the deadline that stops a challenge holding a conviction for ever
/// (wienerlabs/leviathan#15, finding 10).
pub(super) const APPEAL_WINDOW: i64 = 10_000;
pub(super) const EPOCH_TIME: u64 = 300;

pub(super) struct CommitteeNow {
    pub(super) verifiers: Vec<(usize, u64)>,
    pub(super) tie_breakers: Vec<(usize, u64)>,
    pub(super) trainers: Vec<(usize, u64)>,
    pub(super) verifier_quorum: usize,
    pub(super) tie_breaker_quorum: usize,
}

pub(super) struct AppealsHarness {
    pub(super) endpoint: ToolboxEndpoint,
    pub(super) payer: Keypair,
    pub(super) ticker: Keypair,
    pub(super) run: Pubkey,
    pub(super) run_id: String,
    pub(super) main_authority: Keypair,
    pub(super) join_authority: Pubkey,
    pub(super) coordinator_instance: Pubkey,
    pub(super) coordinator_account: Pubkey,
    pub(super) collateral_mint: Pubkey,
    pub(super) clients: Vec<Keypair>,
    /// The p2p key each client joined with, which signs its payloads.
    pub(super) p2ps: Vec<Keypair>,
    pub(super) clients_collateral: Vec<Pubkey>,
    pub(super) verifiers: Vec<(usize, u64)>,
    pub(super) tie_breakers: Vec<(usize, u64)>,
    pub(super) target_client_idx: usize,
    pub(super) target_key: Pubkey,
    pub(super) target_index: u64,
    /// What the target signed for the batch it trained in the first round,
    /// which the suite's verdicts convict it on.
    pub(super) target_commitment: SignedCommitment,
    pub(super) verifier_quorum: u64,
    pub(super) tie_breaker_quorum: u64,
}

pub(super) async fn setup(run_id: &str, index: u64) -> AppealsHarness {
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
    let p2ps: Vec<Keypair> = clients.iter().map(|_| Keypair::new()).collect();
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
            .process_spl_token_mint_to(
                &payer,
                &collateral_mint,
                &mint_authority,
                &ata,
                BOND,
            )
            .await
            .unwrap();
        process_treasurer_participant_create(
            &mut endpoint,
            &payer,
            client,
            &run,
        )
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
                waiting_for_members_extra_time:
                    WAITING_FOR_MEMBERS_EXTRA_SECONDS as u8,
            }),
            model: Some(Model::LLM(LLM {
                architecture: LLMArchitecture::HfLlama,
                checkpoint: Checkpoint::Dummy(HubRepo::dummy()),
                max_seq_len: 4096,
                data_type: LLMTrainingDataType::Pretraining,
                data_location: LLMTrainingDataLocation::default(),
                lr_schedule: LearningRateSchedule::Constant(
                    ConstantLR::default(),
                ),
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

    let state =
        get_coordinator_account_state(&mut endpoint, &coordinator_account)
            .await
            .unwrap()
            .unwrap();
    let selection = CommitteeSelection::from_coordinator_with_tie_breakers(
        &state.coordinator,
        0,
        TIE_BREAKER_SIZE,
    )
    .unwrap();
    let verifier_quorum =
        (2 * selection.get_num_verifier_nodes()).div_ceil(3).max(1);
    let tie_breaker_quorum = (2 * selection.get_num_tie_breaker_nodes())
        .div_ceil(3)
        .max(1);

    let mut verifiers: Vec<(usize, u64)> = vec![];
    let mut tie_breakers: Vec<(usize, u64)> = vec![];
    let mut target: Option<(usize, Pubkey, u64)> = None;
    for (epoch_index, client) in
        state.coordinator.epoch_state.clients.iter().enumerate()
    {
        let signer = *client.id.signer();
        let client_idx =
            clients.iter().position(|k| k.pubkey().to_bytes() == signer);
        let Some(client_idx) = client_idx else {
            continue;
        };
        match selection.get_committee(epoch_index as u64).committee {
            Committee::Verifier => {
                verifiers.push((client_idx, epoch_index as u64))
            },
            Committee::TieBreaker => {
                tie_breakers.push((client_idx, epoch_index as u64))
            },
            Committee::Trainer => {
                if target.is_none() {
                    target = Some((
                        client_idx,
                        clients[client_idx].pubkey(),
                        epoch_index as u64,
                    ));
                }
            },
        }
    }

    let (target_client_idx, target_key, target_index) =
        target.expect("no trainer target found");
    let target_commitment = signed_payload(
        &state.coordinator,
        TIE_BREAKER_SIZE,
        &identity(&clients[target_client_idx], &p2ps[target_client_idx]),
        &p2ps[target_client_idx],
        FRAUD_PAYLOAD,
    );
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
        p2ps,
        clients_collateral,
        verifiers,
        tie_breakers,
        target_client_idx,
        target_key,
        target_index,
        target_commitment,
        verifier_quorum,
        tie_breaker_quorum,
    }
}

impl AppealsHarness {
    pub(super) async fn cast_verdict(
        &mut self,
        client_idx: usize,
    ) -> anyhow::Result<()> {
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
            &self.target_commitment,
            replayed_digest(),
        )
        .await
    }

    pub(super) async fn reach_verifier_quorum(&mut self) {
        let quorum = self.verifier_quorum as usize;
        let voters: Vec<usize> = self
            .verifiers
            .iter()
            .take(quorum)
            .map(|(i, _)| *i)
            .collect();
        for client_idx in voters {
            self.cast_verdict(client_idx).await.unwrap();
        }
    }

    pub(super) async fn cast_appeal(
        &mut self,
        client_idx: usize,
        overturn: bool,
    ) -> anyhow::Result<()> {
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
    pub(super) async fn request_withdraw(
        &mut self,
        client_idx: usize,
        amount: u64,
    ) {
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
    pub(super) async fn try_finalize_withdraw(
        &mut self,
        client_idx: usize,
    ) -> anyhow::Result<()> {
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
    /// authority granted it, and the run is ticked into that epoch's warmup,
    /// paced by the run's own config.
    pub(super) async fn rejoin_all(&mut self) {
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
                identity(&client, &self.p2ps[client_idx]),
            )
            .await
            .unwrap();
        }
        // The last join already ticks the run into its warmup once enough
        // clients are pending; tick it there otherwise.
        for _ in 0..8 {
            match self.run_state().await {
                RunState::Warmup => return,
                RunState::WaitingForMembers => {
                    let config = self.config().await;
                    self.tick_after(
                        config.waiting_for_members_extra_time as u64,
                    )
                    .await;
                },
                other => {
                    panic!("the run left for {other} instead of its warmup")
                },
            }
        }
        panic!("the run never reached its warmup");
    }

    /// As `rejoin_all`, and on through the warmup into the first round.
    pub(super) async fn rejoin_all_and_start_epoch(&mut self) {
        self.rejoin_all().await;
        let config = self.config().await;
        self.tick_after(config.warmup_time).await;
        assert_eq!(self.run_state().await, RunState::RoundTrain);
    }

    pub(super) async fn config(&mut self) -> CoordinatorConfig {
        get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap()
        .coordinator
        .config
    }

    pub(super) async fn run_state(&mut self) -> RunState {
        get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap()
        .coordinator
        .run_state
    }

    pub(super) async fn tick_after(&mut self, seconds: u64) {
        self.endpoint
            .forward_clock_unix_timestamp(seconds)
            .await
            .unwrap();
        process_coordinator_tick_and_reveal(
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
    pub(super) async fn finish_epoch_with_witnesses(&mut self) {
        let start = self.epoch().await;
        self.advance_with_witnesses(|coordinator| {
            coordinator.progress.epoch > start
                && coordinator.run_state == RunState::WaitingForMembers
        })
        .await;
    }

    /// Runs the epoch the same way up to its `Cooldown`, the last state in
    /// which its ejections are charged. The committee drawn for the last round
    /// is still the one in force there, and stays in force into
    /// `WaitingForMembers` until the next epoch's first round draws another.
    pub(super) async fn advance_to_cooldown_with_witnesses(&mut self) {
        self.advance_with_witnesses(|coordinator| {
            coordinator.run_state == RunState::Cooldown
        })
        .await;
    }

    /// Runs the epoch on to its final round's witness phase: the round whose
    /// end ends the epoch.
    pub(super) async fn advance_to_final_round_witness(&mut self) {
        self.advance_with_witnesses(|coordinator| {
            coordinator.run_state == RunState::RoundWitness
                && coordinator.epoch_state.last_step_set()
                && coordinator.progress.step + 1
                    == coordinator.epoch_state.last_step
        })
        .await;
    }

    /// Runs the swarm on into the next round's training, in this epoch or the
    /// next, so a committee drawn for another round is in force.
    pub(super) async fn advance_to_next_round(&mut self) {
        let state = get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap();
        let start = (
            state.coordinator.progress.epoch,
            state.coordinator.current_round().map(|round| round.height),
        );
        self.advance_with_witnesses(|coordinator| {
            coordinator.run_state == RunState::RoundTrain
                && (
                    coordinator.progress.epoch,
                    coordinator.current_round().map(|round| round.height),
                ) != start
        })
        .await;
    }

    pub(super) async fn epoch(&mut self) -> u16 {
        get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap()
        .coordinator
        .progress
        .epoch
    }

    pub(super) async fn advance_with_witnesses<F>(&mut self, done: F)
    where
        F: Fn(&psyche_coordinator::Coordinator) -> bool,
    {
        for _ in 0..1_000 {
            let state = get_coordinator_account_state(
                &mut self.endpoint,
                &self.coordinator_account,
            )
            .await
            .unwrap()
            .unwrap();
            let coordinator = state.coordinator;
            let config = coordinator.config;
            if done(&coordinator) {
                return;
            }
            match coordinator.run_state {
                RunState::RoundTrain => {
                    let selection =
                        CommitteeSelection::from_coordinator(&coordinator, 0)
                            .unwrap();
                    let roster: Vec<NodeIdentity> = coordinator
                        .epoch_state
                        .clients
                        .iter()
                        .map(|c| c.id)
                        .collect();
                    for (index, id) in roster.iter().enumerate() {
                        let proof = selection.get_witness(index as u64);
                        if !bool::from(proof.witness) {
                            continue;
                        }
                        let Some(client_idx) =
                            self.clients.iter().position(|k| {
                                k.pubkey().to_bytes() == *id.signer()
                            })
                        else {
                            continue;
                        };
                        let mut participant_bloom = WitnessBloom::random(
                            roster.len(),
                            BLOOM_FALSE_RATE,
                        );
                        for participant in &roster {
                            participant_bloom
                                .add(&sha256(participant.signer()));
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
                                broadcast_bloom: WitnessBloom::random(
                                    1,
                                    BLOOM_FALSE_RATE,
                                ),
                                broadcast_merkle: Default::default(),
                                metadata: Default::default(),
                            },
                        )
                        .await
                        .unwrap();
                    }
                    let now = get_coordinator_account_state(
                        &mut self.endpoint,
                        &self.coordinator_account,
                    )
                    .await
                    .unwrap()
                    .unwrap()
                    .coordinator
                    .run_state;
                    if now == RunState::RoundTrain {
                        self.tick_after(config.max_round_train_time).await;
                    }
                },
                RunState::RoundWitness => {
                    self.tick_after(config.round_witness_time).await
                },
                RunState::Cooldown => {
                    self.tick_after(config.cooldown_time).await
                },
                RunState::Warmup => self.tick_after(config.warmup_time).await,
                RunState::WaitingForMembers => {
                    self.tick_after(
                        config.waiting_for_members_extra_time as u64,
                    )
                    .await
                },
                halted => panic!("the run halted in {halted}"),
            }
        }
        panic!("the run never got where it was driven");
    }

    /// The committee in force for the current round, as harness indices:
    /// verifiers, tie-breakers and trainers, with the two quorums the treasurer
    /// will require.
    pub(super) async fn committee_now(&mut self) -> CommitteeNow {
        let state = get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap();
        // The tie-breaker count is the run's, as the treasurer reads it.
        let tie_breakers = get_run(&mut self.endpoint, &self.run)
            .await
            .unwrap()
            .unwrap()
            .tie_breaker_committee_size;
        let selection = CommitteeSelection::from_coordinator_with_tie_breakers(
            &state.coordinator,
            0,
            tie_breakers,
        )
        .unwrap();
        let mut committee = CommitteeNow {
            verifiers: vec![],
            tie_breakers: vec![],
            trainers: vec![],
            verifier_quorum: (2 * selection.get_num_verifier_nodes())
                .div_ceil(3)
                .max(2) as usize,
            tie_breaker_quorum: (2 * selection.get_num_tie_breaker_nodes())
                .div_ceil(3)
                .max(1) as usize,
        };
        for (index, client) in
            state.coordinator.epoch_state.clients.iter().enumerate()
        {
            let Some(client_idx) = self
                .clients
                .iter()
                .position(|k| k.pubkey().to_bytes() == *client.id.signer())
            else {
                continue;
            };
            match selection.get_committee(index as u64).committee {
                Committee::Verifier => {
                    committee.verifiers.push((client_idx, index as u64))
                },
                Committee::TieBreaker => {
                    committee.tie_breakers.push((client_idx, index as u64))
                },
                Committee::Trainer => {
                    committee.trainers.push((client_idx, index as u64))
                },
            }
        }
        committee
    }

    /// What `client_idx` signed for the batch it trained in the round the
    /// coordinator is in.
    pub(super) async fn signed_payload_of(
        &mut self,
        client_idx: usize,
    ) -> SignedCommitment {
        let state = get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap();
        let tie_breakers = get_run(&mut self.endpoint, &self.run)
            .await
            .unwrap()
            .unwrap()
            .tie_breaker_committee_size;
        signed_payload(
            &state.coordinator,
            tie_breakers,
            &identity(&self.clients[client_idx], &self.p2ps[client_idx]),
            &self.p2ps[client_idx],
            FRAUD_PAYLOAD,
        )
    }

    /// `voter` votes to convict `target`, a client at roster index
    /// `target_index`, on the payload it signed in the round under way.
    pub(super) async fn convict_by(
        &mut self,
        voter: usize,
        target: usize,
        target_index: u64,
    ) -> anyhow::Result<()> {
        let commitment = self.signed_payload_of(target).await;
        self.convict_on(voter, target, target_index, &commitment)
            .await
    }

    /// `voter` votes to convict `target` on `commitment`.
    pub(super) async fn convict_on(
        &mut self,
        voter: usize,
        target: usize,
        target_index: u64,
        commitment: &SignedCommitment,
    ) -> anyhow::Result<()> {
        let verifier = self.clients[voter].insecure_clone();
        let target_key = self.clients[target].pubkey();
        process_treasurer_run_submit_audit_verdict(
            &mut self.endpoint,
            &self.payer,
            &verifier,
            &self.run,
            &self.coordinator_account,
            &self.run_id,
            &target_key,
            target_index,
            commitment,
            replayed_digest(),
        )
        .await
    }

    /// `appellate` rules on `target`'s appeal.
    pub(super) async fn rule_by(
        &mut self,
        appellate: usize,
        target: usize,
        target_index: u64,
        overturn: bool,
    ) -> anyhow::Result<()> {
        let appellate = self.clients[appellate].insecure_clone();
        let target_key = self.clients[target].pubkey();
        process_treasurer_run_submit_appeal_verdict(
            &mut self.endpoint,
            &self.payer,
            &appellate,
            &self.run,
            &self.coordinator_account,
            &self.run_id,
            &target_key,
            target_index,
            overturn,
        )
        .await
    }

    pub(super) async fn verdict_status_of(
        &mut self,
        target: usize,
    ) -> VerdictStatus {
        let key = self.clients[target].pubkey();
        let account = self
            .endpoint
            .get_account(&find_audit_verdict(&self.run, &key))
            .await
            .unwrap()
            .unwrap();
        AuditVerdict::from_pda_account(&account.owner, &account.data)
            .unwrap()
            .unwrap()
            .status
    }

    pub(super) async fn finalize_withdraw_paying(
        &mut self,
        client_idx: usize,
        recipients: &[usize],
    ) -> anyhow::Result<()> {
        let client = self.clients[client_idx].insecure_clone();
        let collateral = self.clients_collateral[client_idx];
        let recipient_collaterals: Vec<Pubkey> = recipients
            .iter()
            .map(|i| self.clients_collateral[*i])
            .collect();
        process_treasurer_participant_bond_finalize_withdraw_with_voters(
            &mut self.endpoint,
            &self.payer,
            &client,
            &collateral,
            &self.collateral_mint,
            &self.run,
            &self.coordinator_account,
            &recipient_collaterals,
        )
        .await
    }

    /// Where `key` sits in the current epoch's roster.
    pub(super) async fn roster_index_of(&mut self, key: &Pubkey) -> u64 {
        get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
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
    pub(super) fn convicting_verifier_collaterals(&self) -> Vec<Pubkey> {
        self.verifiers
            .iter()
            .take(self.verifier_quorum as usize)
            .map(|(client_idx, _)| self.clients_collateral[*client_idx])
            .collect()
    }

    /// The standing run-manager reads before it sends a finalize: the run, its
    /// coordinator account, the client's participant and own verdict accounts
    /// and the cluster clock sysvar, decoded the way the treasurer and the
    /// backend decode them.
    pub(super) async fn observed_standing(
        &mut self,
        client_idx: usize,
    ) -> WithdrawalStanding {
        let user = self.clients[client_idx].pubkey();
        let run = get_run(&mut self.endpoint, &self.run)
            .await
            .unwrap()
            .unwrap();
        let participant = self.participant_of(client_idx).await;
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
            Some(account) => {
                AuditVerdict::from_pda_account(&account.owner, &account.data)
                    .unwrap()
            },
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
            &participant,
            own_verdict.as_ref(),
            &user,
            now,
        )
    }

    /// `client_idx`'s participant account.
    pub(super) async fn participant_of(
        &mut self,
        client_idx: usize,
    ) -> Participant {
        let user = self.clients[client_idx].pubkey();
        get_participant(&mut self.endpoint, &find_participant(&self.run, &user))
            .await
            .unwrap()
            .unwrap()
    }

    /// `target`'s verdict account, which a dispute against it has opened.
    pub(super) async fn verdict_of(&mut self, target: &Pubkey) -> AuditVerdict {
        get_audit_verdict(&mut self.endpoint, &self.run, target)
            .await
            .unwrap()
            .expect("a dispute was opened against the target")
    }

    /// The tie-breakers whose rulings reach the bench's quorum.
    pub(super) fn bench(&self) -> Vec<usize> {
        self.tie_breakers
            .iter()
            .take(self.tie_breaker_quorum as usize)
            .map(|(client_idx, _)| *client_idx)
            .collect()
    }

    /// The verifiers whose votes reach the verdict quorum.
    pub(super) fn quorum_voters(&self) -> Vec<usize> {
        self.verifiers
            .iter()
            .take(self.verifier_quorum as usize)
            .map(|(client_idx, _)| *client_idx)
            .collect()
    }

    /// The verifiers convict the target, it appeals, and the bench overturns.
    pub(super) async fn convict_and_overturn(&mut self) {
        self.reach_verifier_quorum().await;
        let target = self.clients[self.target_client_idx].insecure_clone();
        process_treasurer_run_open_challenge(
            &mut self.endpoint,
            &self.payer,
            &target,
            &self.run,
        )
        .await
        .unwrap();
        for appellate in self.bench() {
            self.cast_appeal(appellate, true).await.unwrap();
        }
        assert_eq!(self.verdict_status().await, VerdictStatus::Overturned);
    }

    /// Anyone cranks the losing verifier at `position` of the target's verdict,
    /// paying the bench out of its forfeit.
    pub(super) async fn crank_loser(
        &mut self,
        position: u16,
    ) -> anyhow::Result<()> {
        let target_key = self.target_key;
        process_treasurer_run_slash_losing_verifier(
            &mut self.endpoint,
            &self.payer,
            &self.run,
            &self.coordinator_account,
            &target_key,
            position,
        )
        .await
    }

    /// Anyone pays the tie-breaker at `seat` of the target's overturned
    /// verdict its share of the losers' forfeits.
    pub(super) async fn pay_bench_seat(
        &mut self,
        seat: u16,
    ) -> anyhow::Result<()> {
        let target_key = self.target_key;
        process_treasurer_run_pay_bench_bounty(
            &mut self.endpoint,
            &self.payer,
            &self.run,
            &target_key,
            seat,
        )
        .await
    }

    /// `authority` (the payer when `None`) releases `voter`'s vote on the
    /// verdict against `target`.
    pub(super) async fn release_vote(
        &mut self,
        authority: Option<usize>,
        target: &Pubkey,
        voter: usize,
    ) -> anyhow::Result<()> {
        let authority = match authority {
            Some(client_idx) => self.clients[client_idx].insecure_clone(),
            None => self.payer.insecure_clone(),
        };
        let voter_key = self.clients[voter].pubkey();
        process_treasurer_participant_release_verdict_vote(
            &mut self.endpoint,
            &self.payer,
            &authority,
            &self.run,
            &self.coordinator_account,
            target,
            &voter_key,
        )
        .await
    }

    pub(super) async fn verdict_status(&mut self) -> VerdictStatus {
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
    pub(super) async fn finalize_target_withdrawal(
        &mut self,
    ) -> anyhow::Result<()> {
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

    pub(super) async fn collateral_of(&mut self, client_idx: usize) -> u64 {
        let account = self.clients_collateral[client_idx];
        self.endpoint
            .get_spl_token_account(&account)
            .await
            .unwrap()
            .unwrap()
            .amount
    }

    pub(super) async fn client_state(
        &mut self,
        key: &Pubkey,
    ) -> Option<ClientState> {
        let state = get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
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

    pub(super) async fn settle_epoch(&mut self) {
        for _ in 0..15 {
            self.endpoint
                .forward_clock_unix_timestamp(60)
                .await
                .unwrap();
            let _ = process_coordinator_tick_and_reveal(
                &mut self.endpoint,
                &self.payer,
                &self.ticker,
                &self.coordinator_instance,
                &self.coordinator_account,
            )
            .await;
        }
    }

    pub(super) async fn slashed_of(&mut self, key: &Pubkey) -> u64 {
        let state = get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
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
