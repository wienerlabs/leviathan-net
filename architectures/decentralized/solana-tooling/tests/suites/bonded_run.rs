//! A bond-gated treasurer run on the real genesis rehearsal config, driven
//! through whole epochs on the real programs. Shared by the suites that check
//! what a bond allows: joining (wienerlabs/leviathan-net#34) and leaving
//! (wienerlabs/leviathan-net#36).

use psyche_coordinator::model::Model;
use psyche_coordinator::get_data_index_for_step;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::CoordinatorProgress;
use psyche_coordinator::RunState;
use psyche_coordinator::WitnessBloom;
use psyche_coordinator::BLOOM_FALSE_RATE;
use psyche_core::sha256;
use psyche_core::NodeIdentity;
use psyche_solana_authorizer::find_authorization;
use psyche_solana_coordinator::instruction::Witness;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::get_accounts::get_authorization;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick_and_reveal;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_witness;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_claim;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_join_run;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::find_run;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_treasurer::state::Participant;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::signed_payload::replayed_digest;
use super::signed_payload::signed_payload;
use super::signed_payload::FRAUD_PAYLOAD;

/// The bond a node must keep posted to be admitted, and how long a withdrawal
/// waits so a conviction can land first.
pub const BOND: u64 = 500;
pub const WITHDRAW_DELAY: i64 = 100;
/// The per-epoch economics the genesis rehearsal is stood up with
/// (`config/leviathan/README.md`). A slash of 200 points leaves a 500 bond
/// below its own minimum until it is topped up.
pub const EARNING_RATE_TOTAL_SHARED: u64 = 1_000;
pub const SLASHING_RATE: u64 = 200;
/// Two honest nodes keep the run at the genesis config's `min_clients` while
/// the third is the one whose seat is in question.
pub const NODES: usize = 3;
pub const SUBJECT: usize = 2;
/// A run that has not reached its next epoch in this many transitions is
/// stuck, and the suite fails instead of spinning.
pub const MAX_TRANSITIONS: usize = 1_000;

#[derive(serde::Deserialize)]
pub struct RunConfig {
    pub config: CoordinatorConfig,
    pub model: Model,
}

fn genesis_rehearsal() -> RunConfig {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../config/leviathan/genesis-rehearsal.toml"
    );
    let text = std::fs::read_to_string(path)
        .expect("genesis rehearsal config is readable");
    toml::from_str(&text).expect("genesis rehearsal config parses")
}

/// Who admits nodes to the run.
pub enum JoinGate {
    /// The run PDA is its own join authority: admission is against the bond.
    Bond,
    /// An outside key admits nodes, the way the devnet flagship run does.
    Outside(Pubkey),
}

pub struct Node {
    pub wallet: Keypair,
    pub p2p: Keypair,
    pub collateral: Pubkey,
}

impl Node {
    pub fn identity(&self) -> NodeIdentity {
        NodeIdentity::new(
            self.wallet.pubkey().to_bytes(),
            self.p2p.pubkey().to_bytes(),
        )
    }
}

pub struct Harness {
    pub endpoint: ToolboxEndpoint,
    pub payer: Keypair,
    pub main_authority: Keypair,
    pub mint_authority: Keypair,
    pub ticker: Keypair,
    pub collateral_mint: Pubkey,
    pub run_id: String,
    pub run: Pubkey,
    pub run_collateral: Pubkey,
    pub coordinator_instance: Pubkey,
    pub coordinator_account: Pubkey,
    pub config: CoordinatorConfig,
    pub nodes: Vec<Node>,
}

impl Harness {
    /// A run on the genesis rehearsal config with `NODES` participants that
    /// have each bonded `BOND` and hold one more `BOND` they can post later.
    pub async fn new(index: u64, run_id: &str, gate: JoinGate) -> Harness {
        Harness::with(index, run_id, gate, NODES, |_| {}).await
    }

    /// The same run with `nodes` participants and the genesis rehearsal config
    /// changed by `adjust`.
    pub async fn with(
        index: u64,
        run_id: &str,
        gate: JoinGate,
        nodes: usize,
        adjust: impl FnOnce(&mut CoordinatorConfig),
    ) -> Harness {
        let mut endpoint = create_memnet_endpoint().await;
        let payer = Keypair::new();
        endpoint
            .request_airdrop(&payer.pubkey(), 10_000_000_000)
            .await
            .unwrap();
        let mint_authority = Keypair::new();
        let main_authority = Keypair::new();
        let ticker = Keypair::new();
        let mut rehearsal = genesis_rehearsal();
        adjust(&mut rehearsal.config);

        let collateral_mint = endpoint
            .process_spl_token_mint_new(
                &payer,
                &mint_authority.pubkey(),
                None,
                0,
            )
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

        let join_authority = match gate {
            JoinGate::Bond => find_run(index),
            JoinGate::Outside(authority) => authority,
        };
        let (run, coordinator_instance) = process_treasurer_run_create(
            &mut endpoint,
            &payer,
            &collateral_mint,
            &coordinator_account,
            RunCreateParams {
                index,
                run_id: run_id.to_string(),
                main_authority: main_authority.pubkey(),
                join_authority,
                client_version: "latest".to_string(),
            },
        )
        .await
        .unwrap();
        let run_collateral = ToolboxEndpoint::find_spl_associated_token_account(
            &run,
            &collateral_mint,
        );
        process_treasurer_run_bond_config_update(
            &mut endpoint,
            &payer,
            &main_authority,
            &run,
            RunBondConfigUpdateParams {
                bond_minimum_amount: BOND,
                bond_withdraw_delay_seconds: WITHDRAW_DELAY,
            },
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
                config: Some(rehearsal.config),
                model: Some(rehearsal.model),
                progress: None,
                epoch_earning_rate_total_shared: Some(
                    EARNING_RATE_TOTAL_SHARED,
                ),
                epoch_slashing_rate_per_client: Some(SLASHING_RATE),
                paused: Some(false),
                client_version: None,
            },
        )
        .await
        .unwrap();

        let mut harness = Harness {
            endpoint,
            payer,
            main_authority,
            mint_authority,
            ticker,
            collateral_mint,
            run_id: run_id.to_string(),
            run,
            run_collateral,
            coordinator_instance,
            coordinator_account,
            config: rehearsal.config,
            nodes: Vec::new(),
        };
        for _ in 0..nodes {
            let node = harness.add_node().await;
            harness.deposit(node, BOND).await.unwrap();
        }
        harness
    }

    /// A participant of the run holding `2 * BOND` of collateral, none of it
    /// posted yet.
    pub async fn add_node(&mut self) -> usize {
        let wallet = Keypair::new();
        process_treasurer_participant_create(
            &mut self.endpoint,
            &self.payer,
            &wallet,
            &self.run,
        )
        .await
        .unwrap();
        let collateral = self.fund(&wallet.pubkey(), 2 * BOND).await;
        self.nodes.push(Node {
            wallet,
            p2p: Keypair::new(),
            collateral,
        });
        self.nodes.len() - 1
    }

    pub async fn fund(&mut self, owner: &Pubkey, amount: u64) -> Pubkey {
        let account = self
            .endpoint
            .process_spl_associated_token_account_get_or_init(
                &self.payer,
                owner,
                &self.collateral_mint,
            )
            .await
            .unwrap();
        self.endpoint
            .process_spl_token_mint_to(
                &self.payer,
                &self.collateral_mint,
                &self.mint_authority,
                &account,
                amount,
            )
            .await
            .unwrap();
        account
    }

    /// Reward liquidity, paid into the run vault by a funder the way
    /// `run-manager treasurer-top-up-rewards` does, so that claims are not paid
    /// out of the bonds sitting in the same vault.
    pub async fn top_up_rewards(&mut self, amount: u64) {
        let funder = Keypair::new();
        let funder_collateral = self.fund(&funder.pubkey(), amount).await;
        self.endpoint
            .process_spl_token_transfer(
                &self.payer,
                &funder,
                &funder_collateral,
                &self.run_collateral,
                amount,
            )
            .await
            .unwrap();
    }

    pub async fn deposit(
        &mut self,
        node: usize,
        amount: u64,
    ) -> anyhow::Result<()> {
        process_treasurer_participant_bond_deposit(
            &mut self.endpoint,
            &self.payer,
            &self.nodes[node].wallet,
            &self.nodes[node].collateral,
            &self.collateral_mint,
            &self.run,
            amount,
        )
        .await
    }

    pub async fn request_withdraw(&mut self, node: usize, amount: u64) {
        process_treasurer_participant_bond_request_withdraw(
            &mut self.endpoint,
            &self.payer,
            &self.nodes[node].wallet,
            &self.run,
            amount,
        )
        .await
        .unwrap();
    }

    pub async fn try_finalize_withdraw(
        &mut self,
        node: usize,
    ) -> anyhow::Result<()> {
        process_treasurer_participant_bond_finalize_withdraw(
            &mut self.endpoint,
            &self.payer,
            &self.nodes[node].wallet,
            &self.nodes[node].collateral,
            &self.collateral_mint,
            &self.run,
            &self.coordinator_account,
        )
        .await
    }

    pub async fn finalize_withdraw(&mut self, node: usize) {
        self.try_finalize_withdraw(node).await.unwrap();
    }

    pub async fn claim(
        &mut self,
        node: usize,
        points: u64,
    ) -> anyhow::Result<()> {
        process_treasurer_participant_claim(
            &mut self.endpoint,
            &self.payer,
            &self.nodes[node].wallet,
            &self.nodes[node].collateral,
            &self.collateral_mint,
            &self.run,
            &self.coordinator_account,
            points,
        )
        .await
    }

    /// Joins the next epoch the way a node of a bond-gated run does: through
    /// the treasurer.
    pub async fn join(&mut self, node: usize) -> anyhow::Result<()> {
        let identity = self.nodes[node].identity();
        process_treasurer_participant_join_run(
            &mut self.endpoint,
            &self.payer,
            &self.nodes[node].wallet,
            &self.run,
            identity,
        )
        .await
        .map(|_| ())
    }

    /// Joins by calling the coordinator directly with the node's treasurer-made
    /// authorization, skipping the treasurer.
    pub async fn join_directly(&mut self, node: usize) -> anyhow::Result<()> {
        let identity = self.nodes[node].identity();
        let authorization = self.authorization_of(node);
        process_coordinator_join_run(
            &mut self.endpoint,
            &self.payer,
            &self.nodes[node].wallet,
            &authorization,
            &self.coordinator_instance,
            &self.coordinator_account,
            identity,
        )
        .await
    }

    pub fn authorization_of(&self, node: usize) -> Pubkey {
        find_authorization(
            &self.run,
            &self.nodes[node].wallet.pubkey(),
            JOIN_RUN_AUTHORIZATION_SCOPE,
        )
    }

    pub async fn state(&mut self) -> CoordinatorInstanceState {
        get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap()
    }

    pub async fn tick_after(&mut self, seconds: u64) {
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

    /// Every witness drawn this round reports the participants it heard from,
    /// which in this harness is the whole epoch roster.
    pub async fn witness_round(&mut self) {
        let state = self.state().await;
        let coordinator = &state.coordinator;
        let selection =
            CommitteeSelection::from_coordinator(coordinator, 0).unwrap();
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
            let Some(node) =
                self.nodes.iter().position(|n| n.identity() == *id)
            else {
                continue;
            };
            let mut participant_bloom =
                WitnessBloom::random(roster.len(), BLOOM_FALSE_RATE);
            for participant in &roster {
                participant_bloom.add(&sha256(participant.signer()));
            }
            process_coordinator_witness(
                &mut self.endpoint,
                &self.payer,
                &self.nodes[node].wallet,
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
    }

    /// One coordinator transition, paced by the run's own config.
    pub async fn step(&mut self) {
        let run_state = self.state().await.coordinator.run_state;
        match run_state {
            RunState::WaitingForMembers => {
                self.tick_after(
                    self.config.waiting_for_members_extra_time as u64,
                )
                .await
            },
            RunState::Warmup => self.tick_after(self.config.warmup_time).await,
            RunState::RoundTrain => {
                self.witness_round().await;
                if self.state().await.coordinator.run_state
                    == RunState::RoundTrain
                {
                    self.tick_after(self.config.max_round_train_time).await;
                }
            },
            RunState::RoundWitness => {
                self.tick_after(self.config.round_witness_time).await
            },
            RunState::Cooldown => {
                self.tick_after(self.config.cooldown_time).await
            },
            halted => panic!("the run halted in {halted}"),
        }
    }

    pub async fn advance_to(&mut self, target: RunState) {
        for _ in 0..MAX_TRANSITIONS {
            if self.state().await.coordinator.run_state == target {
                return;
            }
            self.step().await;
        }
        panic!("the run never reached {target}");
    }

    /// Runs the current epoch to its end and back to `WaitingForMembers`.
    pub async fn finish_epoch(&mut self) {
        let epoch = self.state().await.coordinator.progress.epoch;
        for _ in 0..MAX_TRANSITIONS {
            let coordinator = self.state().await.coordinator;
            if coordinator.progress.epoch > epoch
                && coordinator.run_state == RunState::WaitingForMembers
            {
                return;
            }
            self.step().await;
        }
        panic!("epoch {epoch} never ended");
    }

    /// The nodes training in the current epoch, by harness index.
    pub async fn roster(&mut self) -> Vec<usize> {
        let ids: Vec<NodeIdentity> = self
            .state()
            .await
            .coordinator
            .epoch_state
            .clients
            .iter()
            .map(|c| c.id)
            .collect();
        ids.iter()
            .filter_map(|id| {
                self.nodes.iter().position(|n| n.identity() == *id)
            })
            .collect()
    }

    pub async fn roster_index(&mut self, node: usize) -> u64 {
        let identity = self.nodes[node].identity();
        self.state()
            .await
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|c| c.id == identity)
            .expect("the node is in this epoch's roster") as u64
    }

    pub async fn ledger(&mut self, node: usize) -> (u64, u64) {
        let signer = self.nodes[node].wallet.pubkey().to_bytes();
        self.state()
            .await
            .clients_state
            .clients
            .iter()
            .find(|c| *c.id.signer() == signer)
            .map(|c| (c.earned, c.slashed))
            .unwrap_or((0, 0))
    }

    pub async fn participant(&mut self, node: usize) -> Participant {
        let participant =
            find_participant(&self.run, &self.nodes[node].wallet.pubkey());
        get_participant(&mut self.endpoint, &participant)
            .await
            .unwrap()
            .unwrap()
    }

    pub async fn bond(&mut self, node: usize) -> u64 {
        self.participant(node).await.bond_amount
    }

    /// Another bond-gated run on the same chain, with coordinator accounts of
    /// its own.
    pub async fn add_other_run(
        &mut self,
        index: u64,
        run_id: &str,
    ) -> (Pubkey, Pubkey) {
        let coordinator_account = self
            .endpoint
            .process_system_new_exempt(
                &self.payer,
                CoordinatorAccount::space_with_discriminator(),
                &psyche_solana_coordinator::ID,
            )
            .await
            .unwrap();
        let (_, coordinator_instance) = process_treasurer_run_create(
            &mut self.endpoint,
            &self.payer,
            &self.collateral_mint,
            &coordinator_account,
            RunCreateParams {
                index,
                run_id: run_id.to_string(),
                main_authority: self.main_authority.pubkey(),
                join_authority: find_run(index),
                client_version: "latest".to_string(),
            },
        )
        .await
        .unwrap();
        (coordinator_instance, coordinator_account)
    }

    pub async fn collateral(&mut self, node: usize) -> u64 {
        let account = self.nodes[node].collateral;
        self.endpoint
            .get_spl_token_account(&account)
            .await
            .unwrap()
            .unwrap()
            .amount
    }

    pub async fn authorization_active(&mut self, node: usize) -> Option<bool> {
        let authorization = self.authorization_of(node);
        get_authorization(&mut self.endpoint, &authorization)
            .await
            .unwrap()
            .map(|authorization| authorization.active)
    }

    /// The run authority convicts `node` in the round under way.
    pub async fn convict(&mut self, node: usize) {
        self.try_convict(node).await.unwrap();
    }

    /// The run authority convicts `node` on the payload it signed in the
    /// round under way.
    pub async fn try_convict(&mut self, node: usize) -> anyhow::Result<()> {
        let commitment = self.signed_payload(node).await;
        self.try_convict_on(node, &commitment).await
    }

    pub async fn try_convict_on(
        &mut self,
        node: usize,
        commitment: &SignedCommitment,
    ) -> anyhow::Result<()> {
        let index = self.roster_index(node).await;
        process_treasurer_run_slash(
            &mut self.endpoint,
            &self.payer,
            &self.main_authority,
            &self.run,
            &self.coordinator_account,
            &self.run_id,
            index,
            commitment,
            replayed_digest(),
        )
        .await
    }

    /// What `node` signed for the cheating payload it broadcast in the round
    /// under way.
    pub async fn signed_payload(&mut self, node: usize) -> SignedCommitment {
        let tie_breakers = get_run(&mut self.endpoint, &self.run)
            .await
            .unwrap()
            .unwrap()
            .tie_breaker_committee_size;
        let coordinator = self.state().await.coordinator;
        signed_payload(
            &coordinator,
            tie_breakers,
            &self.nodes[node].identity(),
            &self.nodes[node].p2p,
            FRAUD_PAYLOAD,
        )
    }

    /// The run authority pauses the run.
    pub async fn pause(&mut self) {
        self.update_run(None, Some(true)).await;
    }

    /// The run authority restarts the run from `step`, the way
    /// `run-manager update-config --restart-from-step` does: it pauses the
    /// run, sets its progress back to `step` and resumes it. The nodes then
    /// join and train from `step` again, in a new epoch.
    pub async fn restart_from_step(&mut self, step: u32) {
        self.pause().await;
        let coordinator = self.state().await.coordinator;
        assert_eq!(coordinator.run_state, RunState::Paused);
        let progress = CoordinatorProgress {
            epoch: coordinator.progress.epoch,
            step,
            epoch_start_data_index: get_data_index_for_step(&coordinator, step),
        };
        self.update_run(Some(progress), None).await;
        self.update_run(None, Some(false)).await;
    }

    async fn update_run(
        &mut self,
        progress: Option<CoordinatorProgress>,
        paused: Option<bool>,
    ) {
        process_treasurer_run_update(
            &mut self.endpoint,
            &self.payer,
            &self.main_authority,
            &self.run,
            &self.coordinator_instance,
            &self.coordinator_account,
            RunUpdateParams {
                metadata: None,
                config: None,
                model: None,
                progress,
                epoch_earning_rate_total_shared: None,
                epoch_slashing_rate_per_client: None,
                paused,
                client_version: None,
            },
        )
        .await
        .unwrap();
    }
}

/// Every node joins and trains one whole epoch.
pub async fn first_epoch(harness: &mut Harness) {
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.finish_epoch().await;
}
