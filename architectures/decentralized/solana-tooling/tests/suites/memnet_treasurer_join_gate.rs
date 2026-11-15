//! The join gate of a bond-gated run has to hold on every join, not only the
//! first one (wienerlabs/leviathan-net#34).
//!
//! A run whose join authority is its own treasurer `Run` PDA admits a node only
//! against a posted bond. These suites drive the real programs through whole
//! epochs on the real genesis rehearsal config, and check the gate each time a
//! node comes back for another epoch.

use anchor_lang::error::ErrorCode as AnchorErrorCode;
use psyche_coordinator::model::Model;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::CoordinatorConfig;
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
use psyche_solana_rpc::instructions::treasurer_participant_join_run;
use psyche_solana_rpc::SolanaBackend;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::get_accounts::get_authorization;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick;
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
use psyche_solana_treasurer::ProgramError;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

/// The bond a node must keep posted to be admitted, and how long a withdrawal
/// waits so a conviction can land first.
const BOND: u64 = 500;
const WITHDRAW_DELAY: i64 = 100;
/// The per-epoch economics the genesis rehearsal is stood up with
/// (`config/leviathan/README.md`). A slash of 200 points leaves a 500 bond
/// below its own minimum until it is topped up.
const EARNING_RATE_TOTAL_SHARED: u64 = 1_000;
const SLASHING_RATE: u64 = 200;
/// Two honest nodes keep the run at the genesis config's `min_clients` while
/// the third is the one whose seat is in question.
const NODES: usize = 3;
const SUBJECT: usize = 2;
/// A run that has not reached its next epoch in this many transitions is
/// stuck, and the suite fails instead of spinning.
const MAX_TRANSITIONS: usize = 1_000;

#[derive(serde::Deserialize)]
struct RunConfig {
    config: CoordinatorConfig,
    model: Model,
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
enum JoinGate {
    /// The run PDA is its own join authority: admission is against the bond.
    Bond,
    /// An outside key admits nodes, the way the devnet flagship run does.
    Outside(Pubkey),
}

struct Node {
    wallet: Keypair,
    p2p: Keypair,
    collateral: Pubkey,
}

impl Node {
    fn identity(&self) -> NodeIdentity {
        NodeIdentity::new(
            self.wallet.pubkey().to_bytes(),
            self.p2p.pubkey().to_bytes(),
        )
    }
}

struct Harness {
    endpoint: ToolboxEndpoint,
    payer: Keypair,
    main_authority: Keypair,
    mint_authority: Keypair,
    ticker: Keypair,
    collateral_mint: Pubkey,
    run_id: String,
    run: Pubkey,
    run_collateral: Pubkey,
    coordinator_instance: Pubkey,
    coordinator_account: Pubkey,
    config: CoordinatorConfig,
    nodes: Vec<Node>,
}

impl Harness {
    /// A run on the genesis rehearsal config with `NODES` participants that
    /// have each bonded `BOND` and hold one more `BOND` they can post later.
    async fn new(index: u64, run_id: &str, gate: JoinGate) -> Harness {
        let mut endpoint = create_memnet_endpoint().await;
        let payer = Keypair::new();
        endpoint
            .request_airdrop(&payer.pubkey(), 10_000_000_000)
            .await
            .unwrap();
        let mint_authority = Keypair::new();
        let main_authority = Keypair::new();
        let ticker = Keypair::new();
        let rehearsal = genesis_rehearsal();

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
        for _ in 0..NODES {
            let node = harness.add_node().await;
            harness.deposit(node, BOND).await.unwrap();
        }
        harness
    }

    /// A participant of the run holding `2 * BOND` of collateral, none of it
    /// posted yet.
    async fn add_node(&mut self) -> usize {
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

    async fn fund(&mut self, owner: &Pubkey, amount: u64) -> Pubkey {
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
    async fn top_up_rewards(&mut self, amount: u64) {
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

    async fn deposit(
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

    async fn request_withdraw(&mut self, node: usize, amount: u64) {
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

    async fn finalize_withdraw(&mut self, node: usize) {
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
        .unwrap();
    }

    async fn claim(&mut self, node: usize, points: u64) -> anyhow::Result<()> {
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
    async fn join(&mut self, node: usize) -> anyhow::Result<()> {
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
    async fn join_directly(&mut self, node: usize) -> anyhow::Result<()> {
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

    fn authorization_of(&self, node: usize) -> Pubkey {
        find_authorization(
            &self.run,
            &self.nodes[node].wallet.pubkey(),
            JOIN_RUN_AUTHORIZATION_SCOPE,
        )
    }

    async fn state(&mut self) -> CoordinatorInstanceState {
        get_coordinator_account_state(
            &mut self.endpoint,
            &self.coordinator_account,
        )
        .await
        .unwrap()
        .unwrap()
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

    /// Every witness drawn this round reports the participants it heard from,
    /// which in this harness is the whole epoch roster.
    async fn witness_round(&mut self) {
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
    async fn step(&mut self) {
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

    async fn advance_to(&mut self, target: RunState) {
        for _ in 0..MAX_TRANSITIONS {
            if self.state().await.coordinator.run_state == target {
                return;
            }
            self.step().await;
        }
        panic!("the run never reached {target}");
    }

    /// Runs the current epoch to its end and back to `WaitingForMembers`.
    async fn finish_epoch(&mut self) {
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
    async fn roster(&mut self) -> Vec<usize> {
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

    async fn roster_index(&mut self, node: usize) -> u64 {
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

    async fn ledger(&mut self, node: usize) -> (u64, u64) {
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

    async fn bond(&mut self, node: usize) -> u64 {
        let participant =
            find_participant(&self.run, &self.nodes[node].wallet.pubkey());
        get_participant(&mut self.endpoint, &participant)
            .await
            .unwrap()
            .unwrap()
            .bond_amount
    }

    async fn collateral(&mut self, node: usize) -> u64 {
        let account = self.nodes[node].collateral;
        self.endpoint
            .get_spl_token_account(&account)
            .await
            .unwrap()
            .unwrap()
            .amount
    }

    async fn authorization_active(&mut self, node: usize) -> Option<bool> {
        let authorization = self.authorization_of(node);
        get_authorization(&mut self.endpoint, &authorization)
            .await
            .unwrap()
            .map(|authorization| authorization.active)
    }

    /// The run authority convicts `node` in the round under way.
    async fn convict(&mut self, node: usize) {
        let index = self.roster_index(node).await;
        process_treasurer_run_slash(
            &mut self.endpoint,
            &self.payer,
            &self.main_authority,
            &self.run,
            &self.coordinator_account,
            &self.run_id,
            index,
        )
        .await
        .unwrap();
    }
}

/// Every node joins and trains one whole epoch.
async fn first_epoch(harness: &mut Harness) {
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.finish_epoch().await;
}

#[tokio::test]
pub async fn a_bonded_node_joins_epoch_after_epoch_through_the_treasurer() {
    let mut harness =
        Harness::new(340, "Leviathan join gate honest", JoinGate::Bond).await;
    for epoch in 0..2 {
        for node in 0..NODES {
            harness.join(node).await.unwrap();
            assert_eq!(
                harness.authorization_active(node).await,
                Some(false),
                "the join authorization stayed open after the join of epoch {epoch}"
            );
        }
        harness.advance_to(RunState::RoundTrain).await;
        assert_eq!(harness.roster().await, (0..NODES).collect::<Vec<_>>());
        harness.finish_epoch().await;
    }
    for node in 0..NODES {
        let (earned, slashed) = harness.ledger(node).await;
        assert_eq!(slashed, 0);
        assert!(
            earned > 0,
            "node {node} trained two epochs and earned nothing"
        );
    }
}

#[tokio::test]
pub async fn a_direct_coordinator_join_is_refused() {
    let mut harness =
        Harness::new(341, "Leviathan join gate direct", JoinGate::Bond).await;
    first_epoch(&mut harness).await;
    assert_eq!(harness.authorization_active(SUBJECT).await, Some(false));
    assert_eq!(
        refusal_code(harness.join_directly(SUBJECT).await),
        u32::from(AnchorErrorCode::ConstraintRaw),
        "the coordinator admitted a node on its treasurer authorization without the treasurer"
    );
    harness.join(SUBJECT).await.unwrap();
}

#[tokio::test]
pub async fn an_unbonded_node_is_refused() {
    let mut harness =
        Harness::new(342, "Leviathan join gate unbonded", JoinGate::Bond).await;
    let unbonded = harness.add_node().await;
    assert_eq!(
        refusal_code(harness.join(unbonded).await),
        treasurer_refusal(ProgramError::BondBelowMinimum)
    );
    assert_eq!(
        harness.authorization_active(unbonded).await,
        None,
        "a refused join still left an authorization behind"
    );
    harness.deposit(unbonded, BOND).await.unwrap();
    harness.join(unbonded).await.unwrap();
}

#[tokio::test]
pub async fn a_withdrawn_bond_does_not_keep_its_seat() {
    let mut harness =
        Harness::new(343, "Leviathan join gate withdrawn", JoinGate::Bond)
            .await;
    first_epoch(&mut harness).await;

    harness.request_withdraw(SUBJECT, BOND).await;
    harness
        .endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    harness.finalize_withdraw(SUBJECT).await;
    assert_eq!(harness.bond(SUBJECT).await, 0);

    harness.join(0).await.unwrap();
    harness.join(1).await.unwrap();
    assert_eq!(
        refusal_code(harness.join(SUBJECT).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a node that withdrew its whole bond was admitted to the bond-gated run again"
    );
    assert_eq!(
        refusal_code(harness.join_directly(SUBJECT).await),
        u32::from(AnchorErrorCode::ConstraintRaw)
    );
    harness.advance_to(RunState::RoundTrain).await;
    assert_eq!(harness.roster().await, vec![0, 1]);
}

#[tokio::test]
pub async fn a_pending_withdrawal_blocks_the_join() {
    let mut harness =
        Harness::new(344, "Leviathan join gate pending", JoinGate::Bond).await;
    first_epoch(&mut harness).await;

    // One unit on its way out is enough: what stays is below the minimum.
    harness.request_withdraw(SUBJECT, 1).await;
    assert_eq!(harness.bond(SUBJECT).await, BOND);
    assert_eq!(
        refusal_code(harness.join(SUBJECT).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a bond already on its way out backed a seat in the next epoch"
    );

    // Posting the shortfall makes the bond that stays whole again.
    harness.deposit(SUBJECT, 1).await.unwrap();
    harness.join(SUBJECT).await.unwrap();
}

#[tokio::test]
pub async fn a_convicted_node_does_not_rejoin_on_its_forfeited_bond() {
    let mut harness =
        Harness::new(345, "Leviathan join gate convicted", JoinGate::Bond)
            .await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness.convict(SUBJECT).await;
    harness.finish_epoch().await;
    assert_eq!(harness.ledger(SUBJECT).await.1, SLASHING_RATE);
    assert_eq!(harness.bond(SUBJECT).await, BOND);

    harness.join(0).await.unwrap();
    harness.join(1).await.unwrap();
    assert_eq!(
        refusal_code(harness.join(SUBJECT).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a convicted node re-joined on a bond its conviction had already forfeited"
    );

    // A fresh bond that covers the forfeit is the price of coming back.
    harness.deposit(SUBJECT, SLASHING_RATE).await.unwrap();
    harness.join(SUBJECT).await.unwrap();
    harness.advance_to(RunState::RoundTrain).await;
    assert_eq!(harness.roster().await, (0..NODES).collect::<Vec<_>>());
}

#[tokio::test]
pub async fn a_claim_needs_the_bond_a_conviction_left() {
    let mut harness =
        Harness::new(346, "Leviathan join gate claim", JoinGate::Bond).await;
    harness.top_up_rewards(2 * EARNING_RATE_TOTAL_SHARED).await;
    first_epoch(&mut harness).await;
    let (earned, _) = harness.ledger(SUBJECT).await;
    assert!(earned > 0);

    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness.convict(SUBJECT).await;
    harness.finish_epoch().await;
    assert_eq!(harness.ledger(SUBJECT).await, (earned, SLASHING_RATE));

    assert_eq!(
        refusal_code(harness.claim(SUBJECT, earned).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a node claimed its earnings against a bond its conviction had already forfeited"
    );

    harness.deposit(SUBJECT, SLASHING_RATE).await.unwrap();
    let before = harness.collateral(SUBJECT).await;
    harness.claim(SUBJECT, earned).await.unwrap();
    assert_eq!(harness.collateral(SUBJECT).await, before + earned);
}

#[tokio::test]
pub async fn a_run_with_an_outside_join_authority_is_not_joined_through_the_treasurer(
) {
    let outside = Keypair::new();
    let mut harness = Harness::new(
        347,
        "Leviathan join gate outside",
        JoinGate::Outside(outside.pubkey()),
    )
    .await;
    assert_eq!(
        refusal_code(harness.join(0).await),
        treasurer_refusal(ProgramError::RunNotBondGated)
    );
    assert_eq!(harness.authorization_active(0).await, None);
}

/// The client decides how to join from chain state alone, and the instruction
/// it builds for a bond-gated run is the one the treasurer accepts.
#[tokio::test]
pub async fn the_client_joins_a_bond_gated_run_through_the_treasurer() {
    let mut harness =
        Harness::new(348, "Leviathan join gate client", JoinGate::Bond).await;
    let run = harness.run;
    let coordinator_instance = harness.coordinator_instance;
    let run_account = harness.endpoint.get_account(&run).await.unwrap();
    assert!(SolanaBackend::is_bond_gated_run(
        &run,
        &coordinator_instance,
        run_account.as_ref()
    )
    .unwrap());
    // The same run account does not gate a coordinator instance it was not
    // created for.
    let elsewhere = Keypair::new().pubkey();
    assert!(!SolanaBackend::is_bond_gated_run(
        &run,
        &elsewhere,
        run_account.as_ref()
    )
    .unwrap());

    // The node pays for its own join, as the client does with its wallet.
    let wallet = harness.nodes[0].wallet.insecure_clone();
    harness
        .endpoint
        .request_airdrop(&wallet.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    let instruction = treasurer_participant_join_run(
        &run,
        &coordinator_instance,
        &harness.coordinator_account,
        &wallet.pubkey(),
        harness.nodes[0].identity(),
    );
    harness
        .endpoint
        .process_instruction(&wallet, instruction)
        .await
        .unwrap();
    assert_eq!(harness.authorization_active(0).await, Some(false));
    let identity = harness.nodes[0].identity();
    assert!(
        harness
            .state()
            .await
            .clients_state
            .clients
            .iter()
            .any(|client| client.id == identity),
        "the client's join did not register the node with the coordinator"
    );
}

/// A run whose join authority is an outside key is joined directly, and the
/// client sees that from the same chain state.
#[tokio::test]
pub async fn the_client_joins_an_outside_authority_run_directly() {
    let outside = Keypair::new();
    let mut harness = Harness::new(
        349,
        "Leviathan join gate client out",
        JoinGate::Outside(outside.pubkey()),
    )
    .await;
    let run = harness.run;
    let coordinator_instance = harness.coordinator_instance;
    let outside_account = harness
        .endpoint
        .get_account(&outside.pubkey())
        .await
        .unwrap();
    assert!(!SolanaBackend::is_bond_gated_run(
        &outside.pubkey(),
        &coordinator_instance,
        outside_account.as_ref()
    )
    .unwrap());
    // The treasurer run itself exists and is a run, but it is not this run's
    // join authority, so it does not gate the join either.
    let run_account = harness.endpoint.get_account(&run).await.unwrap();
    assert!(run_account.is_some());
    assert!(!SolanaBackend::is_bond_gated_run(
        &run,
        &coordinator_instance,
        run_account.as_ref()
    )
    .unwrap());
}
