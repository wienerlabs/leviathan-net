//! A round's seed is revealed from the hash of a slot committed before anyone
//! could know it, read from the real `SlotHashes` sysvar on the real programs.
//! The commitment is made only once a round is due, and from then on nothing
//! that shapes the committee is taken, so nobody who could know the seed can
//! act on it (wienerlabs/leviathan-net#62).

use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::RunState;
use psyche_coordinator::WitnessProof;
use psyche_coordinator::WAITING_FOR_MEMBERS_EXTRA_SECONDS;
use psyche_core::sha256v;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::accounts::PermissionlessCoordinatorAccounts;
use psyche_solana_coordinator::instruction::SetPaused;
use psyche_solana_coordinator::instruction::Tick;
use psyche_solana_coordinator::instruction::WarmupWitness;
use psyche_solana_coordinator::instruction::Witness;
use psyche_solana_coordinator::logic::InitCoordinatorParams;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::round_seed::round_seed;
use psyche_solana_coordinator::round_seed::slot_hash;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_coordinator::ProgramError;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_free;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_init;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_set_paused;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_warmup_witness;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_witness;
use psyche_solana_tooling::process_coordinator_instructions::process_update;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::sysvar;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::chain_state::coordinator_state;
use super::refusal::refusal_code;
use super::signed_payload::identity;

const WARMUP_TIME: u64 = 10;
const ROUND_WITNESS_TIME: u64 = 10;
/// How many slots the `SlotHashes` sysvar keeps.
const SLOT_HASHES_KEPT: usize = 512;

/// One coordinator-only run with a single client, its only witness.
struct Run {
    run_id: String,
    main_authority: Keypair,
    join_authority: Keypair,
    client: Keypair,
    coordinator_instance: Pubkey,
    coordinator_account: Pubkey,
}

struct Chain {
    endpoint: ToolboxEndpoint,
    payer: Keypair,
    ticker: Keypair,
}

impl Chain {
    async fn new() -> Chain {
        let mut endpoint = create_memnet_endpoint().await;
        let payer = Keypair::new();
        endpoint
            .request_airdrop(&payer.pubkey(), 10_000_000_000)
            .await
            .unwrap();
        Chain {
            endpoint,
            payer,
            ticker: Keypair::new(),
        }
    }

    async fn state(&mut self, run: &Run) -> Box<CoordinatorInstanceState> {
        coordinator_state(&mut self.endpoint, &run.coordinator_account).await
    }

    async fn slot(&mut self) -> u64 {
        self.endpoint.get_sysvar_clock().await.unwrap().slot
    }

    async fn tick(&mut self, run: &Run) {
        let ticker = self.ticker.insecure_clone();
        self.tick_as(run, &ticker).await;
    }

    /// A tick signed by `ticker`. Two ticks in one slot have to differ in
    /// something, or the second is the first transaction again.
    async fn tick_as(&mut self, run: &Run, ticker: &Keypair) {
        process_coordinator_tick(
            &mut self.endpoint,
            &self.payer,
            ticker,
            &run.coordinator_instance,
            &run.coordinator_account,
        )
        .await
        .unwrap();
    }

    /// The hash of `slot` as the sysvar holds it, read the way the program
    /// reads it, and checked against the sysvar decoded in full.
    async fn slot_hash(&mut self, slot: u64) -> Option<[u8; 32]> {
        let raw = self
            .endpoint
            .get_account_data(&sysvar::slot_hashes::ID)
            .await
            .unwrap()
            .unwrap();
        let read = slot_hash(&raw, slot).expect("a readable sysvar");
        let decoded = self
            .endpoint
            .get_sysvar_slot_hashes()
            .await
            .unwrap()
            .into_iter()
            .find(|(held, _)| *held == slot)
            .map(|(_, hash)| hash.to_bytes());
        assert_eq!(read, decoded, "the program's reading of slot {slot}");
        read
    }

    /// A run in its warmup, its client joined: the warmup's start committed
    /// the first round's seed.
    async fn warming_up(&mut self, run_id: &str) -> Run {
        let main_authority = Keypair::new();
        let join_authority = Keypair::new();
        let client = Keypair::new();
        let client_p2p = Keypair::new();
        let coordinator_account = self
            .endpoint
            .process_system_new_exempt(
                &self.payer,
                CoordinatorAccount::space_with_discriminator(),
                &psyche_solana_coordinator::ID,
            )
            .await
            .unwrap();
        let coordinator_instance = process_coordinator_init(
            &mut self.endpoint,
            &self.payer,
            &coordinator_account,
            InitCoordinatorParams {
                run_id: run_id.to_string(),
                main_authority: main_authority.pubkey(),
                join_authority: join_authority.pubkey(),
                client_version: "test".to_string(),
            },
        )
        .await
        .unwrap();
        process_update(
            &mut self.endpoint,
            &self.payer,
            &main_authority,
            &coordinator_instance,
            &coordinator_account,
            None,
            Some(CoordinatorConfig {
                warmup_time: WARMUP_TIME,
                cooldown_time: 999,
                max_round_train_time: 15,
                round_witness_time: ROUND_WITNESS_TIME,
                min_clients: 1,
                init_min_clients: 1,
                // Room for a second client to join.
                global_batch_size_start: 2,
                global_batch_size_end: 2,
                global_batch_size_warmup_tokens: 0,
                verification_percent: 0,
                witness_nodes: 1,
                epoch_time: 600,
                total_steps: 100,
                waiting_for_members_extra_time: 3,
            }),
            Some(Model::LLM(LLM {
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
            None,
        )
        .await
        .unwrap();
        let authorization = process_authorizer_authorization_create(
            &mut self.endpoint,
            &self.payer,
            &join_authority,
            &client.pubkey(),
            JOIN_RUN_AUTHORIZATION_SCOPE,
        )
        .await
        .unwrap();
        process_authorizer_authorization_grantor_update(
            &mut self.endpoint,
            &self.payer,
            &join_authority,
            &authorization,
            AuthorizationGrantorUpdateParams { active: true },
        )
        .await
        .unwrap();
        process_coordinator_set_paused(
            &mut self.endpoint,
            &self.payer,
            &main_authority,
            &coordinator_instance,
            &coordinator_account,
            false,
        )
        .await
        .unwrap();
        process_coordinator_join_run(
            &mut self.endpoint,
            &self.payer,
            &client,
            &authorization,
            &coordinator_instance,
            &coordinator_account,
            identity(&client, &client_p2p),
        )
        .await
        .unwrap();
        let run = Run {
            run_id: run_id.to_string(),
            main_authority,
            join_authority,
            client,
            coordinator_instance,
            coordinator_account,
        };
        self.endpoint
            .forward_clock_unix_timestamp(WAITING_FOR_MEMBERS_EXTRA_SECONDS)
            .await
            .unwrap();
        self.tick(&run).await;
        let state = self.state(&run).await;
        assert_eq!(state.coordinator.run_state, RunState::Warmup);
        // Entering the warmup commits nothing: the round is not due yet.
        assert_eq!(state.seed_commit_slot, 0);
        run
    }

    /// The tick that finds the round due, its phase over, and commits the
    /// round's seed to its own slot. Returns that slot.
    async fn commit(&mut self, run: &Run) -> u64 {
        let before = self.state(run).await.coordinator.run_state;
        self.tick(run).await;
        let state = self.state(run).await;
        assert_eq!(
            state.coordinator.run_state, before,
            "the round started early"
        );
        let slot = self.slot().await;
        assert_eq!(state.seed_commit_slot, slot);
        slot
    }

    /// The seed the round due now draws, from the run's own state and the
    /// committed slot's hash in the sysvar.
    async fn expected_seed(&mut self, run: &Run, committed: u64) -> u64 {
        let state = self.state(run).await;
        let hash = self
            .slot_hash(committed)
            .await
            .expect("committed slot held");
        round_seed(
            String::from(&state.coordinator.run_id).as_bytes(),
            state.coordinator.progress.step,
            committed,
            &hash,
        )
    }

    /// The run's single client witnesses: in a warmup, the warmup witness
    /// that completes it; in a round, the witness that ends its training.
    async fn witness(&mut self, run: &Run) {
        let proof = WitnessProof {
            witness: true.into(),
            position: 0,
            index: 0,
        };
        let state = self.state(run).await.coordinator.run_state;
        if state == RunState::Warmup {
            process_coordinator_warmup_witness(
                &mut self.endpoint,
                &self.payer,
                &run.client,
                &run.coordinator_instance,
                &run.coordinator_account,
                &WarmupWitness {
                    proof,
                    participant_bloom: Default::default(),
                    broadcast_bloom: Default::default(),
                    broadcast_merkle: Default::default(),
                },
            )
            .await
            .unwrap();
        } else {
            process_coordinator_witness(
                &mut self.endpoint,
                &self.payer,
                &run.client,
                &run.coordinator_instance,
                &run.coordinator_account,
                &Witness {
                    proof,
                    participant_bloom: Default::default(),
                    broadcast_bloom: Default::default(),
                    broadcast_merkle: Default::default(),
                    metadata: Default::default(),
                },
            )
            .await
            .unwrap();
        }
    }

    /// A tick and, behind it in the same transaction, a pause by the run's
    /// authority: what `run-manager set-paused` sends while a round's seed is
    /// committed.
    async fn tick_then_pause(&mut self, run: &Run) -> anyhow::Result<()> {
        let tick = Instruction {
            accounts: PermissionlessCoordinatorAccounts {
                user: self.ticker.pubkey(),
                coordinator_instance: run.coordinator_instance,
                coordinator_account: run.coordinator_account,
                slot_hashes: sysvar::slot_hashes::ID,
            }
            .to_account_metas(None),
            data: Tick {}.data(),
            program_id: psyche_solana_coordinator::ID,
        };
        let pause = Instruction {
            accounts: OwnerCoordinatorAccounts {
                authority: run.main_authority.pubkey(),
                coordinator_instance: run.coordinator_instance,
                coordinator_account: run.coordinator_account,
            }
            .to_account_metas(None),
            data: SetPaused { paused: true }.data(),
            program_id: psyche_solana_coordinator::ID,
        };
        let ticker = self.ticker.insecure_clone();
        self.endpoint
            .process_instructions_with_signers(
                &self.payer,
                &[tick, pause],
                &[&ticker, &run.main_authority],
            )
            .await?;
        Ok(())
    }

    /// The seed of the round under way.
    async fn round_seed(&mut self, run: &Run) -> u64 {
        let state = self.state(run).await;
        assert_eq!(state.coordinator.run_state, RunState::RoundTrain);
        state.coordinator.current_round().unwrap().random_seed
    }
}

/// What the old scheme drew for a round started at the current slot: the hash
/// of the clock that transaction ran under, which anyone can compute for any
/// slot ahead of it.
async fn clock_seed(chain: &mut Chain) -> u64 {
    let clock = chain.endpoint.get_sysvar_clock().await.unwrap();
    let digest = sha256v(&[
        &clock.unix_timestamp.to_ne_bytes(),
        &clock.slot.to_ne_bytes(),
    ]);
    u64::from_ne_bytes(digest[..8].try_into().unwrap())
}

/// A round's seed is the hash of the slot of the first tick that found it due,
/// read from the sysvar by a later tick, however many slots later, and not
/// the clock of the transaction that started it.
#[tokio::test]
pub async fn a_rounds_seed_is_the_hash_of_the_slot_that_found_it_due() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan seed hash").await;
    chain
        .endpoint
        .forward_clock_unix_timestamp(WARMUP_TIME)
        .await
        .unwrap();
    let warmup_commit = chain.commit(&run).await;
    assert!(
        chain.slot_hash(warmup_commit).await.is_none(),
        "the committed slot's hash exists before the slot is done"
    );
    chain.endpoint.forward_clock_slot(1).await.unwrap();
    let expected = chain.expected_seed(&run, warmup_commit).await;
    chain.tick(&run).await;
    assert_eq!(chain.round_seed(&run).await, expected);
    assert_eq!(
        chain.state(&run).await.seed_commit_slot,
        0,
        "the commitment was spent"
    );

    // The round's training ends with its witness, which commits nothing.
    chain.witness(&run).await;
    let witnessing = chain.state(&run).await;
    assert_eq!(witnessing.coordinator.run_state, RunState::RoundWitness);
    assert_eq!(witnessing.seed_commit_slot, 0);

    // Once the witness phase is over, the first tick commits the next round.
    chain
        .endpoint
        .forward_clock_unix_timestamp(ROUND_WITNESS_TIME)
        .await
        .unwrap();
    let committed = chain.commit(&run).await;
    // The round starts several slots on.
    chain.endpoint.forward_clock_slot(7).await.unwrap();
    let expected = chain.expected_seed(&run, committed).await;
    let clock_seed = clock_seed(&mut chain).await;
    chain.tick(&run).await;
    let seed = chain.round_seed(&run).await;
    assert_eq!(seed, expected);
    assert_ne!(
        seed, clock_seed,
        "the round took the clock of the tick that started it"
    );
    assert_eq!(
        chain
            .state(&run)
            .await
            .coordinator
            .current_round()
            .unwrap()
            .height,
        1
    );
}

/// A tick in the committed slot, whoever sends it, cannot start the round:
/// the slot's hash does not exist yet. The next tick, a slot later, does.
#[tokio::test]
pub async fn a_round_does_not_start_in_the_slot_it_was_committed_in() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan same slot").await;
    chain
        .endpoint
        .forward_clock_unix_timestamp(WARMUP_TIME)
        .await
        .unwrap();
    let committed = chain.commit(&run).await;
    chain.tick_as(&run, &Keypair::new()).await;
    let waiting = chain.state(&run).await;
    assert_eq!(waiting.coordinator.run_state, RunState::Warmup);
    assert_eq!(waiting.seed_commit_slot, committed);

    chain.endpoint.forward_clock_slot(1).await.unwrap();
    let expected = chain.expected_seed(&run, committed).await;
    chain.tick(&run).await;
    assert_eq!(chain.round_seed(&run).await, expected);
}

/// Once the seed is committed, a witness is refused: whoever could know the
/// seed could otherwise decide by it whether the round starts.
#[tokio::test]
pub async fn witnesses_are_closed_once_the_seed_is_committed() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan closed witness").await;
    chain
        .endpoint
        .forward_clock_unix_timestamp(WARMUP_TIME)
        .await
        .unwrap();
    chain.commit(&run).await;
    chain.endpoint.forward_clock_slot(1).await.unwrap();
    let refused = process_coordinator_warmup_witness(
        &mut chain.endpoint,
        &chain.payer,
        &run.client,
        &run.coordinator_instance,
        &run.coordinator_account,
        &WarmupWitness {
            proof: WitnessProof {
                witness: true.into(),
                position: 0,
                index: 0,
            },
            participant_bloom: Default::default(),
            broadcast_bloom: Default::default(),
            broadcast_merkle: Default::default(),
        },
    )
    .await;
    assert_eq!(
        refusal_code(refused),
        u32::from(ProgramError::RoundSeedCommitted)
    );
}

/// A commitment the sysvar no longer holds is renewed to the slot of the tick
/// that found it gone, and the round starts on the renewed slot's hash. No
/// seed anyone could pick stands in for it. A pause waits for the renewal to
/// be kept by a tick of its own, as `run-manager set-paused` sends it.
#[tokio::test]
pub async fn a_commitment_the_sysvar_no_longer_holds_is_renewed() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan stale seed").await;
    chain
        .endpoint
        .forward_clock_unix_timestamp(WARMUP_TIME)
        .await
        .unwrap();
    let stale = chain.commit(&run).await;
    // Every forward puts the slot it leaves in the sysvar, the committed one
    // first, so the sysvar still holds it after as many forwards as it keeps
    // slots, and one more pushes it out.
    for _ in 0..SLOT_HASHES_KEPT {
        chain.endpoint.forward_clock_slot(1).await.unwrap();
    }
    assert!(chain.slot_hash(stale).await.is_some());
    chain.endpoint.forward_clock_slot(1).await.unwrap();
    assert!(chain.slot_hash(stale).await.is_none());

    // A pause behind a tick is refused here: the tick renews the commitment,
    // and the refused pause reverts the renewal with it.
    let refused = chain.tick_then_pause(&run).await;
    assert_eq!(
        refusal_code(refused),
        u32::from(ProgramError::RoundSeedCommitted)
    );
    assert_eq!(chain.state(&run).await.seed_commit_slot, stale);

    // A tick of its own keeps the renewal.
    let fresh = chain.commit(&run).await;
    assert!(fresh > stale);

    // A slot later the round starts on the renewed slot's hash, and the pause
    // behind that tick lets it finish.
    chain.endpoint.forward_clock_slot(1).await.unwrap();
    let expected = chain.expected_seed(&run, fresh).await;
    chain.tick_then_pause(&run).await.unwrap();
    assert_eq!(chain.round_seed(&run).await, expected);
    assert!(chain.state(&run).await.coordinator.pending_pause.is_true());
}

/// The slot a round starts in has no say in its seed, and two runs committed
/// in one slot still draw different committees.
#[tokio::test]
pub async fn the_slot_a_round_starts_in_has_no_say_in_its_seed() {
    let mut chain = Chain::new().await;
    let early = chain.warming_up("Leviathan early start").await;
    let late = chain.warming_up("Leviathan late start").await;
    chain
        .endpoint
        .forward_clock_unix_timestamp(WARMUP_TIME)
        .await
        .unwrap();
    let committed = chain.commit(&early).await;
    assert_eq!(chain.commit(&late).await, committed);

    chain.endpoint.forward_clock_slot(1).await.unwrap();
    let early_expected = chain.expected_seed(&early, committed).await;
    let late_expected = chain.expected_seed(&late, committed).await;
    let early_slot = chain.slot().await;
    chain.tick(&early).await;
    chain.endpoint.forward_clock_slot(9).await.unwrap();
    let late_slot = chain.slot().await;
    chain.tick(&late).await;
    assert!(late_slot > early_slot);

    assert_eq!(chain.round_seed(&early).await, early_expected);
    assert_eq!(chain.round_seed(&late).await, late_expected);
    assert_ne!(early_expected, late_expected, "two runs drew one committee");
    assert_ne!(early.run_id, late.run_id);
}

/// A transaction without the sysvar cannot replace a commitment: it cannot
/// tell whether the commitment is gone, and replacing it would draw the round
/// again. A join, which takes no sysvar, leaves it as it was.
#[tokio::test]
pub async fn a_transaction_without_the_sysvar_cannot_replace_a_commitment() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan kept seed").await;
    chain
        .endpoint
        .forward_clock_unix_timestamp(WARMUP_TIME)
        .await
        .unwrap();
    let committed = chain.commit(&run).await;
    chain.endpoint.forward_clock_slot(3).await.unwrap();

    let newcomer = Keypair::new();
    let authorization = process_authorizer_authorization_create(
        &mut chain.endpoint,
        &chain.payer,
        &run.join_authority,
        &newcomer.pubkey(),
        JOIN_RUN_AUTHORIZATION_SCOPE,
    )
    .await
    .unwrap();
    process_authorizer_authorization_grantor_update(
        &mut chain.endpoint,
        &chain.payer,
        &run.join_authority,
        &authorization,
        AuthorizationGrantorUpdateParams { active: true },
    )
    .await
    .unwrap();
    process_coordinator_join_run(
        &mut chain.endpoint,
        &chain.payer,
        &newcomer,
        &authorization,
        &run.coordinator_instance,
        &run.coordinator_account,
        identity(&newcomer, &Keypair::new()),
    )
    .await
    .unwrap();
    let joined = chain.state(&run).await;
    assert_eq!(joined.coordinator.run_state, RunState::Warmup);
    assert_eq!(joined.seed_commit_slot, committed);

    let expected = chain.expected_seed(&run, committed).await;
    chain.tick(&run).await;
    assert_eq!(chain.round_seed(&run).await, expected);
}

/// The seed is read only from the `SlotHashes` sysvar: a tick naming another
/// account in its place is refused.
#[tokio::test]
pub async fn only_the_slot_hashes_sysvar_is_read() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan forged sysvar").await;
    let accounts = PermissionlessCoordinatorAccounts {
        user: chain.ticker.pubkey(),
        coordinator_instance: run.coordinator_instance,
        coordinator_account: run.coordinator_account,
        slot_hashes: sysvar::clock::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: Tick {}.data(),
        program_id: psyche_solana_coordinator::ID,
    };
    let ticker = chain.ticker.insecure_clone();
    let refused = chain
        .endpoint
        .process_instruction_with_signers(&chain.payer, instruction, &[&ticker])
        .await
        .map(|_| ());
    assert_eq!(
        refusal_code(refused),
        anchor_lang::error::ErrorCode::ConstraintAddress as u32
    );
}

/// A run of the current version under way is not closed, its seed committed
/// or not: only a halted one is. (An account of another version, which runs
/// nothing under these programs, is: see `memnet_coordinator_data_layout`.)
#[tokio::test]
pub async fn a_run_under_way_is_not_closed() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan kept open").await;
    for commit in [false, true] {
        if commit {
            chain
                .endpoint
                .forward_clock_unix_timestamp(WARMUP_TIME)
                .await
                .unwrap();
            chain.commit(&run).await;
        }
        let main_authority = run.main_authority.insecure_clone();
        let refused = process_coordinator_free(
            &mut chain.endpoint,
            &chain.payer,
            &main_authority,
            &Keypair::new().pubkey(),
            &run.coordinator_instance,
            &run.coordinator_account,
        )
        .await;
        assert_eq!(
            refusal_code(refused),
            u32::from(ProgramError::CloseCoordinatorNotHalted)
        );
    }
    assert!(chain
        .endpoint
        .get_account(&run.coordinator_account)
        .await
        .unwrap()
        .is_some());
}

/// Once the seed is committed, a pause is refused: whoever can pause could
/// otherwise cancel a round whose seed it dislikes and draw again. A tick in
/// front of it, in the same transaction a slot later, starts the round on
/// its seed, and the pause behind it lets that round finish.
#[tokio::test]
pub async fn a_pause_cannot_cancel_a_round_whose_seed_is_committed() {
    let mut chain = Chain::new().await;
    let run = chain.warming_up("Leviathan no veto").await;
    chain
        .endpoint
        .forward_clock_unix_timestamp(WARMUP_TIME)
        .await
        .unwrap();
    let committed = chain.commit(&run).await;
    let main_authority = run.main_authority.insecure_clone();
    let refused = process_coordinator_set_paused(
        &mut chain.endpoint,
        &chain.payer,
        &main_authority,
        &run.coordinator_instance,
        &run.coordinator_account,
        true,
    )
    .await;
    assert_eq!(
        refusal_code(refused),
        u32::from(ProgramError::RoundSeedCommitted)
    );

    chain.endpoint.forward_clock_slot(1).await.unwrap();
    let expected = chain.expected_seed(&run, committed).await;
    chain.tick_then_pause(&run).await.unwrap();
    assert_eq!(chain.round_seed(&run).await, expected);
    let state = chain.state(&run).await;
    assert!(state.coordinator.pending_pause.is_true());
    assert!(!state.seed_committed());
}
