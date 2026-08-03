//! Reproduces the cross-round quorum accumulation found in the internal review
//! of the on-chain programs (wienerlabs/leviathan#15).
//!
//! The security of the audit committee rests on sampling: a verifier seat is
//! drawn per *round*, from that round's `random_seed`, so to convict an honest
//! node an attacker has to hold two thirds of the seats in one draw.
//!
//! `run_submit_audit_verdict` checks committee membership against the round that
//! is current when the vote lands, but the verdict it writes into only resets
//! when the *epoch* changes - and an epoch is many rounds by construction
//! (`Coordinator::check_config` rejects `max_round_train_time >= epoch_time`).
//! Nothing records which round a vote came from, so votes cast under different,
//! independently seeded draws are added into one counter and compared against a
//! quorum computed from whichever round happens to be current last.
//!
//! An attacker therefore does not need two thirds of one committee. They need
//! two thirds of the *union* of the committees drawn while the epoch runs, which
//! a fixed sybil fraction reaches by waiting.
//!
//! This test convicts a node with a set of verifiers that was never a committee.

use psyche_coordinator::Committee;
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
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
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
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;
use std::collections::HashSet;

const RUN_ID: &str = "Leviathan cross round";
const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;
const COMMITTED: [u8; 32] = [0xAA; 32];
const REPLAYED: [u8; 32] = [0xBB; 32];

type State = psyche_solana_coordinator::CoordinatorInstanceState;

/// A verdict reaches quorum from voters drawn in different rounds, and the
/// target is ejected even though no single round's committee ever contained
/// enough of them to convict.
#[tokio::test]
pub async fn votes_from_different_committees_are_pooled_into_one_quorum() {
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
    let mut clients = vec![];
    for _ in 0..6 {
        clients.push(Keypair::new());
    }
    let warmup_time = 10;
    let round_witness_time = 10u64;

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
            index: 93,
            run_id: RUN_ID.to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: join_authority.pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    endpoint
        .process_spl_associated_token_account_get_or_init(&payer, &run, &collateral_mint)
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

    // Every client is bonded, so committee membership is the only thing standing
    // between a signer and a vote.
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
    for (index, client) in clients.iter().enumerate() {
        process_treasurer_participant_bond_deposit(
            &mut endpoint,
            &payer,
            client,
            &clients_collateral[index],
            &collateral_mint,
            &run,
            BOND,
        )
        .await
        .unwrap();
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
                round_witness_time,
                min_clients: 1,
                init_min_clients: 1,
                global_batch_size_start: 1,
                global_batch_size_end: clients.len() as u16,
                global_batch_size_warmup_tokens: 0,
                // 6 clients, no tie-breakers reserved: 3 verifier seats a round,
                // quorum 2. Two seats is a majority of one draw - the point is
                // that the two votes come from *different* draws.
                verification_percent: 50,
                witness_nodes: 0,
                epoch_time: 3_000,
                total_steps: 1_000,
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
            epoch_earning_rate_total_shared: Some(0),
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
    tick(&mut endpoint, &payer, &ticker, &coordinator_instance, &coordinator_account).await;
    endpoint
        .forward_clock_unix_timestamp(warmup_time)
        .await
        .unwrap();
    tick(&mut endpoint, &payer, &ticker, &coordinator_instance, &coordinator_account).await;

    let state = state_of(&mut endpoint, &coordinator_account).await;
    let quorum = (2 * CommitteeSelection::from_coordinator(&state.coordinator, 0)
        .unwrap()
        .get_num_verifier_nodes())
    .div_ceil(3)
    .max(1);
    assert_eq!(quorum, 2, "this scenario is written for a quorum of two");

    // The target is an honest node: a trainer, never audited, never at fault.
    let first_round_verifiers = verifiers_now(&state, &clients);
    let target_key = clients
        .iter()
        .map(|k| k.pubkey())
        .find(|key| !first_round_verifiers.contains(key))
        .expect("some client is not a verifier this round");

    // One vote, from a verifier legitimately seated in this round.
    let first_voter = *first_round_verifiers
        .iter()
        .next()
        .expect("the round has verifiers");
    cast(
        &mut endpoint,
        &payer,
        keypair_for(&clients, &first_voter),
        &run,
        &coordinator_account,
        &target_key,
    )
    .await
    .expect("a seated verifier may vote");

    assert!(
        !target_is_ejected(&state_of(&mut endpoint, &coordinator_account).await, &target_key),
        "one vote is below quorum, nothing should have happened yet"
    );

    let first_vote_round = round_height(&state_of(&mut endpoint, &coordinator_account).await);

    // Now wait. Each round reseeds the draw, so keep going until somebody who
    // was NOT seated when the first vote was cast is seated today.
    let mut second_voter = None;
    for _ in 0..20 {
        advance_one_round(
            &mut endpoint,
            &payer,
            &ticker,
            &clients,
            &coordinator_instance,
            &coordinator_account,
            round_witness_time,
        )
        .await;
        let now = state_of(&mut endpoint, &coordinator_account).await;
        if now.coordinator.epoch_state.clients.len() != clients.len() {
            panic!("the client set changed; the scenario needs a stable round");
        }
        let seated = verifiers_now(&now, &clients);
        if let Some(fresh) = seated
            .iter()
            .find(|key| !first_round_verifiers.contains(key) && **key != target_key)
        {
            second_voter = Some(*fresh);
            break;
        }
    }
    let second_voter = second_voter.expect("no newly seated verifier appeared in twenty rounds");

    // This is the whole finding in one assertion: the account about to complete
    // the quorum was not on the committee that cast the first vote.
    assert!(
        !first_round_verifiers.contains(&second_voter),
        "the second voter must come from a different draw"
    );

    let latest = state_of(&mut endpoint, &coordinator_account).await;
    assert!(
        round_height(&latest) > first_vote_round,
        "the two votes must land in different rounds for this test to prove anything"
    );
    assert!(
        !target_is_ejected(&latest, &target_key),
        "the target is still healthy going into the second vote"
    );

    cast(
        &mut endpoint,
        &payer,
        keypair_for(&clients, &second_voter),
        &run,
        &coordinator_account,
        &target_key,
    )
    .await
    .expect("BUG: a verifier from a later round is allowed to complete an earlier quorum");

    let convicted = state_of(&mut endpoint, &coordinator_account).await;
    assert!(
        target_is_ejected(&convicted, &target_key),
        "BUG: quorum was reached and an honest node ejected by two verifiers \
         who were never seated on the same committee"
    );
}

fn keypair_for<'a>(clients: &'a [Keypair], key: &Pubkey) -> &'a Keypair {
    clients
        .iter()
        .find(|k| k.pubkey() == *key)
        .expect("key belongs to a test client")
}

/// The verifier seats of the round that is current right now.
fn verifiers_now(state: &State, clients: &[Keypair]) -> HashSet<Pubkey> {
    let Ok(selection) = CommitteeSelection::from_coordinator(&state.coordinator, 0) else {
        return HashSet::new();
    };
    let mut seated = HashSet::new();
    for (index, client) in state.coordinator.epoch_state.clients.iter().enumerate() {
        if selection.get_committee(index as u64).committee != Committee::Verifier {
            continue;
        }
        if let Some(keypair) = clients
            .iter()
            .find(|k| k.pubkey().to_bytes() == *client.id.signer())
        {
            seated.insert(keypair.pubkey());
        }
    }
    seated
}

fn round_height(state: &State) -> u32 {
    state
        .coordinator
        .current_round()
        .map(|round| round.height)
        .unwrap_or_default()
}

fn target_is_ejected(state: &State, target: &Pubkey) -> bool {
    state
        .coordinator
        .epoch_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == target.to_bytes())
        .map(|c| c.state == psyche_coordinator::ClientState::Ejected)
        .unwrap_or(false)
}

async fn state_of(endpoint: &mut ToolboxEndpoint, coordinator_account: &Pubkey) -> State {
    get_coordinator_account_state(endpoint, coordinator_account)
        .await
        .unwrap()
        .unwrap()
}

async fn cast(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    verifier: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> anyhow::Result<()> {
    // The index is looked up fresh, exactly as an honest verifier client would.
    let target_index = state_of(endpoint, coordinator_account)
        .await
        .coordinator
        .epoch_state
        .clients
        .iter()
        .position(|c| *c.id.signer() == target.to_bytes())
        .expect("target is in the epoch") as u64;
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
        1,
        COMMITTED,
        REPLAYED,
    )
    .await
}

async fn tick(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    ticker: &Keypair,
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
) {
    process_coordinator_tick(endpoint, payer, ticker, coordinator_instance, coordinator_account)
        .await
        .unwrap();
}

/// Witnesses with every client, then closes the round. Witnessing is required:
/// a round that produced none is read as "everybody disconnected" and takes the
/// run into Cooldown instead of drawing a fresh committee.
async fn advance_one_round(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    ticker: &Keypair,
    clients: &[Keypair],
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    round_witness_time: u64,
) {
    let state = state_of(endpoint, coordinator_account).await;
    for client in clients {
        let position = state
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|c| *c.id.signer() == client.pubkey().to_bytes());
        let Some(position) = position else {
            continue;
        };
        let Ok(selection) = CommitteeSelection::from_coordinator(&state.coordinator, 0) else {
            continue;
        };
        let witness_proof = selection.get_witness(position as u64);
        if witness_proof.position >= SOLANA_MAX_NUM_WITNESSES as u64 {
            continue;
        }
        let _ = process_coordinator_witness(
            endpoint,
            payer,
            client,
            coordinator_instance,
            coordinator_account,
            &Witness {
                proof: witness_proof,
                participant_bloom: Default::default(),
                broadcast_bloom: Default::default(),
                broadcast_merkle: Default::default(),
                metadata: Default::default(),
            },
        )
        .await;
    }
    endpoint
        .forward_clock_unix_timestamp(round_witness_time)
        .await
        .unwrap();
    let _ = process_coordinator_tick(
        endpoint,
        payer,
        ticker,
        coordinator_instance,
        coordinator_account,
    )
    .await;
}
