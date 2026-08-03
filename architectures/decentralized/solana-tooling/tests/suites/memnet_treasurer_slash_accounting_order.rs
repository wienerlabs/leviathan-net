//! Reproduces the epoch-end slash accounting bug found in the internal review
//! of the on-chain programs (wienerlabs/leviathan#15).
//!
//! Every conviction path in the treasurer - the committee verdict, the appeals
//! bench, the losing-verifier crank and the admin `run_slash` - ends in the same
//! place: a CPI to the coordinator's `slash_client`, which only sets
//! `ClientState::Ejected` on the client. No money moves there. The money moves
//! once per epoch, in `PsycheSolanaCoordinatorAccountState::tick`, where the
//! ejected clients are matched back to their permanent records and charged
//! `slashing_rate_per_client`.
//!
//! That match used to be a forward-only merge walk: one cursor stepping through
//! `clients_state.clients` (permanent, in join order) into
//! `epoch_state.exited_clients` - which is in *exit* order, not join order,
//! because `move_clients_to_exited` appends a fresh batch at the end of every
//! round. When the two orders disagreed the cursor walked past a client and
//! never came back, so that client's slash was dropped in silence and it
//! withdrew its whole bond after a conviction that is on chain.
//!
//! This test builds the smallest disagreement that exists - two clients ejected
//! in different rounds, the earlier-joined one ejected second - and holds the
//! matching to it.

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
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

const RUN_ID: &str = "Leviathan exit order";
const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;
const WITHDRAW_DELAY: i64 = 100;

/// Two clients are ejected in the same epoch, in different rounds, and the one
/// that joined first is ejected second. Both forfeit.
#[tokio::test]
pub async fn exiting_out_of_join_order_still_forfeits() {
    let mut endpoint = create_memnet_endpoint().await;

    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 5_000_000_000)
        .await
        .unwrap();

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let ticker = Keypair::new();

    // Join order is the order of this vector: `clients_state.clients` is
    // append-only and a rejoining client keeps its original slot, so this is
    // also the order the epoch-end walk iterates in.
    //
    //   clients[0] "first_out_last"  - joins first, is ejected second
    //   clients[1] "bystander"       - stays healthy the whole epoch
    //   clients[2] "last_out_first"  - joins last, is ejected first
    let mut clients = vec![];
    for _ in 0..3 {
        clients.push(Keypair::new());
    }
    let first_out_last = 0usize;
    let last_out_first = 2usize;

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
            index: 91,
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
            bond_withdraw_delay_seconds: WITHDRAW_DELAY,
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

    // Both clients that will be ejected are fully bonded, so a correctly
    // accounted slash has something to take from each of them.
    for index in [first_out_last, last_out_first] {
        process_treasurer_participant_bond_deposit(
            &mut endpoint,
            &payer,
            &clients[index],
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
                cooldown_time: 20,
                max_round_train_time: 15,
                round_witness_time,
                min_clients: 1,
                init_min_clients: 1,
                global_batch_size_start: clients.len() as u16,
                global_batch_size_end: clients.len() as u16,
                global_batch_size_warmup_tokens: 0,
                verification_percent: 0,
                witness_nodes: 0,
                epoch_time: 200,
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

    // The permanent record is in join order, which is what the epoch-end walk
    // iterates. Assert that up front: the whole finding rests on it.
    let joined = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let permanent_order: Vec<Pubkey> = joined
        .clients_state
        .clients
        .iter()
        .map(|c| Pubkey::new_from_array(*c.id.signer()))
        .collect();
    assert_eq!(
        permanent_order,
        clients.iter().map(|c| c.pubkey()).collect::<Vec<_>>(),
        "clients_state.clients must be in join order for this test to mean anything"
    );

    // Round N: eject the client that joined LAST.
    let index = live_index(&mut endpoint, &coordinator_account, &clients[last_out_first].pubkey())
        .await
        .expect("last_out_first is live");
    process_treasurer_run_slash(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        &coordinator_account,
        RUN_ID,
        index,
    )
    .await
    .unwrap();

    // Let the round close so `move_clients_to_exited` appends it on its own.
    advance_until(&mut endpoint, &payer, &ticker, &clients, &coordinator_instance, &coordinator_account, round_witness_time, |state| {
        exited_order(state).len() == 1
    })
    .await;

    // Round N+k: now eject the client that joined FIRST. Its live index has
    // shifted down by one, because the round boundary compacted the list.
    let index = live_index(&mut endpoint, &coordinator_account, &clients[first_out_last].pubkey())
        .await
        .expect("first_out_last is live");
    process_treasurer_run_slash(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        &coordinator_account,
        RUN_ID,
        index,
    )
    .await
    .unwrap();

    advance_until(&mut endpoint, &payer, &ticker, &clients, &coordinator_instance, &coordinator_account, round_witness_time, |state| {
        exited_order(state).len() == 2
    })
    .await;

    // The two lists now disagree, which is the whole precondition.
    let staged = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        exited_order(&staged),
        vec![
            clients[last_out_first].pubkey(),
            clients[first_out_last].pubkey()
        ],
        "exited_clients is in exit order, the reverse of join order here"
    );

    // Run the epoch out. This is where `tick` charges the slashing rate.
    advance_until(&mut endpoint, &payer, &ticker, &clients, &coordinator_instance, &coordinator_account, round_witness_time, |state| {
        slashed_points(state, &clients[last_out_first].pubkey()) > 0
    })
    .await;

    let settled = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(
        slashed_points(&settled, &clients[last_out_first].pubkey()),
        SLASHING_RATE,
        "the client that exited first is charged"
    );

    // The one the old cursor walked past. Both were Ejected; both pay.
    assert_eq!(
        slashed_points(&settled, &clients[first_out_last].pubkey()),
        SLASHING_RATE,
        "an ejected client that exited out of join order is charged too"
    );

    // The bystander stayed healthy all epoch and owes nothing.
    assert_eq!(
        slashed_points(&settled, &clients[1].pubkey()),
        0,
        "a client that was never ejected is not charged"
    );

    // And the accounting reaches the money: the forfeit comes out of the bond.
    process_treasurer_participant_bond_request_withdraw(
        &mut endpoint,
        &payer,
        &clients[first_out_last],
        &run,
        BOND,
    )
    .await
    .unwrap();
    endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    process_treasurer_participant_bond_finalize_withdraw(
        &mut endpoint,
        &payer,
        &clients[first_out_last],
        &clients_collateral[first_out_last],
        &collateral_mint,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();

    assert_eq!(
        endpoint
            .get_spl_token_account(&clients_collateral[first_out_last])
            .await
            .unwrap()
            .unwrap()
            .amount,
        BOND - SLASHING_RATE,
        "a convicted, ejected client withdraws its bond less the forfeit"
    );
}

type State = psyche_solana_coordinator::CoordinatorInstanceState;

fn exited_order(state: &State) -> Vec<Pubkey> {
    state
        .coordinator
        .epoch_state
        .exited_clients
        .iter()
        .map(|c| Pubkey::new_from_array(*c.id.signer()))
        .collect()
}

fn slashed_points(state: &State, signer: &Pubkey) -> u64 {
    state
        .clients_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == signer.to_bytes())
        .map(|c| c.slashed)
        .unwrap_or_default()
}

async fn live_index(
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
    signer: &Pubkey,
) -> Option<u64> {
    get_coordinator_account_state(endpoint, coordinator_account)
        .await
        .unwrap()
        .unwrap()
        .coordinator
        .epoch_state
        .clients
        .iter()
        .position(|c| *c.id.signer() == signer.to_bytes())
        .map(|position| position as u64)
}

/// Drives whole rounds until `done` holds.
///
/// Every live client witnesses before the clock is moved on. That is not
/// decoration: `tick_round_witness` treats a round that produced no witnesses as
/// "everybody disconnected", withdraws the whole client set and drops into
/// Cooldown, which would take the run apart before the scenario is set up.
async fn advance_until<F>(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    ticker: &Keypair,
    clients: &[Keypair],
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    round_witness_time: u64,
    done: F,
) where
    F: Fn(&State) -> bool,
{
    for _ in 0..40 {
        let state = get_coordinator_account_state(endpoint, coordinator_account)
            .await
            .unwrap()
            .unwrap();
        if done(&state) {
            return;
        }
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
    panic!("coordinator never reached the expected state");
}
