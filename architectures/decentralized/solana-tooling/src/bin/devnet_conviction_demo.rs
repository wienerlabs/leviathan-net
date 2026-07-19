use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use anyhow::anyhow;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::RunState;
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
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signature::read_keypair_file;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

const BOND: u64 = 500;
const SLASHING_RATE: u64 = 200;
const WITHDRAW_DELAY: i64 = 5;
const WARMUP_TIME: u64 = 3;
const WITNESS_TIME: u64 = 3;
const COOLDOWN_TIME: u64 = 4;
const WAITING_EXTRA: u8 = 3;
const SLEEP_BUFFER: u64 = 3;

async fn sleep_seconds(seconds: u64) {
    tokio::time::sleep(Duration::from_secs(seconds + SLEEP_BUFFER)).await;
}

async fn balance(endpoint: &mut ToolboxEndpoint, account: &Pubkey) -> u64 {
    endpoint
        .get_spl_token_account(account)
        .await
        .unwrap()
        .map(|a| a.amount)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> Result<()> {
    let wallet_path = std::env::var("LEVIATHAN_DEVNET_WALLET").unwrap_or_else(|_| {
        format!("{}/.config/solana/leviathan-devnet.json", std::env::var("HOME").unwrap())
    });
    let payer = read_keypair_file(&wallet_path)
        .map_err(|err| anyhow!("cannot read wallet {}: {}", wallet_path, err))?;
    println!("[+] wallet {}", payer.pubkey());

    let mut endpoint = ToolboxEndpoint::new_devnet().await;

    let index = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_id = format!("leviathan-conviction-{}", index);
    println!("[+] run_id {}", run_id);

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let stranger = Keypair::new();
    let ticker = Keypair::new();
    let clients: Vec<Keypair> = (0..1).map(|_| Keypair::new()).collect();
    let cheater = 0usize;

    println!("[+] creating collateral mint");
    let collateral_mint = endpoint
        .process_spl_token_mint_new(&payer, &mint_authority.pubkey(), None, 0)
        .await
        .unwrap();

    println!("[+] allocating coordinator account");
    let coordinator_account = endpoint
        .process_system_new_exempt(
            &payer,
            CoordinatorAccount::space_with_discriminator(),
            &psyche_solana_coordinator::ID,
        )
        .await
        .unwrap();

    println!("[+] creating run through the treasurer");
    let (run, coordinator_instance) = process_treasurer_run_create(
        &mut endpoint,
        &payer,
        &collateral_mint,
        &coordinator_account,
        RunCreateParams {
            index,
            run_id: run_id.clone(),
            main_authority: main_authority.pubkey(),
            join_authority: join_authority.pubkey(),
            client_version: "demo".to_string(),
        },
    )
    .await
    .unwrap();

    let run_collateral = endpoint
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
            .process_spl_associated_token_account_get_or_init(&payer, &client.pubkey(), &collateral_mint)
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

    println!("[+] cheater posts a bond of {}", BOND);
    process_treasurer_participant_bond_deposit(
        &mut endpoint,
        &payer,
        &clients[cheater],
        &clients_collateral[cheater],
        &collateral_mint,
        &run,
        BOND,
    )
    .await
    .unwrap();
    let participant = get_participant(&mut endpoint, &find_participant(&run, &clients[cheater].pubkey()))
        .await
        .unwrap()
        .unwrap();
    println!("    on-chain bond_amount = {}", participant.bond_amount);

    println!("[+] configuring and unpausing the run (slashing_rate = {})", SLASHING_RATE);
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
                warmup_time: WARMUP_TIME,
                cooldown_time: COOLDOWN_TIME,
                max_round_train_time: 6,
                round_witness_time: WITNESS_TIME,
                min_clients: 1,
                init_min_clients: 1,
                global_batch_size_start: clients.len() as u16,
                global_batch_size_end: clients.len() as u16,
                global_batch_size_warmup_tokens: 0,
                verification_percent: 0,
                witness_nodes: 0,
                epoch_time: 45,
                total_steps: 100,
                waiting_for_members_extra_time: WAITING_EXTRA,
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
            epoch_earning_rate_total_shared: Some(3_000),
            epoch_slashing_rate_per_client: Some(SLASHING_RATE),
            paused: Some(false),
            client_version: None,
        },
    )
    .await
    .unwrap();

    println!("[+] clients join the run");
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

    println!("[+] a stranger tries to open a dispute (must fail)");
    let stranger_result = process_treasurer_run_slash(
        &mut endpoint,
        &payer,
        &stranger,
        &run,
        &coordinator_account,
        &run_id,
        cheater as u64,
    )
    .await;
    println!("    stranger run_slash rejected = {}", stranger_result.is_err());

    println!("[+] driving one epoch; the run authority convicts the cheater while it trains");
    let cheater_key = clients[cheater].pubkey().to_bytes();
    let mut slashed_done = false;
    for step in 0..60 {
        let state = get_coordinator_account_state(&mut endpoint, &coordinator_account)
            .await
            .unwrap()
            .unwrap();
        let run_state = state.coordinator.run_state;
        let cheater_index = state
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|c| *c.id.signer() == cheater_key);
        println!(
            "    step {} run_state {} in_epoch {} cheater_present {}",
            step,
            run_state,
            state.coordinator.epoch_state.clients.len(),
            cheater_index.is_some(),
        );

        if !slashed_done {
            if let Some(live_index) = cheater_index {
                if process_treasurer_run_slash(
                    &mut endpoint,
                    &payer,
                    &main_authority,
                    &run,
                    &coordinator_account,
                    &run_id,
                    live_index as u64,
                )
                .await
                .is_ok()
                {
                    println!("    convicted cheater at live index {} while {}", live_index, run_state);
                    slashed_done = true;
                }
            }
        }

        match run_state {
            RunState::WaitingForMembers => {
                if slashed_done {
                    println!("    epoch closed");
                    break;
                }
                sleep_seconds(WAITING_EXTRA as u64).await;
            }
            RunState::Warmup => sleep_seconds(WARMUP_TIME).await,
            RunState::RoundTrain => sleep_seconds(7).await,
            RunState::RoundWitness => sleep_seconds(WITNESS_TIME).await,
            RunState::Cooldown => sleep_seconds(COOLDOWN_TIME).await,
            _ => sleep_seconds(WARMUP_TIME).await,
        }
        let _ = process_coordinator_tick(&mut endpoint, &payer, &ticker, &coordinator_instance, &coordinator_account).await;
    }

    let state = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let cheater_client = state
        .clients_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == clients[cheater].pubkey().to_bytes())
        .unwrap();
    println!("[+] on-chain conviction: cheater slashed = {}, earned = {}", cheater_client.slashed, cheater_client.earned);

    println!("[+] cheater tries to reclaim the full bond");
    process_treasurer_participant_bond_request_withdraw(&mut endpoint, &payer, &clients[cheater], &run, BOND)
        .await
        .unwrap();
    sleep_seconds(WITHDRAW_DELAY as u64).await;
    process_treasurer_participant_bond_finalize_withdraw(
        &mut endpoint,
        &payer,
        &clients[cheater],
        &clients_collateral[cheater],
        &collateral_mint,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();

    let recovered = balance(&mut endpoint, &clients_collateral[cheater]).await;
    let vault = balance(&mut endpoint, &run_collateral).await;
    println!("[+] settlement: cheater recovered {}, forfeit retained in vault {}", recovered, vault);
    println!();
    println!("Summary");
    println!("  bond posted         {}", BOND);
    println!("  slashed on-chain    {}", cheater_client.slashed);
    println!("  bond recovered      {}", recovered);
    println!("  forfeit in vault    {}", vault);
    println!("  run                 {}", run);
    println!("  coordinator account {}", coordinator_account);

    if cheater_client.slashed != SLASHING_RATE {
        return Err(anyhow!("expected slashed {}, got {}", SLASHING_RATE, cheater_client.slashed));
    }
    if recovered != BOND - SLASHING_RATE {
        return Err(anyhow!("expected recovery {}, got {}", BOND - SLASHING_RATE, recovered));
    }
    println!("[+] live devnet conviction loop verified");
    Ok(())
}
