use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::anyhow;
use anyhow::Result;
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
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::NodeIdentity;
use psyche_core::OptimizerDefinition;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_create;
use psyche_solana_tooling::process_authorizer_instructions::process_authorizer_authorization_grantor_update;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_join_run;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_update;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::read_keypair_file;
use solana_sdk::signature::Keypair;
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

#[tokio::main]
async fn main() -> Result<()> {
    let wallet_path = std::env::var("LEVIATHAN_DEVNET_WALLET").unwrap_or_else(|_| {
        format!(
            "{}/.config/solana/leviathan-devnet.json",
            std::env::var("HOME").unwrap()
        )
    });
    let payer = read_keypair_file(&wallet_path)
        .map_err(|err| anyhow!("cannot read wallet {}: {}", wallet_path, err))?;
    println!("[+] wallet {}", payer.pubkey());

    let mut endpoint = match std::env::var("LEVIATHAN_DEVNET_RPC").ok() {
        Some(url) => {
            println!("[+] rpc {}", url);
            ToolboxEndpoint::new_rpc_with_url_or_moniker_and_commitment(
                &url,
                CommitmentConfig::confirmed(),
            )
        }
        None => ToolboxEndpoint::new_devnet().await,
    };

    let index = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let run_id = format!("leviathan-committee-{}", index);
    println!("[+] run_id {}", run_id);

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let ticker = Keypair::new();
    let clients: Vec<Keypair> = (0..3).map(|_| Keypair::new()).collect();

    println!("[+] creating collateral mint");
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

    println!("[+] every client posts a bond of {BOND} (verifiers need skin in the game)");
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

    println!("[+] configuring the run (verification_percent=67, slashing_rate={SLASHING_RATE})");
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
                min_clients: clients.len() as u16,
                init_min_clients: clients.len() as u16,
                global_batch_size_start: clients.len() as u16,
                global_batch_size_end: clients.len() as u16,
                global_batch_size_warmup_tokens: 0,
                verification_percent: 67,
                witness_nodes: 0,
                epoch_time: 90,
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

    println!("[+] driving into an active epoch");
    sleep_seconds(WAITING_EXTRA as u64).await;
    let _ = process_coordinator_tick(
        &mut endpoint,
        &payer,
        &ticker,
        &coordinator_instance,
        &coordinator_account,
    )
    .await;
    sleep_seconds(WARMUP_TIME).await;
    let _ = process_coordinator_tick(
        &mut endpoint,
        &payer,
        &ticker,
        &coordinator_instance,
        &coordinator_account,
    )
    .await;

    let state = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let selection = CommitteeSelection::from_coordinator(&state.coordinator, 0).unwrap();
    let quorum = (2 * selection.get_num_verifier_nodes()).div_ceil(3).max(1);

    let mut verifiers: Vec<&Keypair> = vec![];
    let mut target: Option<(Pubkey, u64)> = None;
    for (epoch_index, client) in state.coordinator.epoch_state.clients.iter().enumerate() {
        let signer = *client.id.signer();
        let keypair = clients.iter().find(|k| k.pubkey().to_bytes() == signer);
        match selection.get_committee(epoch_index as u64).committee {
            Committee::Verifier => {
                if let Some(keypair) = keypair {
                    verifiers.push(keypair);
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

    let (target_key, target_index) = target.ok_or_else(|| {
        anyhow!(
            "no trainer target in the epoch (verifiers={}, run_state={}, clients={})",
            verifiers.len(),
            state.coordinator.run_state,
            state.coordinator.epoch_state.clients.len()
        )
    })?;
    println!(
        "[+] committee discovered: {} verifiers, quorum {}, target at index {}",
        verifiers.len(),
        quorum,
        target_index
    );
    if (verifiers.len() as u64) < quorum {
        return Err(anyhow!(
            "not enough verifiers ({}) for quorum {}",
            verifiers.len(),
            quorum
        ));
    }

    for (i, verifier) in verifiers.iter().take(quorum as usize).enumerate() {
        println!("[+] verifier {} submits a verdict on target index {target_index}", i + 1);
        process_treasurer_run_submit_audit_verdict(
            &mut endpoint,
            &payer,
            verifier,
            &run,
            &coordinator_account,
            &run_id,
            &target_key,
            target_index,
            0,
            4,
            [0xAA; 32],
            [0xBB; 32],
        )
        .await?;
    }

    let after = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let target_state = after
        .coordinator
        .epoch_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == target_key.to_bytes())
        .map(|c| c.state);
    println!("[+] target state after the quorum verdict: {target_state:?}");

    println!("[+] driving to epoch end to settle the slash");
    let mut target_slashed = 0;
    for _ in 0..20 {
        sleep_seconds(COOLDOWN_TIME).await;
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
            .map(|c| c.slashed)
            .unwrap_or(0);
        if target_slashed > 0 {
            break;
        }
    }

    println!();
    println!("Summary");
    println!("  verifiers        {}", verifiers.len());
    println!("  quorum           {}", quorum);
    println!("  target slashed   {}", target_slashed);
    println!("  run              {}", run);
    println!("  coordinator      {}", coordinator_account);
    if target_slashed != SLASHING_RATE {
        return Err(anyhow!(
            "expected slashed {}, got {}",
            SLASHING_RATE,
            target_slashed
        ));
    }
    println!("[+] live devnet committee vote verified");
    Ok(())
}
