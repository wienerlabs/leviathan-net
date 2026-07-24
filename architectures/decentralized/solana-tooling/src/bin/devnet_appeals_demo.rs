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
use psyche_coordinator::ClientState;
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
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_challenge_config;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash_losing_verifier;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_appeal_verdict;
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
const CHALLENGE_WINDOW: i64 = 60;
const TIE_BREAKER_SIZE: u16 = 2;
const EPOCH_TIME: u64 = 120;

async fn sleep_seconds(seconds: u64) {
    tokio::time::sleep(Duration::from_secs(seconds + SLEEP_BUFFER)).await;
}

async fn slashed_of(
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
    key: &Pubkey,
) -> u64 {
    let state = get_coordinator_account_state(endpoint, coordinator_account)
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
    let run_id = format!("leviathan-appeals-{}", index);
    println!("[+] run_id {}", run_id);

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let ticker = Keypair::new();
    let clients: Vec<Keypair> = (0..6).map(|_| Keypair::new()).collect();

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

    println!("[+] enabling the appeals court (challenge_window={CHALLENGE_WINDOW}s, tie_breakers={TIE_BREAKER_SIZE})");
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

    println!("[+] every client posts a bond of {BOND}");
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

    println!("[+] configuring the run (verification_percent=50, slashing_rate={SLASHING_RATE})");
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
                verification_percent: 50,
                witness_nodes: 0,
                epoch_time: EPOCH_TIME,
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
    let selection = CommitteeSelection::from_coordinator_with_tie_breakers(
        &state.coordinator,
        0,
        TIE_BREAKER_SIZE,
    )
    .unwrap();
    let verifier_quorum = (2 * selection.get_num_verifier_nodes()).div_ceil(3).max(1);
    let tie_breaker_quorum = (2 * selection.get_num_tie_breaker_nodes()).div_ceil(3).max(1);

    let mut verifiers: Vec<&Keypair> = vec![];
    let mut tie_breakers: Vec<&Keypair> = vec![];
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
            Committee::TieBreaker => {
                if let Some(keypair) = keypair {
                    tie_breakers.push(keypair);
                }
            }
            Committee::Trainer => {
                if target.is_none() {
                    if let Some(keypair) = keypair {
                        target = Some((keypair.pubkey(), epoch_index as u64));
                    }
                }
            }
        }
    }

    let (target_key, target_index) = target.ok_or_else(|| {
        anyhow!(
            "no trainer target (verifiers={}, tie_breakers={}, run_state={})",
            verifiers.len(),
            tie_breakers.len(),
            state.coordinator.run_state
        )
    })?;
    println!(
        "[+] committee discovered: {} verifiers (quorum {}), {} tie-breakers (quorum {}), target at index {}",
        verifiers.len(),
        verifier_quorum,
        tie_breakers.len(),
        tie_breaker_quorum,
        target_index
    );
    if (verifiers.len() as u64) < verifier_quorum || (tie_breakers.len() as u64) < tie_breaker_quorum
    {
        return Err(anyhow!(
            "not enough committee members (verifiers={}, tie_breakers={})",
            verifiers.len(),
            tie_breakers.len()
        ));
    }

    let target_keypair = clients
        .iter()
        .find(|k| k.pubkey() == target_key)
        .ok_or_else(|| anyhow!("target keypair not found"))?;

    println!("[+] {} verifiers convict the target (a pending slash opens the challenge window)", verifier_quorum);
    for (i, verifier) in verifiers.iter().take(verifier_quorum as usize).enumerate() {
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
        println!("    verifier {} voted", i + 1);
    }

    let pending = get_coordinator_account_state(&mut endpoint, &coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let target_state_pending = pending
        .coordinator
        .epoch_state
        .clients
        .iter()
        .find(|c| *c.id.signer() == target_key.to_bytes())
        .map(|c| c.state);
    println!("[+] target state while pending: {target_state_pending:?} (must be Healthy, not yet slashed)");
    if target_state_pending != Some(ClientState::Healthy) {
        return Err(anyhow!("target was slashed before the challenge window closed"));
    }

    println!("[+] the accused target posts a challenge, convening the tie-breaker committee");
    process_treasurer_run_open_challenge(&mut endpoint, &payer, target_keypair, &run).await?;

    println!("[+] {} tie-breakers re-audit and vote to OVERTURN the conviction", tie_breaker_quorum);
    for (i, appellate) in tie_breakers.iter().take(tie_breaker_quorum as usize).enumerate() {
        process_treasurer_run_submit_appeal_verdict(
            &mut endpoint,
            &payer,
            appellate,
            &run,
            &coordinator_account,
            &run_id,
            &target_key,
            target_index,
            true,
        )
        .await?;
        println!("    tie-breaker {} voted overturn", i + 1);
    }

    println!("[+] cranking the losing-side penalty against each convicting verifier");
    let verifier_keys: Vec<Pubkey> = verifiers
        .iter()
        .take(verifier_quorum as usize)
        .map(|k| k.pubkey())
        .collect();
    for position in 0..verifier_keys.len() {
        process_treasurer_run_slash_losing_verifier(
            &mut endpoint,
            &payer,
            &run,
            &coordinator_account,
            &run_id,
            &target_key,
            position as u16,
        )
        .await?;
        println!("    verifier at position {position} ejected for a wrongful conviction");
    }

    println!("[+] driving to epoch end to settle the bonds");
    let mut verifier_slashed = vec![0u64; verifier_keys.len()];
    let mut target_slashed = 0;
    for _ in 0..24 {
        sleep_seconds(COOLDOWN_TIME).await;
        let _ = process_coordinator_tick(
            &mut endpoint,
            &payer,
            &ticker,
            &coordinator_instance,
            &coordinator_account,
        )
        .await;
        for (i, key) in verifier_keys.iter().enumerate() {
            verifier_slashed[i] = slashed_of(&mut endpoint, &coordinator_account, key).await;
        }
        target_slashed = slashed_of(&mut endpoint, &coordinator_account, &target_key).await;
        if verifier_slashed.iter().all(|s| *s > 0) {
            break;
        }
    }

    println!();
    println!("Summary");
    println!("  verifiers            {}", verifiers.len());
    println!("  verifier quorum      {}", verifier_quorum);
    println!("  tie-breaker quorum   {}", tie_breaker_quorum);
    println!("  target slashed       {} (expected 0, the wrongful accusation is reversed)", target_slashed);
    for (i, s) in verifier_slashed.iter().enumerate() {
        println!("  verifier {} slashed   {} (expected {}, the losing side forfeits)", i + 1, s, SLASHING_RATE);
    }
    println!("  run                  {}", run);
    println!("  coordinator          {}", coordinator_account);

    if target_slashed != 0 {
        return Err(anyhow!("overturned target must not be slashed, got {}", target_slashed));
    }
    if verifier_slashed.iter().any(|s| *s != SLASHING_RATE) {
        return Err(anyhow!("each losing verifier must forfeit {}, got {:?}", SLASHING_RATE, verifier_slashed));
    }
    println!("[+] live devnet appeals court verified: the losing side lost its bond, the innocent target kept its own");
    Ok(())
}
