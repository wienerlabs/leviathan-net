// Reproduces the bond-gated claim on live devnet, the second of the two flows
// issue #2 asks for and that had only ever run in memnet
// (`memnet_treasurer_bond_gated_claim.rs`).
//
// A claim is the door through which a participant draws what it has earned. The
// gate is the bond: a participant with no bond posted cannot claim, so a client
// cannot take earnings and leave nothing behind to slash. This driver proves the
// gate on chain against the redeployed treasurer: the claim is rejected before
// the bond exists and accepted once it does.
//
// It also carries the finding-17 check: a zero withdraw delay is refused, so the
// dispute window can never be closed retroactively.

use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::anyhow;
use anyhow::Result;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_claim;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::signature::read_keypair_file;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

const BOND: u64 = 500;
const WITHDRAW_DELAY: i64 = 5;

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
    let run_id = format!("leviathan-claim-{}", index);
    println!("[+] run_id {}", run_id);

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let client = Keypair::new();

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
    let (run, _coordinator_instance) = process_treasurer_run_create(
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

    endpoint
        .process_spl_associated_token_account_get_or_init(&payer, &run, &collateral_mint)
        .await
        .unwrap();
    let client_collateral = endpoint
        .process_spl_associated_token_account_get_or_init(&payer, &client.pubkey(), &collateral_mint)
        .await
        .unwrap();
    endpoint
        .process_spl_token_mint_to(
            &payer,
            &collateral_mint,
            &mint_authority,
            &client_collateral,
            BOND,
        )
        .await
        .unwrap();

    // Finding 17: a zero withdraw delay would let a bond leave the moment it is
    // requested, closing the dispute window retroactively. The program refuses
    // it, so this must fail before we can set a real delay.
    println!("[+] a zero withdraw delay must be rejected (finding 17)");
    let zero_delay = process_treasurer_run_bond_config_update(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        RunBondConfigUpdateParams {
            bond_minimum_amount: BOND,
            bond_withdraw_delay_seconds: 0,
        },
    )
    .await;
    println!("    zero-delay bond config rejected = {}", zero_delay.is_err());
    if zero_delay.is_ok() {
        return Err(anyhow!("a zero withdraw delay must be rejected"));
    }

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

    process_treasurer_participant_create(&mut endpoint, &payer, &client, &run)
        .await
        .unwrap();

    // The gate itself: with no bond posted, the claim must be refused.
    println!("[+] a claim with no bond posted must be rejected (the gate)");
    let ungated = process_treasurer_participant_claim(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
        &coordinator_account,
        0,
    )
    .await;
    println!("    claim without a bond rejected = {}", ungated.is_err());
    if ungated.is_ok() {
        return Err(anyhow!("a claim without a bond must be rejected"));
    }

    println!("[+] client posts a bond of {BOND}");
    process_treasurer_participant_bond_deposit(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
        BOND,
    )
    .await
    .unwrap();
    let participant = get_participant(&mut endpoint, &find_participant(&run, &client.pubkey()))
        .await
        .unwrap()
        .unwrap();
    println!("    on-chain bond_amount = {}", participant.bond_amount);

    // Same claim, same client, now with a bond behind it: the gate opens.
    println!("[+] the same claim, now bonded, must be accepted");
    process_treasurer_participant_claim(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
        &coordinator_account,
        0,
    )
    .await
    .unwrap();

    println!();
    println!("Summary");
    println!("  bond_minimum        {}", BOND);
    println!("  bond posted         {}", participant.bond_amount);
    println!("  zero-delay refused  {}", zero_delay.is_err());
    println!("  claim before bond   rejected");
    println!("  claim after bond    accepted");
    println!("  run                 {}", run);
    println!("  coordinator account {}", coordinator_account);
    if participant.bond_amount != BOND {
        return Err(anyhow!(
            "expected bond_amount {}, got {}",
            BOND,
            participant.bond_amount
        ));
    }
    println!("[+] live devnet bond-gated claim verified");
    Ok(())
}
