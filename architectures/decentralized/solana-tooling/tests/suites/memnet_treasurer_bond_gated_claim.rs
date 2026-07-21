use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_claim;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

const RUN_ID: &str = "Leviathan bond gated claim";
const BOND: u64 = 500;
const WITHDRAW_DELAY: i64 = 100;

#[tokio::test]
pub async fn run() {
    let mut endpoint = create_memnet_endpoint().await;

    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 5_000_000_000)
        .await
        .unwrap();

    let mint_authority = Keypair::new();
    let main_authority = Keypair::new();
    let join_authority = Keypair::new();
    let client = Keypair::new();

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

    let (run, _coordinator_instance) = process_treasurer_run_create(
        &mut endpoint,
        &payer,
        &collateral_mint,
        &coordinator_account,
        RunCreateParams {
            index: 72,
            run_id: RUN_ID.to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: join_authority.pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    endpoint
        .process_spl_associated_token_account_get_or_init(
            &payer,
            &run,
            &collateral_mint,
        )
        .await
        .unwrap();
    let client_collateral = endpoint
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
            &client_collateral,
            BOND,
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

    process_treasurer_participant_create(&mut endpoint, &payer, &client, &run)
        .await
        .unwrap();

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
    .unwrap_err();

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
}
