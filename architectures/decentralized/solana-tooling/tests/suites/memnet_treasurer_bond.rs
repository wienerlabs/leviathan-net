use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_request_withdraw;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

const WITHDRAW_DELAY_SECONDS: i64 = 100;

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
    let stranger = Keypair::new();
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

    let (run, _) = process_treasurer_run_create(
        &mut endpoint,
        &payer,
        &collateral_mint,
        &coordinator_account,
        RunCreateParams {
            index: 51,
            run_id: "Leviathan bond suite run".to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: join_authority.pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    process_treasurer_run_bond_config_update(
        &mut endpoint,
        &payer,
        &stranger,
        &run,
        RunBondConfigUpdateParams {
            bond_minimum_amount: 0,
            bond_withdraw_delay_seconds: WITHDRAW_DELAY_SECONDS,
        },
    )
    .await
    .unwrap_err();

    process_treasurer_run_bond_config_update(
        &mut endpoint,
        &payer,
        &main_authority,
        &run,
        RunBondConfigUpdateParams {
            bond_minimum_amount: 0,
            bond_withdraw_delay_seconds: WITHDRAW_DELAY_SECONDS,
        },
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
            1_000,
        )
        .await
        .unwrap();
    let run_collateral = endpoint
        .process_spl_associated_token_account_get_or_init(
            &payer,
            &run,
            &collateral_mint,
        )
        .await
        .unwrap();

    process_treasurer_participant_create(&mut endpoint, &payer, &client, &run)
        .await
        .unwrap();

    process_treasurer_participant_bond_deposit(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
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
        600,
    )
    .await
    .unwrap();

    assert_amount(&mut endpoint, &client_collateral, 400).await;
    assert_amount(&mut endpoint, &run_collateral, 600).await;
    assert_bond_state(&mut endpoint, &run, &client.pubkey(), 600, 0).await;
    assert_eq!(
        get_run(&mut endpoint, &run)
            .await
            .unwrap()
            .unwrap()
            .total_bonded_amount,
        600
    );

    process_treasurer_participant_bond_request_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &run,
        700,
    )
    .await
    .unwrap_err();

    process_treasurer_participant_bond_request_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &run,
        0,
    )
    .await
    .unwrap_err();

    process_treasurer_participant_bond_request_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &run,
        250,
    )
    .await
    .unwrap();

    process_treasurer_participant_bond_finalize_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap_err();

    endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY_SECONDS as u64)
        .await
        .unwrap();

    process_treasurer_participant_bond_finalize_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();

    assert_amount(&mut endpoint, &client_collateral, 650).await;
    assert_amount(&mut endpoint, &run_collateral, 350).await;
    assert_bond_state(&mut endpoint, &run, &client.pubkey(), 350, 0).await;
    assert_eq!(
        get_run(&mut endpoint, &run)
            .await
            .unwrap()
            .unwrap()
            .total_bonded_amount,
        350
    );

    process_treasurer_participant_bond_finalize_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap_err();

    process_treasurer_participant_bond_request_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &run,
        350,
    )
    .await
    .unwrap();
    endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY_SECONDS as u64)
        .await
        .unwrap();
    process_treasurer_participant_bond_finalize_withdraw(
        &mut endpoint,
        &payer,
        &client,
        &client_collateral,
        &collateral_mint,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();

    assert_amount(&mut endpoint, &client_collateral, 1_000).await;
    assert_amount(&mut endpoint, &run_collateral, 0).await;
    assert_bond_state(&mut endpoint, &run, &client.pubkey(), 0, 0).await;
    assert_eq!(
        get_run(&mut endpoint, &run)
            .await
            .unwrap()
            .unwrap()
            .total_bonded_amount,
        0
    );
}

async fn assert_amount(
    endpoint: &mut ToolboxEndpoint,
    account: &Pubkey,
    expected_amount: u64,
) {
    assert_eq!(
        endpoint
            .get_spl_token_account(account)
            .await
            .unwrap()
            .unwrap()
            .amount,
        expected_amount,
    );
}

async fn assert_bond_state(
    endpoint: &mut ToolboxEndpoint,
    run: &Pubkey,
    user: &Pubkey,
    expected_bond_amount: u64,
    expected_pending_amount: u64,
) {
    let participant =
        get_participant(&mut *endpoint, &find_participant(run, user))
            .await
            .unwrap()
            .unwrap();
    assert_eq!(participant.bond_amount, expected_bond_amount);
    assert_eq!(
        participant.bond_withdraw_pending_amount,
        expected_pending_amount
    );
}
