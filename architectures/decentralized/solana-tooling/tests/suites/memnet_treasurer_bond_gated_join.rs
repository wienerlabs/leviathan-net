use psyche_solana_authorizer::find_authorization;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::get_accounts::get_authorization;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_authorize_join;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_deposit;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_bond_config_update;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_create;
use psyche_solana_treasurer::find_run;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

const RUN_ID: &str = "Leviathan bond gated join";
const INDEX: u64 = 73;
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
    let bonded = Keypair::new();
    let unbonded = Keypair::new();

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

    let run_pda = find_run(INDEX);
    let (run, _coordinator_instance) = process_treasurer_run_create(
        &mut endpoint,
        &payer,
        &collateral_mint,
        &coordinator_account,
        RunCreateParams {
            index: INDEX,
            run_id: RUN_ID.to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: run_pda,
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(run, run_pda);

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

    for client in [&bonded, &unbonded] {
        process_treasurer_participant_create(&mut endpoint, &payer, client, &run)
            .await
            .unwrap();
    }

    let bonded_collateral = endpoint
        .process_spl_associated_token_account_get_or_init(
            &payer,
            &bonded.pubkey(),
            &collateral_mint,
        )
        .await
        .unwrap();
    endpoint
        .process_spl_token_mint_to(
            &payer,
            &collateral_mint,
            &mint_authority,
            &bonded_collateral,
            BOND,
        )
        .await
        .unwrap();
    process_treasurer_participant_bond_deposit(
        &mut endpoint,
        &payer,
        &bonded,
        &bonded_collateral,
        &collateral_mint,
        &run,
        BOND,
    )
    .await
    .unwrap();

    let authorization = process_treasurer_participant_authorize_join(
        &mut endpoint,
        &payer,
        &bonded.pubkey(),
        &run,
    )
    .await
    .unwrap();
    let auth = get_authorization(&mut endpoint, &authorization)
        .await
        .unwrap()
        .unwrap();
    assert!(auth.active);
    assert_eq!(auth.grantor, run);
    assert!(auth.is_valid_for(&run, &bonded.pubkey(), JOIN_RUN_AUTHORIZATION_SCOPE));

    process_treasurer_participant_authorize_join(
        &mut endpoint,
        &payer,
        &unbonded.pubkey(),
        &run,
    )
    .await
    .unwrap_err();
    let unbonded_authorization =
        find_authorization(&run, &unbonded.pubkey(), JOIN_RUN_AUTHORIZATION_SCOPE);
    assert!(get_authorization(&mut endpoint, &unbonded_authorization)
        .await
        .unwrap()
        .is_none());
}
