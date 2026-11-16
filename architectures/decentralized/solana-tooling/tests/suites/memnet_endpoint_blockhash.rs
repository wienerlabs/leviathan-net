//! The memnet endpoint signs every transaction with the blockhash it last
//! saw, and renews it only when the clock moves on. Meanwhile
//! `solana-program-test` registers a new blockhash every simulated slot, a few
//! milliseconds of wall time apart, and refuses a transaction whose blockhash
//! is more than `MAX_PROCESSING_AGE` of them old: about a second. A suite that
//! sends transactions for longer than that without a new slot has them
//! refused, more often the more loaded the machine
//! (wienerlabs/leviathan-net#53).

use std::time::Duration;

use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

/// Longer than `MAX_PROCESSING_AGE` simulated slots take, with room.
const PAST_THE_WINDOW: Duration = Duration::from_secs(3);

#[tokio::test]
pub async fn a_blockhash_ages_out_and_a_new_slot_renews_it() {
    let mut endpoint = create_memnet_endpoint().await;
    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 1_000_000_000)
        .await
        .unwrap();

    tokio::time::sleep(PAST_THE_WINDOW).await;
    let refused = endpoint
        .process_spl_token_mint_new(&payer, &Keypair::new().pubkey(), None, 6)
        .await
        .expect_err(
            "a transaction signed with an aged-out blockhash was taken",
        );
    assert!(
        format!("{refused:#}").contains("Blockhash not found"),
        "refused for another reason: {refused:#}"
    );

    endpoint.forward_clock_slot(1).await.unwrap();
    endpoint
        .process_spl_token_mint_new(&payer, &Keypair::new().pubkey(), None, 6)
        .await
        .unwrap();
}
