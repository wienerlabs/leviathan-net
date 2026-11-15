//! The join gate of a bond-gated run has to hold on every join, not only the
//! first one (wienerlabs/leviathan-net#34).
//!
//! A run whose join authority is its own treasurer `Run` PDA admits a node only
//! against a posted bond. These suites drive the real programs through whole
//! epochs on the real genesis rehearsal config, and check the gate each time a
//! node comes back for another epoch.

use anchor_lang::error::ErrorCode as AnchorErrorCode;
use psyche_coordinator::RunState;
use psyche_solana_rpc::instructions::treasurer_participant_join_run;
use psyche_solana_rpc::SolanaBackend;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

use super::bonded_run::first_epoch;
use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::BOND;
use super::bonded_run::EARNING_RATE_TOTAL_SHARED;
use super::bonded_run::NODES;
use super::bonded_run::SLASHING_RATE;
use super::bonded_run::SUBJECT;
use super::bonded_run::WITHDRAW_DELAY;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

#[tokio::test]
pub async fn a_bonded_node_joins_epoch_after_epoch_through_the_treasurer() {
    let mut harness =
        Harness::new(340, "Leviathan join gate honest", JoinGate::Bond).await;
    for epoch in 0..2 {
        for node in 0..NODES {
            harness.join(node).await.unwrap();
            assert_eq!(
                harness.authorization_active(node).await,
                Some(false),
                "the join authorization stayed open after the join of epoch {epoch}"
            );
        }
        harness.advance_to(RunState::RoundTrain).await;
        assert_eq!(harness.roster().await, (0..NODES).collect::<Vec<_>>());
        harness.finish_epoch().await;
    }
    for node in 0..NODES {
        let (earned, slashed) = harness.ledger(node).await;
        assert_eq!(slashed, 0);
        assert!(
            earned > 0,
            "node {node} trained two epochs and earned nothing"
        );
    }
}

#[tokio::test]
pub async fn a_direct_coordinator_join_is_refused() {
    let mut harness =
        Harness::new(341, "Leviathan join gate direct", JoinGate::Bond).await;
    first_epoch(&mut harness).await;
    assert_eq!(harness.authorization_active(SUBJECT).await, Some(false));
    assert_eq!(
        refusal_code(harness.join_directly(SUBJECT).await),
        u32::from(AnchorErrorCode::ConstraintRaw),
        "the coordinator admitted a node on its treasurer authorization without the treasurer"
    );
    harness.join(SUBJECT).await.unwrap();
}

#[tokio::test]
pub async fn an_unbonded_node_is_refused() {
    let mut harness =
        Harness::new(342, "Leviathan join gate unbonded", JoinGate::Bond).await;
    let unbonded = harness.add_node().await;
    assert_eq!(
        refusal_code(harness.join(unbonded).await),
        treasurer_refusal(ProgramError::BondBelowMinimum)
    );
    assert_eq!(
        harness.authorization_active(unbonded).await,
        None,
        "a refused join still left an authorization behind"
    );
    harness.deposit(unbonded, BOND).await.unwrap();
    harness.join(unbonded).await.unwrap();
}

#[tokio::test]
pub async fn a_withdrawn_bond_does_not_keep_its_seat() {
    let mut harness =
        Harness::new(343, "Leviathan join gate withdrawn", JoinGate::Bond)
            .await;
    first_epoch(&mut harness).await;

    harness.request_withdraw(SUBJECT, BOND).await;
    harness
        .endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    harness.finalize_withdraw(SUBJECT).await;
    assert_eq!(harness.bond(SUBJECT).await, 0);

    harness.join(0).await.unwrap();
    harness.join(1).await.unwrap();
    assert_eq!(
        refusal_code(harness.join(SUBJECT).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a node that withdrew its whole bond was admitted to the bond-gated run again"
    );
    assert_eq!(
        refusal_code(harness.join_directly(SUBJECT).await),
        u32::from(AnchorErrorCode::ConstraintRaw)
    );
    harness.advance_to(RunState::RoundTrain).await;
    assert_eq!(harness.roster().await, vec![0, 1]);
}

#[tokio::test]
pub async fn a_pending_withdrawal_blocks_the_join() {
    let mut harness =
        Harness::new(344, "Leviathan join gate pending", JoinGate::Bond).await;
    first_epoch(&mut harness).await;

    // One unit on its way out is enough: what stays is below the minimum.
    harness.request_withdraw(SUBJECT, 1).await;
    assert_eq!(harness.bond(SUBJECT).await, BOND);
    assert_eq!(
        refusal_code(harness.join(SUBJECT).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a bond already on its way out backed a seat in the next epoch"
    );

    // Posting the shortfall makes the bond that stays whole again.
    harness.deposit(SUBJECT, 1).await.unwrap();
    harness.join(SUBJECT).await.unwrap();
}

#[tokio::test]
pub async fn a_convicted_node_does_not_rejoin_on_its_forfeited_bond() {
    let mut harness =
        Harness::new(345, "Leviathan join gate convicted", JoinGate::Bond)
            .await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness.convict(SUBJECT).await;
    harness.finish_epoch().await;
    assert_eq!(harness.ledger(SUBJECT).await.1, SLASHING_RATE);
    assert_eq!(harness.bond(SUBJECT).await, BOND);

    harness.join(0).await.unwrap();
    harness.join(1).await.unwrap();
    assert_eq!(
        refusal_code(harness.join(SUBJECT).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a convicted node re-joined on a bond its conviction had already forfeited"
    );

    // A fresh bond that covers the forfeit is the price of coming back.
    harness.deposit(SUBJECT, SLASHING_RATE).await.unwrap();
    harness.join(SUBJECT).await.unwrap();
    harness.advance_to(RunState::RoundTrain).await;
    assert_eq!(harness.roster().await, (0..NODES).collect::<Vec<_>>());
}

#[tokio::test]
pub async fn a_claim_needs_the_bond_a_conviction_left() {
    let mut harness =
        Harness::new(346, "Leviathan join gate claim", JoinGate::Bond).await;
    harness.top_up_rewards(2 * EARNING_RATE_TOTAL_SHARED).await;
    first_epoch(&mut harness).await;
    let (earned, _) = harness.ledger(SUBJECT).await;
    assert!(earned > 0);

    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness.convict(SUBJECT).await;
    harness.finish_epoch().await;
    assert_eq!(harness.ledger(SUBJECT).await, (earned, SLASHING_RATE));

    assert_eq!(
        refusal_code(harness.claim(SUBJECT, earned).await),
        treasurer_refusal(ProgramError::BondBelowMinimum),
        "a node claimed its earnings against a bond its conviction had already forfeited"
    );

    harness.deposit(SUBJECT, SLASHING_RATE).await.unwrap();
    let before = harness.collateral(SUBJECT).await;
    harness.claim(SUBJECT, earned).await.unwrap();
    assert_eq!(harness.collateral(SUBJECT).await, before + earned);
}

#[tokio::test]
pub async fn a_run_with_an_outside_join_authority_is_not_joined_through_the_treasurer(
) {
    let outside = Keypair::new();
    let mut harness = Harness::new(
        347,
        "Leviathan join gate outside",
        JoinGate::Outside(outside.pubkey()),
    )
    .await;
    assert_eq!(
        refusal_code(harness.join(0).await),
        treasurer_refusal(ProgramError::RunNotBondGated)
    );
    assert_eq!(harness.authorization_active(0).await, None);
}

/// The client decides how to join from chain state alone, and the instruction
/// it builds for a bond-gated run is the one the treasurer accepts.
#[tokio::test]
pub async fn the_client_joins_a_bond_gated_run_through_the_treasurer() {
    let mut harness =
        Harness::new(348, "Leviathan join gate client", JoinGate::Bond).await;
    let run = harness.run;
    let coordinator_instance = harness.coordinator_instance;
    let run_account = harness.endpoint.get_account(&run).await.unwrap();
    assert!(SolanaBackend::is_bond_gated_run(
        &run,
        &coordinator_instance,
        run_account.as_ref()
    )
    .unwrap());
    // The same run account does not gate a coordinator instance it was not
    // created for.
    let elsewhere = Keypair::new().pubkey();
    assert!(!SolanaBackend::is_bond_gated_run(
        &run,
        &elsewhere,
        run_account.as_ref()
    )
    .unwrap());

    // The node pays for its own join, as the client does with its wallet.
    let wallet = harness.nodes[0].wallet.insecure_clone();
    harness
        .endpoint
        .request_airdrop(&wallet.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    let instruction = treasurer_participant_join_run(
        &run,
        &coordinator_instance,
        &harness.coordinator_account,
        &wallet.pubkey(),
        harness.nodes[0].identity(),
    );
    harness
        .endpoint
        .process_instruction(&wallet, instruction)
        .await
        .unwrap();
    assert_eq!(harness.authorization_active(0).await, Some(false));
    let identity = harness.nodes[0].identity();
    assert!(
        harness
            .state()
            .await
            .clients_state
            .clients
            .iter()
            .any(|client| client.id == identity),
        "the client's join did not register the node with the coordinator"
    );
}

/// A run whose join authority is an outside key is joined directly, and the
/// client sees that from the same chain state.
#[tokio::test]
pub async fn the_client_joins_an_outside_authority_run_directly() {
    let outside = Keypair::new();
    let mut harness = Harness::new(
        349,
        "Leviathan join gate client out",
        JoinGate::Outside(outside.pubkey()),
    )
    .await;
    let run = harness.run;
    let coordinator_instance = harness.coordinator_instance;
    let outside_account = harness
        .endpoint
        .get_account(&outside.pubkey())
        .await
        .unwrap();
    assert!(!SolanaBackend::is_bond_gated_run(
        &outside.pubkey(),
        &coordinator_instance,
        outside_account.as_ref()
    )
    .unwrap());
    // The treasurer run itself exists and is a run, but it is not this run's
    // join authority, so it does not gate the join either.
    let run_account = harness.endpoint.get_account(&run).await.unwrap();
    assert!(run_account.is_some());
    assert!(!SolanaBackend::is_bond_gated_run(
        &run,
        &coordinator_instance,
        run_account.as_ref()
    )
    .unwrap());
}

/// What the gates count after a conviction is what settlement leaves: the
/// unforfeited bond read from chain state beforehand is exactly what
/// `participant_bond_finalize_withdraw` then pays out of a full withdrawal.
#[tokio::test]
pub async fn settlement_pays_out_exactly_the_unforfeited_bond() {
    let mut harness =
        Harness::new(350, "Leviathan join gate settle", JoinGate::Bond).await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness.convict(SUBJECT).await;
    harness.finish_epoch().await;

    let (_, slashed) = harness.ledger(SUBJECT).await;
    let predicted =
        harness.participant(SUBJECT).await.unforfeited_bond(slashed);
    assert_eq!(predicted, BOND - SLASHING_RATE);

    let before = harness.collateral(SUBJECT).await;
    harness.request_withdraw(SUBJECT, BOND).await;
    harness
        .endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    harness.finalize_withdraw(SUBJECT).await;
    assert_eq!(harness.collateral(SUBJECT).await - before, predicted);
    assert_eq!(harness.bond(SUBJECT).await, 0);
}

/// The bond is read from the run's own coordinator: a join that names another
/// run's coordinator accounts is refused, whichever of the two it swaps.
#[tokio::test]
pub async fn a_join_against_another_runs_coordinator_is_refused() {
    let mut harness =
        Harness::new(351, "Leviathan join gate mismatch", JoinGate::Bond).await;
    let (other_instance, other_account) = harness
        .add_other_run(352, "Leviathan join gate other")
        .await;
    let wallet = harness.nodes[0].wallet.insecure_clone();
    harness
        .endpoint
        .request_airdrop(&wallet.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    let swaps = [
        (other_instance, harness.coordinator_account),
        (harness.coordinator_instance, other_account),
        (other_instance, other_account),
    ];
    for (coordinator_instance, coordinator_account) in swaps {
        let instruction = treasurer_participant_join_run(
            &harness.run,
            &coordinator_instance,
            &coordinator_account,
            &wallet.pubkey(),
            harness.nodes[0].identity(),
        );
        assert_eq!(
            refusal_code(
                harness
                    .endpoint
                    .process_instruction(&wallet, instruction)
                    .await
            ),
            u32::from(AnchorErrorCode::ConstraintRaw),
        );
    }
    assert_eq!(harness.authorization_active(0).await, None);
    harness.join(0).await.unwrap();
}
