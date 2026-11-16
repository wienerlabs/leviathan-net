//! `leviathan-indexer` reading real runs off the real programs: what it
//! derives from a run id, and every value it exports against the accounts it
//! came from (wienerlabs/leviathan-net#57).

use leviathan_indexer::live::read_run;
use leviathan_indexer::live::RunSnapshot;
use leviathan_indexer::live::TreasurerAddresses;
use leviathan_indexer::metrics::IndexerMetrics;
use leviathan_indexer::observe::RunObservation;
use leviathan_indexer::verdicts::VerdictLedger;
use leviathan_telemetry::exposition::render;
use prometheus_client::registry::Registry;
use psyche_coordinator::RunState;
use psyche_solana_coordinator::logic::InitCoordinatorParams;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_init;
use psyche_solana_treasurer::logic::MIN_VERIFIER_COMMITTEE;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::BOND;
use super::bonded_run::EARNING_RATE_TOTAL_SHARED;
use super::bonded_run::NODES;
use super::bonded_run::SLASHING_RATE;

const RUN_INDEX: u64 = 5_701;
const RUN_ID: &str = "Leviathan indexer";
const SUBJECT: usize = 2;

async fn snapshot(endpoint: &mut ToolboxEndpoint, run_id: &str) -> RunSnapshot {
    read_run(endpoint, None, run_id).await.unwrap()
}

async fn vault_balance(harness: &mut Harness) -> u64 {
    let vault = harness.run_collateral;
    harness
        .endpoint
        .get_spl_token_account(&vault)
        .await
        .unwrap()
        .unwrap()
        .amount
}

fn scrape(observation: &RunObservation) -> String {
    let mut registry = Registry::default();
    IndexerMetrics::register(&mut registry).observe(RUN_ID, observation);
    render(&registry).unwrap()
}

/// A bond-gated treasurer run read from nothing but its run id, through one
/// epoch with a conviction, a claim, and a top-up that pays for its earnings.
#[tokio::test]
pub async fn the_indexer_reads_a_bonded_run_from_its_run_id() {
    let mut h = Harness::new(RUN_INDEX, RUN_ID, JoinGate::Bond).await;
    let mut ledger = VerdictLedger::default();

    // Everything is found from the run id: the instance PDA, the coordinator
    // account it names, and the treasurer run behind it with its vault.
    let first = snapshot(&mut h.endpoint, RUN_ID).await;
    assert_eq!(first.addresses.coordinator_instance, h.coordinator_instance);
    assert_eq!(first.addresses.coordinator_account, h.coordinator_account);
    assert_eq!(
        first.addresses.treasurer,
        Some(TreasurerAddresses {
            run: h.run,
            vault: h.run_collateral,
        })
    );
    // The snapshot's slot is the cluster's, read with the accounts.
    let clock = h.endpoint.get_sysvar_clock().await.unwrap();
    assert_eq!(first.clock.slot, clock.slot);
    assert_eq!(first.clock.unix_timestamp, clock.unix_timestamp);
    assert_eq!(first.clock.slot_lag(), 0);

    let before = first.observe(&mut ledger);
    assert_eq!(before.run_state, RunState::WaitingForMembers as u8);
    assert!(!before.training);
    let treasurer = before.treasurer.as_ref().unwrap();
    // Between epochs the rates in force are the ones the next epoch starts
    // with, which `Harness::new` set through `run_update`.
    assert_eq!(treasurer.epoch_earning_rate, EARNING_RATE_TOTAL_SHARED);
    assert_eq!(treasurer.epoch_slashing_rate, SLASHING_RATE);
    assert_eq!(treasurer.bond_minimum, BOND);
    // Every node posted a bond and nothing else is in the vault yet.
    let bonded = BOND * NODES as u64;
    assert_eq!(treasurer.bonded_amount, bonded);
    assert_eq!(treasurer.vault_balance, vault_balance(&mut h).await);
    assert_eq!(treasurer.vault_balance, bonded);
    assert_eq!(treasurer.bond_shortfall, 0);
    assert_eq!(treasurer.obligation_shortfall, 0);
    assert_eq!(treasurer.verdicts.by_status, [0; 6]);

    // Into the first epoch, with every node joined through the bond gate.
    for node in 0..NODES {
        h.join(node).await.unwrap();
    }
    h.advance_to(RunState::RoundTrain).await;
    let training = snapshot(&mut h.endpoint, RUN_ID).await.observe(&mut ledger);
    assert!(training.training);
    assert_eq!(training.registered_clients, NODES as u64);
    assert_eq!(training.epoch_clients, [NODES as u64, 0, 0, 0]);
    let treasurer = training.treasurer.as_ref().unwrap();
    // The genesis rehearsal audits 10% of a round's seats, which over three
    // clients is no seat at all: no committee can convict this run.
    assert_eq!(training.audit_probability, 0.1);
    assert_eq!(treasurer.verifier_committee_size, 0);
    assert!(treasurer.verifier_committee_size < MIN_VERIFIER_COMMITTEE);
    // A third of the epoch's earnings against a 200 slash, one audit in ten:
    // a cheat pays (wienerlabs/leviathan-net#57 reports it instead of a
    // hand-entered verdict).
    let share = EARNING_RATE_TOTAL_SHARED / NODES as u64;
    assert_eq!(treasurer.security.reward, share as f64);
    assert_eq!(treasurer.security.effective_penalty, SLASHING_RATE as f64);
    assert!(!treasurer.security.economically_secure);

    // The run authority convicts one node in the round under way.
    h.convict(SUBJECT).await;
    let convicted =
        snapshot(&mut h.endpoint, RUN_ID).await.observe(&mut ledger);
    // healthy, dropped, withdrawn, ejected
    assert_eq!(convicted.epoch_clients, [NODES as u64 - 1, 0, 0, 1]);
    assert_eq!(convicted.active_clients(), NODES as u64 - 1);
    assert_eq!(convicted.epoch_roster_size(), NODES as u64);
    // The authority's slash writes no verdict account; the ledger shows it at
    // the epoch's end.
    assert_eq!(
        convicted.treasurer.as_ref().unwrap().verdicts.by_status,
        [0; 6]
    );

    h.finish_epoch().await;
    let settled = snapshot(&mut h.endpoint, RUN_ID).await.observe(&mut ledger);
    let honest_share = EARNING_RATE_TOTAL_SHARED / (NODES as u64 - 1);
    assert_eq!(settled.epoch, 1);
    assert_eq!(settled.slashed_points, SLASHING_RATE);
    assert_eq!(settled.convicted_clients, 1);
    assert_eq!(settled.earned_points, honest_share * (NODES as u64 - 1));
    let treasurer = settled.treasurer.as_ref().unwrap();
    assert_eq!(treasurer.unclaimed_earnings, settled.earned_points);
    // Nobody funded the rewards, so the vault holds the bonds and owes the
    // earnings on top of them: underfunded, not drained.
    assert_eq!(treasurer.bond_shortfall, 0);
    assert_eq!(treasurer.obligation_shortfall, settled.earned_points);
    let scraped = scrape(&settled);
    for line in [
        format!(
            r#"leviathan_run_obligation_shortfall{{run_id="{RUN_ID}"}} {}.0"#,
            settled.earned_points
        ),
        format!(r#"leviathan_run_bond_shortfall{{run_id="{RUN_ID}"}} 0.0"#),
        format!(
            r#"leviathan_run_slashed_points{{run_id="{RUN_ID}"}} {SLASHING_RATE}.0"#
        ),
        format!(r#"leviathan_run_treasurer_managed{{run_id="{RUN_ID}"}} 1.0"#),
    ] {
        assert!(scraped.contains(&line), "missing {line} in\n{scraped}");
    }

    // A top-up pays for the earnings, and a claim pays one node out of it.
    h.top_up_rewards(settled.earned_points).await;
    h.claim(0, honest_share).await.unwrap();
    let claimed = snapshot(&mut h.endpoint, RUN_ID).await.observe(&mut ledger);
    let treasurer = claimed.treasurer.as_ref().unwrap();
    assert_eq!(treasurer.claimed_collateral, honest_share);
    assert_eq!(
        treasurer.unclaimed_earnings,
        settled.earned_points - honest_share
    );
    assert_eq!(treasurer.vault_balance, vault_balance(&mut h).await);
    assert_eq!(
        treasurer.vault_balance,
        bonded + settled.earned_points - honest_share
    );
    assert_eq!(treasurer.obligation_shortfall, 0);
    assert_eq!(treasurer.bond_shortfall, 0);
}

/// A run the coordinator manages alone, with no treasurer, has no economics
/// and no vault to watch.
#[tokio::test]
pub async fn a_coordinator_only_run_is_read_without_a_treasurer() {
    let mut endpoint = create_memnet_endpoint().await;
    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 5_000_000_000)
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
    let run_id = "Leviathan indexer, no treasurer";
    let coordinator_instance = process_coordinator_init(
        &mut endpoint,
        &payer,
        &coordinator_account,
        InitCoordinatorParams {
            run_id: run_id.to_string(),
            main_authority: Keypair::new().pubkey(),
            join_authority: Keypair::new().pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    let snapshot = snapshot(&mut endpoint, run_id).await;
    assert_eq!(
        snapshot.addresses.coordinator_instance,
        coordinator_instance
    );
    assert_eq!(snapshot.addresses.coordinator_account, coordinator_account);
    assert_eq!(snapshot.addresses.treasurer, None);
    let observation = snapshot.observe(&mut VerdictLedger::default());
    assert!(observation.treasurer.is_none());
    let scraped = scrape(&observation);
    assert!(scraped.contains(&format!(
        r#"leviathan_run_treasurer_managed{{run_id="{RUN_ID}"}} 0.0"#
    )));
    assert!(!scraped.contains("leviathan_run_vault_balance{"));
}

/// A run id nothing was created for is an error, not an empty run, and so is
/// one too long to be a PDA seed, which no run can have: deriving its address
/// used to panic.
#[tokio::test]
pub async fn a_run_that_does_not_exist_is_refused() {
    let mut endpoint = create_memnet_endpoint().await;
    let Err(err) =
        read_run(&mut endpoint, None, "Leviathan indexer, not made").await
    else {
        panic!("a run that does not exist was read");
    };
    assert!(err.to_string().contains("does not exist"), "{err:#}");

    let too_long = "Leviathan indexer, a run id past 32 bytes";
    assert!(too_long.len() > 32);
    let Err(err) = read_run(&mut endpoint, None, too_long).await else {
        panic!("a run id longer than a seed was read");
    };
    assert!(
        err.to_string().contains("longer than a PDA seed allows"),
        "{err:#}"
    );
}
