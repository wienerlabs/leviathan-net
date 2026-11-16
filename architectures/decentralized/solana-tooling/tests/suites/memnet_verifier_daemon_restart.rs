//! A run restarted from an earlier step trains that step again in a new
//! epoch. The payloads kept from before the restart stay on disk, under names
//! of their own, and the daemon has to leave them be: nothing can convict on
//! them now (wienerlabs/leviathan-net#67).

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::contribution_key;
use psyche_coordinator::RunState;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::signed_commitment::PayloadHeader;
use solana_sdk::signer::Signer;

use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::NODES;
use super::bonded_run::SUBJECT;
use super::signed_payload::fixture;
use super::signed_payload::keep_reference;
use super::signed_payload::keep_signed_payload;
use super::signed_payload::payload_header;
use super::signed_payload::FRAUD_PAYLOAD;
use super::signed_payload::HONEST_PAYLOAD;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "levrestart-{}-{}",
        std::process::id(),
        tag
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The subject's cheating payload of the round under way, kept as a peer
/// keeps it, and the honest result for its batch the daemon compares it to.
/// Returns how a daemon audits them, and the payload's key.
async fn kept_fraud(harness: &mut Harness, tag: &str) -> (AuditConfig, String) {
    kept_fraud_naming(harness, tag, |header| header).await
}

/// As `kept_fraud`, with the header the subject signed made by `header_of`
/// from the one of the round under way.
async fn kept_fraud_naming(
    harness: &mut Harness,
    tag: &str,
    header_of: impl Fn(PayloadHeader) -> PayloadHeader,
) -> (AuditConfig, String) {
    let tie_breakers = get_run(&mut harness.endpoint, &harness.run)
        .await
        .unwrap()
        .expect("the run exists")
        .tie_breaker_committee_size;
    let coordinator = harness.state().await.coordinator;
    let subject = &harness.nodes[SUBJECT];
    let header = header_of(payload_header(
        &coordinator,
        tie_breakers,
        &subject.identity(),
    ));
    let run_id = String::from(&coordinator.run_id);
    let submitted_dir = scratch(&format!("{tag}-submitted"));
    let reference_dir = scratch(&format!("{tag}-reference"));
    let kept = keep_signed_payload(
        &submitted_dir,
        &fixture(FRAUD_PAYLOAD),
        &subject.wallet,
        &subject.p2p,
        &run_id,
        header,
    );
    keep_reference(
        &reference_dir,
        &fixture(HONEST_PAYLOAD),
        &subject.wallet,
        header,
    );
    let config = AuditConfig {
        run_id: harness.run_id.clone(),
        submitted_dir,
        reference_dir: Some(reference_dir),
        band: 0.05,
        audit_assigned: false,
        dry_run: false,
        verdict_mode: false,
    };
    (
        config,
        contribution_key(&kept).expect("a dump the daemon indexes"),
    )
}

/// A bonded run of every node, training, whose authority runs the daemon and
/// pays for what it sends.
async fn training_run(index: u64, run_id: &str) -> Harness {
    let mut harness = Harness::new(index, run_id, JoinGate::Bond).await;
    let authority = harness.main_authority.pubkey();
    harness
        .endpoint
        .request_airdrop(&authority, 1_000_000_000)
        .await
        .unwrap();
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness
}

async fn pass(
    harness: &mut Harness,
    config: &AuditConfig,
    judged: &mut HashSet<String>,
) -> PassReport {
    let authority = harness.main_authority.insecure_clone();
    audit_pass(
        &mut harness.endpoint,
        &authority,
        &harness.coordinator_account,
        &harness.run,
        config,
        None,
        judged,
    )
    .await
    .unwrap()
}

/// A fraud kept from before a restart is from an epoch that is over, though
/// the run trains its step again. The daemon sets it aside, rather than send
/// a conviction the chain refuses and report it, every pass, as a fraud the
/// chain never saw.
#[tokio::test]
pub async fn a_fraud_kept_from_before_a_restart_is_set_aside() {
    let mut harness = training_run(444, "Leviathan restarted daemon").await;
    let (config, key) = kept_fraud(&mut harness, "before").await;
    let step = harness.state().await.coordinator.progress.step;

    harness.finish_epoch().await;
    harness.restart_from_step(step).await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    assert_eq!(harness.state().await.coordinator.progress.step, step);

    let mut judged = HashSet::new();
    assert_eq!(
        pass(&mut harness, &config, &mut judged).await,
        PassReport::default()
    );
    assert!(
        judged.contains(&key),
        "the payload's epoch is over: settled"
    );
    assert_eq!(
        pass(&mut harness, &config, &mut judged).await,
        PassReport::default()
    );

    // The same fraud committed in this epoch is convicted as ever.
    let (current, _) = kept_fraud(&mut harness, "after").await;
    assert_eq!(
        pass(&mut harness, &current, &mut HashSet::new()).await,
        PassReport {
            fraud_found: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
}

/// A payload naming an epoch later than the one the daemon read the run in
/// is ahead of its view, not over: the run moved on after the pass read it,
/// or the RPC lags. It is left for the next pass, which reads the run again,
/// rather than settled for the session and never audited.
#[tokio::test]
pub async fn a_fraud_newer_than_the_daemons_view_is_looked_at_again() {
    let mut harness = training_run(445, "Leviathan daemon view behind").await;
    let (config, key) =
        kept_fraud_naming(&mut harness, "ahead", |header| PayloadHeader {
            epoch_start: header.epoch_start + 1,
            ..header
        })
        .await;
    let mut judged = HashSet::new();
    for _ in 0..2 {
        assert_eq!(
            pass(&mut harness, &config, &mut judged).await,
            PassReport::default()
        );
        assert!(
            !judged.contains(&key),
            "a payload ahead of the view was settled"
        );
    }
}
