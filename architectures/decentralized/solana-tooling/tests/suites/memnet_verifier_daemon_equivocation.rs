//! The verifier daemon convicts a peer that signed two different payloads for
//! one batch. It works from the records a client keeps, with no replay and no
//! committee (wienerlabs/leviathan-net#46).

use leviathan_verifier::PayloadBand;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::contribution_key;
use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::daemon_equivocation::equivocation_key;
use psyche_solana_tooling::signed_commitment::PayloadHeader;
use psyche_solana_treasurer::state::VerdictStatus;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

use super::appeals_harness::*;
use super::chain_state::coordinator_state;
use super::signed_payload::fixture;
use super::signed_payload::identity;
use super::signed_payload::keep_equivocation;
use super::signed_payload::keep_signed_payload;
use super::signed_payload::payload_header;
use super::signed_payload::FRAUD_PAYLOAD;
use super::signed_payload::HONEST_PAYLOAD;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "levequiv-{}-{}",
        std::process::id(),
        tag
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A directory in which a node kept the target's payload for its batch this
/// round, and beside it the record of a second payload for the same batch,
/// signed by `signer`. Returns the directory and the payload's key.
async fn kept_payloads(
    h: &mut AppealsHarness,
    tag: &str,
    signer: &Keypair,
) -> (PathBuf, String) {
    kept_payloads_naming(h, tag, signer, |header| header).await
}

/// As `kept_payloads`, with both payloads signed for the header `header_of`
/// makes from the one of the round under way.
async fn kept_payloads_naming(
    h: &mut AppealsHarness,
    tag: &str,
    signer: &Keypair,
    header_of: impl Fn(PayloadHeader) -> PayloadHeader,
) -> (PathBuf, String) {
    let coordinator =
        coordinator_state(&mut h.endpoint, &h.coordinator_account)
            .await
            .coordinator;
    let target = &h.clients[h.target_client_idx];
    let header = header_of(payload_header(
        &coordinator,
        &identity(target, &h.p2ps[h.target_client_idx]),
    ));
    let dir = scratch(tag);
    let kept = keep_signed_payload(
        &dir,
        &fixture(FRAUD_PAYLOAD),
        target,
        &h.p2ps[h.target_client_idx],
        &h.run_id,
        header,
    );
    keep_equivocation(
        &kept,
        &fixture(HONEST_PAYLOAD),
        signer,
        &h.run_id,
        header,
    );
    let key = contribution_key(&kept).expect("a dump the daemon indexes");
    (dir, key)
}

/// How a daemon audits `submitted_dir`. Its reference directory is empty, so
/// it replays nothing: whatever a pass does, it does from the records alone.
/// It audits only its committee's assignments, which an equivocation does not
/// wait for.
fn config(
    h: &AppealsHarness,
    submitted_dir: PathBuf,
    tag: &str,
    dry_run: bool,
) -> AuditConfig {
    AuditConfig {
        run_id: h.run_id.clone(),
        submitted_dir,
        reference_dir: Some(scratch(&format!("{tag}-reference"))),
        band: PayloadBand::Distance(0.05),
        audit_assigned: true,
        dry_run,
        verdict_mode: true,
        continuity: false,
    }
}

async fn pass(
    h: &mut AppealsHarness,
    daemon: &Keypair,
    config: &AuditConfig,
    judged: &mut HashSet<String>,
) -> PassReport {
    audit_pass(
        &mut h.endpoint,
        daemon,
        &h.coordinator_account,
        &h.run,
        config,
        None,
        judged,
    )
    .await
    .unwrap()
}

/// A verifier with lamports to send what it proves.
async fn funded_daemon(h: &mut AppealsHarness) -> Keypair {
    let daemon = Keypair::new();
    h.endpoint
        .request_airdrop(&daemon.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    daemon
}

/// What a pass reports when it finds a proof and commits to it, to reveal it
/// on a later pass (wienerlabs/leviathan-net#92).
fn committed() -> PassReport {
    PassReport {
        equivocations: 1,
        ..PassReport::default()
    }
}

/// A pass that commits to the proof, leaving the peer as it was, and, a slot
/// later, the pass that reveals it. Returns what the second reported.
async fn commit_then_reveal(
    h: &mut AppealsHarness,
    daemon: &Keypair,
    config: &AuditConfig,
    judged: &mut HashSet<String>,
) -> PassReport {
    let target_key = h.target_key;
    assert_eq!(pass(h, daemon, config, judged).await, committed());
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy),
        "the proof landed before it was revealed"
    );
    // A pass in the slot it committed in waits for a later one, which the
    // chain takes the proof from, and counts no failure.
    assert_eq!(pass(h, daemon, config, judged).await, committed());
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );
    h.endpoint.forward_clock_slot(1).await.unwrap();
    pass(h, daemon, config, judged).await
}

/// A pass finds the second payload beside the first and ejects the peer that
/// signed both. A dry run holds it back, a settled record is not judged again,
/// a daemon that lost what it settled finds the peer ejected, and the epoch
/// charges the peer once.
#[tokio::test]
pub async fn a_pass_convicts_a_peer_that_signed_two_payloads_for_a_batch() {
    let mut h = setup("Leviathan daemon equiv", 160).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let (dir, key) = kept_payloads(&mut h, "convict", &p2p).await;
    let target_key = h.target_key;
    let committer = target_key.to_string();
    let daemon = funded_daemon(&mut h).await;

    let dry_run = config(&h, dir.clone(), "convict-dry", true);
    let dry = pass(&mut h, &daemon, &dry_run, &mut HashSet::new()).await;
    assert_eq!(
        dry,
        PassReport {
            equivocations: 1,
            withheld: 1,
            ..PassReport::default()
        }
    );
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );

    let live = config(&h, dir, "convict", false);
    let mut judged = HashSet::new();
    assert_eq!(
        commit_then_reveal(&mut h, &daemon, &live, &mut judged).await,
        PassReport {
            equivocations: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected)
    );
    assert!(
        judged.contains(&committer),
        "the convicted committer is settled"
    );
    assert!(judged.contains(&equivocation_key(&key)));
    assert_eq!(
        pass(&mut h, &daemon, &live, &mut judged).await,
        PassReport::default()
    );

    let restarted = pass(&mut h, &daemon, &live, &mut HashSet::new()).await;
    assert_eq!(
        restarted,
        PassReport {
            equivocations: 1,
            already_on_chain: 1,
            ..PassReport::default()
        }
    );

    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
}

/// A second payload signed by another key is no contradiction by the target:
/// the record is set aside, and nobody is convicted.
#[tokio::test]
pub async fn an_equivocation_record_that_proves_nothing_convicts_no_one() {
    let mut h = setup("Leviathan daemon equiv forged", 161).await;
    let other =
        h.p2ps[(h.target_client_idx + 1) % h.p2ps.len()].insecure_clone();
    let (dir, key) = kept_payloads(&mut h, "forged", &other).await;
    let target_key = h.target_key;
    let daemon = funded_daemon(&mut h).await;
    let live = config(&h, dir, "forged", false);
    let mut judged = HashSet::new();

    // A node writes a record only after checking the second payload's
    // signature, so one that does not hold is reported as malformed.
    assert_eq!(
        pass(&mut h, &daemon, &live, &mut judged).await,
        PassReport {
            malformed: 1,
            ..PassReport::default()
        }
    );
    assert!(
        judged.contains(&equivocation_key(&key)),
        "the record is settled"
    );
    assert_eq!(
        pass(&mut h, &daemon, &live, &mut judged).await,
        PassReport::default()
    );
    assert!(!judged.contains(&target_key.to_string()));
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );
}

/// A proof the chain did not take is not settled: it is counted as a failed
/// submission, which the kill switch fires on, and sent again every pass
/// until it lands.
#[tokio::test]
pub async fn a_proof_that_fails_to_land_is_sent_again_until_it_does() {
    let mut h = setup("Leviathan daemon equiv retry", 162).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let (dir, key) = kept_payloads(&mut h, "retry", &p2p).await;
    let target_key = h.target_key;
    // A verifier whose wallet ran dry cannot pay for the transaction.
    let daemon = Keypair::new();
    let live = config(&h, dir, "retry", false);
    let mut judged = HashSet::new();
    let failed = PassReport {
        equivocations: 1,
        submission_failures: 1,
        ..PassReport::default()
    };

    for _ in 0..2 {
        assert_eq!(pass(&mut h, &daemon, &live, &mut judged).await, failed);
        assert!(!judged.contains(&equivocation_key(&key)));
        assert_eq!(
            h.client_state(&target_key).await,
            Some(ClientState::Healthy)
        );
    }

    h.endpoint
        .request_airdrop(&daemon.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    assert_eq!(
        commit_then_reveal(&mut h, &daemon, &live, &mut judged).await,
        PassReport {
            equivocations: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected)
    );
}

/// A proof the epoch can no longer charge, here because the epoch ended
/// before it landed, is a fraud the chain never saw: counted once as a failed
/// submission, which the kill switch fires on, and not sent. Once the next
/// epoch starts the record is from a round that is over, and is set aside
/// without being counted again, by this daemon or one restarted with the
/// record still on disk.
#[tokio::test]
pub async fn an_equivocation_its_epoch_can_no_longer_charge_is_reported_once() {
    let mut h = setup("Leviathan daemon equiv late", 163).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let (dir, key) = kept_payloads(&mut h, "late", &p2p).await;
    let target_key = h.target_key;
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.run_state().await, RunState::WaitingForMembers);
    let daemon = funded_daemon(&mut h).await;
    let live = config(&h, dir, "late", false);
    let mut judged = HashSet::new();

    assert_eq!(
        pass(&mut h, &daemon, &live, &mut judged).await,
        PassReport {
            equivocations: 1,
            submission_failures: 1,
            ..PassReport::default()
        }
    );
    assert!(judged.contains(&equivocation_key(&key)));
    assert_eq!(
        pass(&mut h, &daemon, &live, &mut judged).await,
        PassReport::default()
    );

    h.rejoin_all_and_start_epoch().await;
    let mut restarted = HashSet::new();
    assert_eq!(
        pass(&mut h, &daemon, &live, &mut restarted).await,
        PassReport::default()
    );
    assert!(restarted.contains(&equivocation_key(&key)));
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );
    assert_eq!(h.slashed_of(&target_key).await, 0);
}

/// A record of a round the pass's view of the run has not reached, because
/// the run moved on after the pass read it or the RPC lags, is looked at
/// again on the next pass rather than settled for the session.
#[tokio::test]
pub async fn an_equivocation_ahead_of_the_daemons_view_is_looked_at_again() {
    let mut h = setup("Leviathan daemon equiv ahead", 164).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let (dir, key) =
        kept_payloads_naming(&mut h, "ahead", &p2p, |header| PayloadHeader {
            epoch_start: header.epoch_start + 1,
            ..header
        })
        .await;
    let daemon = funded_daemon(&mut h).await;
    let live = config(&h, dir, "ahead", false);
    let mut judged = HashSet::new();
    for _ in 0..2 {
        assert_eq!(
            pass(&mut h, &daemon, &live, &mut judged).await,
            PassReport::default()
        );
        assert!(!judged.contains(&equivocation_key(&key)));
    }
}

/// A daemon that replays instead of reading reference dumps has nothing to
/// judge a payload against once the run is past its first epoch, nor while
/// its replay cannot be prepared: no step there can be replayed
/// (wienerlabs/leviathan-net#48). An equivocation needs neither, and it is
/// the one fraud still provable then, so the pass convicts it anyway.
#[tokio::test]
pub async fn an_equivocation_is_convicted_with_nothing_to_replay_against() {
    let mut h = setup("Leviathan equiv no replay", 170).await;
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let (dir, key) = kept_payloads(&mut h, "no-replay", &p2p).await;
    let target_key = h.target_key;
    let daemon = funded_daemon(&mut h).await;
    let replaying = AuditConfig {
        reference_dir: None,
        ..config(&h, dir, "no-replay", false)
    };

    let mut judged = HashSet::new();
    let report =
        commit_then_reveal(&mut h, &daemon, &replaying, &mut judged).await;
    assert_eq!(
        report,
        PassReport {
            equivocations: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected)
    );
    assert!(judged.contains(&equivocation_key(&key)));
}

/// A pass with nothing to replay against still sweeps the run's open
/// disputes, as every pass of a voting daemon does: a conviction whose
/// challenge window closed is finalised, though no payload is judged.
#[tokio::test]
pub async fn a_pass_with_nothing_to_replay_against_still_settles_open_disputes() {
    let mut h = setup("Leviathan sweep no replay", 173).await;
    h.reach_verifier_quorum().await;
    assert_eq!(h.verdict_status().await, VerdictStatus::SlashPending);
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();

    let daemon = funded_daemon(&mut h).await;
    let replaying = AuditConfig {
        reference_dir: None,
        ..config(&h, scratch("sweep-submitted"), "sweep", false)
    };
    let report = audit_pass(
        &mut h.endpoint,
        &daemon,
        &h.coordinator_account,
        &h.run,
        &replaying,
        None,
        &mut HashSet::new(),
    )
    .await
    .unwrap();
    assert_eq!(report, PassReport::default(), "no payload was judged");
    assert_eq!(
        h.verdict_status().await,
        VerdictStatus::Upheld,
        "the conviction whose window closed was not finalised"
    );
}

