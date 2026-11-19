//! A verifier daemon votes that a target it audits did not reveal the state
//! its payload commits to (wienerlabs/leviathan-net#89,
//! `docs/ROUND_AUDIT.md`).

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::PayloadBand;
use psyche_coordinator::select_audits_for_current_round;
use psyche_coordinator::RunState;
use psyche_network::previous_round_path;
use psyche_network::revealed_state_path;
use psyche_network::PreviousRound;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_reveal_window;
use psyche_solana_tooling::signed_commitment::PayloadHeader;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::state::VerdictKind;
use psyche_solana_treasurer::state::VerdictStatus;
use solana_sdk::signer::Signer;

use super::appeals_harness::*;
use super::signed_payload::fixture;
use super::signed_payload::identity;
use super::signed_payload::keep_signed_payload;
use super::signed_payload::payload_header;
use super::signed_payload::FRAUD_PAYLOAD;

/// Shorter than the harness's round witness time of ten seconds.
const REVEAL_SECONDS: u64 = 4;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "leviathan-daemon-non-reveal-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// An audit the round assigns, by harness index: its verifier, the target,
/// the target's payload kept in `submitted_dir` with the state it commits
/// to, and a client the round does not assign it to.
struct Audited {
    verifier: usize,
    unassigned: usize,
    target: usize,
    dump: PathBuf,
}

/// The round's first audit, its target's payload kept as a verifier's node
/// keeps it, and the run's reveal window set.
async fn audited(
    h: &mut AppealsHarness,
    submitted_dir: &std::path::Path,
) -> Audited {
    let coordinator = h.coordinator_state().await.coordinator;
    let audit = select_audits_for_current_round(&coordinator).unwrap()[0];
    let index_of = |signer: &[u8; 32]| {
        h.clients
            .iter()
            .position(|client| client.pubkey().to_bytes() == *signer)
            .expect("every roster client is the harness's")
    };
    let (verifier, target) = (
        index_of(audit.verifier.signer()),
        index_of(audit.target.signer()),
    );
    let assigned: Vec<usize> = select_audits_for_current_round(&coordinator)
        .unwrap()
        .iter()
        .filter(|other| {
            other.target == audit.target && other.batch_id == audit.batch_id
        })
        .map(|other| index_of(other.verifier.signer()))
        .collect();
    let unassigned = h
        .committee_now()
        .await
        .trainers
        .iter()
        .map(|(client, _)| *client)
        .find(|client| *client != target && !assigned.contains(client))
        .expect("the round trains another client");
    // The daemon pays for what it sends with its verifier's key.
    let fees = h.clients[verifier].pubkey();
    h.endpoint
        .request_airdrop(&fees, 1_000_000_000)
        .await
        .unwrap();
    let header = PayloadHeader {
        batch_id: audit.batch_id,
        ..payload_header(
            &coordinator,
            &identity(&h.clients[target], &h.p2ps[target]),
        )
    };
    assert!(
        header.state_digest.is_some(),
        "the payload commits to a state"
    );
    let dump = keep_signed_payload(
        submitted_dir,
        &fixture(FRAUD_PAYLOAD),
        &h.clients[target],
        &h.p2ps[target],
        &h.run_id,
        header,
    );
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_set_reveal_window(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        &h.coordinator_account,
        REVEAL_SECONDS,
    )
    .await
    .unwrap();
    Audited {
        verifier,
        unassigned,
        target,
        dump,
    }
}

fn config(h: &AppealsHarness, submitted_dir: PathBuf) -> AuditConfig {
    AuditConfig {
        run_id: h.run_id.clone(),
        submitted_dir,
        reference_dir: None,
        band: PayloadBand::Distance(0.05),
        audit_assigned: true,
        dry_run: false,
        verdict_mode: true,
        continuity: false,
    }
}

async fn pass_by(
    h: &mut AppealsHarness,
    verifier: usize,
    config: &AuditConfig,
    judged: &mut HashSet<String>,
) -> PassReport {
    let authority = h.clients[verifier].insecure_clone();
    audit_pass(
        &mut h.endpoint,
        &authority,
        &h.coordinator_account,
        &h.run,
        config,
        None,
        judged,
    )
    .await
    .unwrap()
}

async fn has_verdict(h: &mut AppealsHarness, target: usize) -> bool {
    let verdict = find_audit_verdict(&h.run, &h.clients[target].pubkey());
    h.endpoint.get_account(&verdict).await.unwrap().is_some()
}

/// Runs the round under way on into its witness phase.
async fn advance_to_witness_phase(h: &mut AppealsHarness) {
    let step = h.coordinator_state().await.coordinator.progress.step;
    h.advance_with_witnesses(|coordinator| {
        coordinator.run_state == RunState::RoundWitness
    })
    .await;
    assert_eq!(h.coordinator_state().await.coordinator.progress.step, step);
}

/// The daemon votes a non-reveal only for its own audit whose state its node
/// set out to fetch and did not get, once the window has passed in a
/// witness phase: not while the round trains, not as the witness phase
/// opens, not before its node asked, and not by a client the round does not
/// assign the audit to. Then it votes once, and the verdict records a
/// non-reveal.
#[tokio::test]
pub async fn a_daemon_votes_that_a_target_it_audits_did_not_reveal() {
    let mut h =
        setup_sized("Leviathan daemon non-reveal", 175, SMALL_RUN).await;
    let submitted_dir = scratch("votes");
    let audited = audited(&mut h, &submitted_dir).await;
    let config = config(&h, submitted_dir);
    let mut judged = HashSet::new();
    let quiet = PassReport::default();

    assert_eq!(
        pass_by(&mut h, audited.verifier, &config, &mut judged).await,
        quiet
    );
    advance_to_witness_phase(&mut h).await;
    assert_eq!(
        pass_by(&mut h, audited.verifier, &config, &mut judged).await,
        quiet
    );
    h.endpoint
        .forward_clock_unix_timestamp(REVEAL_SECONDS)
        .await
        .unwrap();
    assert_eq!(
        pass_by(&mut h, audited.verifier, &config, &mut judged).await,
        quiet,
        "its node never set out to fetch the state"
    );
    assert!(!has_verdict(&mut h, audited.target).await);

    // Its node set out to fetch it: what the trainer did the round before is
    // written beside the dump first.
    fs::write(
        previous_round_path(&audited.dump),
        PreviousRound::Idle.encode(),
    )
    .unwrap();
    assert_eq!(
        pass_by(&mut h, audited.unassigned, &config, &mut HashSet::new()).await,
        quiet,
        "the round does not assign it the audit"
    );
    assert!(!has_verdict(&mut h, audited.target).await);

    assert_eq!(
        pass_by(&mut h, audited.verifier, &config, &mut judged).await,
        PassReport {
            fraud_found: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
    let target_key = h.clients[audited.target].pubkey();
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.kind, VerdictKind::NonReveal);
    assert_eq!(verdict.replayed_hash, [0; 32]);
    assert_eq!(verdict.voters, vec![h.clients[audited.verifier].pubkey()]);
    assert_eq!(verdict.status, VerdictStatus::Voting);
    assert_eq!(
        pass_by(&mut h, audited.verifier, &config, &mut judged).await,
        quiet
    );
}

/// A state that arrived is never voted a non-reveal.
#[tokio::test]
pub async fn a_daemon_does_not_vote_a_target_that_revealed() {
    let mut h = setup_sized("Leviathan daemon revealed", 176, SMALL_RUN).await;
    let submitted_dir = scratch("revealed");
    let audited = audited(&mut h, &submitted_dir).await;
    let config = config(&h, submitted_dir);
    fs::write(
        previous_round_path(&audited.dump),
        PreviousRound::Idle.encode(),
    )
    .unwrap();
    fs::write(
        revealed_state_path(&audited.dump),
        b"the state its node fetched",
    )
    .unwrap();
    advance_to_witness_phase(&mut h).await;
    h.endpoint
        .forward_clock_unix_timestamp(REVEAL_SECONDS)
        .await
        .unwrap();
    let mut judged = HashSet::new();
    assert_eq!(
        pass_by(&mut h, audited.verifier, &config, &mut judged).await,
        PassReport::default()
    );
    assert!(!has_verdict(&mut h, audited.target).await);
}
