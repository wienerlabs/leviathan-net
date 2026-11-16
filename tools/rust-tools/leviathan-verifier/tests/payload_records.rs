//! What the verifier reads off disk to convict on: payload dumps, and the
//! commitment each committer signed for one (wienerlabs/leviathan-net#43).
//!
//! The dumps are the real ones the daemon suites use, written by an earlier
//! build from a real model.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use leviathan_verifier::check_commitment;
use leviathan_verifier::check_equivocation;
use leviathan_verifier::contribution_key;
use leviathan_verifier::read_dump;
use leviathan_verifier::CommitmentCheck;
use leviathan_verifier::EquivocationCheck;
use leviathan_verifier::VerifiedCommitment;
use psyche_core::sha256;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_network::commitment_record_path;
use psyche_network::distro_results_to_bytes;
use psyche_network::payload_dump_name;
use psyche_network::results_digest;
use psyche_network::equivocation_record_path;
use psyche_network::CommitmentRecord;
use psyche_network::EquivocationRecord;
use psyche_network::SecretKey;
use psyche_network::TransmittableDistroResult;

const FRAUD: &str = "fraud.vec-postcard";
const HONEST: &str = "honest.vec-postcard";
const RUN: &str = "leviathan payload records";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../architectures/decentralized/solana-tooling/tests/fixtures/daemon")
        .join(name)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("levrecords-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn payload(name: &str) -> TransmittableDistroResult {
    TransmittableDistroResult {
        epoch_start: 1_700_000_000,
        step: 9,
        trainer_nonce: 4,
        batch_id: BatchId(ClosedInterval::new(16, 17)),
        distro_results: read_dump(&fixture(name)).unwrap(),
    }
}

/// Keeps `payload` in `dir` as the client keeps a peer's payload: the record
/// of what `key` signed for it in `run_id`, then the dump.
fn keep(dir: &Path, payload: &TransmittableDistroResult, key: &SecretKey, run_id: &str) -> PathBuf {
    let data_hash = payload.comptue_hash(run_id).unwrap();
    let record = CommitmentRecord::new(payload, data_hash, key.sign(&data_hash).to_bytes());
    let path = dir.join(payload_dump_name(
        "Committer",
        payload.epoch_start,
        payload.step,
        payload.batch_id,
    ));
    fs::write(commitment_record_path(&path), record.to_bytes()).unwrap();
    fs::write(
        &path,
        distro_results_to_bytes(&payload.distro_results).unwrap(),
    )
    .unwrap();
    path
}

/// The digest a commitment takes of a payload's results is the sha256 of the
/// dump bytes. These dumps predate this build, so the check also pins that
/// today's encoding writes their results back byte for byte. A change to it
/// would change every commitment hash.
#[test]
fn a_dumps_results_digest_is_the_sha256_of_its_bytes() {
    for name in [FRAUD, HONEST] {
        let path = fixture(name);
        let results = read_dump(&path).unwrap();
        assert_eq!(
            results_digest(&results).unwrap(),
            sha256(&fs::read(&path).unwrap()),
            "{name}"
        );
        assert_eq!(
            distro_results_to_bytes(&results).unwrap(),
            fs::read(&path).unwrap()
        );
    }
}

#[test]
fn a_genuine_commitment_is_verified() {
    let dir = scratch("genuine");
    let key = SecretKey::generate(&mut rand::rng());
    let payload = payload(FRAUD);
    let path = keep(&dir, &payload, &key, RUN);
    match check_commitment(&path, RUN, key.public().as_bytes()).unwrap() {
        CommitmentCheck::Verified(verified) => {
            assert_eq!(
                verified.record.data_hash,
                payload.comptue_hash(RUN).unwrap()
            );
            assert_eq!(verified.results_digest, sha256(&fs::read(&path).unwrap()));
        }
        other => panic!("a genuine commitment was not verified: {other:?}"),
    }
}

#[test]
fn a_dump_kept_without_a_record_has_nothing_to_convict_on() {
    let dir = scratch("missing");
    let path = dir.join("result-Committer-epoch1700000000-step9-batchB[16, 17].vec-postcard");
    fs::copy(fixture(FRAUD), &path).unwrap();
    let key = SecretKey::generate(&mut rand::rng());
    assert_eq!(
        check_commitment(&path, RUN, key.public().as_bytes()).unwrap(),
        CommitmentCheck::Missing
    );
}

/// Each way a record can fail to hold is final for that pair: another key,
/// another run, another epoch, a dump other than the one hashed, or a file
/// that is not a record at all.
#[test]
fn a_record_that_does_not_hold_is_invalid() {
    let key = SecretKey::generate(&mut rand::rng());
    let other = SecretKey::generate(&mut rand::rng());
    let payload = payload(FRAUD);

    let foreign = keep(&scratch("foreign"), &payload, &other, RUN);
    let elsewhere = keep(&scratch("elsewhere"), &payload, &key, "another run");
    let swapped = keep(&scratch("swapped"), &payload, &key, RUN);
    fs::copy(fixture(HONEST), &swapped).unwrap();
    // The record names an epoch the signature was not made over, as a
    // record of a round trained before a restart, relabelled, would
    // (wienerlabs/leviathan-net#67).
    let relabelled = keep(&scratch("relabelled"), &payload, &key, RUN);
    let record = commitment_record_path(&relabelled);
    let mut held = CommitmentRecord::from_bytes(&fs::read(&record).unwrap()).unwrap();
    held.epoch_start += 90;
    fs::write(&record, held.to_bytes()).unwrap();
    let truncated = keep(&scratch("truncated"), &payload, &key, RUN);
    let record = commitment_record_path(&truncated);
    let bytes = fs::read(&record).unwrap();
    fs::write(&record, &bytes[..bytes.len() - 1]).unwrap();

    for (case, path) in [
        ("signed by another key", foreign),
        ("made in another run", elsewhere),
        ("naming another epoch", relabelled),
        ("kept beside another dump", swapped),
        ("cut short", truncated),
    ] {
        assert!(
            matches!(
                check_commitment(&path, RUN, key.public().as_bytes()).unwrap(),
                CommitmentCheck::Invalid(_)
            ),
            "a record {case} was not refused"
        );
    }
}

/// The committer is base58, which can spell `epoch`. The key starts at the
/// `-epoch` that ends the committer, wherever else the letters appear.
#[test]
fn a_committer_that_spells_epoch_does_not_move_the_key() {
    for (name, key) in [
        (
            "result-AepochB-epoch1700000000-step3-batchB15.vec-postcard",
            Some("epoch1700000000-step3-batchB15"),
        ),
        (
            "result-Abc-epoch1700000000-step7-batchB[12, 19].vec-postcard",
            Some("epoch1700000000-step7-batchB[12, 19]"),
        ),
        (
            "Abc-epoch1700000000-step7-batchB[12, 19].vec-postcard",
            None,
        ),
        (
            "result-Abc-epoch1700000000-step7-batchB[12, 19].commitment",
            None,
        ),
        // A dump named before the epoch named the round cannot be placed.
        ("result-Abc-step7-batchB[12, 19].vec-postcard", None),
    ] {
        assert_eq!(contribution_key(Path::new(name)).as_deref(), key, "{name}");
    }
}

/// Keeps what `key` signed for `second`, a payload for the same name as the
/// dump at `kept`, as the client keeps a contradicting payload: an
/// equivocation record beside the kept pair (wienerlabs/leviathan-net#46).
fn keep_equivocation(
    kept: &Path,
    second: &TransmittableDistroResult,
    key: &SecretKey,
    run_id: &str,
) {
    let data_hash = second.comptue_hash(run_id).unwrap();
    let record = CommitmentRecord::new(second, data_hash, key.sign(&data_hash).to_bytes());
    let equivocation = EquivocationRecord::new(second, record).unwrap();
    fs::write(equivocation_record_path(kept), equivocation.to_bytes()).unwrap();
}

fn verified(path: &Path, key: &SecretKey) -> VerifiedCommitment {
    match check_commitment(path, RUN, key.public().as_bytes()).unwrap() {
        CommitmentCheck::Verified(commitment) => commitment,
        other => panic!("the kept pair does not hold: {other:?}"),
    }
}

#[test]
fn a_second_signed_payload_for_the_batch_proves_equivocation() {
    let dir = scratch("equivocation");
    let key = SecretKey::generate(&mut rand::rng());
    let kept = keep(&dir, &payload(FRAUD), &key, RUN);
    keep_equivocation(&kept, &payload(HONEST), &key, RUN);
    let kept_commitment = verified(&kept, &key);
    match check_equivocation(&kept, &kept_commitment, RUN, key.public().as_bytes()).unwrap() {
        EquivocationCheck::Proven(second) => {
            assert_eq!(
                second.record.data_hash,
                payload(HONEST).comptue_hash(RUN).unwrap()
            );
        }
        other => panic!("not proven: {other:?}"),
    }
}

#[test]
fn a_payload_with_no_contradiction_proves_nothing() {
    let dir = scratch("no-equivocation");
    let key = SecretKey::generate(&mut rand::rng());
    let kept = keep(&dir, &payload(FRAUD), &key, RUN);
    let kept_commitment = verified(&kept, &key);
    assert_eq!(
        check_equivocation(&kept, &kept_commitment, RUN, key.public().as_bytes()).unwrap(),
        EquivocationCheck::None
    );
}

#[test]
fn an_equivocation_record_that_does_not_hold_is_invalid() {
    let dir = scratch("bad-equivocation");
    let key = SecretKey::generate(&mut rand::rng());
    let other = SecretKey::generate(&mut rand::rng());
    let kept = keep(&dir, &payload(FRAUD), &key, RUN);
    let kept_commitment = verified(&kept, &key);
    // Signed by someone else.
    keep_equivocation(&kept, &payload(HONEST), &other, RUN);
    assert!(matches!(
        check_equivocation(&kept, &kept_commitment, RUN, key.public().as_bytes()).unwrap(),
        EquivocationCheck::Invalid(_)
    ));
    // The kept payload again: no contradiction.
    keep_equivocation(&kept, &payload(FRAUD), &key, RUN);
    assert!(matches!(
        check_equivocation(&kept, &kept_commitment, RUN, key.public().as_bytes()).unwrap(),
        EquivocationCheck::Invalid(_)
    ));
    // Not a record at all.
    fs::write(equivocation_record_path(&kept), b"not a record").unwrap();
    assert!(matches!(
        check_equivocation(&kept, &kept_commitment, RUN, key.public().as_bytes()).unwrap(),
        EquivocationCheck::Invalid(_)
    ));
}
