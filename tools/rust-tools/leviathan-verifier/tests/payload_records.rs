//! What the verifier reads off disk to convict on: payload dumps, and the
//! commitment each committer signed for one (wienerlabs/leviathan-net#43).
//!
//! The dumps are the real ones the daemon suites use, written by an earlier
//! build from a real model.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use leviathan_verifier::check_commitment;
use leviathan_verifier::contribution_key;
use leviathan_verifier::read_dump;
use leviathan_verifier::CommitmentCheck;
use psyche_core::sha256;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_network::commitment_record_path;
use psyche_network::distro_results_to_bytes;
use psyche_network::results_digest;
use psyche_network::CommitmentRecord;
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
    let path = dir.join(format!(
        "result-{}-step{}-batch{}.vec-postcard",
        "Committer", payload.step, payload.batch_id
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
    let path = dir.join("result-Committer-step9-batchB[16, 17].vec-postcard");
    fs::copy(fixture(FRAUD), &path).unwrap();
    let key = SecretKey::generate(&mut rand::rng());
    assert_eq!(
        check_commitment(&path, RUN, key.public().as_bytes()).unwrap(),
        CommitmentCheck::Missing
    );
}

/// Each way a record can fail to hold is final for that pair: another key,
/// another run, a dump other than the one hashed, or a file that is not a
/// record at all.
#[test]
fn a_record_that_does_not_hold_is_invalid() {
    let key = SecretKey::generate(&mut rand::rng());
    let other = SecretKey::generate(&mut rand::rng());
    let payload = payload(FRAUD);

    let foreign = keep(&scratch("foreign"), &payload, &other, RUN);
    let elsewhere = keep(&scratch("elsewhere"), &payload, &key, "another run");
    let swapped = keep(&scratch("swapped"), &payload, &key, RUN);
    fs::copy(fixture(HONEST), &swapped).unwrap();
    let truncated = keep(&scratch("truncated"), &payload, &key, RUN);
    let record = commitment_record_path(&truncated);
    let bytes = fs::read(&record).unwrap();
    fs::write(&record, &bytes[..bytes.len() - 1]).unwrap();

    for (case, path) in [
        ("signed by another key", foreign),
        ("made in another run", elsewhere),
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

/// The committer is base58, which can spell `step`. The key starts at the
/// `-step` that ends the committer, wherever else the letters appear.
#[test]
fn a_committer_that_spells_step_does_not_move_the_key() {
    for (name, key) in [
        (
            "result-AstepB-step3-batchB15.vec-postcard",
            Some("step3-batchB15"),
        ),
        (
            "result-Abc-step7-batchB[12, 19].vec-postcard",
            Some("step7-batchB[12, 19]"),
        ),
        ("Abc-step7-batchB[12, 19].vec-postcard", None),
        ("result-Abc-step7-batchB[12, 19].commitment", None),
    ] {
        assert_eq!(contribution_key(Path::new(name)).as_deref(), key, "{name}");
    }
}
