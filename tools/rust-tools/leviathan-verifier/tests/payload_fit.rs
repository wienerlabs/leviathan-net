//! A signed payload that does not fit the run's model is fraud, judged at an
//! infinite distance like one that is not made of numbers, and judging it
//! never takes the verifier down (wienerlabs/leviathan-net#59).
//!
//! The payloads are real dumps a real model produced
//! (`solana-tooling/tests/fixtures/daemon`), cut or reshaped the way a
//! trainer could sign them.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use leviathan_verifier::audit_dirs;
use leviathan_verifier::decompress_dump;
use leviathan_verifier::judge_payload;
use leviathan_verifier::read_dump;
use leviathan_verifier::ContributionOutcome;
use psyche_modeling::Distro;
use psyche_modeling::DistroResult;
use psyche_network::distro_results_to_bytes;
use psyche_network::SerializedDistroResult;
use psyche_verifier::DEFAULT_BAND;
use tch::Device;
use tch::Kind;
use tch::Tensor;

const HONEST: &str = "honest.vec-postcard";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../architectures/decentralized/solana-tooling/tests/fixtures/daemon")
        .join(name)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("levfit-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn honest_results() -> Vec<SerializedDistroResult> {
    read_dump(&fixture(HONEST)).unwrap()
}

fn honest_reference() -> Vec<f32> {
    decompress_dump(&fixture(HONEST), Device::Cpu).unwrap()
}

/// Writes `results` as a dump in a directory of its own.
fn dump(tag: &str, results: &[SerializedDistroResult]) -> PathBuf {
    let path = scratch(tag).join("payload.vec-postcard");
    fs::write(&path, distro_results_to_bytes(results).unwrap()).unwrap();
    path
}

/// The honest results with the first one changed by `change`.
fn with_first(change: impl FnOnce(&mut DistroResult)) -> Vec<SerializedDistroResult> {
    let mut results = honest_results();
    let mut first = DistroResult::try_from(&results[0]).unwrap();
    change(&mut first);
    results[0] = SerializedDistroResult::try_from(&first).unwrap();
    results
}

/// The honest results with the first one's encoded shape halved in its last
/// dimension, under indices that still reach as far as before. Decompressing
/// it as it claims scatters past the end of the tensor it allocates.
fn shrunk() -> Vec<SerializedDistroResult> {
    let mut results = honest_results();
    let target = results
        .iter()
        .position(|r| r.xshape.last().is_some_and(|last| *last > 1))
        .expect("a result with a last dimension above 1");
    let last = results[target].xshape.len() - 1;
    results[target].xshape[last] /= 2;
    results
}

/// A payload a few bytes long claiming a tensor of four billion values, with
/// its one index inside it. It holds together as an encoding, so only its
/// size gives it away.
fn vast() -> Vec<SerializedDistroResult> {
    let claimed = DistroResult {
        sparse_idx: Tensor::from_slice(&[0u8, 0, 0, 0]).reshape([1, 1, 4]),
        sparse_val: Tensor::from_slice(&[0.5f32]).reshape([1, 1, 1]),
        xshape: vec![1, 1, 65535, 65535],
        totalk: 65535 * 65535,
        stats: None,
    };
    vec![SerializedDistroResult::try_from(&claimed).unwrap()]
}

#[test]
fn an_honest_payload_is_judged_by_the_band() {
    let judged = judge_payload(
        &fixture(HONEST),
        &honest_reference(),
        DEFAULT_BAND,
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(judged.unfit, None);
    assert!(!judged.verdict.fraud);
    assert_eq!(judged.verdict.distance, 0.0);
}

/// Each of these is a property of the signed bytes alone, so it convicts the
/// signer and nobody else.
#[test]
fn a_payload_that_does_not_fit_is_fraud_at_an_infinite_distance() {
    let mut short = honest_results();
    short.pop();
    let mut unshaped = honest_results();
    unshaped[0].xshape = vec![0];
    let not_numbers = with_first(|result| {
        result.sparse_val = Tensor::full(
            result.sparse_val.size(),
            f64::NAN,
            (Kind::Float, Device::Cpu),
        );
    });
    let reference = honest_reference();
    for (name, results) in [
        ("one result short", short),
        ("a shape shrunk under its indices", shrunk()),
        ("values that are not numbers", not_numbers),
        ("a shape no parameter can have", unshaped),
    ] {
        let judged = judge_payload(&dump(name, &results), &reference, DEFAULT_BAND, Device::Cpu)
            .unwrap_or_else(|err| panic!("a payload with {name} was not judged: {err:#}"));
        assert!(judged.verdict.fraud, "a payload with {name} was cleared");
        assert!(
            judged.verdict.distance.is_infinite(),
            "a payload with {name} was judged at {}",
            judged.verdict.distance
        );
        assert!(
            judged.unfit.is_some(),
            "a payload with {name} was not named as unfit"
        );
    }
}

/// A payload claiming more values than the honest result is unfit before any
/// of them is allocated. Decompressed as it claims, it would have asked for
/// 17 GB, and taken the verifier down on every pass while it was kept.
#[test]
fn a_payload_claiming_vast_shapes_is_unfit_before_anything_is_allocated() {
    let results = vast();
    let decoded: Vec<DistroResult> = results
        .iter()
        .map(|r| DistroResult::try_from(r).unwrap())
        .collect();
    assert_eq!(Distro::check_encodings(&decoded), Ok(()));
    let judged = judge_payload(
        &dump("vast", &results),
        &honest_reference(),
        DEFAULT_BAND,
        Device::Cpu,
    )
    .unwrap();
    assert!(judged.verdict.fraud && judged.verdict.distance.is_infinite());
    assert!(judged.unfit.is_some());
    assert_eq!(judged.submitted_hash, None);
}

/// A payload whose results are laid out otherwise than the model's, one
/// variable claiming two variables' worth of values and another left out,
/// has the honest length and holds together. It is not unfit: the band
/// judges it, and it is nothing like the honest result.
#[test]
fn a_payload_laid_out_otherwise_is_left_to_the_band() {
    let mut results: Vec<DistroResult> = honest_results()
        .iter()
        .map(|r| DistroResult::try_from(r).unwrap())
        .collect();
    let (twice, dropped) = (0..results.len())
        .flat_map(|i| (0..results.len()).map(move |j| (i, j)))
        .find(|(i, j)| {
            i != j && results[*i].xshape == results[*j].xshape && results[*i].xshape[0] < 32768
        })
        .expect("two variables of one shape");
    let doubled = {
        let r = &results[twice];
        let mut xshape = r.xshape.clone();
        xshape[0] *= 2;
        DistroResult {
            sparse_idx: Tensor::cat(&[&r.sparse_idx, &r.sparse_idx], 0),
            sparse_val: Tensor::cat(&[&r.sparse_val, &r.sparse_val], 0),
            xshape,
            totalk: r.totalk,
            stats: None,
        }
    };
    results[twice] = doubled;
    results.remove(dropped);
    assert_eq!(Distro::check_encodings(&results), Ok(()));
    let serialized: Vec<SerializedDistroResult> = results
        .iter()
        .map(|r| SerializedDistroResult::try_from(r).unwrap())
        .collect();
    let judged = judge_payload(
        &dump("relaid", &serialized),
        &honest_reference(),
        DEFAULT_BAND,
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(judged.unfit, None);
    assert!(
        judged.verdict.fraud,
        "judged at {}",
        judged.verdict.distance
    );
    assert!(judged.verdict.distance.is_finite());
}

/// Bytes after the last result that are not a result do not parse.
#[test]
fn a_payload_with_trailing_bytes_is_unfit() {
    let path = scratch("trailing").join("payload.vec-postcard");
    let mut bytes = fs::read(fixture(HONEST)).unwrap();
    bytes.extend_from_slice(&[0xff; 3]);
    fs::write(&path, bytes).unwrap();
    let judged = judge_payload(&path, &honest_reference(), DEFAULT_BAND, Device::Cpu).unwrap();
    assert!(judged.verdict.fraud && judged.verdict.distance.is_infinite());
    assert!(judged.unfit.is_some());
}

/// Decompressing a payload whose indices reach past its shape is an error,
/// not a panic that takes the verifier down, and down again on every restart
/// while the payload is kept.
#[test]
fn decompressing_a_payload_that_does_not_hold_together_is_an_error() {
    assert!(decompress_dump(&dump("shrunk-decompress", &shrunk()), Device::Cpu).is_err());
}

/// A reference that is not made of numbers is this verifier's own fault, and
/// convicts nobody: it is an error, looked at again.
#[test]
fn a_reference_that_is_not_finite_is_the_verifiers_problem() {
    let mut reference = honest_reference();
    reference[0] = f32::NAN;
    assert!(judge_payload(&fixture(HONEST), &reference, DEFAULT_BAND, Device::Cpu).is_err());
}

/// A dump this verifier cannot read is not the signer's fault either.
#[test]
fn a_dump_that_cannot_be_read_is_an_error() {
    let missing = scratch("missing").join("payload.vec-postcard");
    assert!(judge_payload(&missing, &honest_reference(), DEFAULT_BAND, Device::Cpu).is_err());
}

/// The offline audit an appeal bench re-checks with reaches the daemon's
/// verdict: a payload that does not fit is a fraud with its reason, and it no
/// longer stops the audit of everything else.
#[test]
fn the_offline_audit_convicts_a_payload_that_does_not_fit() {
    let submitted = scratch("offline-submitted");
    let reference = scratch("offline-reference");
    let mut short = honest_results();
    short.pop();
    for (name, results) in [
        ("result-a-epoch1-step1-batchB0", short),
        ("result-b-epoch1-step1-batchB1", honest_results()),
    ] {
        fs::write(
            submitted.join(format!("{name}.vec-postcard")),
            distro_results_to_bytes(&results).unwrap(),
        )
        .unwrap();
        fs::copy(
            fixture(HONEST),
            reference.join(format!("{name}.vec-postcard")),
        )
        .unwrap();
    }
    let summary = audit_dirs(&submitted, &reference, DEFAULT_BAND, Device::Cpu).unwrap();
    assert_eq!(summary.audited(), 2);
    assert_eq!(summary.fraud(), 1);
    assert!(matches!(
        &summary.outcomes[..],
        [ContributionOutcome::Unfit { key, .. }, ContributionOutcome::Ok { distance, .. }]
            if key == "epoch1-step1-batchB0" && *distance == 0.0
    ));
}
