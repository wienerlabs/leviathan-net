use std::path::Path;
use std::path::PathBuf;

use leviathan_verifier::audit_dirs;
use psyche_modeling::CompressDCT;
use psyche_modeling::DistroResult;
use psyche_network::distro_results_to_bytes;
use psyche_network::SerializedDistroResult;
use psyche_verifier::DEFAULT_BAND;
use tch::Device;
use tch::Kind;
use tch::Tensor;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("levverify-{}-{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_dump(dir: &Path, key: &str, dense: &[f32]) {
    let x = Tensor::from_slice(dense)
        .reshape([16, (dense.len() / 16) as i64])
        .to_kind(Kind::Float);
    let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(&x, 8);
    let result = DistroResult {
        sparse_idx,
        sparse_val,
        xshape,
        totalk,
        stats: None,
    };
    let serialized: SerializedDistroResult = (&result).try_into().unwrap();
    let bytes = distro_results_to_bytes(&[serialized]).unwrap();
    let path = dir.join(format!("result-testid-{key}.vec-postcard"));
    std::fs::write(path, bytes).unwrap();
}

fn honest_delta() -> Vec<f32> {
    (0..1024).map(|i| ((i as f32) * 0.017).sin() * 0.03).collect()
}

#[test]
fn honest_replay_passes_on_real_distro_dumps() {
    let honest = honest_delta();
    let submitted = scratch("honest-sub");
    let reference = scratch("honest-ref");
    write_dump(&submitted, "step1-batch0", &honest);
    write_dump(&reference, "step1-batch0", &honest);

    let summary = audit_dirs(&submitted, &reference, DEFAULT_BAND, Device::Cpu).unwrap();
    assert_eq!(summary.audited(), 1);
    assert_eq!(summary.fraud(), 0);
}

#[test]
fn sign_flip_is_caught_on_real_distro_dumps() {
    let honest = honest_delta();
    let tampered: Vec<f32> = honest.iter().map(|v| -5.0 * v).collect();
    let submitted = scratch("fraud-sub");
    let reference = scratch("fraud-ref");
    write_dump(&submitted, "step1-batch0", &tampered);
    write_dump(&reference, "step1-batch0", &honest);

    let summary = audit_dirs(&submitted, &reference, DEFAULT_BAND, Device::Cpu).unwrap();
    assert_eq!(summary.audited(), 1);
    assert_eq!(summary.fraud(), 1);
    assert_eq!(summary.proofs().len(), 1);
}

#[test]
fn missing_reference_is_skipped_not_convicted() {
    let honest = honest_delta();
    let submitted = scratch("skip-sub");
    let reference = scratch("skip-ref");
    write_dump(&submitted, "step9-batch9", &honest);

    let summary = audit_dirs(&submitted, &reference, DEFAULT_BAND, Device::Cpu).unwrap();
    assert_eq!(summary.audited(), 0);
    assert_eq!(summary.fraud(), 0);
}
