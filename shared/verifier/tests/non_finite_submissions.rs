//! Reproduces the non-finite submission bypass found in the second pass of the
//! internal review (wienerlabs/leviathan#15).
//!
//! `verify_within_band` decides fraud with `distance > band`. If the submitted
//! delta contains a NaN, the relative L2 distance is NaN, and every IEEE-754
//! ordered comparison against NaN is false - so `distance > band` is false and
//! the submission is judged honest. The one input a cheater fully controls is
//! the one the check cannot see.
//!
//! Infinity behaves correctly, which is what makes this easy to miss: a quick
//! test of "garbage input" that happens to use inf will pass.

use psyche_verifier::AggregationConfig;
use psyche_verifier::DEFAULT_BAND;
use psyche_verifier::DEFAULT_SAFETY_FACTOR;
use psyche_verifier::audit_contribution;
use psyche_verifier::calibrate_band;
use psyche_verifier::relative_l2_distance;
use psyche_verifier::robust_aggregate;
use psyche_verifier::verify_within_band;

fn honest(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 17) as f32) - 8.0).collect()
}

/// A submission with a single NaN is as far from the honest replay as anything
/// can be, and is judged honest.
#[test]
fn a_single_nan_is_judged_honest() {
    let recomputed = honest(1024);
    let mut submitted = vec![0.0f32; recomputed.len()];
    submitted[0] = f32::NAN;

    let distance = relative_l2_distance(&submitted, &recomputed).unwrap();
    assert!(distance.is_nan(), "distance is NaN, not a number to compare");

    let verdict = verify_within_band(&submitted, &recomputed, DEFAULT_BAND).unwrap();
    assert!(
        !verdict.fraud,
        "BUG: `distance > band` is false for NaN, so the band clears the submission"
    );

    // And so no fraud proof is produced: there is nothing to take to the chain.
    let report = audit_contribution(0, &submitted, &recomputed, DEFAULT_BAND).unwrap();
    assert!(
        report.proof.is_none(),
        "BUG: an all-zero delta with one NaN in it produces no fraud proof"
    );
}

/// The same submission without the NaN - a plain lazy zero - is caught. The NaN
/// is doing all the work.
#[test]
fn the_same_submission_without_the_nan_is_caught() {
    let recomputed = honest(1024);
    let submitted = vec![0.0f32; recomputed.len()];
    let report = audit_contribution(0, &submitted, &recomputed, DEFAULT_BAND).unwrap();
    assert!(report.verdict.fraud);
    assert!(report.proof.is_some());
}

/// Infinity is caught, which is why this gap survives a casual check for
/// "does it handle garbage".
#[test]
fn infinity_is_caught() {
    let recomputed = honest(1024);
    let mut submitted = vec![0.0f32; recomputed.len()];
    submitted[0] = f32::INFINITY;
    let verdict = verify_within_band(&submitted, &recomputed, DEFAULT_BAND).unwrap();
    assert!(verdict.distance.is_infinite());
    assert!(verdict.fraud, "inf > band is true, so this one is rejected");
}

/// Calibration silently drops NaN samples, because `f32::max` returns the other
/// operand when one side is NaN. A drift sample that came back NaN does not
/// widen the band and does not raise an error either - it just is not there.
#[test]
fn calibration_silently_ignores_a_nan_sample() {
    let with_nan = calibrate_band(&[0.001, f32::NAN, 0.002], DEFAULT_SAFETY_FACTOR).unwrap();
    let without = calibrate_band(&[0.001, 0.002], DEFAULT_SAFETY_FACTOR).unwrap();
    assert_eq!(
        with_nan, without,
        "the NaN sample vanishes rather than being reported"
    );
}

/// What the NaN does downstream, recorded rather than assumed: the aggregator's
/// keep-mask is built with `distance <= limit`, which is false for NaN, so the
/// poisoned delta is excised and the aggregate stays finite. The band check is
/// the layer that fails, not the aggregation.
#[test]
fn the_aggregator_excises_what_the_band_check_waved_through() {
    let base = honest(256);
    let mut deltas: Vec<Vec<f32>> = (0..8).map(|_| base.clone()).collect();
    let mut poisoned = base.clone();
    poisoned[0] = f32::NAN;
    deltas.push(poisoned);

    let agg = robust_aggregate(&deltas, AggregationConfig::default()).unwrap();
    assert!(
        !agg.kept[8],
        "the NaN delta is dropped by the keep-mask, since NaN <= limit is false"
    );
    assert!(
        agg.result.iter().all(|v| v.is_finite()),
        "the aggregate stays finite"
    );
}
