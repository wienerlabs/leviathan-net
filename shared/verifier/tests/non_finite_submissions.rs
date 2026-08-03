//! Reproduces the non-finite submission bypass found in the second pass of the
//! internal review (wienerlabs/leviathan#15).
//!
//! `verify_within_band` decides fraud with `distance > band`. If the submitted
//! delta contained a NaN the relative L2 distance was NaN, and every IEEE-754
//! ordered comparison against NaN is false - so `distance > band` was false and
//! the submission was judged honest. The one input a cheater fully controls was
//! the one the check could not see.
//!
//! Infinity behaved correctly, which is what made this easy to miss: a quick
//! test of "garbage input" that happens to use inf passes either way.
//!
//! A submission that is not made of numbers is now fraud, not an error. An error
//! would let the cheater out the same door by another route, because
//! `audit_round` turns one into `Malformed`, which carries no fraud proof.

use psyche_verifier::audit_contribution;
use psyche_verifier::calibrate_band;
use psyche_verifier::relative_l2_distance;
use psyche_verifier::robust_aggregate;
use psyche_verifier::verify_within_band;
use psyche_verifier::AggregationConfig;
use psyche_verifier::DEFAULT_BAND;
use psyche_verifier::DEFAULT_SAFETY_FACTOR;

fn honest(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 17) as f32) - 8.0).collect()
}

/// A submission with a single NaN is as far from the honest replay as anything
/// can be, and is convicted.
#[test]
fn a_single_nan_is_fraud() {
    let recomputed = honest(1024);
    let mut submitted = vec![0.0f32; recomputed.len()];
    submitted[0] = f32::NAN;

    // The distance function reports what is wrong rather than returning a
    // number nothing can be compared against.
    assert!(matches!(
        relative_l2_distance(&submitted, &recomputed),
        Err(psyche_verifier::VerifierError::NonFiniteSubmission { index: 0 })
    ));

    let verdict = verify_within_band(&submitted, &recomputed, DEFAULT_BAND).unwrap();
    assert!(
        verdict.fraud,
        "a delta that is not made of numbers is fraud"
    );

    // And there is a proof to take to the chain, which is the point: an error
    // here would be `Malformed` in `audit_round`, carrying nothing.
    let report = audit_contribution(0, &submitted, &recomputed, DEFAULT_BAND).unwrap();
    assert!(report.proof.is_some(), "the conviction is provable");
}

/// A broken replay is the verifier's problem and must not convict the target.
#[test]
fn a_non_finite_replay_is_an_error_not_a_conviction() {
    let submitted = honest(64);
    let mut recomputed = submitted.clone();
    recomputed[7] = f32::NAN;
    assert!(matches!(
        verify_within_band(&submitted, &recomputed, DEFAULT_BAND),
        Err(psyche_verifier::VerifierError::NonFiniteReplay { index: 7 })
    ));
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

/// Calibration used to drop NaN samples in silence, because `f32::max` returns
/// the other operand when one side is NaN: the sample neither widened the band
/// nor raised anything, it just was not there. A measurement that came back as
/// not-a-number is a broken measurement and the operator should hear about it.
#[test]
fn calibration_reports_a_non_finite_sample() {
    assert!(calibrate_band(&[0.001, f32::NAN, 0.002], DEFAULT_SAFETY_FACTOR).is_err());
    assert!(calibrate_band(&[0.001, f32::INFINITY], DEFAULT_SAFETY_FACTOR).is_err());
    let clean = calibrate_band(&[0.001, 0.002], DEFAULT_SAFETY_FACTOR).unwrap();
    assert_eq!(
        clean,
        0.002 * DEFAULT_SAFETY_FACTOR,
        "ordinary samples still calibrate off the worst one"
    );
}

/// What the NaN does downstream, recorded rather than assumed: the aggregator's
/// keep-mask is built with `distance <= limit`, which is false for NaN, so the
/// poisoned delta is excised and the aggregate stays finite. The band check was
/// the layer that failed, not the aggregation.
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
