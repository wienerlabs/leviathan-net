//! Judging a 1-bit payload by the two ways a replay of it can differ
//! (wienerlabs/leviathan-net#49).
//!
//! A 1-bit DisTrO payload sends, for every chunk of the model, the signs of the
//! top-k values of the node's error-feedback buffer. A replay of the same work
//! can differ from it in two ways:
//!
//! - **churn:** it picks other coordinates for its top-k, so a coordinate the
//!   node sent is one the replay did not send at all;
//! - **flips:** it picks the same coordinate with the other sign.
//!
//! Where both send as many coordinates, as two honest top-k's of one model
//! do, the relative L2 distance over the decompressed signs mixes the two:
//! `distance² = 2 · churn + 4 · (1 − churn) · flips`. A band on the distance
//! wide enough for honest churn across hardware classes lets a cheater flip a
//! quarter of its band's square in signs instead. Judging the two apart lets
//! each have its own band.
//!
//! Churn is counted from both sides, so a payload that sends fewer
//! coordinates than the replay, leaving out the honest ones it does not like,
//! churns as much as it left out (wienerlabs/leviathan-net#49 review).

use crate::VerifierError;

/// How a replay of a 1-bit payload differs from what its node sent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignDrift {
    /// The larger of two shares: of the coordinates the node sent, those the
    /// replay did not send at all, and of those the replay sent, those the
    /// node did not. Where both send as many, the two are the same.
    pub churn: f32,
    /// Of the coordinates both sent, the share the replay sent with the
    /// other sign.
    pub flips: f32,
}

impl SignDrift {
    /// The relative L2 distance these two make over 1-bit payloads that send
    /// as many coordinates each. Payloads that do not send as many are not
    /// tied to it.
    pub fn distance(&self) -> f32 {
        (2.0 * self.churn + 4.0 * (1.0 - self.churn) * self.flips).sqrt()
    }
}

/// How far each of a 1-bit payload's two drifts may go before it is fraud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignBand {
    pub churn: f32,
    pub flips: f32,
}

impl SignBand {
    fn check(&self) -> Result<(), VerifierError> {
        for value in [self.churn, self.flips] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(VerifierError::InvalidSignBand(value));
            }
        }
        Ok(())
    }
}

/// A 1-bit payload judged against a [`SignBand`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignVerdict {
    pub drift: SignDrift,
    pub band: SignBand,
    pub fraud: bool,
}

/// How `recomputed`, a verifier's replay, differs from `submitted`, what the
/// node sent, both decompressed 1-bit payloads of one model.
///
/// A coordinate a payload sent is one it holds a non-zero value at. A
/// submission that sent nothing churned everything the replay sent, if it sent
/// anything.
pub fn sign_drift(submitted: &[f32], recomputed: &[f32]) -> Result<SignDrift, VerifierError> {
    if submitted.len() != recomputed.len() {
        return Err(VerifierError::LengthMismatch {
            submitted: submitted.len(),
            recomputed: recomputed.len(),
        });
    }
    if let Some(index) = submitted.iter().position(|value| !value.is_finite()) {
        return Err(VerifierError::NonFiniteSubmission { index });
    }
    if let Some(index) = recomputed.iter().position(|value| !value.is_finite()) {
        return Err(VerifierError::NonFiniteReplay { index });
    }
    // What the node sent, what the replay sent, and what both sent.
    let (mut held, mut replay_held, mut shared, mut flipped) = (0u64, 0u64, 0u64, 0u64);
    for (sent, replayed) in submitted.iter().zip(recomputed) {
        held += u64::from(*sent != 0.0);
        replay_held += u64::from(*replayed != 0.0);
        if *sent != 0.0 && *replayed != 0.0 {
            shared += 1;
            flipped += u64::from((*sent > 0.0) != (*replayed > 0.0));
        }
    }
    // A share of nothing is nothing, except that a side that sent nothing
    // churned everything the other side sent.
    let share = |part: u64, whole: u64, other: u64| match whole {
        0 => f32::from(u8::from(other > 0)),
        whole => part as f32 / whole as f32,
    };
    Ok(SignDrift {
        churn: share(held - shared, held, replay_held).max(share(
            replay_held - shared,
            replay_held,
            held,
        )),
        flips: match shared {
            0 => 0.0,
            shared => flipped as f32 / shared as f32,
        },
    })
}

/// Judges `submitted` against `recomputed`: fraud once either drift goes past
/// its band. A submission that is not made of numbers is fraud, as it is under
/// [`crate::verify_within_band`]; a replay that is not is the verifier's own
/// error and convicts nobody.
pub fn verify_signs_within_band(
    submitted: &[f32],
    recomputed: &[f32],
    band: SignBand,
) -> Result<SignVerdict, VerifierError> {
    band.check()?;
    // Fraud whatever the band, as an infinite distance is under the relative
    // L2: a band of 1 must not let a submission of no numbers through.
    let (drift, unfit) = match sign_drift(submitted, recomputed) {
        Ok(drift) => (drift, false),
        Err(VerifierError::NonFiniteSubmission { .. }) => (
            SignDrift {
                churn: 1.0,
                flips: 1.0,
            },
            true,
        ),
        Err(other) => return Err(other),
    };
    Ok(SignVerdict {
        drift,
        band,
        fraud: unfit || drift.churn > band.churn || drift.flips > band.flips,
    })
}

/// The band a set of honest drifts calls for, from payloads that send
/// `coordinates` coordinates: each part's worst sample times `safety_factor`
/// (at least 1), never narrower than one coordinate and never wider than
/// everything.
///
/// The two parts are calibrated apart. Honest replays across hardware classes
/// churn and rarely flip, so a band calibrated this way holds flips tight
/// where a band on the distance has to let them through. It never holds them
/// at zero: a sign near zero can cross it in any replay but an exact one, and
/// a sample set too small to have caught one would otherwise convict the
/// honest node that does (wienerlabs/leviathan-net#49).
pub fn calibrate_sign_band(
    drifts: &[SignDrift],
    safety_factor: f32,
    coordinates: u64,
) -> Result<SignBand, VerifierError> {
    if drifts.is_empty() || coordinates == 0 {
        return Err(VerifierError::EmptyCalibration);
    }
    if drifts
        .iter()
        .any(|drift| !drift.churn.is_finite() || !drift.flips.is_finite())
    {
        return Err(VerifierError::NonFiniteCalibration);
    }
    if !safety_factor.is_finite() {
        return Err(VerifierError::InvalidBand(safety_factor));
    }
    let factor = safety_factor.max(1.0);
    let one_coordinate = 1.0 / coordinates as f32;
    let worst = |part: fn(&SignDrift) -> f32| drifts.iter().map(part).fold(0.0f32, f32::max);
    let band = |worst: f32| (worst * factor).max(one_coordinate).min(1.0);
    Ok(SignBand {
        churn: band(worst(|drift| drift.churn)),
        flips: band(worst(|drift| drift.flips)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two chunks of eight, each sending its top two: the first payload's
    /// signs are +1 at 0 and −1 at 3, and +1 at 9 and +1 at 12.
    fn sent() -> Vec<f32> {
        let mut v = vec![0.0; 16];
        v[0] = 1.0;
        v[3] = -1.0;
        v[9] = 1.0;
        v[12] = 1.0;
        v
    }

    #[test]
    fn an_identical_replay_has_no_drift() {
        let drift = sign_drift(&sent(), &sent()).unwrap();
        assert_eq!(
            drift,
            SignDrift {
                churn: 0.0,
                flips: 0.0
            }
        );
        assert_eq!(drift.distance(), 0.0);
    }

    #[test]
    fn a_coordinate_picked_instead_is_churn_and_a_sign_turned_is_a_flip() {
        let mut replay = sent();
        // Churn: the replay keeps coordinate 5 where the node kept 3.
        replay[3] = 0.0;
        replay[5] = -1.0;
        // A flip: the same coordinate 12, the other sign.
        replay[12] = -1.0;
        let drift = sign_drift(&sent(), &replay).unwrap();
        assert_eq!(drift.churn, 0.25);
        assert_eq!(drift.flips, 1.0 / 3.0);
    }

    #[test]
    fn the_two_make_the_relative_l2_distance_over_signs() {
        let mut replay = sent();
        replay[3] = 0.0;
        replay[5] = -1.0;
        replay[12] = -1.0;
        let drift = sign_drift(&sent(), &replay).unwrap();
        let l2 = crate::relative_l2_distance(&sent(), &replay).unwrap();
        assert!((drift.distance() - l2).abs() < 1e-6, "{drift:?} vs {l2}");
    }

    #[test]
    fn a_payload_that_leaves_out_honest_coordinates_churns_what_it_left_out() {
        // It sends one coordinate a chunk where the replay sends two, each of
        // the ones it sends right in place and sign.
        let mut fewer = sent();
        fewer[3] = 0.0;
        fewer[12] = 0.0;
        let drift = sign_drift(&fewer, &sent()).unwrap();
        assert_eq!(
            drift,
            SignDrift {
                churn: 0.5,
                flips: 0.0
            }
        );
        // Held to no churn, it is fraud, as its distance always made it.
        let band = SignBand {
            churn: 0.0,
            flips: 0.0,
        };
        assert!(
            verify_signs_within_band(&fewer, &sent(), band)
                .unwrap()
                .fraud
        );
    }

    #[test]
    fn a_payload_that_sends_extra_coordinates_churns_what_it_added() {
        let mut more = sent();
        more[1] = 1.0;
        more[14] = -1.0;
        let drift = sign_drift(&more, &sent()).unwrap();
        assert!((drift.churn - 2.0 / 6.0).abs() < 1e-6, "{drift:?}");
        assert_eq!(drift.flips, 0.0);
    }

    #[test]
    fn a_payload_that_sent_nothing_churned_what_the_replay_sent() {
        let nothing = vec![0.0; 16];
        assert_eq!(
            sign_drift(&nothing, &sent()).unwrap(),
            SignDrift {
                churn: 1.0,
                flips: 0.0
            }
        );
        assert_eq!(
            sign_drift(&nothing, &nothing).unwrap(),
            SignDrift {
                churn: 0.0,
                flips: 0.0
            }
        );
    }

    #[test]
    fn a_band_convicts_once_either_drift_passes_its_own_limit() {
        let band = SignBand {
            churn: 0.25,
            flips: 0.0,
        };
        let mut churned = sent();
        churned[3] = 0.0;
        churned[5] = -1.0;
        let verdict = verify_signs_within_band(&sent(), &churned, band).unwrap();
        assert!(!verdict.fraud, "{verdict:?}");

        // One flip, with no flip allowed.
        let mut flipped = sent();
        flipped[12] = -1.0;
        let verdict = verify_signs_within_band(&sent(), &flipped, band).unwrap();
        assert!(verdict.fraud, "{verdict:?}");

        // Churn past its limit.
        let mut more = churned.clone();
        more[0] = 0.0;
        more[1] = 1.0;
        let verdict = verify_signs_within_band(&sent(), &more, band).unwrap();
        assert!(verdict.fraud, "{verdict:?}");
    }

    #[test]
    fn a_submission_that_is_not_made_of_numbers_is_fraud_and_a_bad_replay_is_an_error() {
        let band = SignBand {
            churn: 1.0,
            flips: 1.0,
        };
        let mut bad = sent();
        bad[0] = f32::NAN;
        assert!(verify_signs_within_band(&bad, &sent(), band).unwrap().fraud);
        assert_eq!(
            verify_signs_within_band(&sent(), &bad, band),
            Err(VerifierError::NonFiniteReplay { index: 0 })
        );
    }

    #[test]
    fn a_band_outside_zero_to_one_is_refused() {
        for band in [
            SignBand {
                churn: -0.1,
                flips: 0.0,
            },
            SignBand {
                churn: 0.0,
                flips: 1.5,
            },
            SignBand {
                churn: f32::NAN,
                flips: 0.0,
            },
        ] {
            assert!(matches!(
                verify_signs_within_band(&sent(), &sent(), band),
                Err(VerifierError::InvalidSignBand(_))
            ));
        }
    }

    #[test]
    fn a_band_is_calibrated_part_by_part_from_the_worst_honest_drift() {
        let drifts = [
            SignDrift {
                churn: 0.02,
                flips: 0.0,
            },
            SignDrift {
                churn: 0.04,
                flips: 0.0,
            },
            SignDrift {
                churn: 0.01,
                flips: 0.001,
            },
        ];
        let band = calibrate_sign_band(&drifts, 2.0, 10_000).unwrap();
        assert!((band.churn - 0.08).abs() < 1e-6, "{band:?}");
        assert!((band.flips - 0.002).abs() < 1e-6, "{band:?}");
        // A factor below 1 does not narrow the band below what honest
        // replays showed, and no band goes past everything.
        let band = calibrate_sign_band(&drifts, 0.5, 10_000).unwrap();
        assert!((band.churn - 0.04).abs() < 1e-6, "{band:?}");
        let band = calibrate_sign_band(&drifts, 100.0, 10_000).unwrap();
        assert_eq!(band.churn, 1.0);
        // Every honest drift it was calibrated on passes it.
        let band = calibrate_sign_band(&drifts, 1.0, 10_000).unwrap();
        for drift in drifts {
            assert!(drift.churn <= band.churn && drift.flips <= band.flips);
        }
    }

    #[test]
    fn a_part_no_sample_drifted_in_is_still_given_one_coordinate() {
        // No sample flipped, and none churned: an exact-looking calibration.
        let drifts = [SignDrift {
            churn: 0.0,
            flips: 0.0,
        }];
        let band = calibrate_sign_band(&drifts, 5.0, 82).unwrap();
        assert_eq!(band.churn, 1.0 / 82.0);
        assert_eq!(band.flips, 1.0 / 82.0);
        // So one sign crossing zero in a payload of 82 coordinates passes it.
        let one_flip = SignDrift {
            churn: 0.0,
            flips: 1.0 / 82.0,
        };
        assert!(one_flip.flips <= band.flips);
        assert_eq!(
            calibrate_sign_band(&drifts, 5.0, 0),
            Err(VerifierError::EmptyCalibration)
        );
    }

    #[test]
    fn a_calibration_needs_samples_made_of_numbers() {
        assert_eq!(
            calibrate_sign_band(&[], 2.0, 10_000),
            Err(VerifierError::EmptyCalibration)
        );
        assert_eq!(
            calibrate_sign_band(
                &[SignDrift {
                    churn: f32::NAN,
                    flips: 0.0
                }],
                2.0,
                10_000
            ),
            Err(VerifierError::NonFiniteCalibration)
        );
        assert!(matches!(
            calibrate_sign_band(
                &[SignDrift {
                    churn: 0.1,
                    flips: 0.0
                }],
                f32::NAN,
                10_000
            ),
            Err(VerifierError::InvalidBand(_))
        ));
    }

    #[test]
    fn payloads_of_different_lengths_are_an_error() {
        assert_eq!(
            sign_drift(&sent(), &sent()[..8]),
            Err(VerifierError::LengthMismatch {
                submitted: 16,
                recomputed: 8
            })
        );
    }
}
