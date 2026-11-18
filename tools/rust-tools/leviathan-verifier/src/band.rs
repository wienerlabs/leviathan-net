//! What a payload is held to (wienerlabs/leviathan-net#49).
//!
//! A band on the relative L2 distance fits payloads of values. Every Leviathan
//! run sends 1-bit signs, and over signs a replay differs in two ways the
//! distance mixes: a coordinate picked instead of the node's (churn), and a
//! coordinate picked with the other sign (flips). Honest replays across
//! hardware classes churn and do not flip, so a 1-bit run is held to a band on
//! each (`docs/BAND_CALIBRATION.md`).

use std::fmt;

use anyhow::anyhow;
use anyhow::Result;
use psyche_verifier::verify_signs_within_band;
use psyche_verifier::verify_within_band;
use psyche_verifier::SignBand;
use psyche_verifier::SignDrift;
use psyche_verifier::DEFAULT_BAND;

/// What a payload is held to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PayloadBand {
    /// A band on the relative L2 distance from the honest result.
    Distance(f32),
    /// For a run that sends 1-bit signs, a band on each of churn and flips.
    /// A band on the signs cannot see magnitudes, so under it a payload that
    /// carries values rather than packed signs is unfit.
    Signs(SignBand),
}

impl PayloadBand {
    /// The largest relative L2 distance a payload within this band can be
    /// at. For a band on the distance it is that band. For a band on the
    /// signs it is the largest of `2 · churn + 4 · (1 − churn) · flips` over
    /// churn and flips within it, square-rooted: with flips above a half it
    /// is reached with no churn at all. Like that identity, it holds for
    /// payloads that send as many coordinates as the replay.
    pub fn distance_bound(&self) -> f32 {
        match self {
            PayloadBand::Distance(band) => *band,
            PayloadBand::Signs(band) => {
                let churned = SignDrift {
                    churn: band.churn,
                    flips: band.flips,
                }
                .distance();
                let unchurned = SignDrift {
                    churn: 0.0,
                    flips: band.flips,
                }
                .distance();
                churned.max(unchurned)
            }
        }
    }
}

impl fmt::Display for PayloadBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayloadBand::Distance(band) => write!(f, "band {band:.4}"),
            PayloadBand::Signs(band) => {
                write!(
                    f,
                    "churn band {:.4}, flip band {:.6}",
                    band.churn, band.flips
                )
            }
        }
    }
}

/// The band a command line names. `--churn-band` and `--flip-band` together
/// hold a 1-bit payload to a band on each of its two drifts; otherwise
/// `--band`, or the shipped default, holds it to the distance.
pub fn payload_band(
    band: Option<f32>,
    churn_band: Option<f32>,
    flip_band: Option<f32>,
) -> Result<PayloadBand> {
    let band = match (band, churn_band, flip_band) {
        (None, Some(churn), Some(flips)) => PayloadBand::Signs(SignBand { churn, flips }),
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            return Err(anyhow!(
                "--band holds a payload to its distance, --churn-band and --flip-band to its signs: pass one or the other"
            ))
        }
        (_, Some(_), None) | (_, None, Some(_)) => {
            return Err(anyhow!(
                "--churn-band and --flip-band go together: a 1-bit payload is held to both"
            ))
        }
        (band, None, None) => PayloadBand::Distance(band.unwrap_or(DEFAULT_BAND)),
    };
    // A band that is not one is refused here, where it is given, rather
    // than on every payload it is used on.
    match band {
        PayloadBand::Distance(distance) => verify_within_band(&[], &[], distance).map(|_| ())?,
        PayloadBand::Signs(signs) => verify_signs_within_band(&[], &[], signs).map(|_| ())?,
    }
    Ok(band)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sign_bands_together_make_a_band_on_the_signs() {
        assert_eq!(
            payload_band(None, Some(0.05), Some(0.001)).unwrap(),
            PayloadBand::Signs(SignBand {
                churn: 0.05,
                flips: 0.001
            })
        );
    }

    #[test]
    fn without_them_the_band_is_on_the_distance_and_defaults_as_it_always_has() {
        assert_eq!(
            payload_band(Some(0.2), None, None).unwrap(),
            PayloadBand::Distance(0.2)
        );
        assert_eq!(
            payload_band(None, None, None).unwrap(),
            PayloadBand::Distance(DEFAULT_BAND)
        );
    }

    #[test]
    fn a_band_on_both_or_half_a_sign_band_is_refused() {
        assert!(payload_band(Some(0.2), Some(0.05), Some(0.001)).is_err());
        assert!(payload_band(Some(0.2), Some(0.05), None).is_err());
        assert!(payload_band(None, Some(0.05), None).is_err());
        assert!(payload_band(None, None, Some(0.001)).is_err());
    }

    #[test]
    fn a_band_that_is_not_one_is_refused_where_it_is_given() {
        assert!(payload_band(Some(-0.1), None, None).is_err());
        assert!(payload_band(Some(f32::NAN), None, None).is_err());
        assert!(payload_band(None, Some(1.5), Some(0.0)).is_err());
        assert!(payload_band(None, Some(0.05), Some(f32::NAN)).is_err());
        assert!(payload_band(None, Some(0.0), Some(0.0)).is_ok());
    }

    #[test]
    fn a_sign_band_admits_the_distance_its_churn_and_flips_make() {
        let band = PayloadBand::Signs(SignBand {
            churn: 0.08,
            flips: 0.0,
        });
        assert!((band.distance_bound() - 0.4).abs() < 1e-6);
        assert_eq!(PayloadBand::Distance(0.05).distance_bound(), 0.05);
        // With flips above a half, the most distance is reached by flipping
        // every shared sign it may and churning nothing.
        let band = PayloadBand::Signs(SignBand {
            churn: 0.5,
            flips: 0.9,
        });
        assert!((band.distance_bound() - (4.0f32 * 0.9).sqrt()).abs() < 1e-6);
    }
}
