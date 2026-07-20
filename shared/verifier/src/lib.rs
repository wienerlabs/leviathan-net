use sha2::{Digest, Sha256};
use thiserror::Error;

pub const DEFAULT_BAND: f32 = 0.05;
pub const DEFAULT_SAFETY_FACTOR: f32 = 5.0;
const NORM_FLOOR: f32 = 1e-12;

#[derive(Debug, Error, PartialEq)]
pub enum VerifierError {
    #[error("submitted and recomputed deltas have different lengths ({submitted} vs {recomputed})")]
    LengthMismatch { submitted: usize, recomputed: usize },
    #[error("band must be finite and non-negative, got {0}")]
    InvalidBand(f32),
    #[error("no drift samples to calibrate from")]
    EmptyCalibration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandVerdict {
    pub distance: f32,
    pub band: f32,
    pub fraud: bool,
}

impl BandVerdict {
    pub fn margin(&self) -> f32 {
        self.distance - self.band
    }
}

pub fn relative_l2_distance(submitted: &[f32], recomputed: &[f32]) -> Result<f32, VerifierError> {
    if submitted.len() != recomputed.len() {
        return Err(VerifierError::LengthMismatch {
            submitted: submitted.len(),
            recomputed: recomputed.len(),
        });
    }
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (s, r) in submitted.iter().zip(recomputed.iter()) {
        let d = (*s - *r) as f64;
        diff_sq += d * d;
        ref_sq += (*r as f64) * (*r as f64);
    }
    let denom = ref_sq.sqrt().max(NORM_FLOOR as f64);
    Ok((diff_sq.sqrt() / denom) as f32)
}

pub fn verify_within_band(
    submitted: &[f32],
    recomputed: &[f32],
    band: f32,
) -> Result<BandVerdict, VerifierError> {
    if !band.is_finite() || band < 0.0 {
        return Err(VerifierError::InvalidBand(band));
    }
    let distance = relative_l2_distance(submitted, recomputed)?;
    Ok(BandVerdict {
        distance,
        band,
        fraud: distance > band,
    })
}

pub fn calibrate_band(
    drift_distances: &[f32],
    safety_factor: f32,
) -> Result<f32, VerifierError> {
    if drift_distances.is_empty() {
        return Err(VerifierError::EmptyCalibration);
    }
    let worst = drift_distances
        .iter()
        .copied()
        .fold(0.0f32, |acc, d| acc.max(d.abs()));
    Ok(worst * safety_factor.max(1.0))
}

pub fn hash_delta(delta: &[f32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for value in delta {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FraudProof {
    pub target_index: u64,
    pub committed_hash: [u8; 32],
    pub replayed_hash: [u8; 32],
    pub distance: f32,
    pub band: f32,
}

pub struct AuditReport {
    pub verdict: BandVerdict,
    pub proof: Option<FraudProof>,
}

pub fn audit_contribution(
    target_index: u64,
    submitted: &[f32],
    recomputed: &[f32],
    band: f32,
) -> Result<AuditReport, VerifierError> {
    let verdict = verify_within_band(submitted, recomputed, band)?;
    let proof = if verdict.fraud {
        Some(FraudProof {
            target_index,
            committed_hash: hash_delta(submitted),
            replayed_hash: hash_delta(recomputed),
            distance: verdict.distance,
            band: verdict.band,
        })
    } else {
        None
    };
    Ok(AuditReport { verdict, proof })
}

#[derive(Debug, Error, PartialEq)]
pub enum ReplayError {
    #[error("target {0} is not assigned to this verifier")]
    NotAssigned(u64),
    #[error("checkpoint {0} unavailable for replay")]
    CheckpointUnavailable(u64),
    #[error("replay backend failed: {0}")]
    Backend(String),
}

pub trait ReplayEngine {
    fn replay(&self, target_index: u64) -> Result<Vec<f32>, ReplayError>;
}

#[derive(Debug, Clone)]
pub struct Contribution {
    pub target_index: u64,
    pub submitted: Vec<f32>,
}

pub enum AuditOutcome {
    Judged(AuditReport),
    ReplayFailed { target_index: u64, error: ReplayError },
    Malformed { target_index: u64, error: VerifierError },
}

pub fn audit_round<E: ReplayEngine>(
    engine: &E,
    contributions: &[Contribution],
    band: f32,
) -> Vec<AuditOutcome> {
    contributions
        .iter()
        .map(|c| match engine.replay(c.target_index) {
            Ok(recomputed) => match audit_contribution(c.target_index, &c.submitted, &recomputed, band) {
                Ok(report) => AuditOutcome::Judged(report),
                Err(error) => AuditOutcome::Malformed {
                    target_index: c.target_index,
                    error,
                },
            },
            Err(error) => AuditOutcome::ReplayFailed {
                target_index: c.target_index,
                error,
            },
        })
        .collect()
}

pub fn fraud_proofs(outcomes: &[AuditOutcome]) -> Vec<FraudProof> {
    outcomes
        .iter()
        .filter_map(|o| match o {
            AuditOutcome::Judged(report) => report.proof,
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    struct RecordedTruth {
        deltas: std::collections::HashMap<u64, Vec<f32>>,
    }

    impl ReplayEngine for RecordedTruth {
        fn replay(&self, target_index: u64) -> Result<Vec<f32>, ReplayError> {
            self.deltas
                .get(&target_index)
                .cloned()
                .ok_or(ReplayError::NotAssigned(target_index))
        }
    }

    fn honest_delta(seed: u64, len: usize) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    fn with_drift(base: &[f32], drift: f32, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let base_norm = (base.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
        let noise: Vec<f32> = (0..base.len()).map(|_| rng.random_range(-1.0..1.0)).collect();
        let noise_norm = (noise.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()).sqrt();
        let scale = (drift as f64) * base_norm / noise_norm.max(1e-12);
        base.iter()
            .zip(noise.iter())
            .map(|(b, n)| b + (*n as f64 * scale) as f32)
            .collect()
    }

    #[test]
    fn honest_replay_scores_zero() {
        let delta = honest_delta(1, 4096);
        let verdict = verify_within_band(&delta, &delta, DEFAULT_BAND).unwrap();
        assert!(verdict.distance < 1e-6);
        assert!(!verdict.fraud);
    }

    #[test]
    fn benign_drift_passes_the_band() {
        let recomputed = honest_delta(2, 4096);
        let submitted = with_drift(&recomputed, 0.01, 99);
        let verdict = verify_within_band(&submitted, &recomputed, DEFAULT_BAND).unwrap();
        assert!(verdict.distance < DEFAULT_BAND, "distance {}", verdict.distance);
        assert!(!verdict.fraud);
    }

    #[test]
    fn sign_flip_is_caught() {
        let recomputed = honest_delta(3, 4096);
        let submitted: Vec<f32> = recomputed.iter().map(|v| -5.0 * v).collect();
        let report = audit_contribution(7, &submitted, &recomputed, DEFAULT_BAND).unwrap();
        assert!(report.verdict.fraud);
        assert!((report.verdict.distance - 6.0).abs() < 1e-3);
        let proof = report.proof.unwrap();
        assert_eq!(proof.target_index, 7);
        assert_ne!(proof.committed_hash, proof.replayed_hash);
    }

    #[test]
    fn lazy_zero_is_caught() {
        let recomputed = honest_delta(4, 4096);
        let submitted = vec![0.0f32; recomputed.len()];
        let verdict = verify_within_band(&submitted, &recomputed, DEFAULT_BAND).unwrap();
        assert!(verdict.fraud);
        assert!((verdict.distance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn gaussian_forgery_is_caught() {
        let recomputed = honest_delta(5, 4096);
        let submitted = honest_delta(6, 4096);
        let verdict = verify_within_band(&submitted, &recomputed, DEFAULT_BAND).unwrap();
        assert!(verdict.fraud);
        assert!(verdict.distance > 1.0);
    }

    #[test]
    fn honest_report_carries_no_proof() {
        let delta = honest_delta(8, 1024);
        let report = audit_contribution(0, &delta, &delta, DEFAULT_BAND).unwrap();
        assert!(!report.verdict.fraud);
        assert!(report.proof.is_none());
    }

    #[test]
    fn calibration_sits_above_observed_drift() {
        let recomputed = honest_delta(10, 4096);
        let drift: Vec<f32> = (0..8)
            .map(|s| {
                let submitted = with_drift(&recomputed, 0.008, 200 + s);
                relative_l2_distance(&submitted, &recomputed).unwrap()
            })
            .collect();
        let band = calibrate_band(&drift, DEFAULT_SAFETY_FACTOR).unwrap();
        assert!(band > drift.iter().cloned().fold(0.0, f32::max));
        for d in &drift {
            let verdict = verify_within_band_from_distance(*d, band);
            assert!(!verdict.fraud);
        }
    }

    fn verify_within_band_from_distance(distance: f32, band: f32) -> BandVerdict {
        BandVerdict {
            distance,
            band,
            fraud: distance > band,
        }
    }

    #[test]
    fn length_mismatch_errors() {
        let err = relative_l2_distance(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert_eq!(
            err,
            VerifierError::LengthMismatch {
                submitted: 2,
                recomputed: 1,
            }
        );
    }

    #[test]
    fn invalid_band_errors() {
        let delta = honest_delta(11, 16);
        assert_eq!(
            verify_within_band(&delta, &delta, -0.1).unwrap_err(),
            VerifierError::InvalidBand(-0.1)
        );
    }

    #[test]
    fn empty_calibration_errors() {
        assert_eq!(
            calibrate_band(&[], DEFAULT_SAFETY_FACTOR).unwrap_err(),
            VerifierError::EmptyCalibration
        );
    }

    #[test]
    fn hash_is_deterministic_and_tamper_sensitive() {
        let delta = honest_delta(12, 256);
        assert_eq!(hash_delta(&delta), hash_delta(&delta));
        let mut tampered = delta.clone();
        tampered[0] += 1e-3;
        assert_ne!(hash_delta(&delta), hash_delta(&tampered));
    }

    #[test]
    fn audit_round_convicts_only_the_cheater() {
        let honest = honest_delta(20, 2048);
        let cheater = honest_delta(21, 2048);
        let mut truth = std::collections::HashMap::new();
        truth.insert(1u64, honest.clone());
        truth.insert(2u64, cheater.clone());
        let engine = RecordedTruth { deltas: truth };
        let contributions = vec![
            Contribution {
                target_index: 1,
                submitted: with_drift(&honest, 0.01, 500),
            },
            Contribution {
                target_index: 2,
                submitted: cheater.iter().map(|v| -5.0 * v).collect(),
            },
        ];
        let outcomes = audit_round(&engine, &contributions, DEFAULT_BAND);
        let proofs = fraud_proofs(&outcomes);
        assert_eq!(proofs.len(), 1);
        assert_eq!(proofs[0].target_index, 2);
    }

    #[test]
    fn audit_round_reports_unassigned_replay_failure() {
        let engine = RecordedTruth {
            deltas: std::collections::HashMap::new(),
        };
        let contributions = vec![Contribution {
            target_index: 9,
            submitted: vec![0.0; 4],
        }];
        let outcomes = audit_round(&engine, &contributions, DEFAULT_BAND);
        assert!(matches!(
            outcomes[0],
            AuditOutcome::ReplayFailed {
                target_index: 9,
                error: ReplayError::NotAssigned(9)
            }
        ));
        assert!(fraud_proofs(&outcomes).is_empty());
    }
}
