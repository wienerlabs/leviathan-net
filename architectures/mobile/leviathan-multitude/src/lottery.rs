//! The audit lottery.
//!
//! Each round, every counted phone becomes a verifier with probability
//! `verification_percent / 100`, and each verifier draws one other phone to
//! replay. Both draws come from the round seed, the same construction as the
//! coordinator's `audit_selection` (same `"audit"` salt), so anyone can check
//! who was supposed to audit whom.

use crate::rng::{digest_u64, sha256_parts};

pub const AUDIT_SALT: &[u8] = b"audit";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuditAssignment {
    pub verifier: usize,
    pub target: usize,
}

/// Assignments over `participants` phones indexed `0..participants`, in the
/// claim's contributor order.
pub fn select_audits(
    seed: &[u8; 32],
    participants: usize,
    verification_percent: u8,
) -> Vec<AuditAssignment> {
    if participants < 2 {
        return Vec::new();
    }
    (0..participants)
        .filter_map(|verifier| {
            let index = (verifier as u64).to_le_bytes();
            let role = digest_u64(&sha256_parts(&[seed, b"verifier", &index]));
            if role % 100 >= verification_percent as u64 {
                return None;
            }
            let pick =
                digest_u64(&sha256_parts(&[seed, AUDIT_SALT, &index])) % (participants as u64 - 1);
            // Skip over the verifier itself, so every other phone is equally likely.
            let target = if pick as usize >= verifier {
                pick as usize + 1
            } else {
                pick as usize
            };
            Some(AuditAssignment { verifier, target })
        })
        .collect()
}

/// Probability that a given phone is audited at least once in a round.
/// For a large swarm this approaches `verification_percent / 100`, the `p` in
/// Leviathan's break-even bond `reward * (1 - p) / p`.
pub fn audit_probability(participants: usize, verification_percent: u8) -> f64 {
    if participants < 2 {
        return 0.0;
    }
    let p = verification_percent as f64 / 100.0;
    let others = participants as f64 - 1.0;
    1.0 - (1.0 - p / others).powf(others)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_never_self() {
        let seed = [42u8; 32];
        let a = select_audits(&seed, 20, 30);
        assert_eq!(a, select_audits(&seed, 20, 30));
        assert!(a.iter().all(|x| x.verifier != x.target && x.target < 20));
    }

    #[test]
    fn rate_follows_verification_percent() {
        let mut verifiers = 0;
        for round in 0..400u32 {
            let seed = sha256_parts(&[b"t", &round.to_le_bytes()]);
            verifiers += select_audits(&seed, 50, 10).len();
        }
        let rate = verifiers as f64 / (400.0 * 50.0);
        assert!((rate - 0.10).abs() < 0.02, "rate {rate}");
    }

    #[test]
    fn too_few_phones_means_no_audits() {
        assert!(select_audits(&[0u8; 32], 1, 100).is_empty());
        assert_eq!(audit_probability(1, 100), 0.0);
    }
}
