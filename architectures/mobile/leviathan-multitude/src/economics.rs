//! Whether cheating pays, with the formula the Leviathan indexer publishes.
//!
//! Mirrors `leviathan_indexer::assess_security` (that crate pulls in the
//! Solana programs, so the arithmetic is repeated here rather than depended
//! on): a cheater gains the reward when it is not audited and loses the
//! effective penalty, the smaller of the slash and the bond, when it is.

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SecurityAssessment {
    pub audit_probability: f64,
    pub break_even_penalty: Option<f64>,
    pub effective_penalty: f64,
    pub expected_fraud_value_per_round: f64,
    pub economically_secure: bool,
}

pub fn assess(
    audit_probability: f64,
    reward_per_round: f64,
    bond: f64,
    slash_when_caught: f64,
) -> SecurityAssessment {
    let p = audit_probability.clamp(0.0, 1.0);
    let break_even_penalty = (p > 0.0)
        .then(|| reward_per_round * (1.0 - p) / p)
        .filter(|value| value.is_finite());
    let effective_penalty = slash_when_caught.min(bond);
    let expected_fraud_value_per_round = (1.0 - p) * reward_per_round - p * effective_penalty;
    SecurityAssessment {
        audit_probability: p,
        break_even_penalty,
        effective_penalty,
        expected_fraud_value_per_round,
        economically_secure: p > 0.0 && expected_fraud_value_per_round <= 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_indexer_cases() {
        let secure = assess(0.1, 1000.0, 100_000.0, 9000.0);
        assert_eq!(secure.break_even_penalty, Some(9000.0));
        assert!(secure.economically_secure);

        let capped = assess(0.1, 1000.0, 500.0, 1_000_000.0);
        assert_eq!(capped.effective_penalty, 500.0);
        assert!(!capped.economically_secure);

        assert!(!assess(0.0, 1000.0, 1e6, 1e6).economically_secure);
    }
}
