use serde::{Deserialize, Serialize};

use crate::aggregate::AggregateClaim;
use crate::config::RunConfig;
use crate::merkle;
use crate::protocol::Contribution;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunInfo {
    pub cfg: RunConfig,
    pub run: [u8; 32],
    pub corpus: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundInfo {
    pub run: [u8; 32],
    pub round: u32,
    pub seed: [u8; 32],
    pub base_checkpoint: [u8; 32],
    pub closes_in_ms: u64,
    pub global_loss: f32,
    pub phones_last_round: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundResult {
    pub claim: AggregateClaim,
    pub digests: Vec<[u8; 32]>,
    pub audit_targets: Vec<[u8; 32]>,
}

impl RoundResult {
    pub fn counts(&self, contribution: &Contribution) -> bool {
        let phone = contribution.header.phone;
        let Some(index) = self.claim.contributors.iter().position(|p| *p == phone) else {
            return false;
        };
        self.digests.len() == self.claim.contributors.len()
            && self.digests[index] == contribution.digest()
            && merkle::root(&self.digests) == self.claim.contributions_root
    }

    pub fn audits(&self, phone: &[u8; 32]) -> bool {
        self.audit_targets.contains(phone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::{aggregate, Expectation};
    use crate::compress::compress;
    use crate::config::{CompressionConfig, InnerConfig, ModelConfig};
    use crate::protocol::{signing_key, ContributionHeader};

    fn contribution(seed: u8, cfg: &RunConfig, expect: &Expectation) -> Contribution {
        let len = cfg.model.num_params();
        let delta: Vec<f32> = (0..len)
            .map(|i| ((i * seed as usize) % 7) as f32 - 3.0)
            .collect();
        let (update, _) = compress(&delta, &vec![0.0; len], &cfg.compression).unwrap();
        let header = ContributionHeader {
            run: expect.run,
            round: expect.round,
            phone: [0u8; 32],
            base_checkpoint: expect.base_checkpoint,
            inner_steps: cfg.inner.min_steps,
            tokens: cfg.inner.min_steps as u64 * cfg.tokens_per_step(),
            mean_loss: 1.0,
            error_commitment: [0u8; 32],
            update_hash: [0u8; 32],
        };
        Contribution::sign(header, update, &signing_key(&[seed; 32]))
    }

    #[test]
    fn a_phone_sees_whether_it_was_counted() {
        let cfg = RunConfig {
            model: ModelConfig {
                d_model: 8,
                n_heads: 2,
                n_layers: 1,
                context: 8,
                vocab_size: 16,
                mlp_ratio: 2,
            },
            inner: InnerConfig::default(),
            compression: CompressionConfig::default(),
            ..RunConfig::default()
        };
        let expect = Expectation::new(&cfg, [9u8; 32], 3, [4u8; 32]);
        let counted = contribution(1, &cfg, &expect);
        let other = contribution(2, &cfg, &expect);
        let late = contribution(3, &cfg, &expect);
        let aggregation = aggregate(&expect, &[counted.clone(), other.clone()]);
        let result = RoundResult {
            claim: aggregation.claim.clone(),
            digests: aggregation.digests.clone(),
            audit_targets: vec![other.header.phone],
        };

        assert!(result.counts(&counted));
        assert!(result.counts(&other));
        assert!(!result.counts(&late));
        assert!(result.audits(&other.header.phone));
        assert!(!result.audits(&counted.header.phone));

        let mut forged = result.clone();
        forged.claim.contributions_root = [0u8; 32];
        assert!(!forged.counts(&counted));
    }
}
