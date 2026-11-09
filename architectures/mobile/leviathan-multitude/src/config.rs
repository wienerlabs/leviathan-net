//! Run configuration.
//!
//! The names follow the Leviathan configs where the meaning is the same:
//! `verification_percent` is the audit rate, `band` is the replay tolerance,
//! `chunk` and `topk` are the SparseLoCo transport of
//! `config/leviathan/genesis-rehearsal.toml` (`compression_chunk = 64`,
//! `compression_topk = 8`, `quantize_1bit = true`).

use serde::{Deserialize, Serialize};

use crate::rng::sha256_parts;
use crate::{Error, Result};

/// The transformer the swarm trains. Byte-level, so no tokenizer has to ship
/// to the phone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub context: usize,
    pub mlp_ratio: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            context: 64,
            mlp_ratio: 4,
        }
    }
}

impl ModelConfig {
    /// Parameter count, matching the tensors [`crate::model::TinyLm`] creates.
    pub fn num_params(&self) -> usize {
        let d = self.d_model;
        let hidden = self.mlp_ratio * d;
        let block = 2 * d + 3 * d * d + d * d + 2 * d * hidden;
        self.vocab_size * d + self.context * d + self.n_layers * block + d + d * self.vocab_size
    }

    pub fn validate(&self) -> Result<()> {
        let all_positive = [
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.context,
            self.mlp_ratio,
        ]
        .iter()
        .all(|v| *v > 0);
        if !all_positive {
            return Err(Error::Invalid(
                "every model dimension must be positive".into(),
            ));
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(Error::Invalid(format!(
                "d_model {} is not divisible by n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        Ok(())
    }
}

/// One phone's local work in a round: the DiLoCo inner loop.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InnerConfig {
    pub lr: f64,
    pub micro_batch: usize,
    pub weight_decay: f64,
    /// Fewest inner steps a contribution may claim. A phone that cannot fit
    /// this many before the deadline sits the round out.
    pub min_steps: u16,
    /// Most inner steps a contribution may claim. Capping it bounds what an
    /// audit has to replay.
    pub max_steps: u16,
}

impl Default for InnerConfig {
    fn default() -> Self {
        Self {
            lr: 3e-3,
            micro_batch: 8,
            weight_decay: 0.0,
            min_steps: 2,
            max_steps: 12,
        }
    }
}

/// SparseLoCo transport: chunked top-k, one sign bit each, error feedback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub chunk: usize,
    pub topk: usize,
    /// Multiplies the error-feedback buffer before the new pseudo-gradient is
    /// added. 1.0 is classic error feedback.
    pub error_decay: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            chunk: 64,
            topk: 8,
            error_decay: 1.0,
        }
    }
}

/// The outer step every replica applies to the tally.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OuterConfig {
    pub lr: f32,
    /// The learning rate decays on a cosine to `lr * final_fraction` over
    /// `total_rounds`.
    pub final_fraction: f32,
    pub total_rounds: u32,
}

impl Default for OuterConfig {
    fn default() -> Self {
        Self {
            lr: 0.006,
            final_fraction: 0.1,
            total_rounds: 30,
        }
    }
}

impl OuterConfig {
    pub fn lr_at(&self, round: u32) -> f32 {
        if self.total_rounds == 0 {
            return self.lr;
        }
        let progress = (round.min(self.total_rounds) as f32) / self.total_rounds as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.lr * (self.final_fraction + (1.0 - self.final_fraction) * cosine)
    }
}

/// Audit parameters, named as in the Leviathan coordinator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditConfig {
    pub verification_percent: u8,
    pub band: f32,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            verification_percent: 10,
            band: psyche_verifier::DEFAULT_BAND,
        }
    }
}

/// Everything a phone needs to know to take part in a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunConfig {
    pub run_id: String,
    pub model: ModelConfig,
    pub inner: InnerConfig,
    pub compression: CompressionConfig,
    pub outer: OuterConfig,
    pub audit: AuditConfig,
    /// Wall-clock budget for one round. Phones plan their inner steps to fit.
    pub round_seconds: f32,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            run_id: "multitude-devnet".into(),
            model: ModelConfig::default(),
            inner: InnerConfig::default(),
            compression: CompressionConfig::default(),
            outer: OuterConfig::default(),
            audit: AuditConfig::default(),
            round_seconds: 60.0,
        }
    }
}

impl RunConfig {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        let c = &self.compression;
        if c.chunk == 0 || c.chunk > 256 || c.topk == 0 || c.topk > c.chunk {
            return Err(Error::Invalid(format!(
                "chunk {} must be in 1..=256 and topk {} in 1..=chunk",
                c.chunk, c.topk
            )));
        }
        if self.inner.min_steps == 0 || self.inner.min_steps > self.inner.max_steps {
            return Err(Error::Invalid(format!(
                "inner steps must satisfy 1 <= min ({}) <= max ({})",
                self.inner.min_steps, self.inner.max_steps
            )));
        }
        if self.audit.verification_percent > 100 {
            return Err(Error::Invalid(
                "verification_percent is a percentage".into(),
            ));
        }
        if !(self.audit.band.is_finite() && self.audit.band >= 0.0) {
            return Err(Error::Invalid(
                "band must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }

    /// Tokens one inner step consumes.
    pub fn tokens_per_step(&self) -> u64 {
        (self.inner.micro_batch * self.model.context) as u64
    }

    /// What every contribution commits to: this config and the corpus.
    /// Two phones with different fingerprints are not in the same run.
    pub fn fingerprint(&self, corpus: &[u8; 32]) -> [u8; 32] {
        let encoded = postcard::to_stdvec(self).expect("run config always serializes");
        sha256_parts(&[b"multitude/run/v1", &encoded, corpus])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        RunConfig::default().validate().unwrap();
    }

    #[test]
    fn outer_lr_decays_on_a_cosine() {
        let outer = OuterConfig::default();
        assert!((outer.lr_at(0) - outer.lr).abs() < 1e-9);
        let end = outer.lr_at(outer.total_rounds);
        assert!((end - outer.lr * outer.final_fraction).abs() < 1e-6);
        assert!(outer.lr_at(10) < outer.lr_at(5));
    }

    #[test]
    fn fingerprint_changes_with_the_config() {
        let a = RunConfig::default();
        let mut b = a.clone();
        b.audit.verification_percent = 20;
        let corpus = [7u8; 32];
        assert_ne!(a.fingerprint(&corpus), b.fingerprint(&corpus));
    }
}
