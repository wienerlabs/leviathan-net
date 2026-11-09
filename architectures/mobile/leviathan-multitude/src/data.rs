//! Training data.
//!
//! The corpus is bytes, and which windows a phone trains on is a function of a
//! public seed. A verifier that replays a phone's round reads exactly the
//! windows the phone read, with no file order or shuffle state to disagree on.
//! (Leviathan learned this the hard way: trainers and the verifier once read
//! shards in different orders and could slash an honest node, fixed in
//! 90c27535.)

use crate::rng::{sha256_parts, SplitMix64};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Corpus {
    bytes: Vec<u8>,
    fingerprint: [u8; 32],
}

impl Corpus {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let fingerprint = sha256_parts(&[b"multitude/corpus/v1", &bytes]);
        Self { bytes, fingerprint }
    }

    pub fn from_text(text: &str) -> Self {
        Self::from_bytes(text.as_bytes().to_vec())
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }

    pub fn check_context(&self, context: usize) -> Result<()> {
        if self.bytes.len() < context + 2 {
            return Err(Error::Invalid(format!(
                "corpus has {} bytes, a window of {} needs at least {}",
                self.bytes.len(),
                context,
                context + 2
            )));
        }
        Ok(())
    }

    fn span(&self, context: usize) -> u64 {
        (self.bytes.len() - context - 1) as u64
    }

    /// The `(inputs, targets)` a phone trains on at inner step `step`, each of
    /// length `micro_batch * context`. Targets are inputs shifted by one byte.
    pub fn batch(
        &self,
        seed: &[u8; 32],
        step: u32,
        micro_batch: usize,
        context: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let digest = sha256_parts(&[b"multitude/batch", seed, &step.to_le_bytes()]);
        let mut rng = SplitMix64::from_digest(&digest);
        let span = self.span(context);
        let mut inputs = Vec::with_capacity(micro_batch * context);
        let mut targets = Vec::with_capacity(micro_batch * context);
        for _ in 0..micro_batch {
            let start = rng.below(span) as usize;
            inputs.extend(self.bytes[start..start + context].iter().map(|&b| b as u32));
            targets.extend(
                self.bytes[start + 1..start + context + 1]
                    .iter()
                    .map(|&b| b as u32),
            );
        }
        (inputs, targets)
    }

    /// Evenly spaced windows the whole swarm scores the global model on.
    pub fn eval_windows(&self, context: usize, count: usize) -> (Vec<u32>, Vec<u32>) {
        let span = self.span(context) as usize;
        let mut inputs = Vec::with_capacity(count * context);
        let mut targets = Vec::with_capacity(count * context);
        for i in 0..count {
            let start = i * span / count.max(1);
            inputs.extend(self.bytes[start..start + context].iter().map(|&b| b as u32));
            targets.extend(
                self.bytes[start + 1..start + context + 1]
                    .iter()
                    .map(|&b| b as u32),
            );
        }
        (inputs, targets)
    }
}

/// The seed a phone's windows come from in a round: the round seed and the
/// phone's public key, nothing the phone chooses.
pub fn data_seed(round_seed: &[u8; 32], phone: &[u8; 32]) -> [u8; 32] {
    sha256_parts(&[b"multitude/data", round_seed, phone])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_are_reproducible_and_shifted() {
        let corpus = Corpus::from_text("the quick brown fox jumps over the lazy dog, twice over");
        let seed = [1u8; 32];
        let (x1, y1) = corpus.batch(&seed, 3, 4, 8);
        let (x2, y2) = corpus.batch(&seed, 3, 4, 8);
        assert_eq!((x1.clone(), y1.clone()), (x2, y2));
        assert_eq!(x1.len(), 32);
        assert_eq!(&x1[1..8], &y1[0..7]);
        let (x3, _) = corpus.batch(&seed, 4, 4, 8);
        assert_ne!(x1, x3);
    }

    #[test]
    fn different_phones_read_different_windows() {
        let round = [5u8; 32];
        assert_ne!(data_seed(&round, &[1u8; 32]), data_seed(&round, &[2u8; 32]));
    }

    #[test]
    fn short_corpus_is_rejected() {
        assert!(Corpus::from_text("tiny").check_context(8).is_err());
    }
}
