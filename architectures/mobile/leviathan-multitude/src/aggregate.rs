//! Verifiable aggregation.
//!
//! Phones cannot download every other phone's update each round, so a relay
//! collects them and publishes one tally. The tally is integer vote counts per
//! parameter (net votes and how many phones voted), which is exactly what
//! Leviathan's `CompressDCT::batch_decompress` averages with its scatter-mean,
//! held as integers instead of floats. Integers make the relay checkable bit
//! for bit: anyone with the same contributions recomputes the same counts, and
//! a relay that miscounts is caught with no tolerance band and no appeal.
//!
//! Applying the tally is one division and one multiply-subtract per parameter
//! in a fixed order, so every honest replica of the global model stays
//! bit-identical, and its checkpoint hash means the same thing everywhere.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::compress::CompressedUpdate;
use crate::config::{CompressionConfig, RunConfig};
use crate::merkle;
use crate::protocol::{hex, short_address, Contribution};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tally {
    net: Vec<i32>,
    count: Vec<u32>,
}

impl Tally {
    pub fn new(len: usize) -> Self {
        Self {
            net: vec![0; len],
            count: vec![0; len],
        }
    }

    pub fn len(&self) -> usize {
        self.net.len()
    }

    pub fn is_empty(&self) -> bool {
        self.net.is_empty()
    }

    pub fn net(&self) -> &[i32] {
        &self.net
    }

    pub fn count(&self) -> &[u32] {
        &self.count
    }

    pub(crate) fn net_mut(&mut self) -> &mut [i32] {
        &mut self.net
    }

    /// Count one update's votes. The update must already have been validated
    /// against this tally's length.
    pub fn add(&mut self, update: &CompressedUpdate) {
        for (index, vote) in update.votes() {
            self.net[index] += vote as i32;
            self.count[index] += 1;
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(8 + self.net.len() * 8);
        bytes.extend((self.net.len() as u64).to_le_bytes());
        for net in &self.net {
            bytes.extend(net.to_le_bytes());
        }
        for count in &self.count {
            bytes.extend(count.to_le_bytes());
        }
        let mut hasher = Sha256::new();
        hasher.update(b"multitude/tally/v1");
        hasher.update(&bytes);
        hasher.finalize().into()
    }

    /// Fraction of parameters at least one phone voted on.
    pub fn coverage(&self) -> f32 {
        if self.count.is_empty() {
            return 0.0;
        }
        self.count.iter().filter(|c| **c > 0).count() as f32 / self.count.len() as f32
    }

    /// The outer step: each parameter moves by `lr` times the mean of the
    /// votes cast on it.
    pub fn apply(&self, params: &mut [f32], lr: f32) {
        assert_eq!(
            params.len(),
            self.net.len(),
            "tally and weights differ in length"
        );
        for ((param, net), count) in params.iter_mut().zip(&self.net).zip(&self.count) {
            if *count > 0 {
                let mean = *net as f32 / *count as f32;
                *param -= lr * mean;
            }
        }
    }
}

/// What a round accepts. The relay, the watchers and the phones all derive
/// this from the run config and the global model.
#[derive(Clone, Debug)]
pub struct Expectation {
    pub run: [u8; 32],
    pub round: u32,
    pub base_checkpoint: [u8; 32],
    pub num_params: usize,
    pub compression: CompressionConfig,
    pub min_steps: u16,
    pub max_steps: u16,
    pub tokens_per_step: u64,
}

impl Expectation {
    pub fn new(cfg: &RunConfig, run: [u8; 32], round: u32, base_checkpoint: [u8; 32]) -> Self {
        Self {
            run,
            round,
            base_checkpoint,
            num_params: cfg.model.num_params(),
            compression: cfg.compression.clone(),
            min_steps: cfg.inner.min_steps,
            max_steps: cfg.inner.max_steps,
            tokens_per_step: cfg.tokens_per_step(),
        }
    }

    /// Everything about a contribution that can be checked without
    /// replaying it. The signature is checked by the caller.
    pub fn check(&self, contribution: &Contribution) -> Result<(), String> {
        let header = &contribution.header;
        if header.run != self.run {
            return Err("belongs to another run".into());
        }
        if header.round != self.round {
            return Err(format!("is for round {}, not {}", header.round, self.round));
        }
        if header.base_checkpoint != self.base_checkpoint {
            return Err("starts from a different checkpoint".into());
        }
        if header.inner_steps < self.min_steps || header.inner_steps > self.max_steps {
            return Err(format!(
                "claims {} inner steps, the run allows {}..={}",
                header.inner_steps, self.min_steps, self.max_steps
            ));
        }
        let tokens = header.inner_steps as u64 * self.tokens_per_step;
        if header.tokens != tokens {
            return Err(format!(
                "claims {} tokens, its steps make {}",
                header.tokens, tokens
            ));
        }
        contribution
            .update
            .validate(self.num_params, &self.compression)
            .map_err(|err| err.to_string())
    }
}

/// What the relay publishes besides the tally itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateClaim {
    pub round: u32,
    pub base_checkpoint: [u8; 32],
    /// Counted phones in ascending key order.
    pub contributors: Vec<[u8; 32]>,
    /// Merkle root over the counted contributions' digests, in the same order.
    pub contributions_root: [u8; 32],
    pub tally_hash: [u8; 32],
}

impl AggregateClaim {
    /// Inclusion proof for one counted phone, checkable against
    /// `contributions_root`.
    pub fn inclusion_proof(
        &self,
        digests: &[[u8; 32]],
        phone: &[u8; 32],
    ) -> Option<Vec<merkle::Step>> {
        let index = self.contributors.iter().position(|p| p == phone)?;
        merkle::proof(digests, index)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    pub phone: [u8; 32],
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct Aggregation {
    pub tally: Tally,
    pub claim: AggregateClaim,
    /// Digests of the counted contributions, in claim order.
    pub digests: Vec<[u8; 32]>,
    pub rejected: Vec<Rejection>,
    /// Phones that signed two different contributions for this round. Two
    /// valid signatures are their own proof; neither contribution is counted.
    pub equivocators: Vec<[u8; 32]>,
}

enum Selection<'a> {
    Counted(&'a Contribution),
    Rejected(String),
    Equivocated,
}

/// Decide what to do with one phone's contributions for the round.
fn select<'a>(expect: &Expectation, contributions: &[&'a Contribution]) -> Selection<'a> {
    let valid: Vec<&Contribution> = contributions
        .iter()
        .copied()
        .filter(|c| c.verify_signature().is_ok())
        .collect();
    let Some(first) = valid.first() else {
        return Selection::Rejected("no contribution with a valid signature".into());
    };
    if valid.iter().any(|c| c.digest() != first.digest()) {
        return Selection::Equivocated;
    }
    match expect.check(first) {
        Ok(()) => Selection::Counted(first),
        Err(reason) => Selection::Rejected(reason),
    }
}

fn group(contributions: &[Contribution]) -> BTreeMap<[u8; 32], Vec<&Contribution>> {
    let mut by_phone: BTreeMap<[u8; 32], Vec<&Contribution>> = BTreeMap::new();
    for contribution in contributions {
        by_phone
            .entry(contribution.header.phone)
            .or_default()
            .push(contribution);
    }
    by_phone
}

fn tally_of(expect: &Expectation, counted: &[&Contribution]) -> (Tally, Vec<[u8; 32]>) {
    let mut tally = Tally::new(expect.num_params);
    for contribution in counted {
        tally.add(&contribution.update);
    }
    (tally, counted.iter().map(|c| c.digest()).collect())
}

/// What an honest relay does: count every valid contribution once, in key
/// order, and publish the result.
pub fn aggregate(expect: &Expectation, contributions: &[Contribution]) -> Aggregation {
    let mut counted = Vec::new();
    let mut rejected = Vec::new();
    let mut equivocators = Vec::new();
    for (phone, list) in group(contributions) {
        match select(expect, &list) {
            Selection::Counted(contribution) => counted.push(contribution),
            Selection::Rejected(reason) => rejected.push(Rejection { phone, reason }),
            Selection::Equivocated => equivocators.push(phone),
        }
    }
    let (tally, digests) = tally_of(expect, &counted);
    let claim = AggregateClaim {
        round: expect.round,
        base_checkpoint: expect.base_checkpoint,
        contributors: counted.iter().map(|c| c.header.phone).collect(),
        contributions_root: merkle::root(&digests),
        tally_hash: tally.hash(),
    };
    Aggregation {
        tally,
        claim,
        digests,
        rejected,
        equivocators,
    }
}

/// A relay caught publishing something the contributions do not support.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AggregateFraud {
    #[error("claim is for round {claimed}, expected {expected}")]
    WrongRound { claimed: u32, expected: u32 },
    #[error("claim starts from a different checkpoint")]
    WrongBase,
    #[error("published tally does not hash to the claimed tally hash")]
    PublishedTallyMismatch,
    #[error("contributors are not strictly ascending")]
    NonCanonicalContributors,
    #[error("claimed contributor {0} has no contribution anyone can see")]
    MissingContribution(String),
    #[error("contributor {phone} should not have been counted: {reason}")]
    InvalidContributionCounted { phone: String, reason: String },
    #[error("contribution root does not match the counted contributions")]
    RootMismatch,
    #[error("tally hash {claimed} does not match the recomputed {recomputed}")]
    TallyMismatch { claimed: String, recomputed: String },
}

/// What a watcher does: recompute the relay's claim from the contributions it
/// can see.
///
/// Returns the phones whose valid contributions the relay left out. Omission
/// is not proof of fraud (a contribution can arrive late), so it is reported
/// rather than convicted; the phone can resubmit or complain.
pub fn verify_aggregate(
    expect: &Expectation,
    pool: &[Contribution],
    claim: &AggregateClaim,
    published: &Tally,
) -> Result<Vec<[u8; 32]>, AggregateFraud> {
    if claim.round != expect.round {
        return Err(AggregateFraud::WrongRound {
            claimed: claim.round,
            expected: expect.round,
        });
    }
    if claim.base_checkpoint != expect.base_checkpoint {
        return Err(AggregateFraud::WrongBase);
    }
    if published.len() != expect.num_params || published.hash() != claim.tally_hash {
        return Err(AggregateFraud::PublishedTallyMismatch);
    }
    if claim.contributors.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(AggregateFraud::NonCanonicalContributors);
    }

    let by_phone = group(pool);
    let mut counted = Vec::with_capacity(claim.contributors.len());
    for phone in &claim.contributors {
        let Some(list) = by_phone.get(phone) else {
            return Err(AggregateFraud::MissingContribution(short_address(phone)));
        };
        match select(expect, list) {
            Selection::Counted(contribution) => counted.push(contribution),
            Selection::Rejected(reason) => {
                return Err(AggregateFraud::InvalidContributionCounted {
                    phone: short_address(phone),
                    reason,
                })
            }
            Selection::Equivocated => {
                return Err(AggregateFraud::InvalidContributionCounted {
                    phone: short_address(phone),
                    reason: "equivocated".into(),
                })
            }
        }
    }

    let (tally, digests) = tally_of(expect, &counted);
    if merkle::root(&digests) != claim.contributions_root {
        return Err(AggregateFraud::RootMismatch);
    }
    let recomputed = tally.hash();
    if recomputed != claim.tally_hash {
        return Err(AggregateFraud::TallyMismatch {
            claimed: hex(&claim.tally_hash[..8]),
            recomputed: hex(&recomputed[..8]),
        });
    }

    let omitted = by_phone
        .iter()
        .filter(|(phone, _)| claim.contributors.binary_search(phone).is_err())
        .filter(|(_, list)| matches!(select(expect, list), Selection::Counted(_)))
        .map(|(phone, _)| *phone)
        .collect();
    Ok(omitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::compress;
    use crate::protocol::{signing_key, ContributionHeader};

    fn setup(phones: u8) -> (Expectation, Vec<Contribution>) {
        let cfg = RunConfig {
            model: crate::config::ModelConfig {
                d_model: 8,
                n_heads: 2,
                n_layers: 1,
                context: 4,
                vocab_size: 16,
                mlp_ratio: 2,
            },
            ..RunConfig::default()
        };
        let expect = Expectation::new(&cfg, [9u8; 32], 2, [8u8; 32]);
        let n = expect.num_params;
        let contributions = (1..=phones)
            .map(|p| {
                let delta: Vec<f32> = (0..n)
                    .map(|i| ((i * 7 + p as usize * 13) % 17) as f32 - 8.0)
                    .collect();
                let (update, _) = compress(&delta, &vec![0.0; n], &cfg.compression).unwrap();
                let header = ContributionHeader {
                    run: expect.run,
                    round: expect.round,
                    phone: [0u8; 32],
                    base_checkpoint: expect.base_checkpoint,
                    inner_steps: 4,
                    tokens: 4 * expect.tokens_per_step,
                    mean_loss: 1.0,
                    error_commitment: [0u8; 32],
                    update_hash: [0u8; 32],
                };
                Contribution::sign(header, update, &signing_key(&[p; 32]))
            })
            .collect();
        (expect, contributions)
    }

    #[test]
    fn honest_aggregation_verifies() {
        let (expect, contributions) = setup(4);
        let aggregation = aggregate(&expect, &contributions);
        assert_eq!(aggregation.claim.contributors.len(), 4);
        let total: u32 = aggregation.tally.count().iter().sum();
        assert_eq!(total as usize, 4 * contributions[0].update.selected());
        let omitted = verify_aggregate(
            &expect,
            &contributions,
            &aggregation.claim,
            &aggregation.tally,
        )
        .unwrap();
        assert!(omitted.is_empty());
    }

    #[test]
    fn order_of_arrival_does_not_matter() {
        let (expect, mut contributions) = setup(4);
        let a = aggregate(&expect, &contributions);
        contributions.reverse();
        let b = aggregate(&expect, &contributions);
        assert_eq!(a.claim, b.claim);
    }

    #[test]
    fn a_miscounting_relay_is_caught() {
        let (expect, contributions) = setup(3);
        let mut aggregation = aggregate(&expect, &contributions);
        aggregation.tally.net_mut()[0] += 1;
        aggregation.claim.tally_hash = aggregation.tally.hash();
        let fraud = verify_aggregate(
            &expect,
            &contributions,
            &aggregation.claim,
            &aggregation.tally,
        )
        .unwrap_err();
        assert!(matches!(fraud, AggregateFraud::TallyMismatch { .. }));
    }

    #[test]
    fn a_relay_serving_a_different_tally_is_caught() {
        let (expect, contributions) = setup(3);
        let mut aggregation = aggregate(&expect, &contributions);
        aggregation.tally.net_mut()[1] -= 1;
        let fraud = verify_aggregate(
            &expect,
            &contributions,
            &aggregation.claim,
            &aggregation.tally,
        )
        .unwrap_err();
        assert_eq!(fraud, AggregateFraud::PublishedTallyMismatch);
    }

    #[test]
    fn omission_is_reported_not_convicted() {
        let (expect, contributions) = setup(3);
        let partial = aggregate(&expect, &contributions[..2]);
        let omitted =
            verify_aggregate(&expect, &contributions, &partial.claim, &partial.tally).unwrap();
        assert_eq!(omitted, vec![contributions[2].header.phone]);
    }

    #[test]
    fn equivocation_is_detected_and_not_counted() {
        let (expect, mut contributions) = setup(2);
        let key = signing_key(&[1u8; 32]);
        let mut second = contributions[0].clone();
        second.update.flip_signs();
        let second = Contribution::sign(second.header, second.update, &key);
        contributions.push(second);
        let aggregation = aggregate(&expect, &contributions);
        assert_eq!(
            aggregation.equivocators,
            vec![key.verifying_key().to_bytes()]
        );
        assert_eq!(aggregation.claim.contributors.len(), 1);
    }

    #[test]
    fn invalid_contributions_are_rejected() {
        let (expect, mut contributions) = setup(2);
        contributions[1].header.tokens += 1;
        let aggregation = aggregate(&expect, &contributions);
        assert_eq!(aggregation.claim.contributors.len(), 1);
        assert_eq!(aggregation.rejected.len(), 1);
    }

    #[test]
    fn every_counted_phone_can_prove_inclusion() {
        let (expect, contributions) = setup(5);
        let aggregation = aggregate(&expect, &contributions);
        for (phone, digest) in aggregation
            .claim
            .contributors
            .iter()
            .zip(&aggregation.digests)
        {
            let steps = aggregation
                .claim
                .inclusion_proof(&aggregation.digests, phone)
                .unwrap();
            assert!(merkle::verify(
                digest,
                &steps,
                &aggregation.claim.contributions_root
            ));
        }
    }

    #[test]
    fn apply_moves_by_the_mean_vote() {
        let mut tally = Tally::new(3);
        tally.net = vec![2, -1, 0];
        tally.count = vec![2, 3, 0];
        let mut params = vec![1.0, 1.0, 1.0];
        tally.apply(&mut params, 0.5);
        assert_eq!(params, vec![0.5, 1.0 + 0.5 / 3.0, 1.0]);
    }
}
