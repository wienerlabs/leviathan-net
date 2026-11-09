//! Replay audits.
//!
//! An audited phone reveals its error-feedback buffer; the verifier checks it
//! against the commitment in the signed header, replays the round from the
//! same checkpoint and the same seeded windows, compresses with the revealed
//! buffer, and compares the two votes with Leviathan's own
//! [`psyche_verifier::audit_contribution`]: relative L2 distance against a
//! tolerance band, with a sha256 fraud proof when it is exceeded.
//!
//! Votes are -1, 0 or +1, so the distance counts disagreements. Honest phones
//! on different hardware disagree only where two coefficients nearly tie for a
//! top-k slot. A phone that flipped its signs sits at distance 2, one that
//! voted at random near 1.3, one that trained fewer steps than it claimed well
//! above any calibrated band.

use candle_core::Device;
use psyche_verifier::FraudProof;
pub use psyche_verifier::{calibrate_band, DEFAULT_BAND, DEFAULT_SAFETY_FACTOR};

use crate::compress::{compress, CompressedUpdate};
use crate::config::RunConfig;
use crate::data::{data_seed, Corpus};
use crate::protocol::{weights_hash, Contribution, ErrorReveal};
use crate::train::{pseudo_gradient, train_local};
use crate::Result;

/// The public state a replay needs.
pub struct AuditContext<'a> {
    pub cfg: &'a RunConfig,
    pub run: [u8; 32],
    pub round: u32,
    pub round_seed: [u8; 32],
    pub global: &'a [f32],
    pub base_checkpoint: [u8; 32],
    pub corpus: &'a Corpus,
    pub band: f32,
    pub device: &'a Device,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FraudReason {
    /// The replay disagrees with the submitted vote by more than the band.
    BeyondBand {
        distance: f32,
        band: f32,
    },
    /// The phone could not produce the error buffer it committed to.
    ErrorCommitmentMismatch,
    /// The phone did not answer the audit.
    NoReveal,
    TokenClaimMismatch {
        claimed: u64,
        expected: u64,
    },
    StepsOutOfRange {
        steps: u16,
    },
    WrongRound,
}

impl std::fmt::Display for FraudReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeyondBand { distance, band } => {
                write!(f, "distance {distance:.4} against band {band:.4}")
            }
            Self::ErrorCommitmentMismatch => {
                write!(f, "revealed buffer does not match its commitment")
            }
            Self::NoReveal => write!(f, "did not answer the audit"),
            Self::TokenClaimMismatch { claimed, expected } => {
                write!(f, "claimed {claimed} tokens, its steps make {expected}")
            }
            Self::StepsOutOfRange { steps } => write!(f, "claimed {steps} inner steps"),
            Self::WrongRound => write!(f, "signed for another round or checkpoint"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Honest {
        distance: f32,
        band: f32,
    },
    Fraud {
        reason: FraudReason,
        proof: Option<FraudProof>,
    },
}

impl Verdict {
    pub fn is_fraud(&self) -> bool {
        matches!(self, Self::Fraud { .. })
    }
}

fn fraud(reason: FraudReason) -> Verdict {
    Verdict::Fraud {
        reason,
        proof: None,
    }
}

/// Recompute what an honest phone would have sent for this contribution.
pub fn replay(
    ctx: &AuditContext,
    contribution: &Contribution,
    buffer: &[f32],
) -> Result<CompressedUpdate> {
    let seed = data_seed(&ctx.round_seed, &contribution.header.phone);
    let local = train_local(
        ctx.cfg,
        ctx.global,
        ctx.corpus,
        &seed,
        contribution.header.inner_steps,
        ctx.device,
    )?;
    let delta = pseudo_gradient(ctx.global, &local.params);
    let (update, _) = compress(&delta, buffer, &ctx.cfg.compression)?;
    Ok(update)
}

/// Audit one contribution. `reveal` is `None` when the phone did not answer.
pub fn audit_contribution(
    ctx: &AuditContext,
    contribution: &Contribution,
    reveal: Option<&ErrorReveal>,
    target_index: u64,
) -> Result<Verdict> {
    contribution.verify_signature()?;
    let header = &contribution.header;
    if header.round != ctx.round
        || header.base_checkpoint != ctx.base_checkpoint
        || header.run != ctx.run
    {
        return Ok(fraud(FraudReason::WrongRound));
    }
    if header.inner_steps < ctx.cfg.inner.min_steps || header.inner_steps > ctx.cfg.inner.max_steps
    {
        return Ok(fraud(FraudReason::StepsOutOfRange {
            steps: header.inner_steps,
        }));
    }
    let expected = header.inner_steps as u64 * ctx.cfg.tokens_per_step();
    if header.tokens != expected {
        return Ok(fraud(FraudReason::TokenClaimMismatch {
            claimed: header.tokens,
            expected,
        }));
    }
    let Some(reveal) = reveal else {
        return Ok(fraud(FraudReason::NoReveal));
    };
    if reveal.phone != header.phone
        || reveal.round != header.round
        || reveal.buffer.len() != ctx.global.len()
        || weights_hash(&reveal.buffer) != header.error_commitment
    {
        return Ok(fraud(FraudReason::ErrorCommitmentMismatch));
    }

    let replayed = replay(ctx, contribution, &reveal.buffer)?;
    let report = psyche_verifier::audit_contribution(
        target_index,
        &contribution.update.to_dense(),
        &replayed.to_dense(),
        ctx.band,
    )?;
    Ok(match report.proof {
        Some(proof) => Verdict::Fraud {
            reason: FraudReason::BeyondBand {
                distance: report.verdict.distance,
                band: report.verdict.band,
            },
            proof: Some(proof),
        },
        None => Verdict::Honest {
            distance: report.verdict.distance,
            band: ctx.band,
        },
    })
}

/// Distance between two votes, for calibrating the band on the replay path
/// itself, as Leviathan's `calibrate-band` does per hardware class.
pub fn vote_distance(a: &CompressedUpdate, b: &CompressedUpdate) -> Result<f32> {
    Ok(psyche_verifier::relative_l2_distance(
        &a.to_dense(),
        &b.to_dense(),
    )?)
}
