//! Leviathan Multitude: distributed LLM training for phones.
//!
//! Hobbes drew Leviathan as one giant made of a multitude of people. The main
//! network makes that giant out of bonded GPUs; Multitude makes it out of
//! phones that train while they charge.
//!
//! A phone cannot join the 1B flagship run as a trainer: the run holds the full
//! model, gradients and the DisTrO buffer on every node, its round deadline is
//! set by the slowest card, and its tolerance band is calibrated per GPU class.
//! So Multitude is a phone-class run with its own shape:
//!
//! - **Local rounds (DiLoCo inner loop).** Each phone trains a small
//!   transformer for a few AdamW steps on windows chosen by a public seed
//!   ([`train`]), then sends the difference to the global weights.
//! - **SparseLoCo transport.** The pseudo-gradient plus an error-feedback
//!   buffer is cut into chunks of 64; each chunk sends its top 8 coefficients
//!   as one sign bit each ([`compress`]). About 1.1 bits per parameter.
//! - **Verifiable aggregation.** A relay sums the sign votes as integers
//!   ([`aggregate`]). Anyone holding the same contributions recomputes the tally
//!   bit for bit, so a relay that miscounts is caught without a tolerance band.
//! - **Replay audits.** A seeded lottery ([`lottery`]) makes phones replay each
//!   other's rounds, and the comparison is Leviathan's own
//!   [`psyche_verifier::audit_contribution`] ([`audit`]).
//! - **Phone etiquette.** Training only runs on a charger, on an unmetered
//!   network, with the device idle and cool ([`device`]).
//!
//! Every phone signs its contribution with an ed25519 key, the same key type
//! as a Solana address, so a phone's identity can be its wallet.

pub mod adversary;
pub mod aggregate;
pub mod audit;
pub mod compress;
pub mod config;
pub mod data;
pub mod device;
pub mod economics;
pub mod lottery;
pub mod merkle;
pub mod model;
pub mod phone;
pub mod protocol;
pub mod rng;
pub mod round;
pub mod train;

pub use aggregate::{
    aggregate, verify_aggregate, AggregateClaim, AggregateFraud, Aggregation, Tally,
};
pub use audit::{audit_contribution, AuditContext, FraudReason, Verdict};
pub use candle_core::Device;
pub use compress::{compress, CompressedUpdate};
pub use config::RunConfig;
pub use data::Corpus;
pub use phone::{PhoneNode, RoundJob};
pub use protocol::{Contribution, ContributionHeader, ErrorReveal};
pub use round::GlobalModel;

/// Everything that can go wrong in Multitude.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tensor backend: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("compression: {0}")]
    Compress(#[from] compress::CompressError),
    #[error("protocol: {0}")]
    Protocol(#[from] protocol::ProtocolError),
    #[error("verifier: {0}")]
    Verifier(#[from] psyche_verifier::VerifierError),
    #[error("loss became non-finite at inner step {step}")]
    NonFiniteLoss { step: u16 },
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;
