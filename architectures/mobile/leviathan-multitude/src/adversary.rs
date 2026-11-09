//! Ways a phone or a relay can misbehave, for tests and the simulator.
//!
//! The production app has no reason to build anything but
//! [`Behavior::Honest`]. Leviathan's own client carries the same kind of switch
//! (`LEVIATHAN_FAKE_DELTA`) to prove its verifier catches a forged gradient.

use crate::aggregate::Tally;
use crate::rng::{sha256_parts, SplitMix64};

#[derive(Clone, Debug, PartialEq)]
pub enum Behavior {
    Honest,
    /// Honest, on hardware whose arithmetic differs slightly from the
    /// verifier's: each pseudo-gradient value is scaled by `1 + scale * u`,
    /// `u` uniform in `[-1, 1]`. This is the drift the band must absorb.
    Jitter {
        scale: f32,
    },
    /// Trains honestly and sends the opposite of every sign: a poisoning
    /// attack that pushes the model uphill.
    SignFlip,
    /// Skips training and votes at random: a free rider collecting rewards.
    RandomVotes,
    /// Trains `real_steps` but claims the full step count.
    Lazy {
        real_steps: u16,
    },
    /// Trains honestly but refuses to answer an audit.
    WithholdReveal,
}

impl Behavior {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Honest => "honest",
            Self::Jitter { .. } => "honest (hardware drift)",
            Self::SignFlip => "sign-flip",
            Self::RandomVotes => "random votes",
            Self::Lazy { .. } => "lazy",
            Self::WithholdReveal => "withholds audit",
        }
    }

    pub fn is_honest(&self) -> bool {
        matches!(self, Self::Honest | Self::Jitter { .. })
    }
}

/// Apply hardware drift to a pseudo-gradient, seeded per phone and round.
pub fn jitter(delta: &mut [f32], scale: f32, phone: &[u8; 32], round: u32) {
    let mut rng = SplitMix64::from_digest(&sha256_parts(&[
        b"multitude/jitter",
        phone,
        &round.to_le_bytes(),
    ]));
    for value in delta.iter_mut() {
        let u = rng.next_f32() * 2.0 - 1.0;
        *value *= 1.0 + scale * u;
    }
}

/// A relay that quietly reverses the net vote on a fraction of parameters.
pub fn tamper_tally(tally: &mut Tally, fraction: f32, seed: u64) -> usize {
    let mut rng = SplitMix64::new(seed);
    let mut touched = 0;
    for net in tally.net_mut() {
        if *net != 0 && rng.next_f32() < fraction {
            *net = -*net;
            touched += 1;
        }
    }
    touched
}
