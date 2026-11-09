//! Deterministic randomness.
//!
//! Every random choice a verifier must reproduce (which windows a phone trains
//! on, which phone audits which) is integer arithmetic on a SHA-256 digest, so
//! it comes out bit-identical on an iPhone, an Android phone and the x86 box
//! that replays them. Floating-point randomness appears only in the genesis
//! weights, which are published with their hash rather than recomputed.

use sha2::{Digest, Sha256};

/// SHA-256 over length-prefixed parts.
///
/// Each part carries its length, so `("ab", "c")` and `("a", "bc")` hash
/// differently. The internal security review found Leviathan's Merkle tree
/// telling leaves from nodes only by preimage length (finding 22); prefixing
/// every part keeps that ambiguity out of this crate entirely.
pub fn sha256_parts(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// The first eight bytes of a digest, little-endian.
pub fn digest_u64(digest: &[u8; 32]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

/// SplitMix64: tiny, fast and the same on every platform.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn from_digest(digest: &[u8; 32]) -> Self {
        Self::new(digest_u64(digest))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` by multiply-shift. The bias is below 2^-32 for every
    /// range this crate draws from.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "cannot draw from an empty range");
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform in `[0, 1)` with 24 bits of precision.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Standard normal by Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
        let u2 = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_are_length_prefixed() {
        assert_ne!(sha256_parts(&[b"ab", b"c"]), sha256_parts(&[b"a", b"bc"]));
    }

    #[test]
    fn splitmix_is_reproducible_and_in_range() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..1000 {
            let x = a.below(13);
            assert_eq!(x, b.below(13));
            assert!(x < 13);
        }
        let f = a.next_f32();
        assert!((0.0..1.0).contains(&f));
    }
}
