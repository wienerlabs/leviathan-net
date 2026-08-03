use std::fmt::{Debug, Display};

use anchor_lang::{Space, prelude::*};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(
    Clone,
    Copy,
    Default,
    Zeroable,
    Pod,
    AnchorSerialize,
    AnchorDeserialize,
    Serialize,
    Deserialize,
    TS,
    Eq,
)]
#[repr(C)]
pub struct NodeIdentity {
    signer: [u8; 32],
    p2p_identity: [u8; 32],
}

impl PartialEq for NodeIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.signer == other.signer
    }
}

impl std::hash::Hash for NodeIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.signer.hash(state);
    }
}
impl NodeIdentity {
    pub fn new(signer: [u8; 32], p2p_identity: [u8; 32]) -> Self {
        Self {
            signer,
            p2p_identity,
        }
    }

    /// In non-Solana usage, we don't have a signer - so
    /// both signer and p2p_identity are the same pubkey.
    pub fn from_single_key(key: [u8; 32]) -> Self {
        Self {
            signer: key,
            p2p_identity: key,
        }
    }

    pub fn signer(&self) -> &[u8; 32] {
        &self.signer
    }

    pub fn p2p_identity(&self) -> &[u8; 32] {
        &self.p2p_identity
    }
}

/// The leading characters of the signer's address, in base58.
///
/// Short enough to keep a log line readable, and base58 rather than hex because
/// that is what the operator's wallet shows them: an operator scanning for their
/// own node matches these characters against the address they already know. The
/// hex of the same bytes matches nothing they have.
fn signer_prefix(signer: &[u8; 32]) -> String {
    let address = Pubkey::from(*signer).to_string();
    address.chars().take(8).collect()
}

impl Display for NodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", signer_prefix(&self.signer))
    }
}

impl Debug for NodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The signer is a Solana address and the p2p identity is an iroh key, so
        // each is printed the way its own tooling prints it: base58 and hex.
        write!(f, "NodeIdentity({}/", signer_prefix(&self.signer))?;
        for b in &self.p2p_identity[..4] {
            write!(f, "{:02x}", b)?;
        }
        write!(f, ")")
    }
}

impl Space for NodeIdentity {
    const INIT_SPACE: usize = 64;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real devnet participant: address 8tcQfLmW1ucG5ohoDf3fci8vRz3Kspviu5bKsTm628Un.
    const SIGNER: [u8; 32] = [
        0x75, 0x3a, 0x77, 0x34, 0x61, 0x0d, 0x5e, 0xbf, 0x81, 0xb6, 0x4d, 0xc2, 0xc9, 0xaa, 0xd5,
        0x5b, 0xb2, 0xaa, 0xe3, 0x6c, 0x68, 0x45, 0x7a, 0x82, 0x82, 0xe2, 0x41, 0x30, 0x3c, 0x5f,
        0xca, 0xbf,
    ];

    #[test]
    fn display_matches_what_the_wallet_shows() {
        let id = NodeIdentity::new(SIGNER, [0xce; 32]);
        // The operator's wallet shows 8tcQfLmW1ucG…, so a log line saying
        // 8tcQfLmW is one they recognise. 753a7734 is the same bytes in an
        // encoding that appears nowhere else they look.
        assert_eq!(id.to_string(), "8tcQfLmW");
        assert!(Pubkey::from(SIGNER).to_string().starts_with(&id.to_string()));
    }

    #[test]
    fn debug_keeps_the_p2p_key_in_its_own_encoding() {
        let id = NodeIdentity::new(SIGNER, [0xce; 32]);
        // iroh prints endpoint ids as hex, so the p2p half stays hex.
        assert_eq!(format!("{id:?}"), "NodeIdentity(8tcQfLmW/cececece)");
    }
}
