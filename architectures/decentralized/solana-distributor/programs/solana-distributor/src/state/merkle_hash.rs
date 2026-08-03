use std::fmt;

use anchor_lang::prelude::*;
use anchor_lang::solana_program::hash::hashv;

#[derive(
    InitSpace, AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Default,
)]
pub struct MerkleHash {
    bytes: [u8; 32],
}

impl MerkleHash {
    pub fn from_parts(parts: &[&[u8]]) -> MerkleHash {
        MerkleHash {
            bytes: hashv(parts).to_bytes(),
        }
    }

    pub fn from_pair(a: &MerkleHash, b: &MerkleHash) -> MerkleHash {
        MerkleHash {
            bytes: if a.bytes <= b.bytes {
                hashv(&[&a.bytes, &b.bytes]).to_bytes()
            } else {
                hashv(&[&b.bytes, &a.bytes]).to_bytes()
            },
        }
    }

    pub fn is_valid_proof(
        &self,
        merkle_leaf: &MerkleHash,
        merkle_proof: &[MerkleHash],
    ) -> bool {
        let mut merkle_hash = merkle_leaf.clone();
        for merkle_node in merkle_proof {
            merkle_hash = MerkleHash::from_pair(&merkle_hash, merkle_node);
        }
        merkle_hash == *self
    }
}

impl std::fmt::Debug for MerkleHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        let parts = self.bytes.iter().map(|b| format!("{:02X}", b));
        write!(f, "{}", parts.collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Allocation;
    use crate::state::Vesting;

    /// An internal node is nothing but `from_parts` over two 32-byte halves, so
    /// a leaf and a node come out of the same hash with no tag telling them
    /// apart. Recorded during the internal review (wienerlabs/leviathan#15).
    #[test]
    fn a_node_is_from_parts_over_sixty_four_bytes() {
        let a = MerkleHash { bytes: [0x11; 32] };
        let b = MerkleHash { bytes: [0x22; 32] };
        let node = MerkleHash::from_pair(&a, &b);
        // `from_pair` sorts, and 0x11.. < 0x22.., so the preimage is a then b.
        let same = MerkleHash::from_parts(&[&a.bytes, &b.bytes]);
        assert_eq!(
            node, same,
            "a node hash is reachable from the leaf constructor given 64 bytes"
        );
    }

    /// What keeps that from being exploitable today is only the width of an
    /// allocation: 60 bytes of preimage against a node's 64. Nothing states the
    /// dependency and nothing enforces it, so four more bytes of allocation
    /// would let a claimer present an internal node as their own leaf.
    #[test]
    fn an_allocation_preimage_is_four_bytes_short_of_a_node() {
        let allocation = Allocation {
            claimer: Pubkey::new_unique(),
            nonce: 7,
            vesting: Vesting {
                start_unix_timestamp: 1,
                duration_seconds: 2,
                end_collateral_amount: 3,
            },
        };
        // The parts `to_merkle_hash` feeds in, in order.
        let preimage_len = allocation.claimer.as_ref().len()
            + allocation.nonce.to_le_bytes().len()
            + allocation.vesting.start_unix_timestamp.to_le_bytes().len()
            + allocation.vesting.duration_seconds.to_le_bytes().len()
            + allocation.vesting.end_collateral_amount.to_le_bytes().len();
        assert_eq!(preimage_len, 60);
        assert_ne!(
            preimage_len, 64,
            "a node preimage is 64 bytes; that difference is the whole separation"
        );
    }

    /// An empty proof is accepted and degenerates to `leaf == root`. That is
    /// safe only because reaching the root needs a preimage attack; it is worth
    /// knowing that the proof length itself is never checked.
    #[test]
    fn an_empty_proof_compares_the_leaf_against_the_root() {
        let leaf = MerkleHash { bytes: [0xAB; 32] };
        assert!(leaf.is_valid_proof(&leaf, &[]));
        let other = MerkleHash { bytes: [0xCD; 32] };
        assert!(!other.is_valid_proof(&leaf, &[]));
    }
}
