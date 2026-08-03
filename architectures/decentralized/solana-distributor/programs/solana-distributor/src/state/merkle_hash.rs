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
    /// Tags a leaf preimage, so a leaf can never be mistaken for a node.
    ///
    /// Without it the two are drawn from the same hash and the only thing
    /// separating them is that an allocation happens to be 60 bytes wide while a
    /// node is 64. Four more bytes of allocation and a claimer could present an
    /// internal node as its own leaf and drain the airdrop
    /// (wienerlabs/leviathan#15, finding 22).
    const LEAF_DOMAIN: [u8; 1] = [0x00];
    /// Tags an internal node preimage.
    const NODE_DOMAIN: [u8; 1] = [0x01];

    pub fn from_parts(parts: &[&[u8]]) -> MerkleHash {
        let mut tagged: Vec<&[u8]> = Vec::with_capacity(parts.len() + 1);
        tagged.push(&Self::LEAF_DOMAIN);
        tagged.extend_from_slice(parts);
        MerkleHash {
            bytes: hashv(&tagged).to_bytes(),
        }
    }

    pub fn from_pair(a: &MerkleHash, b: &MerkleHash) -> MerkleHash {
        let (low, high) = if a.bytes <= b.bytes { (a, b) } else { (b, a) };
        MerkleHash {
            bytes: hashv(&[&Self::NODE_DOMAIN, &low.bytes, &high.bytes]).to_bytes(),
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

    /// A leaf and a node are now different hashes of the same bytes, so the
    /// separation no longer rests on an allocation happening to be 60 bytes wide
    /// while a node is 64 (wienerlabs/leviathan#15, finding 22).
    #[test]
    fn a_leaf_and_a_node_over_the_same_bytes_differ() {
        let a = MerkleHash { bytes: [0x11; 32] };
        let b = MerkleHash { bytes: [0x22; 32] };
        let node = MerkleHash::from_pair(&a, &b);
        // `from_pair` sorts, so this is the same 64 bytes in the same order.
        let leaf = MerkleHash::from_parts(&[&a.bytes, &b.bytes]);
        assert_ne!(
            node, leaf,
            "the domain tag keeps the leaf constructor away from node hashes"
        );
    }

    /// So an allocation that is exactly as wide as a node - which four more
    /// bytes of fields would make it - still cannot collide with one.
    #[test]
    fn a_leaf_the_width_of_a_node_still_cannot_be_one() {
        let a = MerkleHash { bytes: [0x33; 32] };
        let b = MerkleHash { bytes: [0x44; 32] };
        let sixty_four_byte_leaf = MerkleHash::from_parts(&[&a.bytes, &b.bytes]);
        assert_ne!(sixty_four_byte_leaf, MerkleHash::from_pair(&a, &b));
    }

    /// The tree still works: a real allocation proves against a root built the
    /// way the airdrop builds it.
    #[test]
    fn an_honest_allocation_still_proves() {
        let allocation = Allocation {
            claimer: Pubkey::new_unique(),
            nonce: 7,
            vesting: Vesting {
                start_unix_timestamp: 1,
                duration_seconds: 2,
                end_collateral_amount: 3,
            },
        };
        let leaf = allocation.to_merkle_hash();
        let sibling = MerkleHash { bytes: [0x55; 32] };
        let root = MerkleHash::from_pair(&leaf, &sibling);
        assert!(root.is_valid_proof(&leaf, &[sibling.clone()]));
        assert!(!root.is_valid_proof(&sibling, &[sibling.clone()]));
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
