//! A SHA-256 Merkle tree over contribution digests.
//!
//! Leaves and inner nodes carry different one-byte prefixes, so a node can
//! never be passed off as a leaf (the gap the internal review found in the
//! main network's tree, finding 22). An odd node at the end of a level is
//! promoted unchanged rather than paired with a copy of itself.
//!
//! The root lets a phone check, from the relay's claim alone, that its own
//! contribution was counted.

use sha2::{Digest, Sha256};

const LEAF: u8 = 0x00;
const NODE: u8 = 0x01;

fn leaf_hash(data: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([LEAF]);
    hasher.update(data);
    hasher.finalize().into()
}

fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([NODE]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn next_level(level: &[[u8; 32]]) -> Vec<[u8; 32]> {
    level
        .chunks(2)
        .map(|pair| match pair {
            [left, right] => node_hash(left, right),
            [single] => *single,
            _ => unreachable!("chunks(2) yields one or two items"),
        })
        .collect()
}

/// Root over `leaves` in the order given. The empty tree has a fixed root.
pub fn root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return Sha256::digest(b"multitude/merkle/empty").into();
    }
    let mut level: Vec<[u8; 32]> = leaves.iter().map(leaf_hash).collect();
    while level.len() > 1 {
        level = next_level(&level);
    }
    level[0]
}

/// One step of an inclusion proof: the sibling and which side it sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Step {
    Left([u8; 32]),
    Right([u8; 32]),
}

/// Inclusion proof for `leaves[index]`.
pub fn proof(leaves: &[[u8; 32]], mut index: usize) -> Option<Vec<Step>> {
    if index >= leaves.len() {
        return None;
    }
    let mut level: Vec<[u8; 32]> = leaves.iter().map(leaf_hash).collect();
    let mut steps = Vec::new();
    while level.len() > 1 {
        if index % 2 == 1 {
            steps.push(Step::Left(level[index - 1]));
        } else if index + 1 < level.len() {
            steps.push(Step::Right(level[index + 1]));
        }
        level = next_level(&level);
        index /= 2;
    }
    Some(steps)
}

pub fn verify(leaf: &[u8; 32], steps: &[Step], expected_root: &[u8; 32]) -> bool {
    let mut hash = leaf_hash(leaf);
    for step in steps {
        hash = match step {
            Step::Left(sibling) => node_hash(sibling, &hash),
            Step::Right(sibling) => node_hash(&hash, sibling),
        };
    }
    &hash == expected_root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<[u8; 32]> {
        (0..n).map(|i| [i as u8; 32]).collect()
    }

    #[test]
    fn every_leaf_proves_inclusion() {
        for n in 1..=9 {
            let leaves = leaves(n);
            let root = root(&leaves);
            for (i, leaf) in leaves.iter().enumerate() {
                let steps = proof(&leaves, i).unwrap();
                assert!(verify(leaf, &steps, &root), "leaf {i} of {n}");
            }
        }
    }

    #[test]
    fn a_node_is_not_a_leaf() {
        let leaves = leaves(2);
        let root = root(&leaves);
        let node = node_hash(&leaf_hash(&leaves[0]), &leaf_hash(&leaves[1]));
        // Presenting the inner node as a one-leaf tree must not reproduce the root.
        assert!(!verify(&node, &[], &root));
    }

    #[test]
    fn wrong_leaf_fails() {
        let leaves = leaves(5);
        let root = root(&leaves);
        let steps = proof(&leaves, 2).unwrap();
        assert!(!verify(&[99u8; 32], &steps, &root));
    }
}
