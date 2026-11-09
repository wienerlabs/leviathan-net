//! What a phone sends, and how anyone checks it came from that phone.
//!
//! A contribution is a header plus a compressed update, signed with the
//! phone's ed25519 key. The key bytes are the same 32 bytes as a Solana
//! address, so a phone can sign with its wallet and its identity on the
//! network is that address.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::compress::CompressedUpdate;
use crate::rng::sha256_parts;

pub const CONTRIBUTION_DOMAIN: &[u8] = b"leviathan-multitude/contribution/v1";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    #[error("signature must be 64 bytes, got {0}")]
    SignatureLength(usize),
    #[error("phone key is not a valid ed25519 public key")]
    PublicKey,
    #[error("signature does not verify")]
    Signature,
    #[error("update does not match the hash in the signed header")]
    UpdateHash,
    #[error("could not decode: {0}")]
    Decode(String),
}

/// The signed part of a contribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContributionHeader {
    /// [`crate::RunConfig::fingerprint`]: config and corpus.
    pub run: [u8; 32],
    pub round: u32,
    /// ed25519 public key; the same bytes as the phone's Solana address.
    pub phone: [u8; 32],
    /// Hash of the global weights the round started from.
    pub base_checkpoint: [u8; 32],
    pub inner_steps: u16,
    /// Tokens trained, which must equal `inner_steps * micro_batch * context`.
    pub tokens: u64,
    pub mean_loss: f32,
    /// Hash of the error-feedback buffer at the start of the round. An audit
    /// asks the phone to reveal the buffer and checks it against this, which
    /// makes the replay a pure function of public inputs plus that buffer.
    pub error_commitment: [u8; 32],
    pub update_hash: [u8; 32],
}

impl ContributionHeader {
    fn signing_bytes(&self) -> Vec<u8> {
        let mut message = CONTRIBUTION_DOMAIN.to_vec();
        message.extend(postcard::to_stdvec(self).expect("a header always serializes"));
        message
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contribution {
    pub header: ContributionHeader,
    pub update: CompressedUpdate,
    pub signature: Vec<u8>,
}

impl Contribution {
    /// Sign `update` under `header`. The phone key and the update hash in the
    /// header are filled in here, so they cannot disagree with what is signed.
    pub fn sign(
        mut header: ContributionHeader,
        update: CompressedUpdate,
        key: &SigningKey,
    ) -> Self {
        header.phone = key.verifying_key().to_bytes();
        header.update_hash = update.hash();
        let signature = key.sign(&header.signing_bytes()).to_bytes().to_vec();
        Self {
            header,
            update,
            signature,
        }
    }

    pub fn verify_signature(&self) -> Result<(), ProtocolError> {
        if self.update.hash() != self.header.update_hash {
            return Err(ProtocolError::UpdateHash);
        }
        let bytes: [u8; 64] = self
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| ProtocolError::SignatureLength(self.signature.len()))?;
        let key =
            VerifyingKey::from_bytes(&self.header.phone).map_err(|_| ProtocolError::PublicKey)?;
        key.verify_strict(&self.header.signing_bytes(), &Signature::from_bytes(&bytes))
            .map_err(|_| ProtocolError::Signature)
    }

    /// Identity of this exact contribution: header and signature.
    pub fn digest(&self) -> [u8; 32] {
        let header = postcard::to_stdvec(&self.header).expect("a header always serializes");
        sha256_parts(&[b"multitude/contribution-digest", &header, &self.signature])
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_stdvec(self).expect("a contribution always serializes")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProtocolError> {
        postcard::from_bytes(bytes).map_err(|err| ProtocolError::Decode(err.to_string()))
    }
}

/// What a phone hands over when it is audited: its error-feedback buffer as
/// it stood when the round began.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorReveal {
    pub round: u32,
    pub phone: [u8; 32],
    pub buffer: Vec<f32>,
}

/// SHA-256 over little-endian f32s. Leviathan's verifier hashes deltas the same
/// way, and Multitude uses it for checkpoints and error buffers.
pub fn weights_hash(values: &[f32]) -> [u8; 32] {
    psyche_verifier::hash_delta(values)
}

pub fn signing_key(secret: &[u8; 32]) -> SigningKey {
    SigningKey::from_bytes(secret)
}

/// Base58, as a Solana wallet shows it.
pub fn address(phone: &[u8; 32]) -> String {
    bs58::encode(phone).into_string()
}

pub fn short_address(phone: &[u8; 32]) -> String {
    let full = address(phone);
    format!("{}…{}", &full[..4], &full[full.len() - 4..])
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::compress;
    use crate::config::CompressionConfig;

    fn sample() -> (Contribution, SigningKey) {
        let key = signing_key(&[3u8; 32]);
        let (update, _) = compress(
            &[0.3, -0.1, 0.2, 0.9],
            &[0.0; 4],
            &CompressionConfig {
                chunk: 2,
                topk: 1,
                error_decay: 1.0,
            },
        )
        .unwrap();
        let header = ContributionHeader {
            run: [1u8; 32],
            round: 4,
            phone: [0u8; 32],
            base_checkpoint: [2u8; 32],
            inner_steps: 3,
            tokens: 96,
            mean_loss: 2.5,
            error_commitment: [5u8; 32],
            update_hash: [0u8; 32],
        };
        (Contribution::sign(header, update, &key), key)
    }

    #[test]
    fn signed_contribution_verifies_and_round_trips() {
        let (contribution, key) = sample();
        assert_eq!(contribution.header.phone, key.verifying_key().to_bytes());
        contribution.verify_signature().unwrap();
        let decoded = Contribution::from_bytes(&contribution.to_bytes()).unwrap();
        assert_eq!(decoded, contribution);
    }

    #[test]
    fn any_change_breaks_the_signature() {
        let (contribution, _) = sample();

        let mut header = contribution.clone();
        header.header.inner_steps += 1;
        assert_eq!(header.verify_signature(), Err(ProtocolError::Signature));

        let mut update = contribution.clone();
        update.update.flip_signs();
        assert_eq!(update.verify_signature(), Err(ProtocolError::UpdateHash));

        let mut impostor = contribution;
        impostor.header.phone = signing_key(&[4u8; 32]).verifying_key().to_bytes();
        assert_eq!(impostor.verify_signature(), Err(ProtocolError::Signature));
    }

    #[test]
    fn addresses_look_like_solana_addresses() {
        let phone = signing_key(&[3u8; 32]).verifying_key().to_bytes();
        let full = address(&phone);
        assert!((32..=44).contains(&full.len()));
        assert!(short_address(&phone).contains('…'));
    }
}
