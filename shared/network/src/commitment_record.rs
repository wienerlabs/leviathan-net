use std::path::{Path, PathBuf};

use psyche_core::{payload_commitment_hash, payload_commitment_run, BatchId, ClosedInterval};
use thiserror::Error;

use crate::authenticable_identity::raw_p2p_verify;
use crate::serialized_distro::{results_digest, SerializedDistroResult, TransmittableDistroResult};

/// The extension a commitment record is kept under, next to the payload dump
/// it belongs to.
pub const COMMITMENT_RECORD_EXTENSION: &str = "commitment";

/// A peer's signed commitment to a payload, as kept by the node that verified
/// it.
///
/// A verifier convicts a peer on what the peer signed. So next to each peer
/// payload a node keeps, it also writes this record: the header the payload
/// hash was built over, and the peer's signature on that hash. Together with
/// the payload's results, that is everything the treasurer needs to check that
/// a conviction names a payload the peer committed to
/// (wienerlabs/leviathan-net#43).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRecord {
    /// When the epoch the payload was trained in started
    /// (wienerlabs/leviathan-net#67).
    pub epoch_start: u64,
    pub step: u32,
    pub trainer_nonce: u32,
    pub batch_id: BatchId,
    pub data_hash: [u8; 32],
    pub signature: [u8; 64],
    /// The digest of the state the payload was trained from, which the hash
    /// binds, if the trainer committed to one (wienerlabs/leviathan-net#89).
    pub state_digest: Option<[u8; 32]>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CommitmentRecordError {
    #[error("a commitment record is {expected} bytes, this one is {actual}")]
    Length { expected: usize, actual: usize },
    #[error("the batch in this commitment record ends before it starts")]
    Batch,
    #[error("the state digest in this commitment record is neither present nor absent")]
    StateDigest,
    #[error("the payload is not the one this commitment's hash was built over")]
    PayloadMismatch,
    #[error("the signature on this commitment is not the committer's")]
    Signature,
    #[error("the payload could not be encoded to hash: {0}")]
    Encoding(postcard::Error),
}

impl CommitmentRecord {
    /// The epoch's start, the step and nonce, the batch bounds, the hash and
    /// the signature, in that order and big-endian; then a tag saying whether
    /// a state digest follows, and the digest, all zeros when it does not.
    pub const SERIALIZED_SIZE: usize = 8 + 4 + 4 + 8 + 8 + 32 + 64 + 1 + 32;

    /// The record of `payload`, committed to as `data_hash` and signed with
    /// `signature`.
    pub fn new(
        payload: &TransmittableDistroResult,
        data_hash: [u8; 32],
        signature: [u8; 64],
    ) -> Self {
        Self {
            epoch_start: payload.epoch_start,
            step: payload.step,
            trainer_nonce: payload.trainer_nonce,
            batch_id: payload.batch_id,
            data_hash,
            signature,
            state_digest: payload.state_digest,
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut bytes = [0u8; Self::SERIALIZED_SIZE];
        bytes[0..8].copy_from_slice(&self.epoch_start.to_be_bytes());
        bytes[8..12].copy_from_slice(&self.step.to_be_bytes());
        bytes[12..16].copy_from_slice(&self.trainer_nonce.to_be_bytes());
        bytes[16..24].copy_from_slice(&self.batch_id.0.start.to_be_bytes());
        bytes[24..32].copy_from_slice(&self.batch_id.0.end.to_be_bytes());
        bytes[32..64].copy_from_slice(&self.data_hash);
        bytes[64..128].copy_from_slice(&self.signature);
        if let Some(digest) = &self.state_digest {
            bytes[128] = 1;
            bytes[129..161].copy_from_slice(digest);
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CommitmentRecordError> {
        let bytes: &[u8; Self::SERIALIZED_SIZE] =
            bytes
                .try_into()
                .map_err(|_| CommitmentRecordError::Length {
                    expected: Self::SERIALIZED_SIZE,
                    actual: bytes.len(),
                })?;
        let u32_at = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_be_bytes(bytes[at..at + 8].try_into().unwrap());
        let (start, end) = (u64_at(16), u64_at(24));
        if end < start {
            return Err(CommitmentRecordError::Batch);
        }
        let digest: [u8; 32] = bytes[129..161].try_into().unwrap();
        // One encoding per record: an absent digest is all zeros.
        let state_digest = match bytes[128] {
            1 => Some(digest),
            0 if digest == [0; 32] => None,
            _ => return Err(CommitmentRecordError::StateDigest),
        };
        Ok(Self {
            epoch_start: u64_at(0),
            step: u32_at(8),
            trainer_nonce: u32_at(12),
            batch_id: BatchId(ClosedInterval::new(start, end)),
            data_hash: bytes[32..64].try_into().unwrap(),
            signature: bytes[64..128].try_into().unwrap(),
            state_digest,
        })
    }

    /// The hash this record's header commits `results` under, in the run
    /// `run_id`.
    pub fn payload_hash(
        &self,
        results: &[SerializedDistroResult],
        run_id: &str,
    ) -> Result<[u8; 32], CommitmentRecordError> {
        let digest = results_digest(results).map_err(CommitmentRecordError::Encoding)?;
        Ok(payload_commitment_hash(
            &payload_commitment_run(run_id),
            self.epoch_start,
            self.step,
            self.trainer_nonce,
            self.batch_id.0.start,
            self.batch_id.0.end,
            &digest,
            self.state_digest.as_ref(),
        ))
    }

    /// Checks that `results` are the payload this record's hash was built
    /// over in the run `run_id`, and that the hash was signed by the holder of
    /// `p2p_identity`.
    pub fn verify(
        &self,
        results: &[SerializedDistroResult],
        run_id: &str,
        p2p_identity: &[u8; 32],
    ) -> Result<(), CommitmentRecordError> {
        if self.payload_hash(results, run_id)? != self.data_hash {
            return Err(CommitmentRecordError::PayloadMismatch);
        }
        if !raw_p2p_verify(p2p_identity, &self.data_hash, &self.signature) {
            return Err(CommitmentRecordError::Signature);
        }
        Ok(())
    }
}

/// Where the commitment record for the payload dump at `payload_path` is kept.
pub fn commitment_record_path(payload_path: &Path) -> PathBuf {
    payload_path.with_extension(COMMITMENT_RECORD_EXTENSION)
}

/// Where a verifier keeps the state the payload dump at `payload_path` was
/// trained from, once its trainer revealed it (wienerlabs/leviathan-net#89).
pub fn revealed_state_path(payload_path: &Path) -> PathBuf {
    payload_path.with_extension(psyche_modeling::ROUND_STATE_EXTENSION)
}

/// The extension a payload dump is kept under.
pub const PAYLOAD_DUMP_EXTENSION: &str = "vec-postcard";

/// The name a node keeps the payload `committer` trained on `batch_id` at
/// `step`, in the epoch that started at `epoch_start`, under.
///
/// The committer is its full signer, base58 and untruncated, which the
/// verifier daemon matches against the roster (wienerlabs/leviathan#15,
/// finding 20). The epoch is in the name because the step alone does not
/// name the round: a run restarted from an earlier step trains that step
/// again, and without the epoch its payload would be kept under the name of
/// the one trained the first time, and never kept itself
/// (wienerlabs/leviathan-net#67).
pub fn payload_dump_name(
    committer: &str,
    epoch_start: u64,
    step: u32,
    batch_id: BatchId,
) -> String {
    format!(
        "result-{committer}-epoch{epoch_start}-step{step}-batch{batch_id}.{PAYLOAD_DUMP_EXTENSION}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serializable_tensor::SerializableTensor;
    use iroh::SecretKey;
    use tch::{Device, Kind, Tensor};

    const RUN: &str = "leviathan commitment record";

    fn results(values: &[f32]) -> Vec<SerializedDistroResult> {
        let tensor = Tensor::from_slice(values)
            .to_kind(Kind::Float)
            .to(Device::Cpu);
        let serialized = SerializableTensor::try_from(&tensor).unwrap();
        vec![SerializedDistroResult {
            sparse_idx: serialized.clone(),
            sparse_val: serialized,
            xshape: vec![values.len() as u16],
            totalk: values.len() as u32,
        }]
    }

    fn payload(results: Vec<SerializedDistroResult>) -> TransmittableDistroResult {
        TransmittableDistroResult {
            epoch_start: 1_700_000_000,
            step: 12,
            trainer_nonce: 2,
            batch_id: BatchId(ClosedInterval::new(40, 47)),
            distro_results: results,
            state_digest: Some(psyche_core::sha256(b"the state trained from")),
        }
    }

    /// A payload signed the way the client signs it.
    fn signed(key: &SecretKey, payload: &TransmittableDistroResult) -> CommitmentRecord {
        let data_hash = payload.comptue_hash(RUN).unwrap();
        let signature = key.sign(&data_hash).to_bytes();
        CommitmentRecord::new(payload, data_hash, signature)
    }

    #[test]
    fn a_genuine_commitment_verifies_against_its_payload_and_signer() {
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(results(&[1.0, -2.0, 3.0]));
        let record = signed(&key, &payload);
        assert_eq!(
            record.payload_hash(&payload.distro_results, RUN),
            Ok(payload.comptue_hash(RUN).unwrap())
        );
        assert_eq!(
            record.verify(&payload.distro_results, RUN, key.public().as_bytes()),
            Ok(())
        );
    }

    #[test]
    fn other_results_are_not_the_committed_payload() {
        let key = SecretKey::generate(&mut rand::rng());
        let record = signed(&key, &payload(results(&[1.0, -2.0, 3.0])));
        assert_eq!(
            record.verify(&results(&[1.0, -2.0, 4.0]), RUN, key.public().as_bytes()),
            Err(CommitmentRecordError::PayloadMismatch)
        );
    }

    #[test]
    fn a_header_other_than_the_signed_one_does_not_verify() {
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(results(&[1.0, -2.0, 3.0]));
        let record = signed(&key, &payload);
        for moved in [
            CommitmentRecord {
                epoch_start: record.epoch_start + 1,
                ..record
            },
            CommitmentRecord {
                step: record.step + 1,
                ..record
            },
            CommitmentRecord {
                trainer_nonce: record.trainer_nonce + 1,
                ..record
            },
            CommitmentRecord {
                batch_id: BatchId(ClosedInterval::new(40, 48)),
                ..record
            },
        ] {
            assert_eq!(
                moved.verify(&payload.distro_results, RUN, key.public().as_bytes()),
                Err(CommitmentRecordError::PayloadMismatch)
            );
        }
    }

    /// A record verified against another run is not the commitment the peer
    /// made, though the peer's key signed it.
    #[test]
    fn a_commitment_made_in_another_run_does_not_verify() {
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(results(&[1.0, -2.0, 3.0]));
        let record = signed(&key, &payload);
        assert_eq!(
            record.verify(
                &payload.distro_results,
                "another run",
                key.public().as_bytes()
            ),
            Err(CommitmentRecordError::PayloadMismatch)
        );
    }

    #[test]
    fn another_key_did_not_sign_it() {
        let key = SecretKey::generate(&mut rand::rng());
        let other = SecretKey::generate(&mut rand::rng());
        let payload = payload(results(&[1.0, -2.0, 3.0]));
        let record = signed(&key, &payload);
        assert_eq!(
            record.verify(&payload.distro_results, RUN, other.public().as_bytes()),
            Err(CommitmentRecordError::Signature)
        );
        let forged = signed(&other, &payload);
        assert_eq!(
            forged.verify(&payload.distro_results, RUN, key.public().as_bytes()),
            Err(CommitmentRecordError::Signature)
        );
    }

    #[test]
    fn a_record_survives_its_bytes() {
        let key = SecretKey::generate(&mut rand::rng());
        let record = signed(&key, &payload(results(&[1.0, -2.0, 3.0])));
        assert_eq!(CommitmentRecord::from_bytes(&record.to_bytes()), Ok(record));
        // A trainer that committed to no state.
        let stateless = signed(
            &key,
            &TransmittableDistroResult {
                state_digest: None,
                ..payload(results(&[1.0, -2.0, 3.0]))
            },
        );
        assert_eq!(stateless.state_digest, None);
        assert_eq!(
            CommitmentRecord::from_bytes(&stateless.to_bytes()),
            Ok(stateless)
        );
    }

    /// The state a payload was trained from is part of what was signed: a
    /// record naming another state, or none, does not verify
    /// (wienerlabs/leviathan-net#89).
    #[test]
    fn a_record_naming_another_state_does_not_verify() {
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(results(&[1.0, -2.0, 3.0]));
        let record = signed(&key, &payload);
        let public = key.public();
        let identity = public.as_bytes();
        assert_eq!(
            record.verify(&payload.distro_results, RUN, identity),
            Ok(())
        );
        for other in [Some(psyche_core::sha256(b"another state")), None] {
            let claimed = CommitmentRecord {
                state_digest: other,
                ..record
            };
            assert_eq!(
                claimed.verify(&payload.distro_results, RUN, identity),
                Err(CommitmentRecordError::PayloadMismatch),
                "{other:?}"
            );
        }
    }

    /// A record says whether a state digest follows in one way only: a tag
    /// other than 0 or 1, or an absent digest that is not all zeros, is
    /// refused.
    #[test]
    fn a_record_with_a_malformed_state_digest_is_refused() {
        let key = SecretKey::generate(&mut rand::rng());
        let bytes = signed(&key, &payload(results(&[1.0]))).to_bytes();
        let mut tag = bytes;
        tag[128] = 2;
        assert_eq!(
            CommitmentRecord::from_bytes(&tag),
            Err(CommitmentRecordError::StateDigest)
        );
        let mut absent_but_set = bytes;
        absent_but_set[128] = 0;
        assert_eq!(
            CommitmentRecord::from_bytes(&absent_but_set),
            Err(CommitmentRecordError::StateDigest)
        );
    }

    #[test]
    fn a_record_of_the_wrong_size_or_an_inverted_batch_is_refused() {
        let key = SecretKey::generate(&mut rand::rng());
        let bytes = signed(&key, &payload(results(&[1.0]))).to_bytes();
        assert_eq!(
            CommitmentRecord::from_bytes(&bytes[1..]),
            Err(CommitmentRecordError::Length {
                expected: CommitmentRecord::SERIALIZED_SIZE,
                actual: CommitmentRecord::SERIALIZED_SIZE - 1,
            })
        );
        let mut inverted = bytes;
        inverted[16..24].copy_from_slice(&48u64.to_be_bytes());
        assert_eq!(
            CommitmentRecord::from_bytes(&inverted),
            Err(CommitmentRecordError::Batch)
        );
    }

    #[test]
    fn the_record_sits_next_to_its_payload() {
        let name = payload_dump_name("Abc", 1_700_000_000, 3, BatchId(ClosedInterval::new(0, 1)));
        assert_eq!(
            name,
            "result-Abc-epoch1700000000-step3-batchB[0, 1].vec-postcard"
        );
        assert_eq!(
            commitment_record_path(&Path::new("/dumps").join(name)),
            PathBuf::from("/dumps/result-Abc-epoch1700000000-step3-batchB[0, 1].commitment")
        );
    }

    /// A step trained in two epochs, as after a restart from an earlier
    /// step, is two payloads under two names (wienerlabs/leviathan-net#67).
    #[test]
    fn the_same_step_in_two_epochs_is_kept_under_two_names() {
        let batch = BatchId(ClosedInterval::new(0, 1));
        assert_ne!(
            payload_dump_name("Abc", 1_700_000_000, 3, batch),
            payload_dump_name("Abc", 1_700_000_090, 3, batch)
        );
    }
}
