use std::path::{Path, PathBuf};

use psyche_core::{payload_commitment_hash, payload_commitment_run};
use thiserror::Error;

use crate::authenticable_identity::raw_p2p_verify;
use crate::commitment_record::{CommitmentRecord, CommitmentRecordError};
use crate::serialized_distro::TransmittableDistroResult;

/// The extension an equivocation record is kept under, next to the payload
/// dump whose name it contradicts.
pub const EQUIVOCATION_RECORD_EXTENSION: &str = "equivocation";

/// A second commitment a peer signed for a batch whose first is already kept.
///
/// A trainer signs exactly one payload per batch it trains, so two different
/// commitments for the same run, round and batch, both signed by its key, prove
/// it told different peers different things about the same work. The round is
/// the epoch and the step: a run restarted from an earlier step trains the
/// step again in a later epoch, and that is other work
/// (wienerlabs/leviathan-net#67). The node keeps
/// the first payload and drops the second (#44). Beside it, it keeps this
/// record of the second: its header and signature, and the digest of its
/// results. That is everything the treasurer needs to rebuild the second hash
/// and check the signature, without the payload itself
/// (wienerlabs/leviathan-net#46).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EquivocationRecord {
    pub record: CommitmentRecord,
    pub results_digest: [u8; 32],
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EquivocationError {
    #[error("an equivocation record is {expected} bytes, this one is {actual}")]
    Length { expected: usize, actual: usize },
    #[error("the commitment in this equivocation record is malformed: {0}")]
    Record(CommitmentRecordError),
    #[error("the payload could not be encoded to hash: {0}")]
    Encoding(postcard::Error),
    #[error("the two commitments are for different epochs, steps or batches")]
    DifferentWork,
    #[error("the two commitments are the same commitment")]
    SameCommitment,
    #[error("a commitment's hash is not the one its header and digest rebuild")]
    HashMismatch,
    #[error("a commitment's signature is not the committer's")]
    Signature,
}

impl EquivocationRecord {
    /// The commitment record, then the results digest.
    pub const SERIALIZED_SIZE: usize = CommitmentRecord::SERIALIZED_SIZE + 32;

    /// The record of `payload`, committed to as `record` says.
    pub fn new(
        payload: &TransmittableDistroResult,
        record: CommitmentRecord,
    ) -> Result<Self, EquivocationError> {
        Ok(Self {
            record,
            results_digest: payload
                .results_digest()
                .map_err(EquivocationError::Encoding)?,
        })
    }

    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut bytes = [0u8; Self::SERIALIZED_SIZE];
        bytes[..CommitmentRecord::SERIALIZED_SIZE].copy_from_slice(&self.record.to_bytes());
        bytes[CommitmentRecord::SERIALIZED_SIZE..].copy_from_slice(&self.results_digest);
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EquivocationError> {
        if bytes.len() != Self::SERIALIZED_SIZE {
            return Err(EquivocationError::Length {
                expected: Self::SERIALIZED_SIZE,
                actual: bytes.len(),
            });
        }
        let (record, digest) = bytes.split_at(CommitmentRecord::SERIALIZED_SIZE);
        Ok(Self {
            record: CommitmentRecord::from_bytes(record).map_err(EquivocationError::Record)?,
            results_digest: digest.try_into().expect("split at the record's size"),
        })
    }

    /// Checks that this record and `kept`, whose payload has `kept_digest`,
    /// prove equivocation in the run `run_id` by the holder of `p2p_identity`:
    /// the same epoch, step and batch, two different hashes, each the one its
    /// header and digest rebuild, each signed by that key.
    pub fn proves_against(
        &self,
        kept: &CommitmentRecord,
        kept_digest: &[u8; 32],
        run_id: &str,
        p2p_identity: &[u8; 32],
    ) -> Result<(), EquivocationError> {
        let second = &self.record;
        if (second.epoch_start, second.step, second.batch_id)
            != (kept.epoch_start, kept.step, kept.batch_id)
        {
            return Err(EquivocationError::DifferentWork);
        }
        if second.data_hash == kept.data_hash {
            return Err(EquivocationError::SameCommitment);
        }
        let run = payload_commitment_run(run_id);
        for (record, digest) in [(kept, kept_digest), (second, &self.results_digest)] {
            let rebuilt = payload_commitment_hash(
                &run,
                record.epoch_start,
                record.step,
                record.trainer_nonce,
                record.batch_id.0.start,
                record.batch_id.0.end,
                digest,
            );
            if rebuilt != record.data_hash {
                return Err(EquivocationError::HashMismatch);
            }
            if !raw_p2p_verify(p2p_identity, &record.data_hash, &record.signature) {
                return Err(EquivocationError::Signature);
            }
        }
        Ok(())
    }
}

/// Where the equivocation record for the payload dump at `payload_path` is
/// kept.
pub fn equivocation_record_path(payload_path: &Path) -> PathBuf {
    payload_path.with_extension(EQUIVOCATION_RECORD_EXTENSION)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serializable_tensor::SerializableTensor;
    use crate::serialized_distro::SerializedDistroResult;
    use iroh::SecretKey;
    use psyche_core::{BatchId, ClosedInterval};
    use tch::{Device, Kind, Tensor};

    const RUN: &str = "leviathan equivocation record";

    fn payload(values: &[f32], nonce: u32) -> TransmittableDistroResult {
        let tensor = Tensor::from_slice(values)
            .to_kind(Kind::Float)
            .to(Device::Cpu);
        let serialized = SerializableTensor::try_from(&tensor).unwrap();
        TransmittableDistroResult {
            epoch_start: 1_700_000_000,
            step: 12,
            trainer_nonce: nonce,
            batch_id: BatchId(ClosedInterval::new(40, 47)),
            distro_results: vec![SerializedDistroResult {
                sparse_idx: serialized.clone(),
                sparse_val: serialized,
                xshape: vec![values.len() as u16],
                totalk: values.len() as u32,
            }],
        }
    }

    /// `payload` signed by `key` the way the client signs it, with its digest.
    fn signed(key: &SecretKey, payload: &TransmittableDistroResult) -> EquivocationRecord {
        let data_hash = payload.comptue_hash(RUN).unwrap();
        let signature = key.sign(&data_hash).to_bytes();
        EquivocationRecord::new(
            payload,
            CommitmentRecord::new(payload, data_hash, signature),
        )
        .unwrap()
    }

    #[test]
    fn two_signed_payloads_for_one_batch_prove_equivocation() {
        let key = SecretKey::generate(&mut rand::rng());
        let kept = signed(&key, &payload(&[1.0, -2.0], 2));
        // Another payload, and another payload under another nonce.
        for second in [
            signed(&key, &payload(&[1.0, 2.0], 2)),
            signed(&key, &payload(&[1.0, -2.0], 3)),
        ] {
            assert_eq!(
                second.proves_against(
                    &kept.record,
                    &kept.results_digest,
                    RUN,
                    key.public().as_bytes()
                ),
                Ok(())
            );
        }
    }

    #[test]
    fn the_same_commitment_twice_is_not_equivocation() {
        let key = SecretKey::generate(&mut rand::rng());
        let kept = signed(&key, &payload(&[1.0, -2.0], 2));
        let again = signed(&key, &payload(&[1.0, -2.0], 2));
        assert_eq!(
            again.proves_against(
                &kept.record,
                &kept.results_digest,
                RUN,
                key.public().as_bytes()
            ),
            Err(EquivocationError::SameCommitment)
        );
    }

    #[test]
    fn commitments_for_other_work_are_not_equivocation() {
        let key = SecretKey::generate(&mut rand::rng());
        let kept = signed(&key, &payload(&[1.0, -2.0], 2));
        let mut later = payload(&[1.0, 2.0], 2);
        later.step += 1;
        let mut elsewhere = payload(&[1.0, 2.0], 2);
        elsewhere.batch_id = BatchId(ClosedInterval::new(48, 55));
        // The same step trained again after a restart, in a later epoch.
        let mut restarted = payload(&[1.0, 2.0], 2);
        restarted.epoch_start += 90;
        for second in [
            signed(&key, &later),
            signed(&key, &elsewhere),
            signed(&key, &restarted),
        ] {
            assert_eq!(
                second.proves_against(
                    &kept.record,
                    &kept.results_digest,
                    RUN,
                    key.public().as_bytes()
                ),
                Err(EquivocationError::DifferentWork)
            );
        }
    }

    #[test]
    fn a_commitment_signed_by_another_key_proves_nothing() {
        let key = SecretKey::generate(&mut rand::rng());
        let other = SecretKey::generate(&mut rand::rng());
        let kept = signed(&key, &payload(&[1.0, -2.0], 2));
        let forged = signed(&other, &payload(&[1.0, 2.0], 2));
        assert_eq!(
            forged.proves_against(
                &kept.record,
                &kept.results_digest,
                RUN,
                key.public().as_bytes()
            ),
            Err(EquivocationError::Signature)
        );
    }

    #[test]
    fn a_digest_other_than_the_signed_one_proves_nothing() {
        let key = SecretKey::generate(&mut rand::rng());
        let kept = signed(&key, &payload(&[1.0, -2.0], 2));
        let mut second = signed(&key, &payload(&[1.0, 2.0], 2));
        second.results_digest[0] ^= 1;
        assert_eq!(
            second.proves_against(
                &kept.record,
                &kept.results_digest,
                RUN,
                key.public().as_bytes()
            ),
            Err(EquivocationError::HashMismatch)
        );
        // Nor does a commitment made in another run.
        let second = signed(&key, &payload(&[1.0, 2.0], 2));
        assert_eq!(
            second.proves_against(
                &kept.record,
                &kept.results_digest,
                "another run",
                key.public().as_bytes()
            ),
            Err(EquivocationError::HashMismatch)
        );
    }

    #[test]
    fn a_record_survives_its_bytes_and_a_short_one_is_refused() {
        let key = SecretKey::generate(&mut rand::rng());
        let record = signed(&key, &payload(&[1.0, -2.0], 2));
        assert_eq!(
            EquivocationRecord::from_bytes(&record.to_bytes()),
            Ok(record)
        );
        assert_eq!(
            EquivocationRecord::from_bytes(&record.to_bytes()[1..]),
            Err(EquivocationError::Length {
                expected: EquivocationRecord::SERIALIZED_SIZE,
                actual: EquivocationRecord::SERIALIZED_SIZE - 1,
            })
        );
    }
}
