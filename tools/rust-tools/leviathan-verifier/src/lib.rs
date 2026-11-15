use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use psyche_modeling::CompressDCT;
use psyche_modeling::DistroResult;
use psyche_network::commitment_record_path;
use psyche_network::distro_results_from_reader;
use psyche_network::results_digest;
use psyche_network::CommitmentRecord;
use psyche_network::SerializedDistroResult;
use psyche_verifier::audit_contribution;
use psyche_verifier::FraudProof;

pub mod replay;

pub use replay::build_replay_trainer;
pub use replay::parse_assignment_key;
pub use replay::ReplayAssignment;
pub use replay::ReplayTrainerConfig;
pub use replay::TrainerReplayEngine;

/// The step and batch part of a dump named `result-<committer>-step<N>-batch<B>.vec-postcard`.
///
/// The committer is base58, which has no `-`, so the first `-step` after the
/// prefix is where it ends. Searching for `step` alone matched inside a
/// committer that happened to spell it.
pub fn contribution_key(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let rest = name
        .strip_prefix("result-")?
        .strip_suffix(".vec-postcard")?;
    let start = rest.find("-step")?;
    Some(rest[start + 1..].to_string())
}

pub fn index_dir(dir: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut map = BTreeMap::new();
    for entry in fs::read_dir(dir).map_err(|e| anyhow!("cannot read {}: {}", dir.display(), e))? {
        let path = entry?.path();
        if let Some(key) = contribution_key(&path) {
            map.insert(key, path);
        }
    }
    Ok(map)
}

pub fn decompress_dump(path: &Path, device: tch::Device) -> Result<Vec<f32>> {
    let results = read_dump(path)?
        .iter()
        .map(DistroResult::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    decompress_results(&results, device)
}

/// The results in a payload dump, as they went over the wire.
pub fn read_dump(path: &Path) -> Result<Vec<SerializedDistroResult>> {
    let bytes = fs::read(path)?;
    distro_results_from_reader(Cursor::new(bytes))
        .map(|result| result.map_err(|err| anyhow!("{}: {}", path.display(), err)))
        .collect()
}

/// A peer's commitment to a payload dump, checked against the dump and the
/// peer's key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedCommitment {
    pub record: CommitmentRecord,
    /// The digest of the dump's results, which the record's hash covers.
    pub results_digest: [u8; 32],
}

/// What the commitment kept next to a payload dump says about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitmentCheck {
    /// No record was kept with the dump: the node's own result, or a payload
    /// kept before records were.
    Missing,
    /// A record is there and does not hold: the wrong size, a payload other
    /// than the one hashed, another run, or another key. The node keeping the
    /// dump checked all of this before writing the pair, so a failure means
    /// the files were tampered with or belong to someone else.
    Invalid(String),
    /// The dump is the payload the record's hash was built over, in the run,
    /// and the committer signed that hash.
    Verified(VerifiedCommitment),
}

/// Checks the commitment kept next to the payload dump at `path`, for the run
/// `run_id` and the committer's key `p2p_identity` (wienerlabs/leviathan-net#43).
///
/// Only a `Verified` dump can have a conviction on it proven on chain. An error
/// is a failure to read the files, which may pass, so the caller can look again.
/// `Missing` and `Invalid` are final for this pair: the client writes the
/// record before it renames the dump into place, and keeps the first payload
/// for a committer, step and batch.
pub fn check_commitment(
    path: &Path,
    run_id: &str,
    p2p_identity: &[u8; 32],
) -> Result<CommitmentCheck> {
    let record_path = commitment_record_path(path);
    let size = match fs::metadata(&record_path) {
        Ok(metadata) => metadata.len(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CommitmentCheck::Missing)
        }
        Err(err) => return Err(anyhow!("cannot read {}: {}", record_path.display(), err)),
    };
    // Sized before it is read, so a stray file cannot make the daemon load it.
    if size != CommitmentRecord::SERIALIZED_SIZE as u64 {
        return Ok(CommitmentCheck::Invalid(format!(
            "{} is {size} bytes, a record is {}",
            record_path.display(),
            CommitmentRecord::SERIALIZED_SIZE
        )));
    }
    let bytes = fs::read(&record_path)
        .map_err(|err| anyhow!("cannot read {}: {}", record_path.display(), err))?;
    let record = match CommitmentRecord::from_bytes(&bytes) {
        Ok(record) => record,
        Err(err) => {
            return Ok(CommitmentCheck::Invalid(format!(
                "{}: {err}",
                record_path.display()
            )))
        }
    };
    let dump = fs::read(path).map_err(|err| anyhow!("cannot read {}: {}", path.display(), err))?;
    let results = match distro_results_from_reader(Cursor::new(dump)).collect::<Result<Vec<_>, _>>()
    {
        Ok(results) => results,
        Err(err) => {
            return Ok(CommitmentCheck::Invalid(format!(
                "{}: {err}",
                path.display()
            )))
        }
    };
    if let Err(err) = record.verify(&results, run_id, p2p_identity) {
        return Ok(CommitmentCheck::Invalid(format!(
            "{}: {err}",
            record_path.display()
        )));
    }
    Ok(CommitmentCheck::Verified(VerifiedCommitment {
        record,
        results_digest: results_digest(&results)?,
    }))
}

/// Writes a tokenized dataset a small model can actually be trained on. Real
/// corpora are tokenized for a full vocabulary, so feeding one to a tiny model
/// indexes past its embedding table; this keeps every token inside vocab_size so
/// the dataset and the model agree. Tokens come from a fixed linear congruential
/// sequence, so the same seed always produces the same corpus and a replay can
/// be reproduced exactly.
pub fn write_token_dataset(
    path: &Path,
    vocab_size: u16,
    sequence_length: usize,
    sequences: usize,
    seed: u64,
) -> Result<()> {
    if vocab_size == 0 {
        return Err(anyhow!("a dataset needs a non-empty vocabulary"));
    }
    let mut state = seed | 1;
    let total_tokens = sequence_length * (sequences + 1);
    let mut bytes = Vec::with_capacity(total_tokens * 2);
    for _ in 0..total_tokens {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let token = ((state >> 33) % vocab_size as u64) as u16;
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

pub fn write_results_dump(path: &Path, results: &[DistroResult]) -> Result<()> {
    let serialized: Vec<psyche_network::SerializedDistroResult> = results
        .iter()
        .map(|result| result.try_into())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    fs::write(path, psyche_network::distro_results_to_bytes(&serialized)?)?;
    Ok(())
}

pub fn write_dense_dump(path: &Path, dense: &[f32], rows: i64) -> Result<()> {
    if rows <= 0 || dense.len() as i64 % rows != 0 {
        return Err(anyhow!(
            "{} values do not reshape into {} rows",
            dense.len(),
            rows
        ));
    }
    let x = tch::Tensor::from_slice(dense)
        .reshape([rows, dense.len() as i64 / rows])
        .to_kind(tch::Kind::Float);
    let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(&x, 8);
    let result = DistroResult {
        sparse_idx,
        sparse_val,
        xshape,
        totalk,
        stats: None,
    };
    let serialized: psyche_network::SerializedDistroResult = (&result).try_into()?;
    let bytes = psyche_network::distro_results_to_bytes(&[serialized])?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn decompress_results(results: &[DistroResult], device: tch::Device) -> Result<Vec<f32>> {
    let mut dense: Vec<f32> = Vec::new();
    for result in results {
        let sparse_idx = result.sparse_idx.to_device(device);
        let sparse_val = result.sparse_val.to_device(device);
        let decompressed = CompressDCT::decompress(
            &sparse_idx,
            &sparse_val,
            &result.xshape,
            result.totalk,
            tch::Kind::Float,
            device,
        );
        let flat: Vec<f32> = (&decompressed.flatten(0, -1)).try_into()?;
        dense.extend(flat);
    }
    Ok(dense)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContributionOutcome {
    Ok { key: String, distance: f32 },
    Fraud { key: String, proof: FraudProof },
    LengthMismatch { key: String, submitted: usize, reference: usize },
    NoReference { key: String },
}

pub struct AuditSummary {
    pub outcomes: Vec<ContributionOutcome>,
}

impl AuditSummary {
    pub fn audited(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| matches!(o, ContributionOutcome::Ok { .. } | ContributionOutcome::Fraud { .. }))
            .count()
    }

    pub fn fraud(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| {
                matches!(
                    o,
                    ContributionOutcome::Fraud { .. } | ContributionOutcome::LengthMismatch { .. }
                )
            })
            .count()
    }

    pub fn proofs(&self) -> Vec<&FraudProof> {
        self.outcomes
            .iter()
            .filter_map(|o| match o {
                ContributionOutcome::Fraud { proof, .. } => Some(proof),
                _ => None,
            })
            .collect()
    }
}

pub fn audit_with_replay<E: psyche_verifier::ReplayEngine>(
    submitted: &Path,
    engine: &E,
    band: f32,
    device: tch::Device,
) -> Result<AuditSummary> {
    let submitted_index = index_dir(submitted)?;
    let mut outcomes = Vec::new();
    for (target_index, (key, submitted_path)) in submitted_index.iter().enumerate() {
        let target_index = target_index as u64;
        let submitted_delta = decompress_dump(submitted_path, device)?;
        let reference_delta = match engine.replay(target_index) {
            Ok(delta) => delta,
            Err(_) => {
                outcomes.push(ContributionOutcome::NoReference { key: key.clone() });
                continue;
            }
        };
        if submitted_delta.len() != reference_delta.len() {
            outcomes.push(ContributionOutcome::LengthMismatch {
                key: key.clone(),
                submitted: submitted_delta.len(),
                reference: reference_delta.len(),
            });
            continue;
        }
        let report =
            audit_contribution(target_index, &submitted_delta, &reference_delta, band)?;
        if report.verdict.fraud {
            outcomes.push(ContributionOutcome::Fraud {
                key: key.clone(),
                proof: report.proof.unwrap(),
            });
        } else {
            outcomes.push(ContributionOutcome::Ok {
                key: key.clone(),
                distance: report.verdict.distance,
            });
        }
    }
    Ok(AuditSummary { outcomes })
}

pub fn audit_dirs(
    submitted: &Path,
    reference: &Path,
    band: f32,
    device: tch::Device,
) -> Result<AuditSummary> {
    let submitted_index = index_dir(submitted)?;
    let reference_index = index_dir(reference)?;
    let mut outcomes = Vec::new();
    for (target_index, (key, submitted_path)) in submitted_index.iter().enumerate() {
        let Some(reference_path) = reference_index.get(key) else {
            outcomes.push(ContributionOutcome::NoReference { key: key.clone() });
            continue;
        };
        let submitted_delta = decompress_dump(submitted_path, device)?;
        let reference_delta = decompress_dump(reference_path, device)?;
        if submitted_delta.len() != reference_delta.len() {
            outcomes.push(ContributionOutcome::LengthMismatch {
                key: key.clone(),
                submitted: submitted_delta.len(),
                reference: reference_delta.len(),
            });
            continue;
        }
        let report = audit_contribution(
            target_index as u64,
            &submitted_delta,
            &reference_delta,
            band,
        )?;
        if report.verdict.fraud {
            outcomes.push(ContributionOutcome::Fraud {
                key: key.clone(),
                proof: report.proof.unwrap(),
            });
        } else {
            outcomes.push(ContributionOutcome::Ok {
                key: key.clone(),
                distance: report.verdict.distance,
            });
        }
    }
    Ok(AuditSummary { outcomes })
}
