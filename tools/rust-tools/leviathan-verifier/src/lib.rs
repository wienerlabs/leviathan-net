use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use psyche_modeling::CompressDCT;
use psyche_modeling::Distro;
use psyche_modeling::DistroResult;
use psyche_network::commitment_record_path;
use psyche_network::distro_results_from_reader;
use psyche_network::equivocation_record_path;
use psyche_network::results_digest;
use psyche_network::CommitmentRecord;
use psyche_network::EquivocationError;
use psyche_network::EquivocationRecord;
use psyche_network::PAYLOAD_DUMP_EXTENSION;
use psyche_network::SerializedDistroResult;
use psyche_verifier::hash_delta;
use psyche_verifier::relative_l2_distance;
use psyche_verifier::verify_signs_within_band;
use psyche_verifier::verify_within_band;
use psyche_verifier::BandVerdict;
use psyche_verifier::FraudProof;
use psyche_verifier::SignDrift;
use psyche_verifier::VerifierError;

pub mod band;
pub mod error_feedback;
pub mod fidelity;
pub mod hardware;
pub mod replay;
pub mod run_data;

pub use band::payload_band;
pub use band::PayloadBand;
pub use hardware::device_label;
pub use hardware::parse_device;
pub use hardware::parse_dtype;

pub use replay::build_replay_trainer;
pub use replay::checkpoint_repo_files;
pub use replay::cold_start_warmup_bounds;
pub use replay::published_checkpoint;
pub use replay::read_run_global_batch;
pub use replay::read_run_verification_percent;
pub use replay::read_run_model;
pub use replay::replay_data_layout;
pub use replay::parse_assignment_key;
pub use replay::replay_epoch;
pub use replay::sends_signs;
pub use replay::DumpKey;
pub use replay::ReplayAssignment;
pub use replay::ReplayTrainerConfig;
pub use replay::ReplayWork;
pub use replay::TrainerReplayEngine;

/// The epoch, step and batch part of a dump named
/// `result-<committer>-epoch<T>-step<N>-batch<B>.vec-postcard`
/// (`psyche_network::payload_dump_name`).
///
/// The committer is base58, which has no `-`, so the first `-epoch` after the
/// prefix is where it ends. Searching for `epoch` alone matched inside a
/// committer that happened to spell it.
pub fn contribution_key(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let rest = name
        .strip_prefix("result-")?
        .strip_suffix(&format!(".{PAYLOAD_DUMP_EXTENSION}"))?;
    let start = rest.find("-epoch")?;
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

/// What `judge_payload` found of a signed payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PayloadJudgement {
    /// The payload's distance from the honest result, and whether it is
    /// fraud under the band: infinite, and fraud, for one that does not fit.
    /// Under a band on the signs, `band` is the largest distance that band
    /// admits, and fraud is decided by the signs.
    pub verdict: BandVerdict,
    /// How the payload's signs differ from the honest result's, when it is
    /// held to a band on the signs (wienerlabs/leviathan-net#49).
    pub signs: Option<SignDrift>,
    /// Why the payload cannot be the honest result whatever it is compared
    /// with, if it cannot.
    pub unfit: Option<String>,
    /// `hash_delta` of what the payload decompresses to, when it does.
    pub submitted_hash: Option<[u8; 32]>,
}

/// Judges the signed payload dump at `path` against `reference`, the honest
/// result for its batch as the run's own model gives it: replayed, or trained
/// by this verifier's own node.
///
/// A payload that does not decode, does not hold together as a DisTrO encoding
/// of its own shapes, or does not have the reference's length is not the
/// honest result. Every honest node's `Distro::apply` refuses it, so its
/// sender did not do the work it claims. It is judged at an infinite distance,
/// as fraud, as a payload that is not made of numbers is (wienerlabs/leviathan#15
/// finding 18, wienerlabs/leviathan-net#59). The first two are properties of
/// the signed bytes alone. The length, like the band, rests on the reference
/// being the run's own model's result. An appeal bench re-auditing the same
/// dump against the same replay reaches the same verdict.
///
/// An error is this verifier's own: the dump cannot be read, the band is not
/// one, or the reference is not made of numbers. None of them convicts
/// anyone, and each may pass.
pub fn judge_payload(
    path: &Path,
    reference: &[f32],
    band: PayloadBand,
    device: tch::Device,
) -> Result<PayloadJudgement> {
    // A band that is not one is refused before anything is judged, by the
    // same checks that judge against it.
    match band {
        PayloadBand::Distance(band) => {
            verify_within_band(&[], &[], band)?;
        }
        PayloadBand::Signs(band) => {
            verify_signs_within_band(&[], &[], band)?;
        }
    }
    if let Some(index) = reference.iter().position(|value| !value.is_finite()) {
        return Err(anyhow!("the reference is not finite at index {index}"));
    }
    let bytes = fs::read(path).map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;
    let one_bit = matches!(band, PayloadBand::Signs(_));
    match fitting_values(&bytes, reference.len(), one_bit, device)? {
        Ok(submitted) => {
            let (verdict, signs) = judge_values(&submitted, reference, band)?;
            Ok(PayloadJudgement {
                verdict,
                signs,
                unfit: None,
                submitted_hash: Some(hash_delta(&submitted)),
            })
        }
        Err(reason) => Ok(PayloadJudgement {
            verdict: BandVerdict {
                distance: f32::INFINITY,
                band: band.distance_bound(),
                fraud: true,
            },
            signs: None,
            unfit: Some(reason),
            submitted_hash: None,
        }),
    }
}

/// `submitted`, values that fit the model, judged against `reference` under
/// `band`.
fn judge_values(
    submitted: &[f32],
    reference: &[f32],
    band: PayloadBand,
) -> Result<(BandVerdict, Option<SignDrift>)> {
    match band {
        PayloadBand::Distance(band) => Ok((verify_within_band(submitted, reference, band)?, None)),
        PayloadBand::Signs(sign_band) => {
            let judged = verify_signs_within_band(submitted, reference, sign_band)?;
            let distance = match relative_l2_distance(submitted, reference) {
                Ok(distance) => distance,
                Err(VerifierError::NonFiniteSubmission { .. }) => f32::INFINITY,
                Err(err) => return Err(err.into()),
            };
            let verdict = BandVerdict {
                distance,
                band: band.distance_bound(),
                fraud: judged.fraud,
            };
            Ok((verdict, Some(judged.drift)))
        }
    }
}

/// The values the signed payload `bytes` decompresses to, if it fits a model
/// whose honest result for the batch has `honest_len` values; why it does not,
/// if it does not.
///
/// The outer error is this verifier's own. Once the payload holds together at
/// the honest size, decompressing it can only fail for reasons of the
/// verifier's, a device it cannot reach for one, and that convicts nobody.
fn fitting_values(
    bytes: &[u8],
    honest_len: usize,
    one_bit: bool,
    device: tch::Device,
) -> Result<std::result::Result<Vec<f32>, String>> {
    let results = match distro_results_from_reader(Cursor::new(bytes))
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(results) => results,
        Err(err) => return Ok(Err(format!("it does not parse: {err}"))),
    };
    let results = match results
        .iter()
        .map(DistroResult::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(results) => results,
        Err(err) => return Ok(Err(format!("it does not decode: {err}"))),
    };
    // A run that sends 1-bit signs packs them as booleans. A payload that
    // carries values instead is not what an honest node of the run sends, and
    // its magnitudes are exactly what a band on the signs cannot see: every
    // node's `Distro::apply` would weigh them (wienerlabs/leviathan-net#49).
    if one_bit {
        if let Some(index) = results
            .iter()
            .position(|result| result.sparse_val.kind() != tch::Kind::Bool)
        {
            return Ok(Err(format!(
                "result {index} carries values, and the run sends 1-bit signs"
            )));
        }
    }
    // Compared before anything is allocated: a payload of a few bytes can
    // claim shapes of billions of values, every index inside them, and
    // decompressing it as it claims would exhaust the verifier's memory
    // before its length was ever checked.
    let claimed = claimed_values(&results);
    if claimed != honest_len as u128 {
        return Ok(Err(format!(
            "it claims {claimed} values, the honest result has {honest_len}"
        )));
    }
    if let Err(err) = Distro::check_encodings(&results) {
        return Ok(Err(format!("it does not hold together: {err}")));
    }
    let submitted = decompress_results(&results, device)?;
    if submitted.len() != honest_len {
        return Err(anyhow!(
            "{claimed} values decompressed to {}",
            submitted.len()
        ));
    }
    Ok(Ok(submitted))
}

/// How many values `results` decompress to, as their shapes claim, counted
/// without allocating any of them.
pub fn claimed_values(results: &[DistroResult]) -> u128 {
    results
        .iter()
        .map(|result| {
            // Decoding refuses a dimension that is not positive
            // (`validate_xshape`); one that slipped through would count as no
            // values, which no honest result has.
            result
                .xshape
                .iter()
                .map(|dim| u128::try_from(*dim).unwrap_or(0))
                .product::<u128>()
        })
        .sum()
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

/// What an equivocation record kept beside a payload dump proves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquivocationCheck {
    /// No equivocation record is kept beside the dump.
    None,
    /// A record is there and does not prove anything: the wrong size, the
    /// kept commitment again, other work, or another key's signature.
    Invalid(String),
    /// The committer signed this second commitment for the same step and
    /// batch as the kept one, which differs from it.
    Proven(EquivocationRecord),
}

/// Checks the equivocation record kept beside the payload dump at `path`,
/// against `kept`, the dump's own verified commitment, in the run `run_id`
/// for the committer's key `p2p_identity` (wienerlabs/leviathan-net#46).
///
/// A `Proven` record and `kept` are two signed commitments to one batch, which
/// convict the committer on their own. An error is a failure to read the file,
/// which may pass.
pub fn check_equivocation(
    path: &Path,
    kept: &VerifiedCommitment,
    run_id: &str,
    p2p_identity: &[u8; 32],
) -> Result<EquivocationCheck> {
    let record_path = equivocation_record_path(path);
    let size = match fs::metadata(&record_path) {
        Ok(metadata) => metadata.len(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(EquivocationCheck::None)
        }
        Err(err) => return Err(anyhow!("cannot read {}: {}", record_path.display(), err)),
    };
    // Sized before it is read, so a stray file cannot make the daemon load it.
    if size != EquivocationRecord::SERIALIZED_SIZE as u64 {
        return Ok(EquivocationCheck::Invalid(format!(
            "{} is {size} bytes, a record is {}",
            record_path.display(),
            EquivocationRecord::SERIALIZED_SIZE
        )));
    }
    let bytes = fs::read(&record_path)
        .map_err(|err| anyhow!("cannot read {}: {}", record_path.display(), err))?;
    let invalid = |err: EquivocationError| {
        EquivocationCheck::Invalid(format!("{}: {err}", record_path.display()))
    };
    let second = match EquivocationRecord::from_bytes(&bytes) {
        Ok(second) => second,
        Err(err) => return Ok(invalid(err)),
    };
    Ok(
        match second.proves_against(&kept.record, &kept.results_digest, run_id, p2p_identity) {
            Ok(()) => EquivocationCheck::Proven(second),
            Err(err) => invalid(err),
        },
    )
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

/// The dense values a payload's results carry, in the DCT domain the replay is
/// compared in. A 1-bit payload carries signs packed as booleans, read as the
/// -1 and +1 that `Distro::apply` applies. Read as numbers they would come out
/// as 1 for a negative value and 0 for a positive one
/// (wienerlabs/leviathan-net#47).
///
/// Each result is checked as an encoding of its own shape first.
/// `CompressDCT::decompress` scatters with libtorch's infallible operations,
/// so a result whose indices reach past its shape used to panic the verifier,
/// and again on every restart while the payload was kept
/// (wienerlabs/leviathan-net#59). It allocates every value the shapes claim,
/// so a stranger's payload has to be held to a size first, as
/// `judge_payload` holds it to the honest result's.
pub fn decompress_results(results: &[DistroResult], device: tch::Device) -> Result<Vec<f32>> {
    Distro::check_encodings(results)?;
    let mut dense: Vec<f32> = Vec::new();
    for result in results {
        let sparse_idx = result.sparse_idx.to_device(device);
        let sparse_val = result.sparse_val.to_device(device);
        let sparse_val = match sparse_val.kind() {
            tch::Kind::Bool => {
                Distro::unpack_tensor_sign_from_boolean(sparse_val, tch::Kind::Float)
            }
            _ => sparse_val,
        };
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
    /// Within the band. `signs` is how its signs differed, when it was held
    /// to a band on the signs.
    Ok {
        key: String,
        distance: f32,
        signs: Option<SignDrift>,
    },
    Fraud {
        key: String,
        proof: FraudProof,
        signs: Option<SignDrift>,
    },
    /// A payload that cannot be the honest result whatever it is compared
    /// with: fraud, at an infinite distance (`judge_payload`,
    /// wienerlabs/leviathan-net#59).
    Unfit {
        key: String,
        reason: String,
    },
    NoReference {
        key: String,
    },
}

pub struct AuditSummary {
    pub outcomes: Vec<ContributionOutcome>,
}

impl AuditSummary {
    pub fn audited(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| {
                matches!(
                    o,
                    ContributionOutcome::Ok { .. }
                        | ContributionOutcome::Fraud { .. }
                        | ContributionOutcome::Unfit { .. }
                )
            })
            .count()
    }

    pub fn fraud(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| {
                matches!(
                    o,
                    ContributionOutcome::Fraud { .. } | ContributionOutcome::Unfit { .. }
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

/// What the submitted dump at `path`, judged against `reference`, comes to.
/// A payload that does not fit is its signer's fault; anything that stops it
/// being judged at all is this verifier's, and fails the audit.
fn outcome(
    key: &str,
    target_index: u64,
    path: &Path,
    reference: &[f32],
    band: PayloadBand,
    device: tch::Device,
) -> Result<ContributionOutcome> {
    let judged = judge_payload(path, reference, band, device)?;
    let key = key.to_string();
    Ok(match (judged.unfit, judged.submitted_hash) {
        (Some(reason), _) => ContributionOutcome::Unfit { key, reason },
        (None, Some(committed_hash)) if judged.verdict.fraud => ContributionOutcome::Fraud {
            key,
            signs: judged.signs,
            proof: FraudProof {
                target_index,
                committed_hash,
                replayed_hash: hash_delta(reference),
                distance: judged.verdict.distance,
                band: judged.verdict.band,
            },
        },
        (None, _) => ContributionOutcome::Ok {
            key,
            distance: judged.verdict.distance,
            signs: judged.signs,
        },
    })
}

/// Audits every dump in `submitted` against `engine`'s replay of the work the
/// dump names. A dump is judged against the replay of its own epoch, step and
/// batch, looked up by those, so what else is in the directory, or lands in it
/// while the engine is built, cannot pair it with another dump's replay.
pub fn audit_with_replay<E: ReplayWork + ?Sized>(
    submitted: &Path,
    engine: &E,
    band: PayloadBand,
    device: tch::Device,
) -> Result<AuditSummary> {
    let submitted_index = index_dir(submitted)?;
    let mut outcomes = Vec::new();
    for (target_index, (key, submitted_path)) in submitted_index.iter().enumerate() {
        // Offline there is no roster, so the proof's index is the dump's place
        // in this listing: a label for the report, never what is replayed.
        let target_index = target_index as u64;
        let replayed = parse_assignment_key(key)
            .ok_or_else(|| anyhow!("{key} names no epoch, step and batch"))
            .and_then(|work| engine.replay_work(&work).map_err(|e| anyhow!(e)));
        let reference_delta = match replayed {
            Ok(delta) => delta,
            Err(_) => {
                outcomes.push(ContributionOutcome::NoReference { key: key.clone() });
                continue;
            }
        };
        outcomes.push(outcome(
            key,
            target_index,
            submitted_path,
            &reference_delta,
            band,
            device,
        )?);
    }
    Ok(AuditSummary { outcomes })
}

pub fn audit_dirs(
    submitted: &Path,
    reference: &Path,
    band: PayloadBand,
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
        let reference_delta = decompress_dump(reference_path, device)?;
        outcomes.push(outcome(
            key,
            target_index as u64,
            submitted_path,
            &reference_delta,
            band,
            device,
        )?);
    }
    Ok(AuditSummary { outcomes })
}
