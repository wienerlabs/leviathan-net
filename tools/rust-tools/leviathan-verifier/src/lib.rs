use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use psyche_modeling::CompressDCT;
use psyche_modeling::DistroResult;
use psyche_network::distro_results_from_reader;
use psyche_verifier::audit_contribution;
use psyche_verifier::FraudProof;

pub fn contribution_key(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    if !name.ends_with(".vec-postcard") {
        return None;
    }
    let start = name.find("step")?;
    Some(name[start..].trim_end_matches(".vec-postcard").to_string())
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
    let bytes = fs::read(path)?;
    let mut dense: Vec<f32> = Vec::new();
    for serialized in distro_results_from_reader(Cursor::new(bytes)) {
        let mut result: DistroResult = (&serialized?).try_into()?;
        result.sparse_idx = result.sparse_idx.to_device(device);
        result.sparse_val = result.sparse_val.to_device(device);
        let decompressed = CompressDCT::decompress(
            &result.sparse_idx,
            &result.sparse_val,
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
