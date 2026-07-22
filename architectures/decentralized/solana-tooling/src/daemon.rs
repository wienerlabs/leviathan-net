use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use leviathan_verifier::decompress_dump;
use leviathan_verifier::index_dir;
use psyche_coordinator::select_audits_for_current_round;
use psyche_verifier::hash_delta;
use psyche_verifier::verify_within_band;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use tch::Device;

use crate::get_accounts::get_coordinator_account_state;
use crate::process_treasurer_instructions::process_treasurer_run_slash_with_hashes;
use crate::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;

pub struct AuditConfig {
    pub run_id: String,
    pub submitted_dir: PathBuf,
    pub reference_dir: PathBuf,
    pub band: f32,
    pub audit_assigned: bool,
    pub dry_run: bool,
    pub verdict_mode: bool,
}

pub fn hex8(bytes: &[u8; 32]) -> String {
    bytes[..4].iter().map(|b| format!("{b:02x}")).collect()
}

pub fn parse_committer(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let rest = name.strip_prefix("result-")?;
    let end = rest.find("-step")?;
    Some(rest[..end].to_string())
}

pub fn parse_batch_bounds(text: &str) -> Option<(u64, u64)> {
    let lb = text.find('[')?;
    let rb = text.find(']')?;
    let inner = &text[lb + 1..rb];
    let mut parts = inner.split(',');
    let start = parts.next()?.trim().parse().ok()?;
    let end = parts.next()?.trim().parse().ok()?;
    Some((start, end))
}

pub async fn audit_pass(
    endpoint: &mut solana_toolbox_endpoint::ToolboxEndpoint,
    authority: &Keypair,
    coordinator_account: &Pubkey,
    run: &Pubkey,
    config: &AuditConfig,
    convicted: &mut HashSet<String>,
) -> Result<usize> {
    let state = get_coordinator_account_state(endpoint, coordinator_account)
        .await?
        .ok_or_else(|| anyhow!("coordinator account {} not found", coordinator_account))?;
    let coordinator = &state.coordinator;

    let assigned: Option<HashSet<(String, u64, u64)>> = if config.audit_assigned {
        match select_audits_for_current_round(coordinator) {
            Ok(audits) => Some(
                audits
                    .iter()
                    .filter_map(|a| {
                        let (start, end) = parse_batch_bounds(&format!("{}", a.batch_id))?;
                        Some((format!("{}", a.target), start, end))
                    })
                    .collect(),
            ),
            Err(err) => {
                println!("[verifier-daemon] no audit assignments this round: {err:?}");
                Some(HashSet::new())
            }
        }
    } else {
        None
    };

    let submitted = index_dir(&config.submitted_dir)?;
    let reference = index_dir(&config.reference_dir)?;
    let device = Device::Cpu;
    let mut new_convictions = 0usize;

    for (key, submitted_path) in &submitted {
        let Some(reference_path) = reference.get(key) else {
            continue;
        };
        let Some(committer) = parse_committer(submitted_path) else {
            continue;
        };
        let (batch_start, batch_end) = parse_batch_bounds(key).unwrap_or((0, 0));

        if let Some(assigned) = &assigned {
            if !assigned.contains(&(committer.clone(), batch_start, batch_end)) {
                continue;
            }
        }

        let submitted_delta = decompress_dump(submitted_path, device)?;
        let reference_delta = decompress_dump(reference_path, device)?;
        let verdict = verify_within_band(&submitted_delta, &reference_delta, config.band)?;

        if !verdict.fraud {
            println!(
                "[verifier-daemon] ok    {key} committer {committer} distance {:.4} within band {:.4}",
                verdict.distance, config.band
            );
            continue;
        }

        let committed_hash = hash_delta(&submitted_delta);
        let replayed_hash = hash_delta(&reference_delta);
        println!(
            "[verifier-daemon] FRAUD {key} committer {committer} distance {:.4} exceeds band {:.4} (committed {} replayed {})",
            verdict.distance,
            config.band,
            hex8(&committed_hash),
            hex8(&replayed_hash)
        );

        if convicted.contains(&committer) {
            println!("  already convicted {committer} this session, skipping");
            continue;
        }

        let index = coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| format!("{}", client.id) == committer);
        let Some(index) = index else {
            println!("  committer {committer} is not in the current epoch roster, cannot act yet");
            continue;
        };

        let action = if config.verdict_mode { "verdict" } else { "slash" };
        if config.dry_run {
            println!("  [dry-run] would submit {action} for {committer} at epoch index {index}");
            new_convictions += 1;
            continue;
        }

        let result = if config.verdict_mode {
            let target = Pubkey::new_from_array(
                *coordinator
                    .epoch_state
                    .clients
                    .iter()
                    .nth(index)
                    .expect("index was just found by position")
                    .id
                    .signer(),
            );
            process_treasurer_run_submit_audit_verdict(
                endpoint,
                authority,
                authority,
                run,
                coordinator_account,
                &config.run_id,
                &target,
                index as u64,
                batch_start,
                batch_end,
                committed_hash,
                replayed_hash,
            )
            .await
        } else {
            process_treasurer_run_slash_with_hashes(
                endpoint,
                authority,
                authority,
                run,
                coordinator_account,
                &config.run_id,
                index as u64,
                batch_start,
                batch_end,
                committed_hash,
                replayed_hash,
            )
            .await
        };

        match result {
            Ok(()) => {
                println!("  {action} submitted for {committer} at epoch index {index}");
                convicted.insert(committer);
                new_convictions += 1;
            }
            Err(err) => {
                println!("  {action} submission failed for {committer} at index {index}: {err:#}")
            }
        }
    }

    Ok(new_convictions)
}
