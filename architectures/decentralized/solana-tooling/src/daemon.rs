use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use leviathan_verifier::decompress_dump;
use leviathan_verifier::index_dir;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::check_commitment;
use leviathan_verifier::CommitmentCheck;
use psyche_coordinator::select_audits_for_current_round;
use psyche_coordinator::Coordinator;
use psyche_core::BatchId;
use psyche_solana_treasurer::state::commitment_run;
use psyche_solana_treasurer::state::committee_matches_roster;
use psyche_solana_treasurer::state::CommitmentEvidence;
use psyche_verifier::hash_delta;
use psyche_verifier::verify_within_band;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use tch::Device;

use crate::get_accounts::get_coordinator_account_state;
use crate::get_accounts::get_run;
use crate::process_treasurer_instructions::process_treasurer_run_slash;
use crate::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use crate::signed_commitment::SignedCommitment;

pub struct AuditConfig {
    pub run_id: String,
    pub submitted_dir: PathBuf,
    /// Honest dumps to compare against. Leave it out and pass a replay engine to
    /// audit_pass instead, so this verifier establishes the reference itself
    /// rather than trusting dumps someone else produced.
    pub reference_dir: Option<PathBuf>,
    pub band: f32,
    pub audit_assigned: bool,
    pub dry_run: bool,
    pub verdict_mode: bool,
    /// With a replay engine, the one step whose contributions it can reproduce:
    /// the step trained from the published checkpoint with fresh optimizer
    /// state. Contributions from any other step are left alone, because judging
    /// them against that model would convict honest nodes.
    pub replay_step: Option<u32>,
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

/// The contributions this round's committee assigns for audit, as the committer
/// named in a dump and the batch it trained.
fn audit_assignments(
    coordinator: &Coordinator,
    tie_breaker_committee_size: u16,
) -> HashSet<(String, BatchId)> {
    // In a warmup, or once the roster has moved under the last round's draw,
    // no committee stands over the roster. Drawing one over it anyway indexes
    // past the round it was drawn for.
    if !committee_matches_roster(coordinator) {
        println!(
            "[verifier-daemon] no audit assignments this round: no committee drawn over the roster ({})",
            coordinator.run_state
        );
        return HashSet::new();
    }
    match select_audits_for_current_round(
        coordinator,
        tie_breaker_committee_size,
    ) {
        Ok(audits) => audits
            .iter()
            // The full signer, as the dump names carry it (finding 20).
            .map(|a| (bs58::encode(a.target.signer()).into_string(), a.batch_id))
            .collect(),
        Err(err) => {
            println!(
                "[verifier-daemon] no audit assignments this round: {err:?}"
            );
            HashSet::new()
        },
    }
}

pub async fn audit_pass(
    endpoint: &mut solana_toolbox_endpoint::ToolboxEndpoint,
    authority: &Keypair,
    coordinator_account: &Pubkey,
    run: &Pubkey,
    config: &AuditConfig,
    replay: Option<&(dyn psyche_verifier::ReplayEngine + Sync)>,
    // What this session has settled: convicted committers by signer, and
    // contributions already found within band by file key, so a long-running
    // daemon neither slashes twice nor replays the same file every pass.
    judged: &mut HashSet<String>,
) -> Result<usize> {
    // The tie-breaker count comes from the run, because that is what the
    // treasurer gates votes with. Deriving it any other way picks a different
    // verifier set and every verdict is refused as `VerifierNotAssigned`
    // (wienerlabs/leviathan#15, finding 4).
    let tie_breaker_committee_size = get_run(endpoint, run)
        .await?
        .ok_or_else(|| anyhow!("run {} not found", run))?
        .tie_breaker_committee_size;

    let state = get_coordinator_account_state(endpoint, coordinator_account)
        .await?
        .ok_or_else(|| anyhow!("coordinator account {} not found", coordinator_account))?;
    let coordinator = &state.coordinator;
    // The run every commitment here has to have been made in.
    let run_id = String::from(&coordinator.run_id);

    let assigned = config
        .audit_assigned
        .then(|| audit_assignments(coordinator, tie_breaker_committee_size));

    let submitted = index_dir(&config.submitted_dir)?;
    let reference = match &config.reference_dir {
        Some(dir) => Some(index_dir(dir)?),
        None => None,
    };
    if reference.is_none() && replay.is_none() {
        return Err(anyhow!(
            "the daemon needs either a reference directory or a replay engine to audit against"
        ));
    }
    let device = Device::Cpu;
    let mut new_convictions = 0usize;

    for (key, submitted_path) in &submitted {
        let Some(committer) = parse_committer(submitted_path) else {
            continue;
        };
        if judged.contains(key) {
            continue;
        }
        let Some((step, batch_id)) = parse_assignment_key(key) else {
            println!("[verifier-daemon] skip  {key}: not a step and batch this daemon can read");
            continue;
        };

        if let Some(assigned) = &assigned {
            if !assigned.contains(&(committer.clone(), batch_id)) {
                continue;
            }
        }

        // Matched on the whole 32-byte signer. `Display` is eight characters
        // meant for a human reading a log line; using it here made two clients
        // sharing a prefix indistinguishable, and tied the conviction path to
        // how identities happen to be printed (finding 20).
        let Some(index) =
            coordinator.epoch_state.clients.iter().position(|client| {
                bs58::encode(client.id.signer()).into_string() == committer
            })
        else {
            println!(
                "[verifier-daemon] skip  {key} committer {committer} is not in the epoch roster, nothing to judge it against"
            );
            continue;
        };
        let target = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(index)
            .expect("index was just found by position")
            .id;

        // What the committer signed comes first. A conviction is proven on chain
        // against it, so a payload kept without one, or with one that does not
        // hold, has nothing to convict on and is not worth replaying
        // (wienerlabs/leviathan-net#43).
        let commitment = match check_commitment(
            submitted_path,
            &run_id,
            target.p2p_identity(),
        ) {
            Ok(CommitmentCheck::Verified(commitment)) => commitment,
            Ok(CommitmentCheck::Missing) => {
                println!(
                    "[verifier-daemon] skip  {key} committer {committer} has no signed commitment kept with it, nothing to convict on"
                );
                judged.insert(key.clone());
                continue;
            },
            Ok(CommitmentCheck::Invalid(reason)) => {
                println!(
                    "[verifier-daemon] skip  {key} committer {committer}: its commitment does not hold: {reason}"
                );
                judged.insert(key.clone());
                continue;
            },
            // A failure to read the files may pass, so the pair is looked at
            // again on the next pass rather than settled now.
            Err(err) => {
                println!(
                    "[verifier-daemon] retry {key} committer {committer}: cannot read its commitment yet: {err:#}"
                );
                continue;
            },
        };
        let record = commitment.record;
        if (record.step, record.batch_id) != (step, batch_id) {
            println!(
                "[verifier-daemon] skip  {key} committer {committer}: its commitment is for step {} batch {}",
                record.step, record.batch_id
            );
            judged.insert(key.clone());
            continue;
        }

        let reference_delta = match (&reference, replay) {
            (Some(reference), _) => {
                let Some(reference_path) = reference.get(key) else {
                    continue;
                };
                decompress_dump(reference_path, device)?
            }
            (None, Some(engine)) => {
                if let Some(step) = config.replay_step {
                    if record.step != step {
                        continue;
                    }
                }
                match engine.replay(index as u64) {
                    Ok(delta) => delta,
                    Err(err) => {
                        println!(
                            "[verifier-daemon] skip  {key} committer {committer} cannot be replayed: {err}"
                        );
                        continue;
                    }
                }
            }
            (None, None) => unreachable!("checked above"),
        };

        let submitted_delta = decompress_dump(submitted_path, device)?;
        let verdict = verify_within_band(&submitted_delta, &reference_delta, config.band)?;

        if !verdict.fraud {
            println!(
                "[verifier-daemon] ok    {key} committer {committer} distance {:.4} within band {:.4}",
                verdict.distance, config.band
            );
            judged.insert(key.clone());
            continue;
        }

        let signed = SignedCommitment {
            run: commitment_run(coordinator),
            batch_start: record.batch_id.0.start,
            batch_end: record.batch_id.0.end,
            evidence: CommitmentEvidence {
                step: record.step,
                trainer_nonce: record.trainer_nonce,
                results_digest: commitment.results_digest,
            },
            signer: *target.p2p_identity(),
            signature: record.signature,
        };
        let replayed_hash = hash_delta(&reference_delta);
        println!(
            "[verifier-daemon] FRAUD {key} committer {committer} distance {:.4} exceeds band {:.4} (committed {} replayed {})",
            verdict.distance,
            config.band,
            hex8(&signed.committed_hash()),
            hex8(&replayed_hash)
        );

        if judged.contains(&committer) {
            println!("  already convicted {committer} this session, skipping");
            continue;
        }

        let action = if config.verdict_mode {
            "verdict"
        } else {
            "slash"
        };
        if config.dry_run {
            println!("  [dry-run] would submit {action} for {committer} at epoch index {index}");
            new_convictions += 1;
            continue;
        }

        let result = if config.verdict_mode {
            process_treasurer_run_submit_audit_verdict(
                endpoint,
                authority,
                authority,
                run,
                coordinator_account,
                &config.run_id,
                &Pubkey::new_from_array(*target.signer()),
                index as u64,
                &signed,
                replayed_hash,
            )
            .await
        } else {
            process_treasurer_run_slash(
                endpoint,
                authority,
                authority,
                run,
                coordinator_account,
                &config.run_id,
                index as u64,
                &signed,
                replayed_hash,
            )
            .await
        };

        match result {
            Ok(()) => {
                println!("  {action} submitted for {committer} at epoch index {index}");
                judged.insert(committer);
                judged.insert(key.clone());
                new_convictions += 1;
            }
            Err(err) => {
                println!("  {action} submission failed for {committer} at index {index}: {err:#}")
            }
        }
    }

    Ok(new_convictions)
}
