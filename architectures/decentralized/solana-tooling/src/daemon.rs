use std::collections::BTreeMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::decompress_dump;
use leviathan_verifier::index_dir;
use leviathan_verifier::judge_payload;
use leviathan_verifier::sends_signs;
use leviathan_verifier::PayloadBand;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::check_commitment;
use leviathan_verifier::CommitmentCheck;
use leviathan_verifier::ReplayWork;
use leviathan_verifier::VerifiedCommitment;
use psyche_coordinator::select_audits_for_current_round;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_coordinator::model::Model;
use psyche_core::BatchId;
use psyche_core::OptimizerDefinition;
use psyche_network::CommitmentRecord;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::state::commitment_run;
use psyche_solana_treasurer::state::committee_matches_roster;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::CommitmentEvidence;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_verifier::hash_delta;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use tch::Device;

use crate::daemon_equivocation::EquivocationPass;
use crate::get_accounts::get_coordinator_account_state;
use crate::process_treasurer_instructions::process_treasurer_run_slash;
use crate::process_treasurer_instructions::process_treasurer_run_submit_audit_verdict;
use crate::slash_reports::settle_slash_reports;
use crate::verdict_votes::settle_open_disputes;
use crate::verdict_votes::settle_previous_dispute;
use crate::verdict_votes::DisputeSettlement;
use crate::signed_commitment::SignedCommitment;

pub struct AuditConfig {
    pub run_id: String,
    pub submitted_dir: PathBuf,
    /// Honest dumps to compare against. Leave it out and pass a replay engine to
    /// audit_pass instead, so this verifier establishes the reference itself
    /// rather than trusting dumps someone else produced.
    pub reference_dir: Option<PathBuf>,
    /// What a payload is held to: a band on its distance, or for a run that
    /// sends 1-bit signs, a band on each of churn and flips
    /// (wienerlabs/leviathan-net#49).
    pub band: PayloadBand,
    pub audit_assigned: bool,
    pub dry_run: bool,
    pub verdict_mode: bool,
}

pub fn hex8(bytes: &[u8; 32]) -> String {
    bytes[..4].iter().map(|b| format!("{b:02x}")).collect()
}

/// The committer a payload dump is named for
/// (`psyche_network::payload_dump_name`). Base58 has no `-`, so the first
/// `-epoch` after the prefix is where it ends.
pub fn parse_committer(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let rest = name.strip_prefix("result-")?;
    let end = rest.find("-epoch")?;
    Some(rest[..end].to_string())
}

/// The contributions this round's committee assigns for audit, as the committer
/// named in a dump and the batch it trained.
fn audit_assignments(coordinator: &Coordinator) -> HashSet<(String, BatchId)> {
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
    // Drawn with the tie-breaker seats the round reserves, as the round's
    // clients and the treasurer draw it (wienerlabs/leviathan#15, finding 4).
    match select_audits_for_current_round(coordinator) {
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

/// Whether the round `record` names ended before the epoch the coordinator
/// is in: an earlier epoch, or a step before this epoch's first. A record the
/// coordinator has not reached yet is not over, only ahead of this view of it
/// (wienerlabs/leviathan-net#67).
pub fn epoch_is_over(
    coordinator: &Coordinator,
    record: &CommitmentRecord,
) -> bool {
    let epoch_start = coordinator.epoch_state.start_timestamp;
    record.epoch_start < epoch_start
        || (record.epoch_start == epoch_start
            && record.step < coordinator.epoch_state.start_step)
}

/// What `p2p` signed for a kept payload, as the treasurer takes it in a
/// conviction.
pub fn signed_commitment_of(
    coordinator: &Coordinator,
    commitment: &VerifiedCommitment,
    p2p: [u8; 32],
) -> SignedCommitment {
    let record = commitment.record;
    SignedCommitment {
        run: commitment_run(coordinator),
        batch_start: record.batch_id.0.start,
        batch_end: record.batch_id.0.end,
        evidence: CommitmentEvidence {
            epoch_start: record.epoch_start,
            step: record.step,
            trainer_nonce: record.trainer_nonce,
            results_digest: commitment.results_digest,
        },
        signer: p2p,
        signature: record.signature,
    }
}

/// Whether `target` is ejected, on the roster or, once a round ended, among
/// the clients that left it.
fn is_ejected(coordinator: &Coordinator, target: &Pubkey) -> bool {
    let roster = coordinator.epoch_state.clients.iter();
    let exited = coordinator.epoch_state.exited_clients.iter();
    roster.chain(exited).any(|client| {
        *client.id.signer() == target.to_bytes()
            && client.state == ClientState::Ejected
    })
}

/// Whether the chain has ejected `target`, read now.
pub async fn target_ejected(
    endpoint: &mut solana_toolbox_endpoint::ToolboxEndpoint,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<bool> {
    let state = get_coordinator_account_state(endpoint, coordinator_account)
        .await?
        .ok_or_else(|| {
            anyhow!("coordinator account {coordinator_account} not found")
        })?;
    Ok(is_ejected(&state.coordinator, target))
}

/// Whether the chain already holds what a refused submission would have put
/// there: a conviction of `target` in this epoch, landed or still in its
/// windows, or `verifier`'s own vote against it in the round under way. A
/// refusal for that reason is a proof the chain already has, not one it never
/// saw (wienerlabs/leviathan-net#57).
async fn already_on_chain(
    endpoint: &mut solana_toolbox_endpoint::ToolboxEndpoint,
    coordinator_account: &Pubkey,
    run: &Pubkey,
    target: &Pubkey,
    verifier: &Pubkey,
) -> Result<bool> {
    let (epoch, round_height, ejected) = {
        let state =
            get_coordinator_account_state(endpoint, coordinator_account)
                .await?
                .ok_or_else(|| {
                    anyhow!(
                        "coordinator account {coordinator_account} not found"
                    )
                })?;
        let coordinator = &state.coordinator;
        let ejected = is_ejected(coordinator, target);
        let round_height =
            coordinator.current_round().map_or(0, |round| round.height);
        (coordinator.progress.epoch, round_height, ejected)
    };
    if ejected {
        return Ok(true);
    }
    let Some(account) = endpoint
        .get_account(&find_audit_verdict(run, target))
        .await?
    else {
        return Ok(false);
    };
    let Some(verdict) =
        AuditVerdict::from_pda_account(&account.owner, &account.data)?
    else {
        return Ok(false);
    };
    let this_epoch = verdict.epoch == epoch;
    Ok(match verdict.status {
        VerdictStatus::SlashPending
        | VerdictStatus::Challenged
        | VerdictStatus::Upheld
        | VerdictStatus::Due => this_epoch,
        VerdictStatus::Voting => {
            this_epoch
                && verdict.round_height == round_height
                && verdict.voters.contains(verifier)
        },
        // The chain found the target innocent of its last dispute, and holds
        // nothing of this one.
        VerdictStatus::Overturned => false,
    })
}

/// Whether `config`'s band is the kind the run calls for: a band on the signs
/// for a run that sends 1-bit signs, a band on the distance for one that sends
/// values (wienerlabs/leviathan-net#49). The run's optimizer is read every
/// pass, because a run can be configured after its daemon starts. A run not
/// configured yet has nothing to judge, and a daemon that only watches may
/// hold either kind to either. Says why when it is not.
fn band_fits_run(config: &AuditConfig, coordinator: &Coordinator) -> bool {
    let Model::LLM(llm) = &coordinator.model;
    if matches!(llm.optimizer, OptimizerDefinition::Dummy) {
        return false;
    }
    if config.dry_run {
        return true;
    }
    match (sends_signs(&llm.optimizer), config.band) {
        (true, PayloadBand::Distance(_)) => {
            println!(
                "[verifier-daemon] the run sends 1-bit signs, and this daemon holds payloads to a band on their distance: judging none until it is given --churn-band and --flip-band, calibrated with calibrate-band for its hardware (docs/BAND_CALIBRATION.md)"
            );
            false
        },
        (false, PayloadBand::Signs(_)) => {
            println!(
                "[verifier-daemon] the run sends values, not 1-bit signs, and a band on the signs cannot see their magnitudes: judging none until this daemon is given --band"
            );
            false
        },
        _ => true,
    }
}

pub async fn audit_pass(
    endpoint: &mut solana_toolbox_endpoint::ToolboxEndpoint,
    authority: &Keypair,
    coordinator_account: &Pubkey,
    run: &Pubkey,
    config: &AuditConfig,
    replay: Option<&(dyn ReplayWork + Sync)>,
    // What this session has settled: convicted committers by signer, and
    // contributions by file key once judged for good (within band, convicted,
    // withheld, already on chain, or malformed), so a long-running daemon
    // neither slashes twice nor replays the same file every pass. A
    // contribution whose submission failed is not settled, and is judged and
    // submitted again on the next pass.
    judged: &mut HashSet<String>,
) -> Result<PassReport> {
    let state = get_coordinator_account_state(endpoint, coordinator_account)
        .await?
        .ok_or_else(|| anyhow!("coordinator account {} not found", coordinator_account))?;
    let coordinator = &state.coordinator;
    // The run every commitment here has to have been made in.
    let run_id = String::from(&coordinator.run_id);

    let assigned = config
        .audit_assigned
        .then(|| audit_assignments(coordinator));

    let submitted = index_dir(&config.submitted_dir)?;
    let reference = match &config.reference_dir {
        Some(dir) => Some(index_dir(dir)?),
        None => None,
    };
    let device = Device::Cpu;
    let mut report = PassReport::default();

    // Equivocation first: it needs no reference and no committee seat, so it
    // is proven for every payload kept, assigned or not, and a committer it
    // convicts is not convicted again below (wienerlabs/leviathan-net#46).
    EquivocationPass {
        endpoint,
        authority,
        coordinator_account,
        run,
        config,
        coordinator,
        report: &mut report,
        judged,
    }
    .convict_all(&submitted)
    .await;

    // A daemon that replays has nothing to judge a payload against past the
    // run's first epoch, where no step can be replayed
    // (wienerlabs/leviathan-net#48), nor in a pass whose replay could not be
    // prepared. Equivocation is the fraud still provable then, and it has
    // just been judged. The rest of the pass still runs: the sweep of the
    // run's open disputes below is what finalises and pays the convictions
    // this daemon voted on before.
    let nothing_to_judge_against = BTreeMap::new();
    let band_fits = band_fits_run(config, coordinator);
    if !band_fits {
        // Counted as payloads this verifier cannot judge yet, which is what
        // they are until it is given the band the run calls for.
        report.unreplayable += submitted
            .keys()
            .filter(|key| !judged.contains(*key))
            .count() as u64;
    }
    let to_judge = match (&reference, replay, band_fits) {
        (None, None, _) | (_, _, false) => &nothing_to_judge_against,
        _ => &submitted,
    };

    for (key, submitted_path) in to_judge {
        let Some(committer) = parse_committer(submitted_path) else {
            continue;
        };
        if judged.contains(key) {
            continue;
        }
        let Some(dump) = parse_assignment_key(key) else {
            println!("[verifier-daemon] skip  {key}: not an epoch, step and batch this daemon can read");
            continue;
        };
        let batch_id = dump.batch_id;

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
        if (record.epoch_start, record.step, record.batch_id)
            != (dump.epoch_start, dump.step, dump.batch_id)
        {
            println!(
                "[verifier-daemon] skip  {key} committer {committer}: its commitment is for the epoch that started at {}, step {} batch {}",
                record.epoch_start, record.step, record.batch_id
            );
            judged.insert(key.clone());
            continue;
        }
        let signed = signed_commitment_of(
            coordinator,
            &commitment,
            *target.p2p_identity(),
        );
        // A payload from an epoch that is over convicts nobody now: the
        // treasurer refuses it, by the same rule. One kept from before a
        // restart from an earlier step is such a payload, though its step is
        // trained again. Judged and sent anyway, it would be refused every
        // pass and reported as a fraud the chain never saw
        // (wienerlabs/leviathan-net#67).
        if !signed.evidence.is_from_current_epoch(coordinator) {
            if epoch_is_over(coordinator, &record) {
                println!(
                    "[verifier-daemon] skip  {key} committer {committer}: its epoch is over, nothing can convict on it now"
                );
                judged.insert(key.clone());
            } else {
                // Work this pass's view of the run has not reached: the run
                // moved on after it was read, or the RPC lags the node that
                // kept the payload. The next pass reads the run again.
                println!(
                    "[verifier-daemon] retry {key} committer {committer}: it is newer than this pass's view of the run"
                );
            }
            continue;
        }

        let reference_delta = match (&reference, replay) {
            (Some(reference), _) => {
                let Some(reference_path) = reference.get(key) else {
                    continue;
                };
                // This verifier's own reference: a dump it cannot read is its
                // problem, not the target's, and is retried.
                match decompress_dump(reference_path, device) {
                    Ok(delta) => delta,
                    Err(err) => {
                        println!(
                            "[verifier-daemon] retry {key} committer {committer}: cannot read the reference: {err:#}"
                        );
                        report.unreplayable += 1;
                        continue;
                    },
                }
            }
            (None, Some(engine)) => {
                // A replay reproduces only the first step of a cold-start
                // epoch, trained from zeroed error feedback, and the run says
                // which step that is. Judging any other step against it would
                // convict honest nodes (wienerlabs/leviathan-net#47, #48), and
                // so would the same step trained again after a restart, from
                // other weights (wienerlabs/leviathan-net#67).
                if !coordinator.is_replayable(record.epoch_start, record.step) {
                    continue;
                }
                // The replay of the work this payload is signed for, found by
                // that work. The roster index is this pass's view of where the
                // committer sits, and only names it to the chain: the engine
                // was built from an earlier read, and anyone leaving the epoch
                // in between moves the committer to a seat whose batch is
                // someone else's.
                match engine.replay_work(&dump) {
                    Ok(delta) => delta,
                    Err(err) => {
                        println!(
                            "[verifier-daemon] skip  {key} committer {committer} cannot be replayed: {err}"
                        );
                        report.unreplayable += 1;
                        continue;
                    }
                }
            }
            (None, None) => unreachable!("checked above"),
        };

        // A signed payload that does not decode, does not hold together, or
        // does not have the honest result's length cannot be that result: it
        // is fraud at an infinite distance, and convicted like any other
        // (wienerlabs/leviathan-net#59). What stops it being judged at all is
        // this verifier's own trouble, looked at again on the next pass:
        // failing the pass instead would let one payload stop this verifier
        // judging anyone (wienerlabs/leviathan-net#57).
        let judged_payload = match judge_payload(
            submitted_path,
            &reference_delta,
            config.band,
            device,
        ) {
            Ok(judged_payload) => judged_payload,
            Err(err) => {
                println!(
                    "[verifier-daemon] retry {key} committer {committer}: cannot judge its payload yet: {err:#}"
                );
                report.unreplayable += 1;
                continue;
            },
        };
        if let Some(reason) = &judged_payload.unfit {
            println!(
                "[verifier-daemon] {key} committer {committer}: its payload does not fit the run's model: {reason}"
            );
        }
        let verdict = judged_payload.verdict;

        let signs = match judged_payload.signs {
            Some(drift) => {
                format!(", churn {:.4}, flips {:.6}", drift.churn, drift.flips)
            },
            None => String::new(),
        };
        if !verdict.fraud {
            println!(
                "[verifier-daemon] ok    {key} committer {committer} distance {:.4}{signs} within {}",
                verdict.distance, config.band
            );
            report.cleared += 1;
            judged.insert(key.clone());
            continue;
        }

        let replayed_hash = hash_delta(&reference_delta);
        report.fraud_found += 1;
        println!(
            "[verifier-daemon] FRAUD {key} committer {committer} distance {:.4}{signs} outside {} (committed {} replayed {})",
            verdict.distance,
            config.band,
            hex8(&signed.committed_hash()),
            hex8(&replayed_hash)
        );

        if judged.contains(&committer) {
            println!("  already convicted {committer} this session, skipping");
            judged.insert(key.clone());
            continue;
        }

        let action = if config.verdict_mode {
            "verdict"
        } else {
            "slash"
        };
        if config.dry_run {
            println!("  [dry-run] would submit {action} for {committer} at epoch index {index}");
            report.withheld += 1;
            judged.insert(key.clone());
            continue;
        }

        let target_signer = Pubkey::new_from_array(*target.signer());
        if config.verdict_mode {
            // The target's verdict account holds one dispute at a time, and it
            // is taken over only once the last one holds nobody's bond and
            // owes nobody a bounty. What nobody's consent is needed for is
            // settled first, so a lapsed, standing or overturned dispute does
            // not refuse this one (wienerlabs/leviathan-net#39). Anything still
            // at stake stays, and the vote below is refused as before.
            match settle_previous_dispute(
                endpoint,
                authority,
                run,
                coordinator_account,
                &target_signer,
            )
            .await
            {
                Ok(settled) if settled == DisputeSettlement::default() => {},
                Ok(settled) => println!(
                    "  settled the last dispute against {committer}: conviction finalised {}, conviction charged {}, {} vote(s) freed, {} loser(s) charged, {} recipient(s) credited",
                    settled.conviction_finalized,
                    settled.conviction_charged,
                    settled.votes_freed,
                    settled.losers_charged,
                    settled.seats_credited
                ),
                Err(err) => println!(
                    "  could not settle the last dispute against {committer}: {err:#}"
                ),
            }
        }
        let result = if config.verdict_mode {
            process_treasurer_run_submit_audit_verdict(
                endpoint,
                authority,
                authority,
                run,
                coordinator_account,
                &config.run_id,
                &target_signer,
                index as u64,
                &signed,
                replayed_hash,
            )
            .await
        } else {
            // The run authority slashes as itself and names no reporter, so
            // the bounty stays in the vault (wienerlabs/leviathan-net#72).
            process_treasurer_run_slash(
                endpoint,
                authority,
                authority,
                run,
                coordinator_account,
                &config.run_id,
                &target_signer,
                None,
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
                report.submitted += 1;
            }
            Err(err) => {
                match already_on_chain(
                    endpoint,
                    coordinator_account,
                    run,
                    &target_signer,
                    &authority.pubkey(),
                )
                .await
                {
                    Ok(true) => {
                        println!(
                            "  {action} for {committer} refused, the chain already holds it: {err:#}"
                        );
                        judged.insert(committer);
                        judged.insert(key.clone());
                        report.already_on_chain += 1;
                    },
                    Ok(false) => {
                        println!(
                            "  {action} submission failed for {committer} at index {index}, retrying next pass: {err:#}"
                        );
                        report.submission_failures += 1;
                    },
                    Err(check) => {
                        println!(
                            "  {action} submission failed for {committer} at index {index}, and the chain could not be read to tell why: {err:#}; {check:#}"
                        );
                        report.submission_failures += 1;
                    },
                }
            }
        }
    }

    // Then every dispute in the run is settled as far as nobody's consent is
    // needed: a conviction's bounty is credited to its voters only once the
    // target is charged and each seat paid, and a target nobody disputes again
    // would otherwise leave them unpaid (wienerlabs/leviathan-net#64). It runs
    // after this verifier's own judging, so a backlog never costs it a vote in
    // the round under way. Only a daemon that votes on verdicts sweeps them,
    // and a dry run sends nothing.
    if config.verdict_mode && !config.dry_run {
        match settle_open_disputes(endpoint, authority, run, coordinator_account).await {
            Ok(sweep) if sweep.is_quiet() => {},
            Ok(sweep) => {
                println!(
                    "[verifier-daemon] settled open disputes: {} open, {} conviction(s) finalised, {} charged, {} vote(s) freed, {} loser(s) charged, {} recipient(s) credited",
                    sweep.open,
                    sweep.convictions_finalized,
                    sweep.convictions_charged,
                    sweep.votes_freed,
                    sweep.losers_charged,
                    sweep.seats_credited
                );
                for (target, err) in &sweep.failures {
                    println!("[verifier-daemon] could not settle the dispute against {target}: {err:#}");
                }
            },
            Err(err) => println!("[verifier-daemon] could not look up the run's open disputes: {err:#}"),
        }
    }

    // A slash no verdict accounts for, from the run authority or an
    // equivocation proof, is charged against its target once its epoch is,
    // and its reporter credited (wienerlabs/leviathan-net#72). A dry run sends
    // nothing.
    if !config.dry_run {
        match settle_slash_reports(endpoint, authority, run, coordinator_account).await {
            Ok(sweep) if sweep.charged == 0 && sweep.credited == 0 && sweep.failures.is_empty() => {},
            Ok(sweep) => {
                println!(
                    "[verifier-daemon] settled slash reports: {} open, {} charged, {} reporter(s) credited, {} reporter(s) with no account to credit yet",
                    sweep.open, sweep.charged, sweep.credited, sweep.awaiting_reporter
                );
                for (target, epoch_start, err) in &sweep.failures {
                    println!("[verifier-daemon] could not settle the slash of {target} in the epoch from {epoch_start}: {err:#}");
                }
            },
            Err(err) => println!("[verifier-daemon] could not look up the run's slash reports: {err:#}"),
        }
    }

    Ok(report)
}
