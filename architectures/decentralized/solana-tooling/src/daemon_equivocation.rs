//! The verifier daemon's side of equivocation (wienerlabs/leviathan-net#46).
//!
//! A client keeps the first payload a peer signs for a batch and, beside it,
//! the record of a second, different payload the peer signed for the same
//! batch. The two signatures convict the peer on their own, with no replay
//! and no committee. So every daemon looks for them in every pass, whatever
//! batches its committee assigns it, and submits the proof. The first proof
//! to land ejects the peer, and the daemons after it find the peer ejected.
//!
//! A proof is taken only from a reporter that committed to it in an earlier
//! slot, so that nobody who sees it can copy it and land it as its own
//! (wienerlabs/leviathan-net#92). A daemon commits to a proof in one pass and
//! reveals it in the next.

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use anchor_lang::AccountDeserialize;
use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::check_commitment;
use leviathan_verifier::check_equivocation;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::CommitmentCheck;
use leviathan_verifier::DumpKey;
use leviathan_verifier::EquivocationCheck;
use leviathan_verifier::VerifiedCommitment;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_network::equivocation_record_path;
use psyche_solana_treasurer::find_equivocation_commitment;
use psyche_solana_treasurer::state::conviction_landing;
use psyche_solana_treasurer::state::EquivocationCommitment;
use psyche_solana_treasurer::state::Landing;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use crate::daemon::hex8;
use crate::daemon::parse_committer;
use crate::daemon::signed_commitment_of;
use crate::daemon::target_ejected;
use crate::daemon::AuditConfig;
use crate::equivocation::equivocation_commitment;
use crate::equivocation::process_treasurer_run_close_equivocation_commitment;
use crate::equivocation::process_treasurer_run_commit_equivocation;
use crate::equivocation::process_treasurer_run_slash_equivocation;
use crate::signed_commitment::SignedCommitment;

/// The key a pass settles a payload's equivocation record under. Auditing
/// the payload and convicting on its record are settled apart, so judging
/// one does not stop the other.
pub fn equivocation_key(key: &str) -> String {
    format!("equivocation:{key}")
}

/// What became of an equivocation record in a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Judgement {
    /// Nothing more can come of it this session: the proof was put on chain,
    /// held back, found already landed or past landing, or found to prove
    /// nothing.
    Settled,
    /// A file or the chain could not be read, the proof is ahead of this
    /// pass's view of the run, or the submission failed. It is judged again on
    /// the next pass.
    Retry,
    /// The record is there and proves nothing. A node writes one only after
    /// checking the second payload's signature, so one that does not hold was
    /// damaged or tampered with on disk. Settled, and counted as malformed so
    /// it is not lost in the log.
    Unproven,
}

/// Where a payload's round stands against the run as a pass read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Round {
    /// The epoch the coordinator is in, at a step it has reached.
    Current,
    /// An epoch that is over, or a step before this epoch's first: nothing
    /// can convict on it now.
    Over,
    /// Work the pass's view has not reached: the run moved on after it was
    /// read, or the RPC lags the node that kept the payload.
    Ahead,
}

fn round_of(coordinator: &Coordinator, dump: &DumpKey) -> Round {
    let epoch_start = coordinator.epoch_state.start_timestamp;
    match dump.epoch_start.cmp(&epoch_start) {
        std::cmp::Ordering::Less => Round::Over,
        std::cmp::Ordering::Greater => Round::Ahead,
        std::cmp::Ordering::Equal
            if dump.step < coordinator.epoch_state.start_step =>
        {
            Round::Over
        },
        std::cmp::Ordering::Equal if dump.step > coordinator.progress.step => {
            Round::Ahead
        },
        std::cmp::Ordering::Equal => Round::Current,
    }
}

/// A committer's two commitments to one round and batch, proven to differ and
/// both signed by its roster key.
struct Proof<'a> {
    key: &'a str,
    committer: &'a str,
    signer: Pubkey,
    /// Where the committer sits on the roster, or `None` once a round's end
    /// moved it to the clients that left.
    index: Option<u64>,
    state: ClientState,
    first: SignedCommitment,
    second: SignedCommitment,
}

/// What an equivocation pass works with: the chain and the daemon's key, the
/// run as the pass read it, and what the pass reports and settles.
pub struct EquivocationPass<'a> {
    pub endpoint: &'a mut ToolboxEndpoint,
    pub authority: &'a Keypair,
    pub coordinator_account: &'a Pubkey,
    pub run: &'a Pubkey,
    pub config: &'a AuditConfig,
    pub coordinator: &'a Coordinator,
    pub report: &'a mut PassReport,
    pub judged: &'a mut HashSet<String>,
}

impl EquivocationPass<'_> {
    /// Convicts, for every kept payload with an equivocation record beside
    /// it, the peer that signed both. Each record is settled once, under
    /// `equivocation_key`. A committer convicted here is settled too, so the
    /// audit after this pass does not convict it a second time this session.
    pub async fn convict_all(&mut self, submitted: &BTreeMap<String, PathBuf>) {
        for (key, path) in submitted {
            let settled_key = equivocation_key(key);
            if self.judged.contains(&settled_key) {
                continue;
            }
            // Only a stat per payload, so the dump is read only where a
            // record says there is something to prove.
            match equivocation_record_path(path).try_exists() {
                Ok(true) => {},
                Ok(false) => continue,
                Err(err) => {
                    println!(
                        "[verifier-daemon] retry {key}: cannot tell whether it has an equivocation record: {err}"
                    );
                    continue;
                },
            }
            if self.judge(key, path).await == Judgement::Settled {
                self.judged.insert(settled_key);
            }
        }
    }

    async fn judge(&mut self, key: &str, path: &Path) -> Judgement {
        let (Some(dump), Some(committer)) =
            (parse_assignment_key(key), parse_committer(path))
        else {
            return Judgement::Settled;
        };
        match round_of(self.coordinator, &dump) {
            Round::Current => {},
            // Counted nowhere: a daemon restarted with old records on disk
            // would otherwise report every one of them again.
            Round::Over => {
                println!(
                    "[verifier-daemon] skip  {key} committer {committer}: its equivocation record is from a round that is over"
                );
                return Judgement::Settled;
            },
            Round::Ahead => return Judgement::Retry,
        }
        match self.prove(key, &committer, path, &dump) {
            Ok(proof) => {
                self.report.equivocations += 1;
                self.submit(&proof).await
            },
            Err(Judgement::Unproven) => {
                self.report.malformed += 1;
                Judgement::Settled
            },
            Err(judgement) => judgement,
        }
    }

    /// Reads and checks the equivocation record beside the payload at
    /// `path`, kept under `dump`, against the roster key of `committer`.
    fn prove<'k>(
        &self,
        key: &'k str,
        committer: &'k str,
        path: &Path,
        dump: &DumpKey,
    ) -> Result<Proof<'k>, Judgement> {
        let skip = |reason: String| {
            println!(
                "[verifier-daemon] skip  {key} committer {committer}: its equivocation record proves nothing: {reason}"
            );
            Judgement::Unproven
        };
        let retry = |err: anyhow::Error| {
            println!(
                "[verifier-daemon] retry {key} committer {committer}: cannot read its equivocation record yet: {err:#}"
            );
            Judgement::Retry
        };
        // The committer trained in this epoch, so it is on its roster, or
        // among the clients a round's end moved off it.
        let coordinator = self.coordinator;
        let roster = coordinator.epoch_state.clients.iter().enumerate();
        let exited = coordinator.epoch_state.exited_clients.iter();
        let Some((index, client)) = roster
            .map(|(index, client)| (Some(index as u64), client))
            .chain(exited.map(|client| (None, client)))
            .find(|(_, client)| {
                bs58::encode(client.id.signer()).into_string() == committer
            })
        else {
            return Err(skip("its committer is not in this epoch".into()));
        };
        let p2p = *client.id.p2p_identity();
        let run_id = String::from(&coordinator.run_id);
        let kept = match check_commitment(path, &run_id, &p2p) {
            Ok(CommitmentCheck::Verified(kept)) => kept,
            Ok(CommitmentCheck::Missing) => {
                return Err(skip(
                    "no commitment is kept with the payload".into(),
                ))
            },
            Ok(CommitmentCheck::Invalid(reason)) => return Err(skip(reason)),
            Err(err) => return Err(retry(err)),
        };
        let record = kept.record;
        if (record.epoch_start, record.step, record.batch_id)
            != (dump.epoch_start, dump.step, dump.batch_id)
        {
            return Err(skip(
                "the kept commitment is for other work than its name".into(),
            ));
        }
        let second = match check_equivocation(path, &kept, &run_id, &p2p) {
            Ok(EquivocationCheck::Proven(second)) => VerifiedCommitment {
                record: second.record,
                results_digest: second.results_digest,
            },
            // Removed between the look and the read: nothing to judge.
            Ok(EquivocationCheck::None) => return Err(Judgement::Settled),
            Ok(EquivocationCheck::Invalid(reason)) => return Err(skip(reason)),
            Err(err) => return Err(retry(err)),
        };
        Ok(Proof {
            key,
            committer,
            signer: Pubkey::new_from_array(*client.id.signer()),
            index,
            state: client.state,
            first: signed_commitment_of(coordinator, &kept, p2p),
            second: signed_commitment_of(coordinator, &second, p2p),
        })
    }

    /// Puts `proof` on chain, unless the chain already ejected its committer
    /// or can no longer charge it, or the daemon runs dry.
    async fn submit(&mut self, proof: &Proof<'_>) -> Judgement {
        let Proof { key, committer, .. } = proof;
        println!(
            "[verifier-daemon] EQUIVOCATION {key} committer {committer} signed {} and {} for one batch",
            hex8(&proof.first.committed_hash()),
            hex8(&proof.second.committed_hash()),
        );
        if proof.state == ClientState::Ejected {
            println!("  {committer} is already ejected, the chain holds a conviction of it");
            self.report.already_on_chain += 1;
            self.judged.insert(committer.to_string());
            self.withdraw_commitment(proof).await;
            return Judgement::Settled;
        }
        // The treasurer lands the proof only as an ejection the epoch will
        // charge. One it cannot land now it can never land in this epoch:
        // the committer withdrew or left the roster, or the epoch stopped
        // charging. That is a fraud the chain never saw, counted once.
        let index = match proof.index.map(|index| {
            conviction_landing(self.coordinator, &proof.signer, index)
        }) {
            Some(Ok(Landing::Eject(index))) => index,
            _ => {
                println!(
                    "  equivocation by {committer} can no longer be charged in this epoch: it is {}, and the run is in {}",
                    proof.state, self.coordinator.run_state
                );
                self.report.submission_failures += 1;
                self.withdraw_commitment(proof).await;
                return Judgement::Settled;
            },
        };
        if self.config.dry_run {
            println!(
                "  [dry-run] would submit equivocation for {committer} at epoch index {index}"
            );
            self.report.withheld += 1;
            return Judgement::Settled;
        }
        // The proof is the daemon's only once it committed to it in an
        // earlier slot: commit now, and reveal on a later pass, once the slot
        // has moved on.
        match self.committed(proof).await {
            Ok(Some(commitment)) => {
                match self.endpoint.get_sysvar_clock().await {
                    Ok(clock) if commitment.revealable_in(clock.slot) => {},
                    Ok(_) => {
                        println!(
                        "  the commitment to the equivocation proof for {committer} is from this slot, revealing it next pass"
                    );
                        return Judgement::Retry;
                    },
                    Err(err) => {
                        println!(
                        "  could not read the slot to reveal the equivocation proof for {committer}, retrying next pass: {err:#}"
                    );
                        self.report.submission_failures += 1;
                        return Judgement::Retry;
                    },
                }
            },
            Ok(None) => return self.commit(proof).await,
            Err(err) => {
                println!(
                    "  could not read the commitment to the equivocation proof for {committer}, retrying next pass: {err:#}"
                );
                self.report.submission_failures += 1;
                return Judgement::Retry;
            },
        }
        let result = process_treasurer_run_slash_equivocation(
            self.endpoint,
            self.authority,
            self.authority,
            self.run,
            self.coordinator_account,
            &self.config.run_id,
            &proof.signer,
            index,
            &proof.first,
            &proof.second,
        )
        .await;
        match result {
            Ok(()) => {
                println!(
                    "  equivocation submitted for {committer} at epoch index {index}"
                );
                self.report.submitted += 1;
                self.judged.insert(committer.to_string());
                Judgement::Settled
            },
            Err(err) => self.refused(proof, index, err).await,
        }
    }

    /// The daemon's commitment to `proof`, the hash it reveals it against.
    fn commitment(&self, proof: &Proof<'_>) -> [u8; 32] {
        equivocation_commitment(
            self.run,
            &self.authority.pubkey(),
            &proof.signer,
            &proof.first,
            &proof.second,
        )
    }

    /// The daemon's commitment to `proof` on chain, if it made one.
    async fn committed(
        &mut self,
        proof: &Proof<'_>,
    ) -> anyhow::Result<Option<EquivocationCommitment>> {
        let address = find_equivocation_commitment(
            self.run,
            &self.authority.pubkey(),
            &self.commitment(proof),
        );
        let Some(account) = self.endpoint.get_account(&address).await? else {
            return Ok(None);
        };
        let commitment =
            EquivocationCommitment::try_deserialize(&mut &account.data[..])
                .map_err(|error| {
                    anyhow::anyhow!(
                        "Unable to decode the equivocation commitment {address}: {error}"
                    )
                })?;
        Ok(Some(commitment))
    }

    /// Commits to `proof`, to reveal it on a later pass, once the commitment
    /// is on chain from a slot before anyone could see the proof.
    async fn commit(&mut self, proof: &Proof<'_>) -> Judgement {
        let committer = proof.committer;
        match process_treasurer_run_commit_equivocation(
            self.endpoint,
            self.authority,
            self.authority,
            self.run,
            &proof.signer,
            &proof.first,
            &proof.second,
        )
        .await
        {
            Ok(_) => {
                println!(
                    "  committed to the equivocation proof for {committer}, to reveal it next pass"
                );
                Judgement::Retry
            },
            Err(err) => {
                println!(
                    "  could not commit to the equivocation proof for {committer}, retrying next pass: {err:#}"
                );
                self.report.submission_failures += 1;
                Judgement::Retry
            },
        }
    }

    /// Closes the daemon's commitment to `proof`, which another reporter's
    /// proof made moot, so its rent comes back. A failure costs only the rent.
    async fn withdraw_commitment(&mut self, proof: &Proof<'_>) {
        let commitment = self.commitment(proof);
        match self.committed(proof).await {
            Ok(Some(_)) => {},
            Ok(None) => return,
            Err(err) => {
                println!(
                    "  could not read the commitment to the equivocation proof for {}: {err:#}",
                    proof.committer
                );
                return;
            },
        }
        if let Err(err) = process_treasurer_run_close_equivocation_commitment(
            self.endpoint,
            self.authority,
            self.authority,
            self.run,
            &commitment,
        )
        .await
        {
            println!(
                "  could not close the commitment to the equivocation proof for {}: {err:#}",
                proof.committer
            );
        }
    }

    /// What a refused submission of `proof` comes to: the chain already
    /// ejected its committer, or a failure retried on the next pass.
    async fn refused(
        &mut self,
        proof: &Proof<'_>,
        index: u64,
        err: anyhow::Error,
    ) -> Judgement {
        let committer = proof.committer;
        match target_ejected(
            self.endpoint,
            self.coordinator_account,
            &proof.signer,
        )
        .await
        {
            Ok(true) => {
                println!(
                    "  equivocation for {committer} refused, the chain already ejected it: {err:#}"
                );
                self.report.already_on_chain += 1;
                self.judged.insert(committer.to_string());
                self.withdraw_commitment(proof).await;
                Judgement::Settled
            },
            Ok(false) => {
                println!(
                    "  equivocation submission failed for {committer} at index {index}, retrying next pass: {err:#}"
                );
                self.report.submission_failures += 1;
                Judgement::Retry
            },
            Err(check) => {
                println!(
                    "  equivocation submission failed for {committer} at index {index}, and the chain could not be read to tell why: {err:#}; {check:#}"
                );
                self.report.submission_failures += 1;
                Judgement::Retry
            },
        }
    }
}
