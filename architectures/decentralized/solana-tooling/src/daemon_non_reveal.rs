//! A verifier daemon's vote that a target did not reveal
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! The verifier's node asks each payload's trainer it audits for the state
//! the payload commits to, and keeps it beside the payload's dump. A payload
//! the node set out to fetch the state of, whose state is still not there
//! once the run's reveal window has passed in a witness phase, is voted a
//! non-reveal. Only this daemon's own audits are: its node fetches no other,
//! and a missing state there says nothing of its trainer.

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::check_commitment;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::CommitmentCheck;
use psyche_coordinator::select_audits_for_current_round;
use psyche_coordinator::Coordinator;
use psyche_coordinator::RunState;
use psyche_core::BatchId;
use psyche_network::previous_round_path;
use psyche_network::revealed_state_path;
use psyche_solana_treasurer::find_reveal_window;
use psyche_solana_treasurer::state::committee_matches_roster;
use psyche_solana_treasurer::state::RevealWindow;
use psyche_solana_treasurer::state::VerdictKind;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use crate::daemon::epoch_is_over;
use crate::daemon::parse_committer;
use crate::daemon::signed_commitment_of;
use crate::daemon::AuditConfig;
use crate::daemon::Convicting;
use crate::daemon::Conviction;

/// The key a payload's non-reveal is settled under.
pub fn non_reveal_key(key: &str) -> String {
    format!("non-reveal {key}")
}

/// The payloads the round's audit lottery assigns `verifier`, as the
/// committer named in a dump and the batch it trained: the ones its node
/// fetches the states of.
pub fn own_audits(
    coordinator: &Coordinator,
    verifier: &Pubkey,
) -> HashSet<(String, BatchId)> {
    if !committee_matches_roster(coordinator) {
        return HashSet::new();
    }
    match select_audits_for_current_round(coordinator) {
        Ok(audits) => audits
            .iter()
            .filter(|audit| audit.verifier.signer() == &verifier.to_bytes())
            .map(|audit| {
                (
                    bs58::encode(audit.target.signer()).into_string(),
                    audit.batch_id,
                )
            })
            .collect(),
        Err(_) => HashSet::new(),
    }
}

/// Whether the run's reveal window of `reveal_seconds` has passed, at
/// `now_unix`, in the phase the coordinator is in, as the treasurer judges
/// it: a witness phase, or the cooldown after the epoch's final round.
pub fn reveal_window_passed(
    coordinator: &Coordinator,
    reveal_seconds: u64,
    now_unix: i64,
) -> bool {
    matches!(
        coordinator.run_state,
        RunState::RoundWitness | RunState::Cooldown
    ) && now_unix >= 0
        && now_unix as u64
            >= coordinator
                .run_state_start_unix_timestamp
                .saturating_add(reveal_seconds)
}

/// What a non-reveal pass works with: the chain and the daemon's key, the run
/// as the pass read it, and what the pass reports and settles.
pub struct NonRevealPass<'a> {
    pub endpoint: &'a mut ToolboxEndpoint,
    pub authority: &'a Keypair,
    pub coordinator_account: &'a Pubkey,
    pub run: &'a Pubkey,
    pub config: &'a AuditConfig,
    pub coordinator: &'a Coordinator,
    pub report: &'a mut PassReport,
    pub judged: &'a mut HashSet<String>,
}

impl NonRevealPass<'_> {
    /// Votes a non-reveal for each of this daemon's own audits whose state
    /// its node set out to fetch and did not get within the run's window.
    /// Nothing is voted in a run with no window, or before it has passed.
    pub async fn vote_all(&mut self, submitted: &BTreeMap<String, PathBuf>) {
        let window = match self
            .endpoint
            .get_account(&find_reveal_window(self.run))
            .await
        {
            Ok(Some(account)) => {
                match RevealWindow::from_pda_account(
                    &account.owner,
                    &account.data,
                ) {
                    Ok(window) => window,
                    Err(err) => {
                        println!("[verifier-daemon] the run's reveal window does not read: {err:#}");
                        return;
                    },
                }
            },
            Ok(None) => None,
            Err(err) => {
                println!("[verifier-daemon] cannot read the run's reveal window: {err:#}");
                return;
            },
        };
        let Some(window) = window else {
            return;
        };
        let now = match self.endpoint.get_sysvar_clock().await {
            Ok(clock) => clock.unix_timestamp,
            Err(err) => {
                println!(
                    "[verifier-daemon] cannot read the chain's clock: {err:#}"
                );
                return;
            },
        };
        if !reveal_window_passed(self.coordinator, window.reveal_seconds, now) {
            return;
        }
        let audits = own_audits(self.coordinator, &self.authority.pubkey());
        for (key, path) in submitted {
            let settled_key = non_reveal_key(key);
            if self.judged.contains(&settled_key) {
                continue;
            }
            let Some(conviction) = self.check(key, &settled_key, path, &audits)
            else {
                continue;
            };
            self.report.fraud_found += 1;
            Convicting {
                endpoint: &mut *self.endpoint,
                authority: self.authority,
                coordinator_account: self.coordinator_account,
                run: self.run,
                config: self.config,
                report: &mut *self.report,
                judged: &mut *self.judged,
            }
            .submit(&conviction)
            .await;
        }
    }

    /// The non-reveal to vote for the payload at `path`, if it is one of
    /// `audits`, this epoch's, and its node set out to fetch its state and
    /// did not get it. Settles what is never to be voted.
    fn check(
        &mut self,
        key: &str,
        settled_key: &str,
        path: &Path,
        audits: &HashSet<(String, BatchId)>,
    ) -> Option<Conviction> {
        let (Some(dump), Some(committer)) =
            (parse_assignment_key(key), parse_committer(path))
        else {
            return None;
        };
        if !audits.contains(&(committer.clone(), dump.batch_id)) {
            return None;
        }
        // Revealed: nothing to vote, for good.
        if revealed_state_path(path).exists() {
            self.judged.insert(settled_key.to_string());
            return None;
        }
        // Its node never set out to fetch it, so its absence says nothing of
        // the trainer.
        if !previous_round_path(path).exists() {
            return None;
        }
        let coordinator = self.coordinator;
        let index =
            coordinator.epoch_state.clients.iter().position(|client| {
                bs58::encode(client.id.signer()).into_string() == committer
            })?;
        let target = coordinator.epoch_state.clients.iter().nth(index)?.id;
        let run_id = String::from(&coordinator.run_id);
        let commitment = match check_commitment(
            path,
            &run_id,
            target.p2p_identity(),
        ) {
            Ok(CommitmentCheck::Verified(commitment)) => commitment,
            Ok(CommitmentCheck::Missing) | Ok(CommitmentCheck::Invalid(_)) => {
                self.judged.insert(settled_key.to_string());
                return None;
            },
            Err(err) => {
                println!(
                    "[verifier-daemon] retry {key} committer {committer}: cannot read its commitment yet: {err:#}"
                );
                return None;
            },
        };
        let record = commitment.record;
        if record.state_digest.is_none() {
            self.judged.insert(settled_key.to_string());
            return None;
        }
        let signed = signed_commitment_of(
            coordinator,
            &commitment,
            *target.p2p_identity(),
        );
        if !signed.evidence.is_from_current_epoch(coordinator) {
            if epoch_is_over(coordinator, &record) {
                self.judged.insert(settled_key.to_string());
            }
            return None;
        }
        println!(
            "[verifier-daemon] FRAUD {key} committer {committer}: the state its payload commits to was not revealed within the run's window"
        );
        Some(Conviction {
            key: settled_key.to_string(),
            committer,
            target: Pubkey::new_from_array(*target.signer()),
            index,
            signed,
            replayed_hash: [0; 32],
            kind: VerdictKind::NonReveal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_passes_in_a_witness_phase_or_cooldown_only() {
        let mut coordinator: Coordinator = bytemuck::Zeroable::zeroed();
        coordinator.run_state_start_unix_timestamp = 100;
        for state in [RunState::RoundWitness, RunState::Cooldown] {
            coordinator.run_state = state;
            assert!(!reveal_window_passed(&coordinator, 4, 103));
            assert!(reveal_window_passed(&coordinator, 4, 104));
            assert!(reveal_window_passed(&coordinator, 4, 500));
            assert!(!reveal_window_passed(&coordinator, 4, -1));
        }
        for state in [
            RunState::RoundTrain,
            RunState::Warmup,
            RunState::WaitingForMembers,
            RunState::Paused,
            RunState::Finished,
        ] {
            coordinator.run_state = state;
            assert!(!reveal_window_passed(&coordinator, 4, 500), "{state:?}");
        }
    }
}
