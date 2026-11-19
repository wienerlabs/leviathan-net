//! The continuity check of a verifier daemon's pass
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! A round replayed from the state its trainer revealed, with what the round
//! after applied first applied as this verifier's node applied it, leaves
//! the state the trainer's next round trains from. The trainer's payload of
//! that round commits to the state it trained from: one that names another
//! was trained from a state its trainer's last round did not leave, and is
//! convicted. Its round replayed from that other state would pass the band,
//! so continuity is what catches it.

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::check_commitment;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::revealed::next_round;
use leviathan_verifier::revealed::revealed_round;
use leviathan_verifier::revealed::Continuity;
use leviathan_verifier::CommitmentCheck;
use leviathan_verifier::ReplayWork;
use psyche_coordinator::Coordinator;
use psyche_solana_treasurer::state::VerdictKind;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_toolbox_endpoint::ToolboxEndpoint;

use crate::daemon::epoch_is_over;
use crate::daemon::hex8;
use crate::daemon::parse_committer;
use crate::daemon::signed_commitment_of;
use crate::daemon::AuditConfig;
use crate::daemon::Convicting;
use crate::daemon::Conviction;

/// The key a payload's continuity check is settled under, apart from the
/// judgement of the payload itself.
pub fn continuity_key(key: &str) -> String {
    format!("continuity {key}")
}

/// What a continuity pass works with: the chain and the daemon's key, the
/// run as the pass read it, the replay, and what the pass reports and
/// settles.
pub struct ContinuityPass<'a> {
    pub endpoint: &'a mut ToolboxEndpoint,
    pub authority: &'a Keypair,
    pub coordinator_account: &'a Pubkey,
    pub run: &'a Pubkey,
    pub config: &'a AuditConfig,
    pub coordinator: &'a Coordinator,
    pub replay: &'a (dyn ReplayWork + Sync),
    pub report: &'a mut PassReport,
    pub judged: &'a mut HashSet<String>,
}

enum Check {
    /// Checked for good: held, or never checkable here.
    Settled,
    /// Looked at again on the next pass.
    Retry,
    /// The trainer's next payload commits to a state its round did not
    /// leave.
    Broken(Box<Conviction>),
}

impl ContinuityPass<'_> {
    /// Checks every kept payload whose state was revealed, each settled once
    /// under `continuity_key`.
    pub async fn check_all(&mut self, submitted: &BTreeMap<String, PathBuf>) {
        for (key, path) in submitted {
            let settled_key = continuity_key(key);
            if self.judged.contains(&settled_key) {
                continue;
            }
            match self.check(key, &settled_key, path) {
                Check::Settled => {
                    self.judged.insert(settled_key);
                },
                Check::Retry => {},
                Check::Broken(conviction) => {
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
                },
            }
        }
    }

    fn check(&mut self, key: &str, settled_key: &str, path: &Path) -> Check {
        let (Some(dump), Some(committer)) =
            (parse_assignment_key(key), parse_committer(path))
        else {
            return Check::Settled;
        };
        let coordinator = self.coordinator;
        // Only this epoch's work convicts anyone; work newer than this
        // pass's view of the run is looked at again.
        let epoch_start = coordinator.epoch_state.start_timestamp;
        if dump.epoch_start != epoch_start {
            return match dump.epoch_start < epoch_start {
                true => Check::Settled,
                false => Check::Retry,
            };
        }
        // A payload whose state is not revealed has nothing to replay, until
        // it is.
        let round = match revealed_round(path) {
            Ok(Some(round)) => round,
            Ok(None) => return Check::Retry,
            Err(err) => {
                println!(
                    "[verifier-daemon] skip  {key} committer {committer}: its round cannot be replayed to check what it leaves: {err}"
                );
                return Check::Settled;
            },
        };
        let next = match next_round(path) {
            Continuity::Ready(next) => next,
            Continuity::Pending => return Check::Retry,
            Continuity::Unchecked(why) => {
                println!(
                    "[verifier-daemon] skip  {key} committer {committer}: what its round leaves cannot be checked here: {why}"
                );
                return Check::Settled;
            },
        };
        let Some(index) =
            coordinator.epoch_state.clients.iter().position(|client| {
                bs58::encode(client.id.signer()).into_string() == committer
            })
        else {
            println!(
                "[verifier-daemon] skip  {key} committer {committer} is not in the epoch roster, nothing to convict"
            );
            return Check::Settled;
        };
        let target = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(index)
            .expect("index was just found by position")
            .id;

        // The trainer's next payload is what a broken continuity convicts,
        // so what it signed for it is checked first.
        let run_id = String::from(&coordinator.run_id);
        let commitment = match check_commitment(
            &next.payload,
            &run_id,
            target.p2p_identity(),
        ) {
            Ok(CommitmentCheck::Verified(commitment)) => commitment,
            Ok(CommitmentCheck::Missing) => {
                println!(
                        "[verifier-daemon] skip  {key} committer {committer}: its next payload has no signed commitment kept with it"
                    );
                return Check::Settled;
            },
            Ok(CommitmentCheck::Invalid(reason)) => {
                println!(
                        "[verifier-daemon] skip  {key} committer {committer}: its next payload's commitment does not hold: {reason}"
                    );
                return Check::Settled;
            },
            Err(err) => {
                println!(
                        "[verifier-daemon] retry {key} committer {committer}: cannot read its next payload's commitment yet: {err:#}"
                    );
                return Check::Retry;
            },
        };
        let record = commitment.record;
        if record.step != next.step
            || record.state_digest != Some(next.committed)
        {
            println!(
                "[verifier-daemon] retry {key} committer {committer}: its next payload's commitment changed under the check"
            );
            return Check::Retry;
        }
        let signed = signed_commitment_of(
            coordinator,
            &commitment,
            *target.p2p_identity(),
        );
        if !signed.evidence.is_from_current_epoch(coordinator) {
            return match epoch_is_over(coordinator, &record) {
                true => Check::Settled,
                false => Check::Retry,
            };
        }

        let left = match self.replay.replay_continuity(&dump, &round, &next) {
            Ok(left) => left,
            Err(err) => {
                println!(
                    "[verifier-daemon] skip  {key} committer {committer}: its round cannot be replayed on to the next: {err}"
                );
                self.report.unreplayable += 1;
                return Check::Retry;
            },
        };
        if left == next.committed {
            println!(
                "[verifier-daemon] ok    {key} committer {committer}: its round leaves the state its step {} trained from",
                next.step
            );
            return Check::Settled;
        }
        println!(
            "[verifier-daemon] FRAUD {key} committer {committer}: its round leaves the state {}, and its step {} committed to {}",
            hex8(&left),
            next.step,
            hex8(&next.committed)
        );
        Check::Broken(Box::new(Conviction {
            key: settled_key.to_string(),
            committer,
            target: Pubkey::new_from_array(*target.signer()),
            index,
            signed,
            replayed_hash: left,
            kind: VerdictKind::Replay,
        }))
    }
}
