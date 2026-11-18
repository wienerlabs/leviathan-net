//! A verifier fetches, from its trainer, the state each payload it audits
//! commits to, and keeps it beside the payload's dump for the verifier daemon
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use psyche_coordinator::{Coordinator, select_audits_for_current_round};
use psyche_core::{BatchId, NodeIdentity};
use psyche_network::{
    CommitmentRecord, Endpoint, EndpointId, commitment_record_path, reveal_round_state,
    revealed_state_path,
};
use tracing::{debug, info, warn};

/// The payloads this node is to audit in the coordinator's current round:
/// those the round's audit lottery assigns it, by committer and batch, drawn
/// as the verifier daemon and the treasurer draw them. None while the roster
/// has moved under the round's draw, as the treasurer then refuses every vote
/// (`committee_matches_roster` in the treasurer).
pub(crate) fn own_audits(
    state: &Coordinator,
    identity: &NodeIdentity,
) -> HashSet<(NodeIdentity, BatchId)> {
    let drawn_over_roster = state
        .current_round()
        .is_some_and(|round| round.clients_len as usize == state.epoch_state.clients.len());
    if !drawn_over_roster {
        return HashSet::new();
    }
    match select_audits_for_current_round(state) {
        Ok(audits) => audits
            .into_iter()
            .filter(|audit| audit.verifier == *identity)
            .map(|audit| (audit.target, audit.batch_id))
            .collect(),
        Err(err) => {
            warn!("No audit assignments this round: {err}");
            HashSet::new()
        }
    }
}

/// What a verifier needs to fetch the state a payload it audits commits to.
#[derive(Debug, Clone)]
pub(crate) struct StateToReveal {
    pub endpoint: Endpoint,
    pub trainer: EndpointId,
    pub digest: [u8; 32],
    /// The length of a state of the run's model, as this node trains it.
    pub expected_len: u64,
    /// The payload's dump, which the state is kept beside.
    pub dump: PathBuf,
    pub retry_every: Duration,
    pub time_to_reveal: Duration,
}

impl StateToReveal {
    /// How long a verifier asks for the state of a payload trained in the
    /// epoch `state` is in: until the epoch is over, as the treasurer convicts
    /// only on a commitment from it. It asks again every round witness time,
    /// the span a round's results are broadcast in.
    pub fn windows(state: &Coordinator) -> (Duration, Duration) {
        let retry_every = Duration::from_secs(state.config.round_witness_time.max(1));
        let epoch_over = state.epoch_state.start_timestamp
            + state.config.epoch_time
            + state.config.cooldown_time;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(epoch_over);
        (
            retry_every,
            Duration::from_secs(epoch_over.saturating_sub(now)),
        )
    }
}

/// Whether the commitment record beside the dump at `dump` names the state
/// `digest`: the payload kept under that name is the one whose state is
/// about to be fetched. A payload that lost its name to another the same
/// committer signed for the batch is kept as an equivocation, not as this.
fn record_names(dump: &Path, digest: &[u8; 32]) -> bool {
    let record_path = commitment_record_path(dump);
    match std::fs::read(&record_path)
        .ok()
        .and_then(|bytes| CommitmentRecord::from_bytes(&bytes).ok())
    {
        Some(record) => record.state_digest.as_ref() == Some(digest),
        None => false,
    }
}

/// Fetches from its trainer the state a payload this node audits commits to,
/// beside the payload's dump (`revealed_state_path`), asking until the
/// payload's epoch is over.
pub(crate) async fn reveal_audited_state(reveal: StateToReveal) {
    let dump = reveal.dump.display().to_string();
    if !record_names(&reveal.dump, &reveal.digest) {
        debug!(
            dump,
            "Not revealing a state for a payload not kept under its name"
        );
        return;
    }
    let dest = revealed_state_path(&reveal.dump);
    match reveal_round_state(
        &reveal.endpoint,
        reveal.trainer,
        reveal.digest,
        reveal.expected_len,
        &dest,
        reveal.retry_every,
        reveal.time_to_reveal,
    )
    .await
    {
        Ok(attempts) => info!(
            dump,
            attempts, "Revealed the state an audited payload was trained from"
        ),
        Err(err) => warn!(
            dump,
            "The state an audited payload was trained from was not revealed: {err}"
        ),
    }
}
