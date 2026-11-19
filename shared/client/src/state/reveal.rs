//! A verifier fetches, from its trainer, the state each payload it audits
//! commits to, and keeps it beside the payload's dump for the verifier daemon
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use psyche_coordinator::{Coordinator, select_audits_for_current_round};
use psyche_core::{BatchId, NodeIdentity};
use psyche_network::{
    CommitmentRecord, Endpoint, EndpointId, PreviousRound, commitment_record_path,
    previous_round_path, reveal_round_state, revealed_state_path, round_training_path,
};

use super::round_state::RoundState;
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
    /// What the trainer did in the round before the payload's, kept beside it
    /// too: a replay needs it to know what the trainer's step took out of its
    /// buffer first.
    pub previous: PreviousRound,
    /// What the trainer trained in the payload's own round, kept beside it
    /// as well: a continuity check needs to know the payload's step is all
    /// the trainer's next state follows from.
    pub trained: PreviousRound,
    pub retry_every: Duration,
    pub time_to_reveal: Duration,
}

/// How often a verifier asks again for a state it has not got. Its daemon
/// votes a non-reveal once the run's reveal window has passed in a witness
/// phase, a window shorter than the witness phase, so an honest trainer's
/// state has to be asked for again well within it: the first request can
/// come before the trainer kept it (wienerlabs/leviathan-net#89).
pub(crate) const REVEAL_RETRY: Duration = Duration::from_secs(1);

impl StateToReveal {
    /// How long a verifier asks for the state of a payload trained in the
    /// epoch `state` is in: until the epoch is over, as the treasurer convicts
    /// only on a commitment from it. It asks again every `REVEAL_RETRY`.
    pub fn windows(state: &Coordinator) -> (Duration, Duration) {
        let retry_every = REVEAL_RETRY;
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

/// What this node kept of a round's draw: enough to say who trained what in
/// it.
#[derive(Debug, Clone)]
pub(crate) struct RoundDraw {
    height: u32,
    epoch_start: u64,
    data_assignments: BTreeMap<BatchId, NodeIdentity>,
}

impl RoundDraw {
    pub fn of(round: &RoundState) -> Self {
        Self {
            height: round.height,
            epoch_start: round.epoch_start,
            data_assignments: round.data_assignments.clone(),
        }
    }
}

/// The batches `trainer` was assigned in `round`, the round of a payload of
/// its this node audits. Unknown if the draw this node kept assigns it none.
pub(crate) fn round_training(round: &RoundState, trainer: &NodeIdentity) -> PreviousRound {
    let batches: Vec<BatchId> = round
        .data_assignments
        .iter()
        .filter(|(_, assignee)| *assignee == trainer)
        .map(|(batch, _)| *batch)
        .collect();
    match batches.is_empty() {
        true => PreviousRound::Unknown,
        false => PreviousRound::Trained(batches),
    }
}

/// What `trainer` did in `previous`, the round before `round`, as this node
/// saw the rounds drawn. A round that opens its epoch starts every node's
/// rounds afresh, so its step takes nothing out first; otherwise the trainer
/// takes out its results for the batches it was assigned there, if any.
pub(crate) fn previous_training(
    previous: &RoundDraw,
    round: &RoundState,
    trainer: &NodeIdentity,
) -> PreviousRound {
    if round.height == 0 {
        return PreviousRound::Idle;
    }
    if previous.epoch_start != round.epoch_start || previous.height + 1 != round.height {
        return PreviousRound::Unknown;
    }
    let batches: Vec<BatchId> = previous
        .data_assignments
        .iter()
        .filter(|(_, assignee)| *assignee == trainer)
        .map(|(batch, _)| *batch)
        .collect();
    match batches.is_empty() {
        true => PreviousRound::Idle,
        false => PreviousRound::Trained(batches),
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
    // Beside the dump before the state is, so a reader that finds the state
    // finds these too.
    let previous_path = previous_round_path(&reveal.dump);
    if let Err(err) = write_whole(&previous_path, reveal.previous.encode().as_bytes()).await {
        warn!(
            dump,
            "Cannot keep what an audited payload's trainer did the round before: {err}"
        );
        return;
    }
    let trained_path = round_training_path(&reveal.dump);
    if let Err(err) = write_whole(&trained_path, reveal.trained.encode().as_bytes()).await {
        warn!(
            dump,
            "Cannot keep what an audited payload's trainer trained in its round: {err}"
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

/// Writes `bytes` to `path` under a temporary name first, so a reader never
/// finds it partly written.
async fn write_whole(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut partial = path.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    tokio::fs::write(&partial, bytes).await?;
    tokio::fs::rename(&partial, path).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use psyche_core::ClosedInterval;

    fn identity(seed: u8) -> NodeIdentity {
        NodeIdentity::from_single_key([seed; 32])
    }

    fn batch(start: u64) -> BatchId {
        BatchId(ClosedInterval::new(start, start + 3))
    }

    fn round(height: u32, epoch_start: u64, assignments: &[(BatchId, NodeIdentity)]) -> RoundState {
        RoundState {
            height,
            epoch_start,
            data_assignments: assignments.iter().copied().collect::<BTreeMap<_, _>>(),
            ..RoundState::new()
        }
    }

    /// The batches the trainer was assigned in its payload's own round, and
    /// only its own; none assigned is a draw this node does not know.
    #[test]
    fn what_a_trainer_trained_in_its_round_is_its_own_batches() {
        let (trainer, other) = (identity(1), identity(2));
        let one = round(5, 90, &[(batch(0), other), (batch(4), trainer)]);
        assert_eq!(
            round_training(&one, &trainer),
            PreviousRound::Trained(vec![batch(4)])
        );
        let two = round(
            5,
            90,
            &[(batch(0), trainer), (batch(4), trainer), (batch(8), other)],
        );
        assert_eq!(
            round_training(&two, &trainer),
            PreviousRound::Trained(vec![batch(0), batch(4)])
        );
        let none = round(5, 90, &[(batch(0), other)]);
        assert_eq!(round_training(&none, &trainer), PreviousRound::Unknown);
    }

    /// The batches the trainer was assigned in the round before, and only
    /// its own.
    /// A state is asked for again every second, however long the run's
    /// witness phase, until the payload's epoch is over.
    #[test]
    fn a_state_is_asked_for_again_every_second_until_the_epoch_is_over() {
        let mut state: Coordinator = bytemuck::Zeroable::zeroed();
        state.config.round_witness_time = 30;
        state.config.epoch_time = 100;
        state.config.cooldown_time = 10;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        state.epoch_state.start_timestamp = now;
        let (retry_every, time_to_reveal) = StateToReveal::windows(&state);
        assert_eq!(retry_every, Duration::from_secs(1));
        assert!(
            (108..=110).contains(&time_to_reveal.as_secs()),
            "{time_to_reveal:?}"
        );
        state.epoch_state.start_timestamp = now - 200;
        assert_eq!(StateToReveal::windows(&state).1, Duration::ZERO);
    }

    #[test]
    fn a_trainer_that_trained_the_round_before_took_its_results_out() {
        let (trainer, other) = (identity(1), identity(2));
        let previous = round(4, 90, &[(batch(0), other), (batch(4), trainer)]);
        assert_eq!(
            previous_training(&RoundDraw::of(&previous), &round(5, 90, &[]), &trainer),
            PreviousRound::Trained(vec![batch(4)])
        );
    }

    /// A trainer not assigned in the round before, a verifier there, took
    /// nothing out.
    #[test]
    fn a_trainer_that_verified_the_round_before_took_nothing_out() {
        let previous = round(4, 90, &[(batch(0), identity(2))]);
        assert_eq!(
            previous_training(&RoundDraw::of(&previous), &round(5, 90, &[]), &identity(1)),
            PreviousRound::Idle
        );
    }

    /// The round that opens an epoch starts afresh, whatever was kept
    /// before it.
    #[test]
    fn the_first_round_of_an_epoch_takes_nothing_out() {
        let trainer = identity(1);
        let previous = round(7, 45, &[(batch(0), trainer)]);
        assert_eq!(
            previous_training(&RoundDraw::of(&previous), &round(0, 90, &[]), &trainer),
            PreviousRound::Idle
        );
    }

    /// A round before that is not the one before, of another epoch or
    /// height, says nothing.
    #[test]
    fn a_round_that_is_not_the_one_before_says_nothing() {
        let trainer = identity(1);
        let previous = round(3, 90, &[(batch(0), trainer)]);
        assert_eq!(
            previous_training(&RoundDraw::of(&previous), &round(5, 90, &[]), &trainer),
            PreviousRound::Unknown
        );
        let other_epoch = round(4, 45, &[(batch(0), trainer)]);
        assert_eq!(
            previous_training(&RoundDraw::of(&other_epoch), &round(5, 90, &[]), &trainer),
            PreviousRound::Unknown
        );
    }
}
