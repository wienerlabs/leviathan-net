//! A payload's round replayed from the state its trainer revealed
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! A verifier keeps, beside each payload it audits, the state the payload
//! commits to (`<dump>.state`) and what its trainer did the round before
//! (`<dump>.previous`). A node's step takes its own results of the round
//! before out of its buffer first; the values follow from the revealed buffer
//! and the indices of the trainer's payload of that round, which the verifier
//! kept with every other payload.

use std::path::{Path, PathBuf};

use psyche_network::{
    applied_results_path, commitment_record_path, payload_dump_name, previous_round_path,
    revealed_state_path, round_training_path, AppliedResults, CommitmentRecord, PreviousRound,
};
use thiserror::Error;

use crate::fidelity::{client_dump, ClientDump};

/// What a replay of a payload's round from its revealed state needs, beside
/// the payload's dump.
#[derive(Debug, Clone)]
pub struct RevealedRound {
    /// The state the payload commits to, as its trainer revealed it.
    pub state: PathBuf,
    /// The trainer's own payload of the round before, if it trained there.
    pub previous_payload: Option<PathBuf>,
}

/// Why a payload whose state was revealed cannot be replayed here. Each is
/// this verifier's gap, not the trainer's fault.
#[derive(Debug, Error)]
pub enum UnreplayableRound {
    #[error("this verifier did not see what the trainer did the round before")]
    UnknownPrevious,
    #[error("the trainer trained {0} batches the round before, and only the last one's values are its buffer's")]
    SeveralPrevious(usize),
    #[error("the trainer's payload of the round before is not kept here: {}", .0.display())]
    MissingPrevious(PathBuf),
    #[error("what was kept beside the payload does not read: {0}")]
    Unreadable(String),
}

/// What a replay of the payload dump at `dump` from its revealed state needs,
/// if its trainer revealed one: the state, and the trainer's payload of the
/// round before if it trained there, kept beside it. None without a
/// revealed state.
pub fn revealed_round(dump: &Path) -> Result<Option<RevealedRound>, UnreplayableRound> {
    let state = revealed_state_path(dump);
    if !state.exists() {
        return Ok(None);
    }
    let previous_path = previous_round_path(dump);
    let text = std::fs::read_to_string(&previous_path).map_err(|err| {
        UnreplayableRound::Unreadable(format!("{}: {err}", previous_path.display()))
    })?;
    let previous = PreviousRound::decode(&text).map_err(UnreplayableRound::Unreadable)?;
    let previous_payload = match previous {
        PreviousRound::Idle => None,
        PreviousRound::Unknown => return Err(UnreplayableRound::UnknownPrevious),
        PreviousRound::Trained(batches) if batches.len() > 1 => {
            return Err(UnreplayableRound::SeveralPrevious(batches.len()));
        }
        PreviousRound::Trained(batches) => {
            let dumped = client_dump(dump).ok_or_else(|| {
                UnreplayableRound::Unreadable(format!("{} is not a payload dump", dump.display()))
            })?;
            let step = dumped.key.step.checked_sub(1).ok_or_else(|| {
                UnreplayableRound::Unreadable("a first step has no round before".to_string())
            })?;
            let path = dump.with_file_name(payload_dump_name(
                &dumped.signer,
                dumped.key.epoch_start,
                step,
                batches[0],
            ));
            if !path.exists() {
                return Err(UnreplayableRound::MissingPrevious(path));
            }
            Some(path)
        }
    };
    Ok(Some(RevealedRound {
        state,
        previous_payload,
    }))
}

/// What a continuity check of a payload's round needs beside the replay of
/// the round: what the round after applied first, and the state the
/// trainer's payload of that round commits to (wienerlabs/leviathan-net#89).
/// The state a round leaves, once those results are applied, is the state
/// the trainer's next round trains from.
#[derive(Debug, Clone)]
pub struct NextRound {
    /// The step of the round after.
    pub step: u32,
    /// The payloads the round after applied first, in the order applied, as
    /// this verifier's own node applied them; None for a round that applied
    /// nothing (an epoch's first two).
    pub applied: Option<Vec<PathBuf>>,
    /// The trainer's payload of the round after.
    pub payload: PathBuf,
    /// The state that payload commits to.
    pub committed: [u8; 32],
}

/// Whether the round of a payload whose state was revealed can be checked
/// for continuity here.
#[derive(Debug, Clone)]
pub enum Continuity {
    /// The trainer's payload of the round after is not here yet.
    Pending,
    /// It can never be, here, and why: this verifier's gap, or a round that
    /// does not leave the next one's state on its own.
    Unchecked(String),
    Ready(NextRound),
}

/// Whether the round of the payload dump at `dump`, which `revealed_round`
/// found revealed, can be checked for continuity: its trainer trained that
/// batch alone in the round, its payload of the round after is kept here,
/// with a commitment to a state, and this verifier's node kept what it
/// applied at the start of that round, with every payload it names.
pub fn next_round(dump: &Path) -> Continuity {
    match next_round_or_gap(dump) {
        Ok(Some(next)) => Continuity::Ready(next),
        Ok(None) => Continuity::Pending,
        Err(gap) => Continuity::Unchecked(gap),
    }
}

fn next_round_or_gap(dump: &Path) -> Result<Option<NextRound>, String> {
    let dumped =
        client_dump(dump).ok_or_else(|| format!("{} is not a payload dump", dump.display()))?;
    let trained_path = round_training_path(dump);
    let trained = std::fs::read_to_string(&trained_path).map_err(|err| {
        format!(
            "what the trainer trained in its round is not kept: {}: {err}",
            trained_path.display()
        )
    })?;
    match PreviousRound::decode(&trained)? {
        PreviousRound::Trained(batches) if batches == [dumped.key.batch_id] => {}
        PreviousRound::Trained(batches) => {
            return Err(format!(
                "the trainer trained {} batches in the round, and its next state follows from all of them",
                batches.len()
            ));
        }
        other => {
            return Err(format!(
                "this verifier did not see what the trainer trained in the round: {other:?}"
            ))
        }
    }
    let dir = dump
        .parent()
        .ok_or_else(|| format!("{} is in no directory", dump.display()))?;
    let step = dumped
        .key
        .step
        .checked_add(1)
        .ok_or_else(|| "no step follows the last".to_string())?;
    let Some((payload, committed)) = next_payload(dir, &dumped, step)? else {
        return Ok(None);
    };
    let applied_path = applied_results_path(dir, dumped.key.epoch_start, step);
    let applied = std::fs::read_to_string(&applied_path).map_err(|err| {
        format!(
            "this verifier did not keep what the round at step {step} applied: {}: {err}",
            applied_path.display()
        )
    })?;
    let applied = match AppliedResults::decode(&applied)? {
        AppliedResults::Skipped => None,
        AppliedResults::Applied(names) => Some(
            names
                .iter()
                .map(|name| {
                    let path = dir.join(name);
                    match path.exists() {
                        true => Ok(path),
                        false => Err(format!(
                            "an applied payload is not kept here: {}",
                            path.display()
                        )),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    Ok(Some(NextRound {
        step,
        applied,
        payload,
        committed,
    }))
}

/// The trainer of `dumped`'s payload of the round at `step`, and the state it
/// commits to, if one is kept in `dir`. Every payload of a trainer's round
/// commits to the same state; the first by name is taken.
fn next_payload(
    dir: &Path,
    dumped: &ClientDump,
    step: u32,
) -> Result<Option<(PathBuf, [u8; 32])>, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|err| format!("cannot list {}: {err}", dir.display()))?;
    let mut next: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|err| format!("cannot list {}: {err}", dir.display()))?
            .path();
        if let Some(other) = client_dump(&path) {
            if other.signer == dumped.signer
                && other.key.epoch_start == dumped.key.epoch_start
                && other.key.step == step
            {
                next.push(path);
            }
        }
    }
    next.sort();
    let Some(payload) = next.into_iter().next() else {
        return Ok(None);
    };
    let record_path = commitment_record_path(&payload);
    let record = match std::fs::read(&record_path) {
        Ok(bytes) => CommitmentRecord::from_bytes(&bytes)
            .map_err(|err| format!("the next payload's commitment does not read: {err}"))?,
        // Kept beside the payload once it is written; not there yet.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(format!("cannot read {}: {err}", record_path.display())),
    };
    let committed = record
        .state_digest
        .ok_or_else(|| "the trainer's next payload commits to no state".to_string())?;
    Ok(Some((payload, committed)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use psyche_core::{BatchId, ClosedInterval};

    const SIGNER: &str = "5rvGuv911jzq1AvHXzam5EUp4UVqKQxktBS2LEB6FENb";
    const OTHER: &str = "BXwRgCYXaJjpw8QiRCpBEkUZU64RGhu8bYb1tCZRtZxa";
    const EPOCH: u64 = 1_795_040_087;

    fn batch(start: u64) -> BatchId {
        BatchId(ClosedInterval::new(start, start + 3))
    }

    /// Keeps a payload of `signer` at `step` in `dir`, with a commitment
    /// record naming `state`, and returns its dump.
    fn keep(
        dir: &Path,
        signer: &str,
        step: u32,
        batch_id: BatchId,
        state: Option<[u8; 32]>,
    ) -> PathBuf {
        let dump = dir.join(payload_dump_name(signer, EPOCH, step, batch_id));
        std::fs::write(&dump, b"payload").unwrap();
        let record = CommitmentRecord {
            epoch_start: EPOCH,
            step,
            trainer_nonce: 0,
            batch_id,
            data_hash: [1; 32],
            signature: [2; 64],
            state_digest: state,
        };
        std::fs::write(commitment_record_path(&dump), record.to_bytes()).unwrap();
        dump
    }

    fn trained(dump: &Path, batches: Vec<BatchId>) {
        std::fs::write(
            round_training_path(dump),
            PreviousRound::Trained(batches).encode(),
        )
        .unwrap();
    }

    fn applied(dir: &Path, step: u32, applied: AppliedResults) {
        std::fs::write(applied_results_path(dir, EPOCH, step), applied.encode()).unwrap();
    }

    fn ready(continuity: Continuity) -> NextRound {
        match continuity {
            Continuity::Ready(next) => next,
            other => panic!("not ready: {other:?}"),
        }
    }

    /// The trainer's next payload, the state it commits to, and what the
    /// verifier's node applied at the start of that round, in its order.
    #[test]
    fn a_round_whose_trainer_trained_the_next_is_ready() {
        let dir = tempfile::tempdir().unwrap();
        let dump = keep(dir.path(), SIGNER, 3, batch(20), Some([3; 32]));
        trained(&dump, vec![batch(20)]);
        let next_payload = keep(dir.path(), SIGNER, 4, batch(28), Some([4; 32]));
        // Another node's payload of the round after is not the trainer's.
        keep(dir.path(), OTHER, 4, batch(24), Some([9; 32]));
        let first = keep(dir.path(), OTHER, 2, batch(8), Some([5; 32]));
        let second = keep(dir.path(), SIGNER, 2, batch(12), Some([6; 32]));
        let names = |paths: &[&PathBuf]| -> Vec<String> {
            paths
                .iter()
                .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        };
        applied(
            dir.path(),
            4,
            AppliedResults::Applied(names(&[&second, &first])),
        );
        let next = ready(next_round(&dump));
        assert_eq!(next.step, 4);
        assert_eq!(next.payload, next_payload);
        assert_eq!(next.committed, [4; 32]);
        assert_eq!(next.applied, Some(vec![second, first]));

        // A round after that applied nothing.
        applied(dir.path(), 4, AppliedResults::Skipped);
        assert_eq!(ready(next_round(&dump)).applied, None);
    }

    /// Until the trainer's next payload is kept, with its commitment, there
    /// is nothing to check yet.
    #[test]
    fn a_round_whose_next_payload_is_not_here_is_pending() {
        let dir = tempfile::tempdir().unwrap();
        let dump = keep(dir.path(), SIGNER, 3, batch(20), Some([3; 32]));
        trained(&dump, vec![batch(20)]);
        applied(dir.path(), 4, AppliedResults::Skipped);
        assert!(matches!(next_round(&dump), Continuity::Pending));

        let next = dir
            .path()
            .join(payload_dump_name(SIGNER, EPOCH, 4, batch(28)));
        std::fs::write(&next, b"payload").unwrap();
        assert!(matches!(next_round(&dump), Continuity::Pending));
    }

    /// What cannot be checked here: a round with other batches of the
    /// trainer's, no record of what it trained, a next payload committing to
    /// no state, and a next round whose application this verifier did not
    /// keep, or whose payloads it did not.
    #[test]
    fn a_round_this_verifier_cannot_follow_is_unchecked() {
        let unchecked = |continuity: Continuity| match continuity {
            Continuity::Unchecked(why) => why,
            other => panic!("not unchecked: {other:?}"),
        };
        let dir = tempfile::tempdir().unwrap();
        let dump = keep(dir.path(), SIGNER, 3, batch(20), Some([3; 32]));
        assert!(unchecked(next_round(&dump)).contains("not kept"));
        trained(&dump, vec![batch(20), batch(24)]);
        assert!(unchecked(next_round(&dump)).contains("2 batches"));
        std::fs::write(round_training_path(&dump), PreviousRound::Unknown.encode()).unwrap();
        unchecked(next_round(&dump));

        trained(&dump, vec![batch(20)]);
        keep(dir.path(), SIGNER, 4, batch(28), None);
        assert!(unchecked(next_round(&dump)).contains("no state"));

        keep(dir.path(), SIGNER, 4, batch(28), Some([4; 32]));
        assert!(unchecked(next_round(&dump)).contains("did not keep"));
        let missing = payload_dump_name(OTHER, EPOCH, 2, batch(8));
        applied(dir.path(), 4, AppliedResults::Applied(vec![missing]));
        assert!(unchecked(next_round(&dump)).contains("not kept here"));
    }
}
