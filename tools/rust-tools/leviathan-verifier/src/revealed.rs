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

use psyche_network::{payload_dump_name, previous_round_path, revealed_state_path, PreviousRound};
use thiserror::Error;

use crate::fidelity::client_dump;

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
