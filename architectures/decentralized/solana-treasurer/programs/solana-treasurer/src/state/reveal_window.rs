use anchor_lang::prelude::*;
use psyche_coordinator::RunState;

use crate::state::CommitmentEvidence;
use crate::ProgramError;

/// How long a trainer has, into a round's witness phase, to reveal the state
/// a payload commits to before verifiers may vote that it did not
/// (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
///
/// Every payload a vote can cite was broadcast before the witness phase it
/// is cast in began, and its trainer serves the state from its broadcast on.
/// A vote is taken only from the committee of the round under way, so the
/// window has to close within the witness phase: it is shorter than the
/// run's `round_witness_time`. Kept in an account of its own rather than in
/// `Run`, whose length tells the dashboard's decoders its generations apart.
/// A run that sets none takes no non-reveal vote.
#[account()]
#[derive(Debug)]
pub struct RevealWindow {
    pub bump: u8,
    pub run: Pubkey,
    pub reveal_seconds: u64,
}

impl RevealWindow {
    pub const SEEDS_PREFIX: &'static [u8] = b"RevealWindow";

    /// Counted against the Borsh layout.
    pub fn space_with_discriminator() -> usize {
        8 + 1 // bump
            + 32 // run
            + 8 // reveal_seconds
    }

    /// The window a run sets, read from the account at its PDA. A run that
    /// set none has no window.
    pub fn from_pda_account(
        owner: &Pubkey,
        data: &[u8],
    ) -> Result<Option<RevealWindow>> {
        if data.is_empty() || owner != &crate::ID {
            return Ok(None);
        }
        Ok(Some(RevealWindow::try_deserialize(&mut &data[..])?))
    }
}

/// Refuses a reveal window that is empty, or that a round's witness phase of
/// `round_witness_time` seconds cannot hold.
pub fn check_reveal_window(
    reveal_seconds: u64,
    round_witness_time: u64,
) -> Result<()> {
    if reveal_seconds == 0 || reveal_seconds >= round_witness_time {
        return err!(ProgramError::RevealWindowInvalid);
    }
    Ok(())
}

/// Whether a vote that the target did not reveal can be taken now: the
/// payload commits to a state, the vote replayed nothing, the run set a
/// window, and the window has passed in the witness phase under way, or in
/// the cooldown that follows the epoch's final round.
pub fn check_non_reveal_vote(
    evidence: &CommitmentEvidence,
    replayed_hash: &[u8; 32],
    reveal_seconds: Option<u64>,
    run_state: RunState,
    phase_start_unix: u64,
    now_unix: i64,
) -> Result<()> {
    if !evidence.state_committed {
        return err!(ProgramError::NonRevealWithoutState);
    }
    if replayed_hash != &[0u8; 32] {
        return err!(ProgramError::NonRevealReplayed);
    }
    let Some(reveal_seconds) = reveal_seconds else {
        return err!(ProgramError::RevealWindowUnset);
    };
    let in_phase =
        matches!(run_state, RunState::RoundWitness | RunState::Cooldown);
    let closes = phase_start_unix.saturating_add(reveal_seconds);
    if !in_phase || now_unix < 0 || (now_unix as u64) < closes {
        return err!(ProgramError::RevealWindowOpen);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn committed() -> CommitmentEvidence {
        CommitmentEvidence {
            state_committed: true,
            state_digest: [7; 32],
            ..CommitmentEvidence::default()
        }
    }

    #[test]
    fn a_window_has_to_close_within_the_witness_phase() {
        assert!(check_reveal_window(1, 10).is_ok());
        assert!(check_reveal_window(9, 10).is_ok());
        for (seconds, witness) in [(0, 10), (10, 10), (11, 10), (1, 0), (0, 0)]
        {
            assert_eq!(
                check_reveal_window(seconds, witness).unwrap_err(),
                error!(ProgramError::RevealWindowInvalid),
                "{seconds} of {witness}"
            );
        }
    }

    #[test]
    fn a_non_reveal_vote_waits_for_the_window_in_a_witness_phase() {
        let vote = |state, now| {
            check_non_reveal_vote(
                &committed(),
                &[0; 32],
                Some(5),
                state,
                100,
                now,
            )
        };
        assert!(vote(RunState::RoundWitness, 105).is_ok());
        assert!(vote(RunState::RoundWitness, 400).is_ok());
        assert!(vote(RunState::Cooldown, 105).is_ok());
        for (state, now) in [
            (RunState::RoundWitness, 104),
            (RunState::RoundWitness, 99),
            (RunState::RoundWitness, -1),
            (RunState::Cooldown, 104),
            (RunState::RoundTrain, 400),
            (RunState::Warmup, 400),
            (RunState::WaitingForMembers, 400),
            (RunState::Paused, 400),
        ] {
            assert_eq!(
                vote(state, now).unwrap_err(),
                error!(ProgramError::RevealWindowOpen),
                "{state:?} at {now}"
            );
        }
    }

    #[test]
    fn a_non_reveal_vote_needs_a_state_a_window_and_no_replay() {
        let uncommitted = CommitmentEvidence::default();
        assert_eq!(
            check_non_reveal_vote(
                &uncommitted,
                &[0; 32],
                Some(5),
                RunState::RoundWitness,
                100,
                200,
            )
            .unwrap_err(),
            error!(ProgramError::NonRevealWithoutState)
        );
        assert_eq!(
            check_non_reveal_vote(
                &committed(),
                &[1; 32],
                Some(5),
                RunState::RoundWitness,
                100,
                200,
            )
            .unwrap_err(),
            error!(ProgramError::NonRevealReplayed)
        );
        assert_eq!(
            check_non_reveal_vote(
                &committed(),
                &[0; 32],
                None,
                RunState::RoundWitness,
                100,
                200,
            )
            .unwrap_err(),
            error!(ProgramError::RevealWindowUnset)
        );
    }
}
