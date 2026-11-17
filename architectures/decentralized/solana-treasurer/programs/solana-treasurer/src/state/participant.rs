use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::ProgramError;

/// The points the coordinator has charged against `user`, read from its
/// permanent client ledger. A signer that never joined the run owes nothing.
///
/// Points and collateral base units are the same thing by the run's own
/// convention (`RunUpdateParams`, wienerlabs/leviathan#15 finding 14), which is
/// what lets them be subtracted from a bond.
pub fn coordinator_slashed_points(
    coordinator_account: &CoordinatorAccount,
    user: &Pubkey,
) -> u64 {
    coordinator_account
        .state
        .clients_state
        .clients
        .iter()
        .find(|client| *client.id.signer() == user.to_bytes())
        .map(|client| client.slashed)
        .unwrap_or(0)
}

#[account()]
#[derive(Debug)]
pub struct Participant {
    pub bump: u8,

    pub claimed_collateral_amount: u64,
    pub claimed_earned_points: u64,

    pub bond_amount: u64,
    pub bond_withdraw_pending_amount: u64,
    pub bond_withdraw_requested_at: i64,
    pub bond_settled_slashed_points: u64,

    /// The withdraw delay as it stood when this withdrawal was requested.
    ///
    /// The unlock time used to be computed from the run's *current* delay, so
    /// setting it to zero made every pending withdrawal claimable at once,
    /// including those mid-dispute. The whole point of the window is that a
    /// cheater cannot leave between committing fraud and being convicted, and a
    /// window that can be closed retroactively is not that window
    /// (wienerlabs/leviathan#15, finding 17).
    pub bond_withdraw_delay_snapshot: i64,

    /// Votes this participant cast on audit verdicts that can still cost it
    /// its bond.
    ///
    /// A verdict that is overturned charges every verifier that voted for it,
    /// and that liability is recorded only on the target's verdict. A count here
    /// is what lets a withdrawal see it: a vote is counted when it is cast, and
    /// stops counting once the verdict stands, the vote is withdrawn from a
    /// verdict that never reached quorum, or the losing-verifier crank has
    /// charged it. Until then the bond cannot leave
    /// (wienerlabs/leviathan-net#39).
    pub open_verdict_votes: u32,
}

impl Participant {
    pub const SEEDS_PREFIX: &'static [u8] = b"Participant";

    /// The participant held by the account at a participant PDA, given that
    /// account's owner and data. An empty account, or one the treasurer does
    /// not own, is no participant: the wallet never created one.
    pub fn from_pda_account(
        owner: &Pubkey,
        data: &[u8],
    ) -> Result<Option<Participant>> {
        if data.is_empty() || owner != &crate::ID {
            return Ok(None);
        }
        Ok(Some(Participant::try_deserialize(&mut &data[..])?))
    }

    /// Points a conviction has charged that this bond has not paid yet.
    ///
    /// A slash reaches `bond_amount` only when the participant itself finalizes
    /// a withdrawal, so until then the gross bond still counts what the
    /// conviction already took (wienerlabs/leviathan-net#34).
    pub fn unsettled_slashed_points(
        &self,
        coordinator_slashed_points: u64,
    ) -> u64 {
        coordinator_slashed_points
            .saturating_sub(self.bond_settled_slashed_points)
    }

    /// What a slash could still take: the bond, less what convictions already
    /// owe. Settlement forfeits exactly this much first, so this is the bond as
    /// it will stand once `participant_bond_finalize_withdraw` runs.
    pub fn unforfeited_bond(&self, coordinator_slashed_points: u64) -> u64 {
        self.bond_amount.saturating_sub(
            self.unsettled_slashed_points(coordinator_slashed_points),
        )
    }

    /// The bond that stays behind new work: unforfeited, and not already on its
    /// way out. A withdrawal that has been requested leaves when its delay
    /// elapses, whatever the participant does in the meantime, so it cannot
    /// back a seat in a later epoch or a vote whose penalty lands later.
    pub fn committed_bond(&self, coordinator_slashed_points: u64) -> u64 {
        self.unforfeited_bond(coordinator_slashed_points)
            .saturating_sub(self.bond_withdraw_pending_amount)
    }

    /// `unforfeited_bond`, less what the participant's own conviction will
    /// still take that the ledger has not charged
    /// (`unledgered_conviction_charge`): the bond as it will stand once that
    /// conviction is charged too.
    pub fn unforfeited_bond_owing(
        &self,
        coordinator_slashed_points: u64,
        unledgered_conviction: u64,
    ) -> u64 {
        self.unforfeited_bond(coordinator_slashed_points)
            .saturating_sub(unledgered_conviction)
    }

    /// `committed_bond`, less what the participant's own conviction will
    /// still take that the ledger has not charged. This is what backs a seat
    /// or a vote.
    pub fn committed_bond_owing(
        &self,
        coordinator_slashed_points: u64,
        unledgered_conviction: u64,
    ) -> u64 {
        self.committed_bond(coordinator_slashed_points)
            .saturating_sub(unledgered_conviction)
    }

    /// Counted against the Borsh layout rather than taken from
    /// `std::mem::size_of`, which reports the padded Rust layout (finding 13).
    pub fn space_with_discriminator() -> usize {
        8 + 1 // bump
            + 8 // claimed_collateral_amount
            + 8 // claimed_earned_points
            + 8 // bond_amount
            + 8 // bond_withdraw_pending_amount
            + 8 // bond_withdraw_requested_at
            + 8 // bond_settled_slashed_points
            + 8 // bond_withdraw_delay_snapshot
            + 4 // open_verdict_votes
    }

    /// Counts a vote this participant just cast.
    pub fn open_verdict_vote(&mut self) -> Result<()> {
        self.open_verdict_votes =
            self.open_verdict_votes.checked_add(1).ok_or_else(|| {
                error!(ProgramError::OpenVerdictVotesInconsistent)
            })?;
        Ok(())
    }

    /// Stops counting a vote that can no longer cost this participant.
    ///
    /// Every vote is counted once when it is cast and closed at most once,
    /// through the verdict's own record of it, so a count that would go below
    /// zero means that record and this count disagree. It is refused rather
    /// than clamped, so the disagreement shows.
    pub fn close_verdict_vote(&mut self) -> Result<()> {
        self.open_verdict_votes =
            self.open_verdict_votes.checked_sub(1).ok_or_else(|| {
                error!(ProgramError::OpenVerdictVotesInconsistent)
            })?;
        Ok(())
    }

    /// Pays `amount` of the points the ledger charged out of the bond, as
    /// settlement does: the bond loses it and it counts as settled, so it is
    /// never charged again. A pending withdrawal is cut down to what is left.
    ///
    /// The caller passes no more than the unsettled points and the bond.
    pub fn settle_ledger_points(&mut self, amount: u64) {
        self.bond_amount -= amount;
        self.bond_settled_slashed_points += amount;
        self.bond_withdraw_pending_amount =
            self.bond_withdraw_pending_amount.min(self.bond_amount);
    }

    /// What a conviction that ejected this participant takes from its bond
    /// once the ledger charged the ejection `rate`: that charge, out of what is
    /// still unsettled and what the bond holds.
    ///
    /// The ledger charges a client at most once per epoch, so the charge for
    /// the epoch a conviction ejected it in is that conviction's. Taking no
    /// more than `rate` leaves every other unsettled point, such as a
    /// `run_slash` in another epoch, to settlement (wienerlabs/leviathan-net#64).
    pub fn ejected_conviction_forfeit(
        &self,
        coordinator_slashed_points: u64,
        rate: u64,
    ) -> u64 {
        rate.min(self.unsettled_slashed_points(coordinator_slashed_points))
            .min(self.bond_amount)
    }

    /// Forfeits up to `amount` of the bond, out of what convictions the ledger
    /// already charged do not claim, and returns what it took.
    ///
    /// This is a charge the coordinator's ledger never sees, so it is not
    /// counted as a settled point: counting it would hide a later conviction of
    /// the same size. A pending withdrawal is cut down to what is left, so it
    /// never asks for more than the bond holds.
    pub fn forfeit_unclaimed_bond(
        &mut self,
        amount: u64,
        coordinator_slashed_points: u64,
    ) -> u64 {
        let forfeit =
            amount.min(self.unforfeited_bond(coordinator_slashed_points));
        self.bond_amount -= forfeit;
        self.bond_withdraw_pending_amount =
            self.bond_withdraw_pending_amount.min(self.bond_amount);
        forfeit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn participant(
        bond: u64,
        settled_points: u64,
        pending: u64,
    ) -> Participant {
        Participant {
            bump: 0,
            claimed_collateral_amount: 0,
            claimed_earned_points: 0,
            bond_amount: bond,
            bond_withdraw_pending_amount: pending,
            bond_withdraw_requested_at: 0,
            bond_settled_slashed_points: settled_points,
            bond_withdraw_delay_snapshot: 0,
            open_verdict_votes: 0,
        }
    }

    #[test]
    fn a_bond_with_no_conviction_counts_in_full() {
        let p = participant(500, 0, 0);
        assert_eq!(p.unsettled_slashed_points(0), 0);
        assert_eq!(p.unforfeited_bond(0), 500);
        assert_eq!(p.committed_bond(0), 500);
    }

    #[test]
    fn an_unsettled_conviction_comes_off_the_bond() {
        let p = participant(500, 0, 0);
        assert_eq!(p.unsettled_slashed_points(200), 200);
        assert_eq!(p.unforfeited_bond(200), 300);
        assert_eq!(p.committed_bond(200), 300);
    }

    #[test]
    fn a_settled_conviction_is_not_counted_twice() {
        // Settlement moved 200 out of `bond_amount` and recorded it as settled.
        let p = participant(300, 200, 0);
        assert_eq!(p.unsettled_slashed_points(200), 0);
        assert_eq!(p.unforfeited_bond(200), 300);
        // A second conviction is charged on top of the first.
        assert_eq!(p.unforfeited_bond(400), 100);
    }

    #[test]
    fn a_pending_withdrawal_comes_off_the_committed_bond_only() {
        let p = participant(500, 0, 100);
        assert_eq!(p.unforfeited_bond(0), 500);
        assert_eq!(p.committed_bond(0), 400);
        assert_eq!(p.committed_bond(200), 200);
    }

    /// A conviction that has not reached the ledger comes off both, on top of
    /// what the ledger charged and the withdrawal requested.
    #[test]
    fn a_conviction_off_the_ledger_comes_off_what_the_bond_backs() {
        let p = participant(900, 0, 200);
        assert_eq!(p.unforfeited_bond_owing(200, 0), 700);
        assert_eq!(p.unforfeited_bond_owing(200, 200), 500);
        assert_eq!(p.committed_bond_owing(200, 0), 500);
        assert_eq!(p.committed_bond_owing(200, 200), 300);
        assert_eq!(p.committed_bond_owing(200, 1_000), 0);
    }

    #[test]
    fn a_conviction_larger_than_the_bond_leaves_nothing() {
        let p = participant(500, 0, 500);
        assert_eq!(p.unforfeited_bond(900), 0);
        assert_eq!(p.committed_bond(900), 0);
    }

    #[test]
    fn votes_are_counted_open_and_closed_once_each() {
        let mut p = participant(500, 0, 0);
        p.open_verdict_vote().unwrap();
        p.open_verdict_vote().unwrap();
        assert_eq!(p.open_verdict_votes, 2);
        p.close_verdict_vote().unwrap();
        p.close_verdict_vote().unwrap();
        assert_eq!(p.open_verdict_votes, 0);
        assert!(p.close_verdict_vote().is_err());
        assert_eq!(p.open_verdict_votes, 0);
    }

    #[test]
    fn a_full_vote_count_refuses_another_vote() {
        let mut p = participant(500, 0, 0);
        p.open_verdict_votes = u32::MAX;
        assert!(p.open_verdict_vote().is_err());
        assert_eq!(p.open_verdict_votes, u32::MAX);
    }

    #[test]
    fn a_forfeit_takes_the_amount_from_the_bond() {
        let mut p = participant(500, 0, 0);
        assert_eq!(p.forfeit_unclaimed_bond(200, 0), 200);
        assert_eq!(p.bond_amount, 300);
        // Not a ledger charge, so nothing is recorded as settled.
        assert_eq!(p.bond_settled_slashed_points, 0);
    }

    #[test]
    fn a_forfeit_leaves_what_the_ledger_already_claims() {
        // 400 of the bond is already owed to an unsettled conviction.
        let mut p = participant(500, 0, 0);
        assert_eq!(p.forfeit_unclaimed_bond(200, 400), 100);
        assert_eq!(p.bond_amount, 400);
        assert_eq!(p.unforfeited_bond(400), 0);
        assert_eq!(p.forfeit_unclaimed_bond(200, 400), 0);
        assert_eq!(p.bond_amount, 400);
    }

    #[test]
    fn settled_ledger_points_leave_the_bond_once() {
        let mut p = participant(500, 0, 450);
        p.settle_ledger_points(200);
        assert_eq!(p.bond_amount, 300);
        assert_eq!(p.bond_settled_slashed_points, 200);
        assert_eq!(p.bond_withdraw_pending_amount, 300);
        // The points are settled, so the same ledger owes nothing more.
        assert_eq!(p.unsettled_slashed_points(200), 0);
        assert_eq!(p.unforfeited_bond(200), 300);
    }

    #[test]
    fn an_ejected_conviction_takes_its_own_charge_and_no_more() {
        // 200 from another slash, then 200 for the conviction: the conviction
        // takes its 200, the other stays for settlement.
        let p = participant(500, 0, 0);
        assert_eq!(p.ejected_conviction_forfeit(400, 200), 200);
        // What the ledger charged is all the conviction can take.
        assert_eq!(p.ejected_conviction_forfeit(150, 200), 150);
        // Already settled: nothing left to take.
        let settled = participant(300, 200, 0);
        assert_eq!(settled.ejected_conviction_forfeit(200, 200), 0);
        // A bond short of the charge pays what it holds.
        let short = participant(120, 0, 0);
        assert_eq!(short.ejected_conviction_forfeit(200, 200), 120);
    }

    #[test]
    fn a_forfeit_cuts_a_pending_withdrawal_down_to_the_bond() {
        let mut p = participant(500, 0, 500);
        p.forfeit_unclaimed_bond(200, 0);
        assert_eq!(p.bond_withdraw_pending_amount, 300);
        // A smaller request is left alone.
        let mut q = participant(500, 0, 100);
        q.forfeit_unclaimed_bond(200, 0);
        assert_eq!(q.bond_withdraw_pending_amount, 100);
    }
}
