use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

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
}

impl Participant {
    pub const SEEDS_PREFIX: &'static [u8] = b"Participant";

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

    #[test]
    fn a_conviction_larger_than_the_bond_leaves_nothing() {
        let p = participant(500, 0, 500);
        assert_eq!(p.unforfeited_bond(900), 0);
        assert_eq!(p.committed_bond(900), 0);
    }
}
