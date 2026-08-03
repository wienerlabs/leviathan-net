use anchor_lang::prelude::*;

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
