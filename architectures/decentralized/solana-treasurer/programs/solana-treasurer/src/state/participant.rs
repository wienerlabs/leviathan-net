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
}

impl Participant {
    pub const SEEDS_PREFIX: &'static [u8] = b"Participant";

    pub fn space_with_discriminator() -> usize {
        8 + std::mem::size_of::<Participant>()
    }
}
