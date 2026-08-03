use anchor_lang::prelude::*;

#[account()]
#[derive(Debug)]
pub struct Run {
    pub bump: u8,
    pub index: u64,

    pub main_authority: Pubkey,
    pub join_authority: Pubkey,

    pub coordinator_account: Pubkey,
    pub coordinator_instance: Pubkey,

    pub collateral_mint: Pubkey,

    pub total_claimed_collateral_amount: u64,
    pub total_claimed_earned_points: u64,

    pub total_bonded_amount: u64,
    pub bond_minimum_amount: u64,
    pub bond_withdraw_delay_seconds: i64,
    pub slash_bounty_bps: u16,

    pub challenge_window_seconds: i64,
    pub tie_breaker_committee_size: u16,

    /// How long a challenge may hold a conviction open before the verdict the
    /// verifiers already reached is allowed to finalise anyway. Zero means the
    /// appeal never times out, which is the old behaviour and lets a guilty
    /// target park a correct conviction for free
    /// (wienerlabs/leviathan#15, finding 10).
    pub appeal_window_seconds: i64,
}

impl Run {
    pub const SEEDS_PREFIX: &'static [u8] = b"Run";

    /// Counted field by field against the Borsh layout, not taken from
    /// `std::mem::size_of`, which reports the Rust layout with its alignment
    /// padding. The two agree today only by luck (finding 13).
    pub fn space_with_discriminator() -> usize {
        8 + 1 // bump
            + 8 // index
            + 32 // main_authority
            + 32 // join_authority
            + 32 // coordinator_account
            + 32 // coordinator_instance
            + 32 // collateral_mint
            + 8 // total_claimed_collateral_amount
            + 8 // total_claimed_earned_points
            + 8 // total_bonded_amount
            + 8 // bond_minimum_amount
            + 8 // bond_withdraw_delay_seconds
            + 2 // slash_bounty_bps
            + 8 // challenge_window_seconds
            + 2 // tie_breaker_committee_size
            + 8 // appeal_window_seconds
    }
}
