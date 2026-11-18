use crate::state::Run;

/// The bounty share of a forfeit of `forfeited_amount` in `run`.
pub fn bounty_amount(run: &Run, forfeited_amount: u64) -> u64 {
    bounty_share(forfeited_amount, run.slash_bounty_bps)
}

/// `bounty_bps` of a forfeit of `forfeited_amount`, rounded down: what does
/// not divide evenly stays in the vault.
pub fn bounty_share(forfeited_amount: u64, bounty_bps: u16) -> u64 {
    (forfeited_amount as u128 * bounty_bps as u128 / 10_000) as u64
}
