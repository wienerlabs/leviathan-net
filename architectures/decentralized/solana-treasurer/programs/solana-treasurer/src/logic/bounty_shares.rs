use crate::state::Run;

/// The bounty share of a forfeit of `forfeited_amount` in `run`.
pub fn bounty_amount(run: &Run, forfeited_amount: u64) -> u64 {
    (forfeited_amount as u128 * run.slash_bounty_bps as u128 / 10_000) as u64
}
