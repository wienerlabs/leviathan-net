use anchor_lang::prelude::*;

use crate::state::EquivocationCommitment;
use crate::state::Run;

#[derive(Accounts)]
#[instruction(params: RunCloseEquivocationCommitmentParams)]
pub struct RunCloseEquivocationCommitmentAccounts<'info> {
    /// The reporter that made the commitment, which takes its rent back.
    #[account(mut)]
    pub reporter: Signer<'info>,

    #[account()]
    pub run: Box<Account<'info, Run>>,

    #[account(
        mut,
        close = reporter,
        seeds = [
            EquivocationCommitment::SEEDS_PREFIX,
            run.key().as_ref(),
            reporter.key().as_ref(),
            params.commitment.as_ref(),
        ],
        bump = equivocation_commitment.bump,
    )]
    pub equivocation_commitment: Box<Account<'info, EquivocationCommitment>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunCloseEquivocationCommitmentParams {
    pub commitment: [u8; 32],
}

/// Closes a commitment its reporter will not reveal, say because another
/// reporter convicted the target first, and gives the reporter its rent back
/// (wienerlabs/leviathan-net#92). A revealed commitment is closed by the
/// reveal itself.
pub fn run_close_equivocation_commitment_processor(
    _context: Context<RunCloseEquivocationCommitmentAccounts>,
    _params: RunCloseEquivocationCommitmentParams,
) -> Result<()> {
    Ok(())
}
