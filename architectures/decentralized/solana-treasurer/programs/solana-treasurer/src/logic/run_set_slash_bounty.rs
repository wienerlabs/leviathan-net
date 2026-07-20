use anchor_lang::prelude::*;

use crate::ProgramError;
use crate::state::Run;

pub const MAX_SLASH_BOUNTY_BPS: u16 = 10_000;

#[derive(Accounts)]
#[instruction(params: RunSetSlashBountyParams)]
pub struct RunSetSlashBountyAccounts<'info> {
    #[account()]
    pub main_authority: Signer<'info>,

    #[account(
        mut,
        constraint = run.main_authority == main_authority.key(),
    )]
    pub run: Box<Account<'info, Run>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunSetSlashBountyParams {
    pub slash_bounty_bps: u16,
}

pub fn run_set_slash_bounty_processor(
    context: Context<RunSetSlashBountyAccounts>,
    params: RunSetSlashBountyParams,
) -> Result<()> {
    if params.slash_bounty_bps > MAX_SLASH_BOUNTY_BPS {
        return err!(ProgramError::InvalidParameter);
    }
    context.accounts.run.slash_bounty_bps = params.slash_bounty_bps;
    Ok(())
}
