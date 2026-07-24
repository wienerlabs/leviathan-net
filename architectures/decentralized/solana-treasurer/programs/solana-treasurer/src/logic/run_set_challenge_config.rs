use anchor_lang::prelude::*;

use crate::state::Run;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSetChallengeConfigParams)]
pub struct RunSetChallengeConfigAccounts<'info> {
    #[account()]
    pub main_authority: Signer<'info>,

    #[account(
        mut,
        constraint = run.main_authority == main_authority.key(),
    )]
    pub run: Box<Account<'info, Run>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunSetChallengeConfigParams {
    pub challenge_window_seconds: i64,
    pub tie_breaker_committee_size: u16,
}

pub fn run_set_challenge_config_processor(
    context: Context<RunSetChallengeConfigAccounts>,
    params: RunSetChallengeConfigParams,
) -> Result<()> {
    if params.challenge_window_seconds < 0 {
        return err!(ProgramError::InvalidParameter);
    }
    let run = &mut context.accounts.run;
    run.challenge_window_seconds = params.challenge_window_seconds;
    run.tie_breaker_committee_size = params.tie_breaker_committee_size;
    Ok(())
}
