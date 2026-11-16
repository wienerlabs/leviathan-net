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
    /// How long an opened challenge may hold a conviction before the verifiers'
    /// verdict finalises anyway (wienerlabs/leviathan#15, finding 10). It may
    /// be zero only with appeals off. Switching appeals off ends the appeals in
    /// flight in the verifiers' favour: an open challenge finalises at once
    /// rather than holding the conviction, and every vote on it, for good
    /// (wienerlabs/leviathan-net#39).
    pub appeal_window_seconds: i64,
}

pub fn run_set_challenge_config_processor(
    context: Context<RunSetChallengeConfigAccounts>,
    params: RunSetChallengeConfigParams,
) -> Result<()> {
    if params.challenge_window_seconds < 0 || params.appeal_window_seconds < 0 {
        return err!(ProgramError::InvalidParameter);
    }
    // An appeal with no deadline holds the target's bond for as long as no
    // bench resolves it, and once the target or the run is gone no bench ever
    // can. With appeals on, the verdict the verifiers reached has to be able
    // to finalise eventually (wienerlabs/leviathan-net#36).
    if params.challenge_window_seconds > 0 && params.appeal_window_seconds == 0 {
        return err!(ProgramError::AppealWindowRequired);
    }
    let run = &mut context.accounts.run;
    run.challenge_window_seconds = params.challenge_window_seconds;
    run.tie_breaker_committee_size = params.tie_breaker_committee_size;
    run.appeal_window_seconds = params.appeal_window_seconds;
    Ok(())
}
