use anchor_lang::prelude::*;

use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VerdictStatus;
use crate::ProgramError;

#[derive(Accounts)]
pub struct RunOpenChallengeAccounts<'info> {
    #[account()]
    pub challenger: Signer<'info>,

    #[account(
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            challenger.key().as_ref(),
        ],
        bump = challenger_participant.bump,
    )]
    pub challenger_participant: Box<Account<'info, Participant>>,

    #[account()]
    pub run: Box<Account<'info, Run>>,

    #[account(
        mut,
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            challenger.key().as_ref(),
        ],
        bump = audit_verdict.bump,
    )]
    pub audit_verdict: Box<Account<'info, AuditVerdict>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunOpenChallengeParams {}

pub fn run_open_challenge_processor(
    context: Context<RunOpenChallengeAccounts>,
    _params: RunOpenChallengeParams,
) -> Result<()> {
    if context.accounts.challenger_participant.bond_amount
        < context.accounts.run.bond_minimum_amount
    {
        return err!(ProgramError::BondBelowMinimum);
    }

    let challenge_window_seconds = context.accounts.run.challenge_window_seconds;
    let verdict = &mut context.accounts.audit_verdict;

    if verdict.status != VerdictStatus::SlashPending {
        return err!(ProgramError::VerdictNotPending);
    }

    let now = Clock::get()?.unix_timestamp;
    if now >= verdict.pending_since_unix + challenge_window_seconds {
        return err!(ProgramError::ChallengeWindowClosed);
    }

    verdict.status = VerdictStatus::Challenged;
    verdict.challenger = context.accounts.challenger.key();
    verdict.overturn_count = 0;
    verdict.uphold_count = 0;
    verdict.appeal_voters.clear();

    msg!(
        "audit_verdict: challenge opened by target, tie-breaker committee convened"
    );

    Ok(())
}
