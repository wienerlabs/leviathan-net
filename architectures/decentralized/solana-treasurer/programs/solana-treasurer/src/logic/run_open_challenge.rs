use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VerdictStatus;
use crate::state::coordinator_slashed_points;
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

    #[account(
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    #[account()]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

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
    // An upheld appeal is paid for out of the challenger's bond, so what counts
    // is the bond a slash could still take. Earlier convictions it has not
    // settled are already spoken for (wienerlabs/leviathan-net#34). A pending
    // withdrawal is not held against it: the challenger is defending a
    // conviction, not taking on new work, and that bond is still in the vault.
    let slashed_points = coordinator_slashed_points(
        &*context.accounts.coordinator_account.load()?,
        &context.accounts.challenger.key(),
    );
    if context
        .accounts
        .challenger_participant
        .unforfeited_bond(slashed_points)
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
    // Stamped so the appeal has a deadline of its own. Opening a challenge costs
    // the target nothing beyond the bond it already posted, so without one it is
    // strictly dominant for a guilty target: it cannot make their position worse
    // and it parked a correct conviction for good
    // (wienerlabs/leviathan#15, finding 10).
    verdict.challenged_since_unix = now;
    verdict.challenger = context.accounts.challenger.key();
    verdict.overturn_count = 0;
    verdict.uphold_count = 0;
    verdict.appeal_voters.clear();

    msg!(
        "audit_verdict: challenge opened by target, tie-breaker committee convened"
    );

    Ok(())
}
