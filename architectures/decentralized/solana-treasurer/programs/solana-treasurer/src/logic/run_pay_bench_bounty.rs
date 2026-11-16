use anchor_lang::prelude::*;

use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunPayBenchBountyParams)]
pub struct RunPayBenchBountyAccounts<'info> {
    #[account()]
    pub payer: Signer<'info>,

    #[account(mut)]
    pub run: Box<Account<'info, Run>>,

    #[account(
        mut,
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump = audit_verdict.bump,
    )]
    pub audit_verdict: Box<Account<'info, AuditVerdict>>,

    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            params.tie_breaker.as_ref(),
        ],
        bump = tie_breaker_participant.bump,
    )]
    pub tie_breaker_participant: Box<Account<'info, Participant>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunPayBenchBountyParams {
    pub target: Pubkey,
    /// The tie-breaker's position in the overturned verdict's `appeal_voters`.
    pub seat: u16,
    /// The tie-breaker at `seat`, whose bond is credited.
    pub tie_breaker: Pubkey,
}

/// Credits one tie-breaker's bond with its share of what an overturn's losing
/// verifiers forfeited, once the crank has charged every one of them.
///
/// The share never leaves the vault: it moves from forfeited collateral into
/// the tie-breaker's bond, which it withdraws like any other. No token account
/// is involved, so nothing a tie-breaker or a mint does to one can stop the
/// payout, and the overturn always settles. Anyone may send it, each seat is
/// paid once, and one seat per transaction keeps a bench of any size payable
/// (wienerlabs/leviathan-net#39).
pub fn run_pay_bench_bounty_processor(
    context: Context<RunPayBenchBountyAccounts>,
    params: RunPayBenchBountyParams,
) -> Result<()> {
    let seat = params.seat as usize;
    if context.accounts.audit_verdict.appeal_voters.get(seat)
        != Some(&params.tie_breaker)
    {
        return err!(ProgramError::InvalidParameter);
    }
    let share = context.accounts.audit_verdict.pay_bench_seat(seat)?;
    let tie_breaker = &mut context.accounts.tie_breaker_participant;
    tie_breaker.bond_amount = tie_breaker
        .bond_amount
        .checked_add(share)
        .ok_or_else(|| error!(ProgramError::InvalidParameter))?;
    let run = &mut context.accounts.run;
    run.total_bonded_amount = run
        .total_bonded_amount
        .checked_add(share)
        .ok_or_else(|| error!(ProgramError::InvalidParameter))?;
    msg!(
        "pay_bench_bounty: target={} seat={} tie_breaker={} credited {}",
        params.target,
        seat,
        params.tie_breaker,
        share
    );
    Ok(())
}
