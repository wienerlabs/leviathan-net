use anchor_lang::prelude::*;

use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunPayVerdictBountyParams)]
pub struct RunPayVerdictBountyAccounts<'info> {
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
            params.recipient.as_ref(),
        ],
        bump = recipient_participant.bump,
    )]
    pub recipient_participant: Box<Account<'info, Participant>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunPayVerdictBountyParams {
    pub target: Pubkey,
    /// The recipient's position in the verdict's `bounty_recipients`.
    pub seat: u16,
    /// The recipient at `seat`, whose bond is credited.
    pub recipient: Pubkey,
}

/// Credits one recipient's bond with its share of what the verdict's losing
/// side forfeited, once that is final:
///
/// - an overturn pays each tie-breaker out of the losing verifiers' forfeits,
///   once the crank has charged every one (wienerlabs/leviathan-net#39);
/// - a conviction that stands pays each of its voters, or each tie-breaker of
///   the bench that upheld it, out of the target's forfeit, once that is
///   charged (wienerlabs/leviathan-net#64).
///
/// The share never leaves the vault: it moves from forfeited collateral into
/// the recipient's bond, which it withdraws like any other. No token account
/// is involved, so nothing a recipient or a mint does to one can stop the
/// payout. Anyone may send it, each seat is paid once, and one seat per
/// transaction keeps a committee of any size payable.
pub fn run_pay_verdict_bounty_processor(
    context: Context<RunPayVerdictBountyAccounts>,
    params: RunPayVerdictBountyParams,
) -> Result<()> {
    let seat = params.seat as usize;
    if context.accounts.audit_verdict.bounty_recipients().get(seat)
        != Some(&params.recipient)
    {
        return err!(ProgramError::InvalidParameter);
    }
    let share = context.accounts.audit_verdict.pay_bounty_seat(seat)?;
    let recipient = &mut context.accounts.recipient_participant;
    recipient.bond_amount = recipient
        .bond_amount
        .checked_add(share)
        .ok_or_else(|| error!(ProgramError::InvalidParameter))?;
    let run = &mut context.accounts.run;
    run.total_bonded_amount = run
        .total_bonded_amount
        .checked_add(share)
        .ok_or_else(|| error!(ProgramError::InvalidParameter))?;
    msg!(
        "pay_verdict_bounty: target={} seat={} recipient={} credited {}",
        params.target,
        seat,
        params.recipient,
        share
    );
    Ok(())
}
