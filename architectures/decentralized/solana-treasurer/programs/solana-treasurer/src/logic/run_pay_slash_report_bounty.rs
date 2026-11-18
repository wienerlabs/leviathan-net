use anchor_lang::prelude::*;

use crate::state::ConvictionCharge;
use crate::state::Participant;
use crate::state::Run;
use crate::state::SlashReport;
use crate::ProgramError;

/// Credits a charged slash's bounty to the bond of the reporter its report
/// records, once (wienerlabs/leviathan-net#72). Anyone may send it. Like the
/// shares of a verdict's bounty (wienerlabs/leviathan-net#39, #64) it is
/// credited to a bond, never paid to a token account somebody could close or
/// choose.
#[derive(Accounts)]
#[instruction(params: RunPaySlashReportBountyParams)]
pub struct RunPaySlashReportBountyAccounts<'info> {
    #[account()]
    pub payer: Signer<'info>,

    #[account(mut)]
    pub run: Box<Account<'info, Run>>,

    #[account(
        mut,
        seeds = [
            SlashReport::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
            params.charge_epoch_start.to_le_bytes().as_ref(),
        ],
        bump = slash_report.bump,
    )]
    pub slash_report: Box<Account<'info, SlashReport>>,

    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            params.reporter.as_ref(),
        ],
        bump = reporter_participant.bump,
    )]
    pub reporter_participant: Box<Account<'info, Participant>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunPaySlashReportBountyParams {
    pub target: Pubkey,
    pub charge_epoch_start: u64,
    pub reporter: Pubkey,
}

/// What `report` pays `reporter` now, which it then records as paid.
pub fn take_slash_report_bounty(
    report: &mut SlashReport,
    reporter: &Pubkey,
) -> Result<u64> {
    if report.charge != ConvictionCharge::Charged {
        return err!(ProgramError::ConvictionNotCharged);
    }
    match report.reporter {
        None => return err!(ProgramError::SlashReportOwesNoBounty),
        Some(recorded) if recorded != *reporter => {
            return err!(ProgramError::InvalidParameter)
        },
        Some(_) => {},
    }
    if report.bounty_paid {
        return err!(ProgramError::BountySeatPaid);
    }
    if report.bounty_owed == 0 {
        return err!(ProgramError::NothingOwedToSeats);
    }
    report.bounty_paid = true;
    Ok(report.bounty_owed)
}

pub fn run_pay_slash_report_bounty_processor(
    context: Context<RunPaySlashReportBountyAccounts>,
    params: RunPaySlashReportBountyParams,
) -> Result<()> {
    let bounty = take_slash_report_bounty(
        &mut context.accounts.slash_report,
        &params.reporter,
    )?;
    let reporter = &mut context.accounts.reporter_participant;
    reporter.bond_amount = reporter
        .bond_amount
        .checked_add(bounty)
        .ok_or_else(|| error!(ProgramError::InvalidParameter))?;
    let run = &mut context.accounts.run;
    run.total_bonded_amount = run
        .total_bonded_amount
        .checked_add(bounty)
        .ok_or_else(|| error!(ProgramError::InvalidParameter))?;
    Ok(())
}
