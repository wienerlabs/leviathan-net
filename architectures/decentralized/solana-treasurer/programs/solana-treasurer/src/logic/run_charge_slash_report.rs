use anchor_lang::prelude::*;
use psyche_coordinator::Coordinator;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::logic::bounty_share;
use crate::state::coordinator_slashed_points;
use crate::state::equivocation_bounty_bps;
use crate::state::ConvictionCharge;
use crate::state::EquivocationBounty;
use crate::state::Participant;
use crate::state::Run;
use crate::state::SlashKind;
use crate::state::SlashReport;
use crate::ProgramError;

/// Charges a slash no verdict accounts for against its target's bond, once
/// the ledger has charged the ejection, and records what that owes the
/// slash's reporter (wienerlabs/leviathan-net#72). Anyone may send it: the
/// report says what is owed and to whom, and nothing here is chosen by the
/// sender.
#[derive(Accounts)]
#[instruction(params: RunChargeSlashReportParams)]
pub struct RunChargeSlashReportAccounts<'info> {
    #[account()]
    pub payer: Signer<'info>,

    #[account(
        mut,
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    // An account of another version is refused, not loaded: loading one
    // of another size panics (wienerlabs/leviathan-net#62).
    #[account(
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

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

    /// CHECK: the target's participant account at its own PDA. A target with
    /// none has no bond to charge.
    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump,
    )]
    pub target_participant: UncheckedAccount<'info>,

    /// CHECK: the run's equivocation bounty at its own PDA, read only after
    /// this checks that it exists and belongs to this program. A run that set
    /// none gives an equivocation's reporter nothing
    /// (wienerlabs/leviathan-net#92).
    #[account(
        seeds = [EquivocationBounty::SEEDS_PREFIX, run.key().as_ref()],
        bump,
    )]
    pub equivocation_bounty: UncheckedAccount<'info>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunChargeSlashReportParams {
    pub target: Pubkey,
    pub charge_epoch_start: u64,
}

/// The share of the forfeit `report`'s reporter is owed in `run`: a slash's
/// bounty for the run authority's `run_slash`, and the run's equivocation
/// bounty, `equivocation`, for an equivocation, whose reporter can be the
/// target itself (wienerlabs/leviathan-net#92).
pub fn report_bounty_bps(
    report: &SlashReport,
    run: &Run,
    equivocation: Option<&EquivocationBounty>,
) -> u16 {
    match report.kind {
        SlashKind::AuthoritySlash => run.slash_bounty_bps,
        SlashKind::Equivocation => equivocation_bounty_bps(run, equivocation),
    }
}

/// Charges `report`'s slash against `participant`, the target's account if
/// it has one, out of `slashed_points` the ledger charged it, and records
/// `bounty_bps` of what the bond paid as owed to the reporter. Returns what
/// the bond paid.
pub fn charge_slash_report(
    report: &mut SlashReport,
    participant: Option<&mut Participant>,
    coordinator: &Coordinator,
    slashed_points: u64,
    bounty_bps: u16,
) -> Result<u64> {
    if report.charge != ConvictionCharge::Ejected {
        return err!(ProgramError::ConvictionNotChargeable);
    }
    if !report.ejection_charged(coordinator) {
        return err!(ProgramError::EjectionNotCharged);
    }
    let forfeit = match participant {
        Some(participant) => {
            let forfeit = participant
                .ejected_conviction_forfeit(slashed_points, report.charge_rate);
            participant.settle_ledger_points(forfeit);
            if report.counted_on_target {
                participant.close_slash_report()?;
            }
            forfeit
        },
        None => 0,
    };
    report.record_charge(bounty_share(forfeit, bounty_bps));
    Ok(forfeit)
}

pub fn run_charge_slash_report_processor(
    context: Context<RunChargeSlashReportAccounts>,
    params: RunChargeSlashReportParams,
) -> Result<()> {
    let info = context.accounts.target_participant.to_account_info();
    let mut participant = {
        let data = info.try_borrow_data()?;
        Participant::from_pda_account(info.owner, &data)?
    };
    let equivocation = {
        let info = context.accounts.equivocation_bounty.to_account_info();
        let data = info.try_borrow_data()?;
        EquivocationBounty::from_pda_account(info.owner, &data)?
    };
    let bounty_bps = report_bounty_bps(
        &context.accounts.slash_report,
        &context.accounts.run,
        equivocation.as_ref(),
    );
    let forfeit = {
        let account = context.accounts.coordinator_account.load()?;
        charge_slash_report(
            &mut context.accounts.slash_report,
            participant.as_mut(),
            &account.state.coordinator,
            coordinator_slashed_points(&account, &params.target),
            bounty_bps,
        )?
    };
    if let Some(participant) = participant.as_ref() {
        let mut data = info.try_borrow_mut_data()?;
        participant.try_serialize(&mut &mut data[..])?;
    }
    // What the bond paid leaves the bonded total; the vault keeps it, and
    // the reporter's share of it is credited from there.
    let run = &mut context.accounts.run;
    run.total_bonded_amount = run
        .total_bonded_amount
        .checked_sub(forfeit)
        .ok_or_else(|| error!(ProgramError::InvalidParameter))?;
    Ok(())
}
