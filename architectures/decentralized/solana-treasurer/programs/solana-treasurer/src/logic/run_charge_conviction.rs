use anchor_lang::prelude::*;
use psyche_coordinator::Coordinator;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::logic::bounty_amount;
use crate::state::coordinator_slashed_points;
use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunChargeConvictionParams)]
pub struct RunChargeConvictionAccounts<'info> {
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
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump = audit_verdict.bump,
    )]
    pub audit_verdict: Box<Account<'info, AuditVerdict>>,

    /// CHECK: pinned to the target's participant PDA by the seeds below, and
    /// read only after this checks that it exists and belongs to this program.
    ///
    /// Not required to exist. A run whose join authority is not the treasurer
    /// admits clients that never created a participant account, and a verdict
    /// convicts them all the same. Requiring one here left such a conviction
    /// uncharged for good, and its verdict could then never be taken over: the
    /// target could not be disputed again. With no account at this address
    /// there is no bond, so the conviction is charged nothing.
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
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunChargeConvictionParams {
    pub target: Pubkey,
}

/// Charges the target's bond, `participant`, for the conviction `verdict`
/// landed as an ejection, now that the ledger has charged it, and records what
/// that forfeit owes the conviction's recipients. Returns the forfeit, which
/// the caller takes off the run's bonded total.
///
/// The forfeit is the ledger's charge for the epoch the conviction ejected the
/// target in, out of what the bond still owes and holds, settled the way a
/// withdrawal settles it, so it is never charged twice. A target with no
/// participant account has no bond, and is charged nothing.
pub fn charge_conviction(
    verdict: &mut AuditVerdict,
    participant: Option<&mut Participant>,
    run: &Run,
    coordinator: &Coordinator,
    slashed_points: u64,
) -> Result<u64> {
    if !verdict.awaits_charge() {
        return err!(ProgramError::ConvictionNotChargeable);
    }
    if !verdict.ejection_charged(coordinator) {
        return err!(ProgramError::EjectionNotCharged);
    }
    let forfeit = match participant {
        Some(participant) => {
            let forfeit = participant.ejected_conviction_forfeit(
                slashed_points,
                verdict.charge_rate,
            );
            participant.settle_ledger_points(forfeit);
            forfeit
        },
        None => 0,
    };
    verdict.record_charge(bounty_amount(run, forfeit));
    Ok(forfeit)
}

/// Charges a target's bond for a conviction that ejected it, once the epoch it
/// was ejected in has been charged, and records what that owes the
/// conviction's voters, or the bench that upheld it.
///
/// The bounty used to be paid at the target's own withdrawal, to whoever its
/// verdict account named then, in one transaction with every recipient's
/// token account. A later dispute could take the account over first, an
/// overturned verdict named its losers, and a committee larger than a
/// transaction holds left the target unable to withdraw. Now the charge is
/// made here, by anyone, without the target and without a token account, and
/// each recipient is credited through `run_pay_verdict_bounty`
/// (wienerlabs/leviathan-net#64). The target's withdrawal makes the same
/// charge first if nobody has.
pub fn run_charge_conviction_processor(
    context: Context<RunChargeConvictionAccounts>,
    params: RunChargeConvictionParams,
) -> Result<()> {
    let info = context.accounts.target_participant.to_account_info();
    let mut participant = {
        let data = info.try_borrow_data()?;
        Participant::from_pda_account(info.owner, &data)?
    };
    let forfeit = {
        let account = context.accounts.coordinator_account.load()?;
        charge_conviction(
            &mut context.accounts.audit_verdict,
            participant.as_mut(),
            &context.accounts.run,
            &account.state.coordinator,
            coordinator_slashed_points(&account, &params.target),
        )?
    };
    if let Some(participant) = participant.as_ref() {
        let mut data = info.try_borrow_mut_data()?;
        participant.try_serialize(&mut &mut data[..])?;
    }
    let run = &mut context.accounts.run;
    run.total_bonded_amount -= forfeit;
    msg!(
        "charge_conviction: target={} forfeit {} owes {}",
        params.target,
        forfeit,
        context.accounts.audit_verdict.bounty_owed
    );
    Ok(())
}
