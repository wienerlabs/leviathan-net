use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::logic::bounty_amount;
use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VerdictStatus;
use crate::state::coordinator_slashed_points;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSlashLosingVerifierParams)]
pub struct RunSlashLosingVerifierAccounts<'info> {
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

    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            params.loser.as_ref(),
        ],
        bump = loser_participant.bump,
    )]
    pub loser_participant: Box<Account<'info, Participant>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSlashLosingVerifierParams {
    pub target: Pubkey,
    pub voter_position: u16,
    /// The verifier at `voter_position`, whose participant account is charged.
    pub loser: Pubkey,
}

/// Charges the next verifier whose vote an appeal overturned, and records what
/// its forfeit owes the bench.
///
/// The charge used to go through the coordinator: the crank ejected the loser,
/// the epoch's end charged the ejection, and the bond paid at the loser's own
/// withdrawal. That reached a loser only while it was on the roster of an epoch
/// that would charge it, so one that left the run, or a run that halted, was out
/// of reach for good. Its withdrawal paid the bench only if it chose to name the
/// overturned verdict. Now the crank forfeits the epoch's slashing rate from the
/// bond directly, the way an unlanded conviction is settled at a target's
/// withdrawal. It settles every loser wherever it is, and the loser's
/// withdrawal, held while the vote was counted against it, waits for exactly
/// this. The bench's share is paid seat by seat through
/// `run_pay_bench_bounty`, so no payout, and no number of seats, can stop a
/// loser being charged (wienerlabs/leviathan-net#39).
pub fn run_slash_losing_verifier_processor(
    context: Context<RunSlashLosingVerifierAccounts>,
    params: RunSlashLosingVerifierParams,
) -> Result<()> {
    let position = {
        let verdict = &context.accounts.audit_verdict;
        if verdict.status != VerdictStatus::Overturned {
            return err!(ProgramError::VerdictNotOverturned);
        }
        if (verdict.settled_count as usize) >= verdict.voters.len() {
            return err!(ProgramError::AllLosersSettled);
        }
        if params.voter_position != verdict.settled_count {
            return err!(ProgramError::InvalidParameter);
        }
        let position = params.voter_position as usize;
        if verdict.voters[position] != params.loser {
            return err!(ProgramError::InvalidParameter);
        }
        position
    };

    let (slashed_points, slashing_rate) = {
        let account = context.accounts.coordinator_account.load()?;
        (
            coordinator_slashed_points(&account, &params.loser),
            account
                .state
                .clients_state
                .current_epoch_rates
                .slashing_rate_per_client,
        )
    };

    let forfeit = {
        let loser = &mut context.accounts.loser_participant;
        let forfeit =
            loser.forfeit_unclaimed_bond(slashing_rate, slashed_points);
        loser.close_verdict_vote()?;
        forfeit
    };
    let bench_bounty = bounty_amount(&context.accounts.run, forfeit);
    context
        .accounts
        .audit_verdict
        .settle_losing_vote(position, bench_bounty);
    context.accounts.run.total_bonded_amount -= forfeit;

    msg!(
        "slash_losing_verifier: loser={} settled, forfeit {}, bench owed {}",
        params.loser,
        forfeit,
        bench_bounty
    );

    Ok(())
}
