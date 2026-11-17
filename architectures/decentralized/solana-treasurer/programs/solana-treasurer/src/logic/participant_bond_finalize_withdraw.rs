use anchor_lang::prelude::*;
use anchor_spl::token::Token;
use anchor_spl::token::TokenAccount;
use anchor_spl::token::Transfer;
use anchor_spl::token::transfer;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::ProgramError;
use crate::logic::bounty_amount;
use crate::logic::charge_conviction;
use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::WithdrawalHold;
use crate::state::WithdrawalStanding;
use crate::state::coordinator_slashed_points;
use crate::state::unattributed_ledger_forfeit;
use crate::state::withdrawal_standing;

#[derive(Accounts)]
#[instruction(params: ParticipantBondFinalizeWithdrawParams)]
pub struct ParticipantBondFinalizeWithdrawAccounts<'info> {
    #[account()]
    pub user: Signer<'info>,

    #[account(
        mut,
        constraint = user_collateral.mint == run.collateral_mint,
        constraint = user_collateral.owner == user.key(),
        constraint = user_collateral.delegate == None.into(),
    )]
    pub user_collateral: Box<Account<'info, TokenAccount>>,

    #[account(
        mut,
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    #[account(
        mut,
        associated_token::mint = run.collateral_mint,
        associated_token::authority = run,
    )]
    pub run_collateral: Box<Account<'info, TokenAccount>>,

    #[account(
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            user.key().as_ref()
        ],
        bump = participant.bump
    )]
    pub participant: Box<Account<'info, Participant>>,

    /// CHECK: pinned to its PDA by the seeds below, and read only after this
    /// checks that it exists and belongs to this program.
    ///
    /// Not an `Option`. Anchor skips every constraint on an optional account the
    /// client declines to pass, including the seeds - so a participant convicted
    /// by a full verifier quorum could simply leave the slot empty, fall into
    /// the branch that pays an unchecked recipient, and take the whole bounty
    /// itself (wienerlabs/leviathan#15, finding 2). The address is derived here,
    /// so an empty account at it means there really is no verdict.
    ///
    /// Writable because a conviction that ran out its windows without reaching
    /// the ledger is settled here, and recorded as upheld
    /// (wienerlabs/leviathan-net#36), and one that ejected the participant is
    /// charged here if nobody charged it before (wienerlabs/leviathan-net#64).
    /// Either records what the forfeit owes the conviction's recipients.
    #[account(
        mut,
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            user.key().as_ref(),
        ],
        bump,
    )]
    pub audit_verdict: UncheckedAccount<'info>,

    #[account()]
    pub token_program: Program<'info, Token>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ParticipantBondFinalizeWithdrawParams {}

pub fn participant_bond_finalize_withdraw_processor<'info>(
    context: Context<'_, '_, 'info, 'info, ParticipantBondFinalizeWithdrawAccounts<'info>>,
    _params: ParticipantBondFinalizeWithdrawParams,
) -> Result<()> {
    let user_key = context.accounts.user.key();

    // The verdict is read from the account at its own PDA, which the caller
    // cannot substitute or leave out. An account with no data there is the
    // only thing that counts as "no verdict".
    let mut own_verdict = {
        let info = context.accounts.audit_verdict.to_account_info();
        let data = info.try_borrow_data()?;
        AuditVerdict::from_pda_account(info.owner, &data)?
    };

    let now = Clock::get()?.unix_timestamp;
    let coordinator_account = context.accounts.coordinator_account.load()?;
    let participant_slashed_points =
        coordinator_slashed_points(&coordinator_account, &user_key);
    let slashing_rate = coordinator_account
        .state
        .clients_state
        .current_epoch_rates
        .slashing_rate_per_client;
    let standing = withdrawal_standing(
        &context.accounts.run,
        &coordinator_account,
        &context.accounts.participant,
        own_verdict.as_ref(),
        &user_key,
        now,
    );

    // The withdrawal delay is there so a cheater cannot leave between the fraud
    // and the conviction, and it only works if the conviction can reach the
    // bond first. It cannot while it is still inside its challenge window or
    // appeal, or while the epoch it would be charged in is still running:
    // settling then would find nothing owed and pay the whole bond out
    // (wienerlabs/leviathan-net#36).
    let settle_conviction = match standing {
        WithdrawalStanding::Held(WithdrawalHold::ConvictionPending) => {
            return err!(ProgramError::ConvictionPending)
        },
        WithdrawalStanding::Held(WithdrawalHold::EpochNotSettled) => {
            return err!(ProgramError::EpochNotSettled)
        },
        WithdrawalStanding::Held(WithdrawalHold::VotesAtStake) => {
            return err!(ProgramError::VerdictVotesOpen)
        },
        WithdrawalStanding::Free { settle_conviction } => settle_conviction,
    };

    let participant = &mut context.accounts.participant;
    let run = &mut context.accounts.run;

    if participant.bond_withdraw_pending_amount == 0 {
        return err!(ProgramError::InvalidParameter);
    }

    let unlock_unix_timestamp = participant.bond_withdraw_requested_at
        + participant.bond_withdraw_delay_snapshot;
    if now < unlock_unix_timestamp {
        return err!(ProgramError::WithdrawDelayNotElapsed);
    }

    // Every point the ledger charged that no verdict accounts for, such as a
    // `run_slash` or an equivocation, read before anything is charged, the
    // way a client reads it to know whether to name a reporter.
    let ledger_forfeit = unattributed_ledger_forfeit(
        participant,
        participant_slashed_points,
        own_verdict.as_ref(),
    );

    // A conviction that ejected this participant takes its own charge first,
    // and its verdict records what that owes the conviction's recipients,
    // who are credited seat by seat, never paid here. Anyone may have charged
    // it already through `run_charge_conviction`, which this is. Paying them
    // here used to read whoever the verdict account named by now, which a
    // later dispute could have replaced, and to need every recipient's token
    // account in this one transaction (wienerlabs/leviathan-net#64).
    let mut verdict_changed = false;
    let conviction_forfeit = match own_verdict.as_mut() {
        Some(verdict) if verdict.awaits_charge() => {
            verdict_changed = true;
            charge_conviction(
                verdict,
                Some(participant),
                run,
                &coordinator_account.state.coordinator,
                participant_slashed_points,
            )?
        },
        _ => 0,
    };
    drop(coordinator_account);

    // Then the rest of what the ledger charged.
    participant.settle_ledger_points(ledger_forfeit);

    // A conviction that ran out its windows but never reached the ledger, and
    // now never can, because the participant is not in an epoch that will be
    // charged: it left, the run halted, or nobody finalised it in time. It is
    // charged here at the rate the ledger would have charged, so it is neither
    // lost nor left holding the bond forever, and recorded as upheld so it is
    // never charged twice. The ledger never saw it, so it is not counted as a
    // settled point. What it owes its recipients is recorded on the verdict
    // like any other conviction's.
    let unlanded_forfeit = if settle_conviction {
        let forfeit = slashing_rate.min(participant.bond_amount);
        participant.bond_amount -= forfeit;
        if let Some(verdict) = own_verdict.as_mut() {
            verdict.settle_unlanded(bounty_amount(run, forfeit));
            verdict_changed = true;
        }
        msg!(
            "finalize_withdraw: settled a conviction that never reached the ledger, forfeit {}",
            forfeit
        );
        forfeit
    } else {
        0
    };

    if verdict_changed {
        if let Some(verdict) = own_verdict.as_ref() {
            let info = context.accounts.audit_verdict.to_account_info();
            let mut data = info.try_borrow_mut_data()?;
            verdict.try_serialize(&mut &mut data[..])?;
        }
    }

    let payout_amount = participant
        .bond_withdraw_pending_amount
        .min(participant.bond_amount);
    participant.bond_amount -= payout_amount;
    participant.bond_withdraw_pending_amount = 0;
    participant.bond_withdraw_requested_at = 0;

    run.total_bonded_amount -=
        conviction_forfeit + ledger_forfeit + unlanded_forfeit + payout_amount;

    let run_index = run.index;
    let run_bump = run.bump;
    let collateral_mint = run.collateral_mint;
    let reporter_bounty = bounty_amount(run, ledger_forfeit);

    let index_bytes = run_index.to_le_bytes();
    let run_signer_seeds: &[&[&[u8]]] = &[&[Run::SEEDS_PREFIX, &index_bytes, &[run_bump]]];

    if reporter_bounty > 0 {
        // No verdict accounts for this forfeit, so its bounty goes to whoever
        // reported the fraud out of band. That account has to be a token
        // account of the run's collateral, and has to belong to somebody other
        // than the participant being slashed: this branch used to check
        // nothing at all, so the participant named its own token account and
        // took back the bounty half of its own forfeiture
        // (wienerlabs/leviathan#15, finding 2). That the participant still
        // names it is wienerlabs/leviathan-net#72.
        let reporter = context
            .remaining_accounts
            .first()
            .ok_or(error!(ProgramError::MissingReporter))?;
        let reporter_token_account =
            Account::<TokenAccount>::try_from(reporter)
                .map_err(|_| error!(ProgramError::InvalidBountyRecipient))?;
        if reporter_token_account.mint != collateral_mint
            || reporter_token_account.owner == user_key
        {
            return err!(ProgramError::InvalidBountyRecipient);
        }
        transfer(
            CpiContext::new(
                context.accounts.token_program.to_account_info(),
                Transfer {
                    from: context.accounts.run_collateral.to_account_info(),
                    to: reporter.to_account_info(),
                    authority: context.accounts.run.to_account_info(),
                },
            )
            .with_signer(run_signer_seeds),
            reporter_bounty,
        )?;
    }

    if payout_amount > 0 {
        transfer(
            CpiContext::new(
                context.accounts.token_program.to_account_info(),
                Transfer {
                    from: context.accounts.run_collateral.to_account_info(),
                    to: context.accounts.user_collateral.to_account_info(),
                    authority: context.accounts.run.to_account_info(),
                },
            )
            .with_signer(run_signer_seeds),
            payout_amount,
        )?;
    }

    Ok(())
}
