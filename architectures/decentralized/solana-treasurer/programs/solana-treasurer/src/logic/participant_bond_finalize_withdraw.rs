use anchor_lang::prelude::*;
use anchor_spl::token::Token;
use anchor_spl::token::TokenAccount;
use anchor_spl::token::Transfer;
use anchor_spl::token::transfer;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::ProgramError;
use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;

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
        constraint = coordinator_account.load()?.version == CoordinatorAccount::VERSION,
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

    #[account(
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            user.key().as_ref(),
        ],
        bump,
    )]
    pub audit_verdict: Option<Box<Account<'info, AuditVerdict>>>,

    #[account()]
    pub token_program: Program<'info, Token>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ParticipantBondFinalizeWithdrawParams {}

pub fn participant_bond_finalize_withdraw_processor<'info>(
    context: Context<'_, '_, 'info, 'info, ParticipantBondFinalizeWithdrawAccounts<'info>>,
    _params: ParticipantBondFinalizeWithdrawParams,
) -> Result<()> {
    let mut participant_slashed_points = 0;
    for client in context
        .accounts
        .coordinator_account
        .load()?
        .state
        .clients_state
        .clients
        .iter()
    {
        if *client.id.signer() == context.accounts.user.key().to_bytes() {
            participant_slashed_points = client.slashed;
            break;
        }
    }

    let participant = &mut context.accounts.participant;
    let run = &mut context.accounts.run;

    if participant.bond_withdraw_pending_amount == 0 {
        return err!(ProgramError::InvalidParameter);
    }

    let unlock_unix_timestamp = participant.bond_withdraw_requested_at
        + run.bond_withdraw_delay_seconds;
    if Clock::get()?.unix_timestamp < unlock_unix_timestamp {
        return err!(ProgramError::WithdrawDelayNotElapsed);
    }

    let unsettled_slashed_points = participant_slashed_points
        .saturating_sub(participant.bond_settled_slashed_points);
    let forfeited_amount =
        unsettled_slashed_points.min(participant.bond_amount);
    participant.bond_settled_slashed_points += forfeited_amount;
    participant.bond_amount -= forfeited_amount;

    let payout_amount = participant
        .bond_withdraw_pending_amount
        .min(participant.bond_amount);
    participant.bond_amount -= payout_amount;
    participant.bond_withdraw_pending_amount = 0;
    participant.bond_withdraw_requested_at = 0;

    run.total_bonded_amount -= forfeited_amount + payout_amount;

    let run_index = run.index;
    let run_bump = run.bump;
    let collateral_mint = run.collateral_mint;
    let bounty_amount = if run.slash_bounty_bps > 0 {
        (forfeited_amount as u128 * run.slash_bounty_bps as u128 / 10_000) as u64
    } else {
        0
    };

    let index_bytes = run_index.to_le_bytes();
    let run_signer_seeds: &[&[&[u8]]] = &[&[Run::SEEDS_PREFIX, &index_bytes, &[run_bump]]];

    if bounty_amount > 0 {
        let voters: Vec<Pubkey> = context
            .accounts
            .audit_verdict
            .as_ref()
            .map(|verdict| verdict.voters.clone())
            .unwrap_or_default();

        if voters.is_empty() {
            let reporter = context
                .remaining_accounts
                .first()
                .ok_or(error!(ProgramError::MissingReporter))?;
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
                bounty_amount,
            )?;
        } else {
            if context.remaining_accounts.len() < voters.len() {
                return err!(ProgramError::MissingReporter);
            }
            let share = bounty_amount / voters.len() as u64;
            for (position, voter) in voters.iter().enumerate() {
                if share == 0 {
                    break;
                }
                let recipient = &context.remaining_accounts[position];
                let recipient_token_account =
                    Account::<TokenAccount>::try_from(recipient)
                        .map_err(|_| error!(ProgramError::BountyRecipientMismatch))?;
                if recipient_token_account.owner != *voter
                    || recipient_token_account.mint != collateral_mint
                {
                    return err!(ProgramError::BountyRecipientMismatch);
                }
                transfer(
                    CpiContext::new(
                        context.accounts.token_program.to_account_info(),
                        Transfer {
                            from: context.accounts.run_collateral.to_account_info(),
                            to: recipient.to_account_info(),
                            authority: context.accounts.run.to_account_info(),
                        },
                    )
                    .with_signer(run_signer_seeds),
                    share,
                )?;
            }
        }
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
