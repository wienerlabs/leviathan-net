use anchor_lang::prelude::*;
use anchor_spl::token::Token;
use anchor_spl::token::TokenAccount;
use anchor_spl::token::Transfer;
use anchor_spl::token::transfer;

use crate::ProgramError;
use crate::state::Participant;
use crate::state::Run;

#[derive(Accounts)]
#[instruction(params: ParticipantBondDepositParams)]
pub struct ParticipantBondDepositAccounts<'info> {
    #[account()]
    pub user: Signer<'info>,

    #[account(
        mut,
        constraint = user_collateral.mint == run.collateral_mint,
        constraint = user_collateral.owner == user.key(),
        constraint = user_collateral.delegate == None.into(),
    )]
    pub user_collateral: Box<Account<'info, TokenAccount>>,

    #[account(mut)]
    pub run: Box<Account<'info, Run>>,

    #[account(
        mut,
        associated_token::mint = run.collateral_mint,
        associated_token::authority = run,
    )]
    pub run_collateral: Box<Account<'info, TokenAccount>>,

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

    #[account()]
    pub token_program: Program<'info, Token>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct ParticipantBondDepositParams {
    pub collateral_amount: u64,
}

pub fn participant_bond_deposit_processor(
    context: Context<ParticipantBondDepositAccounts>,
    params: ParticipantBondDepositParams,
) -> Result<()> {
    if params.collateral_amount == 0 {
        return err!(ProgramError::InvalidParameter);
    }

    transfer(
        CpiContext::new(
            context.accounts.token_program.to_account_info(),
            Transfer {
                authority: context.accounts.user.to_account_info(),
                from: context.accounts.user_collateral.to_account_info(),
                to: context.accounts.run_collateral.to_account_info(),
            },
        ),
        params.collateral_amount,
    )?;

    let participant = &mut context.accounts.participant;
    let run = &mut context.accounts.run;

    participant.bond_amount += params.collateral_amount;
    run.total_bonded_amount += params.collateral_amount;

    Ok(())
}
