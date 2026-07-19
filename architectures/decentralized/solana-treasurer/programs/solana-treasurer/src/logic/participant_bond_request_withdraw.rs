use anchor_lang::prelude::*;

use crate::ProgramError;
use crate::state::Participant;
use crate::state::Run;

#[derive(Accounts)]
#[instruction(params: ParticipantBondRequestWithdrawParams)]
pub struct ParticipantBondRequestWithdrawAccounts<'info> {
    #[account()]
    pub user: Signer<'info>,

    #[account()]
    pub run: Box<Account<'info, Run>>,

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
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct ParticipantBondRequestWithdrawParams {
    pub collateral_amount: u64,
}

pub fn participant_bond_request_withdraw_processor(
    context: Context<ParticipantBondRequestWithdrawAccounts>,
    params: ParticipantBondRequestWithdrawParams,
) -> Result<()> {
    let participant = &mut context.accounts.participant;

    let available_bond_amount =
        participant.bond_amount - participant.bond_withdraw_pending_amount;
    if params.collateral_amount == 0
        || params.collateral_amount > available_bond_amount
    {
        return err!(ProgramError::InsufficientBond);
    }

    participant.bond_withdraw_pending_amount += params.collateral_amount;
    participant.bond_withdraw_requested_at = Clock::get()?.unix_timestamp;

    Ok(())
}
