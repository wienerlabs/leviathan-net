use anchor_lang::prelude::*;

use crate::ProgramError;
use crate::state::Run;

#[derive(Accounts)]
#[instruction(params: RunBondConfigUpdateParams)]
pub struct RunBondConfigUpdateAccounts<'info> {
    #[account()]
    pub main_authority: Signer<'info>,

    #[account(
        mut,
        constraint = run.main_authority == main_authority.key(),
    )]
    pub run: Box<Account<'info, Run>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunBondConfigUpdateParams {
    pub bond_minimum_amount: u64,
    pub bond_withdraw_delay_seconds: i64,
}

pub fn run_bond_config_update_processor(
    context: Context<RunBondConfigUpdateAccounts>,
    params: RunBondConfigUpdateParams,
) -> Result<()> {
    if params.bond_withdraw_delay_seconds < 0 {
        return err!(ProgramError::InvalidParameter);
    }

    let run = &mut context.accounts.run;
    run.bond_minimum_amount = params.bond_minimum_amount;
    run.bond_withdraw_delay_seconds = params.bond_withdraw_delay_seconds;

    Ok(())
}
