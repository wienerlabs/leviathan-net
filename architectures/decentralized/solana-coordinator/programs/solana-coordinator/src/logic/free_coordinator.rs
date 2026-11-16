use anchor_lang::prelude::*;

use crate::CoordinatorAccount;
use crate::CoordinatorInstance;
use crate::bytes_from_string;
use crate::program_error::ProgramError;

#[derive(Accounts)]
#[instruction(params: FreeCoordinatorParams)]
pub struct FreeCoordinatorAccounts<'info> {
    #[account()]
    pub authority: Signer<'info>,

    /// CHECK: TODO why is this ok to be unchecked?
    #[account(mut)]
    pub spill: SystemAccount<'info>,

    #[account(
        mut,
        seeds = [
            CoordinatorInstance::SEEDS_PREFIX,
            bytes_from_string(&coordinator_instance.run_id)
        ],
        bump = coordinator_instance.bump,
        constraint = coordinator_instance.main_authority == authority.key(),
        close = spill,
    )]
    pub coordinator_instance: Box<Account<'info, CoordinatorInstance>>,

    #[account(
        mut,
        constraint = coordinator_instance.coordinator_account == coordinator_account.key(),
        close = spill,
    )]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct FreeCoordinatorParams {}

pub fn free_coordinator_processor(
    context: Context<FreeCoordinatorAccounts>,
    _params: FreeCoordinatorParams,
) -> Result<()> {
    // An account of another version runs nothing under this program, so its
    // authority may close it and take its rent back; loading it to ask
    // whether it is halted would panic on its size. One of this version
    // closes only once halted (wienerlabs/leviathan-net#62).
    let coordinator_account = &context.accounts.coordinator_account;
    if CoordinatorAccount::is_current(coordinator_account)?
        && !coordinator_account.load()?.state.coordinator.halted()
    {
        return err!(ProgramError::CloseCoordinatorNotHalted);
    }
    Ok(())
}
