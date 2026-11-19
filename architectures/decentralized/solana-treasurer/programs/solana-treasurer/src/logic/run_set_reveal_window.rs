use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::check_reveal_window;
use crate::state::RevealWindow;
use crate::state::Run;

/// Sets how long a trainer has, into a round's witness phase, to reveal the
/// state a payload commits to before verifiers may vote that it did not
/// (wienerlabs/leviathan-net#89).
#[derive(Accounts)]
#[instruction(params: RunSetRevealWindowParams)]
pub struct RunSetRevealWindowAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account()]
    pub main_authority: Signer<'info>,

    #[account(
        constraint = run.main_authority == main_authority.key(),
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    // An account of another version is refused, not loaded: loading one of
    // another size panics (wienerlabs/leviathan-net#62).
    #[account(constraint = CoordinatorAccount::is_current(&coordinator_account)?)]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account(
        init_if_needed,
        payer = payer,
        space = RevealWindow::space_with_discriminator(),
        seeds = [RevealWindow::SEEDS_PREFIX, run.key().as_ref()],
        bump,
    )]
    pub reveal_window: Box<Account<'info, RevealWindow>>,

    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunSetRevealWindowParams {
    pub reveal_seconds: u64,
}

pub fn run_set_reveal_window_processor(
    context: Context<RunSetRevealWindowAccounts>,
    params: RunSetRevealWindowParams,
) -> Result<()> {
    let round_witness_time = context
        .accounts
        .coordinator_account
        .load()?
        .state
        .coordinator
        .config
        .round_witness_time;
    check_reveal_window(params.reveal_seconds, round_witness_time)?;
    let window = &mut context.accounts.reveal_window;
    window.bump = context.bumps.reveal_window;
    window.run = context.accounts.run.key();
    window.reveal_seconds = params.reveal_seconds;
    Ok(())
}
