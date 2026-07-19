use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;

use crate::state::Run;

#[derive(Accounts)]
#[instruction(params: RunSlashParams)]
pub struct RunSlashAccounts<'info> {
    #[account()]
    pub authority: Signer<'info>,

    #[account(
        constraint = run.main_authority == authority.key(),
        constraint = run.coordinator_instance == coordinator_instance.key(),
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    #[account()]
    pub coordinator_instance: Account<'info, CoordinatorInstance>,

    #[account(mut)]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account()]
    pub coordinator_program: Program<'info, PsycheSolanaCoordinator>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSlashParams {
    pub index: u64,
    pub batch_start: u64,
    pub batch_end: u64,
    pub committed_hash: [u8; 32],
    pub replayed_hash: [u8; 32],
}

pub fn run_slash_processor(
    context: Context<RunSlashAccounts>,
    params: RunSlashParams,
) -> Result<()> {
    let run = &context.accounts.run;
    let run_signer_seeds: &[&[&[u8]]] =
        &[&[Run::SEEDS_PREFIX, &run.index.to_le_bytes(), &[run.bump]]];

    slash_client(
        CpiContext::new(
            context.accounts.coordinator_program.to_account_info(),
            OwnerCoordinatorAccounts {
                authority: context.accounts.run.to_account_info(),
                coordinator_instance: context
                    .accounts
                    .coordinator_instance
                    .to_account_info(),
                coordinator_account: context
                    .accounts
                    .coordinator_account
                    .to_account_info(),
            },
        )
        .with_signer(run_signer_seeds),
        SlashClientParams {
            index: params.index,
            batch_start: params.batch_start,
            batch_end: params.batch_end,
            committed_hash: params.committed_hash,
            replayed_hash: params.replayed_hash,
        },
    )?;

    Ok(())
}
