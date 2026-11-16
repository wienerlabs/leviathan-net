use anchor_lang::prelude::*;
use anchor_lang::solana_program::sysvar::instructions as instructions_sysvar;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;

use crate::state::CommitmentEvidence;
use crate::state::Run;
use crate::state::commitment_run;
use crate::state::transaction_proves_signature;
use crate::ProgramError;

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

    // An account of another version is refused, not loaded: loading one
    // of another size panics (wienerlabs/leviathan-net#62).
    #[account(
        mut,
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account()]
    pub coordinator_program: Program<'info, PsycheSolanaCoordinator>,

    /// CHECK: the instructions sysvar, pinned by address. It is read for the
    /// precompile instruction proving the target signed the commitment.
    #[account(address = instructions_sysvar::ID)]
    pub instructions: UncheckedAccount<'info>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSlashParams {
    pub index: u64,
    pub batch_start: u64,
    pub batch_end: u64,
    /// The payload the target committed to for `batch_start..=batch_end`. The
    /// same transaction has to carry an ed25519 precompile instruction with the
    /// target's p2p key signing its commitment hash.
    pub evidence: CommitmentEvidence,
    pub replayed_hash: [u8; 32],
}

pub fn run_slash_processor(
    context: Context<RunSlashAccounts>,
    params: RunSlashParams,
) -> Result<()> {
    // The run authority convicts on the same evidence a verifier quorum
    // does: a payload the target signed, from this epoch, for the batch named
    // (wienerlabs/leviathan-net#43).
    let (target_p2p, run) = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;
        let target = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(params.index as usize)
            .ok_or_else(|| error!(ProgramError::TargetMismatch))?;
        if !params.evidence.is_from_current_epoch(coordinator) {
            return err!(ProgramError::CommitmentOutsideEpoch);
        }
        (*target.id.p2p_identity(), commitment_run(coordinator))
    };
    let committed_hash = params.evidence.committed_hash(
        &run,
        params.batch_start,
        params.batch_end,
    );
    if !transaction_proves_signature(
        &context.accounts.instructions,
        &target_p2p,
        &committed_hash,
    )? {
        return err!(ProgramError::CommitmentNotSigned);
    }

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
            committed_hash,
            replayed_hash: params.replayed_hash,
        },
    )?;

    Ok(())
}
