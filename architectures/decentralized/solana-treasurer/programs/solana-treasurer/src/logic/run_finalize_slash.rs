use anchor_lang::prelude::*;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;

use crate::state::AuditVerdict;
use crate::state::Run;
use crate::state::VerdictStatus;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunFinalizeSlashParams)]
pub struct RunFinalizeSlashAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account(
        constraint = run.coordinator_instance == coordinator_instance.key(),
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    #[account()]
    pub coordinator_instance: Account<'info, CoordinatorInstance>,

    #[account(mut)]
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

    #[account()]
    pub coordinator_program: Program<'info, PsycheSolanaCoordinator>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunFinalizeSlashParams {
    pub target: Pubkey,
    /// Where the target sits in the client list *now*.
    ///
    /// The verdict records the index it was convicted at, but
    /// `epoch_state.clients` is compacted at the end of every round, so anyone
    /// below the target leaving shifts it down. Trusting the stored index left
    /// a correct conviction permanently unfinalisable with `TargetMismatch` and
    /// no instruction to correct it (wienerlabs/leviathan#15, finding 5). Every
    /// other slash path already takes the index fresh and revalidates it; this
    /// one now does too.
    pub target_index: u64,
}

pub fn run_finalize_slash_processor(
    context: Context<RunFinalizeSlashAccounts>,
    params: RunFinalizeSlashParams,
) -> Result<()> {
    let challenge_window_seconds = context.accounts.run.challenge_window_seconds;
    let appeal_window_seconds = context.accounts.run.appeal_window_seconds;

    let target_index = params.target_index;
    let batch_start;
    let batch_end;
    let committed_hash;
    let replayed_hash;
    {
        let verdict = &mut context.accounts.audit_verdict;
        let now = Clock::get()?.unix_timestamp;
        match verdict.status {
            VerdictStatus::SlashPending => {
                if now < verdict.pending_since_unix + challenge_window_seconds {
                    return err!(ProgramError::ChallengeWindowOpen);
                }
            },
            // A challenge that the bench never resolved does not get to hold the
            // conviction for ever. Past the appeal window the verdict the
            // verifiers already reached finalises, which is what a target with
            // nothing to say in its defence would have got anyway (finding 10).
            // `appeal_window_seconds == 0` keeps the old behaviour: no timeout.
            VerdictStatus::Challenged => {
                if appeal_window_seconds == 0
                    || now < verdict.challenged_since_unix + appeal_window_seconds
                {
                    return err!(ProgramError::AppealWindowOpen);
                }
            },
            _ => return err!(ProgramError::VerdictNotPending),
        }
        verdict.status = VerdictStatus::Upheld;
        batch_start = verdict.batch_start;
        batch_end = verdict.batch_end;
        committed_hash = verdict.committed_hash;
        replayed_hash = verdict.replayed_hash;
    }

    {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;
        let target_client = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(target_index as usize)
            .ok_or_else(|| error!(ProgramError::TargetMismatch))?;
        if *target_client.id.signer()
            != context.accounts.audit_verdict.target.to_bytes()
        {
            return err!(ProgramError::TargetMismatch);
        }
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
            index: target_index,
            batch_start,
            batch_end,
            committed_hash,
            replayed_hash,
        },
    )?;

    Ok(())
}
