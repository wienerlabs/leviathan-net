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
use crate::state::Landing;
use crate::state::appeal_window_open;
use crate::state::conviction_landing;
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

    // An account of another version is refused, not loaded: loading one
    // of another size panics (wienerlabs/leviathan-net#62).
    #[account(
        mut,
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
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
    let landing = {
        let account = context.accounts.coordinator_account.load()?;
        conviction_landing(
            &account.state.coordinator,
            &context.accounts.audit_verdict.target,
            target_index,
        )?
    };

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
            // Appeals are on only with a deadline (#36), so a zero window means
            // the run switched appeals off while this one was open: there is no
            // deadline left to wait for. It used to mean no timeout at all,
            // which parked the conviction, and since #39 every vote on it,
            // for good.
            VerdictStatus::Challenged => {
                if appeal_window_open(
                    appeal_window_seconds,
                    verdict.challenged_since_unix,
                    now,
                ) {
                    return err!(ProgramError::AppealWindowOpen);
                }
            },
            // Already standing; it only waits for an epoch that will charge it.
            VerdictStatus::Due => {},
            _ => return err!(ProgramError::VerdictNotPending),
        }
        match landing {
            Landing::Eject(_) => verdict.status = VerdictStatus::Upheld,
            // Finalising where no ejection would be charged (between epochs, in
            // a warmup, or with the target off the roster) used to eject onto a
            // roster nobody charges and lose the conviction. It is recorded as
            // due instead, to land in a later epoch or at the target's
            // withdrawal (wienerlabs/leviathan-net#36).
            Landing::Due => {
                if verdict.status == VerdictStatus::Due {
                    return err!(ProgramError::TargetNotInChargingEpoch);
                }
                verdict.status = VerdictStatus::Due;
                msg!("finalize_slash: target not in a charging epoch, conviction due");
                return Ok(());
            },
        }
        batch_start = verdict.batch_start;
        batch_end = verdict.batch_end;
        committed_hash = verdict.committed_hash;
        replayed_hash = verdict.replayed_hash;
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
