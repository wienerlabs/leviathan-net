use anchor_lang::prelude::*;
use psyche_coordinator::ClientState;
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
#[instruction(params: RunSlashLosingVerifierParams)]
pub struct RunSlashLosingVerifierAccounts<'info> {
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
pub struct RunSlashLosingVerifierParams {
    pub target: Pubkey,
    pub voter_position: u16,
}

pub fn run_slash_losing_verifier_processor(
    context: Context<RunSlashLosingVerifierAccounts>,
    params: RunSlashLosingVerifierParams,
) -> Result<()> {
    let loser = {
        let verdict = &context.accounts.audit_verdict;
        if verdict.status != VerdictStatus::Overturned {
            return err!(ProgramError::VerdictNotOverturned);
        }
        if (verdict.settled_count as usize) >= verdict.voters.len() {
            return err!(ProgramError::AllLosersSettled);
        }
        if params.voter_position != verdict.settled_count {
            return err!(ProgramError::InvalidParameter);
        }
        verdict.voters[verdict.settled_count as usize]
    };

    let (loser_index, slashable) = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;
        let found = coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| *client.id.signer() == loser.to_bytes());
        match found {
            Some(idx) => {
                let state = coordinator.epoch_state.clients[idx].state;
                let slashable =
                    state == ClientState::Healthy || state == ClientState::Dropped;
                (Some(idx as u64), slashable)
            }
            None => (None, false),
        }
    };

    let (batch_start, batch_end, committed_hash, replayed_hash) = {
        let verdict = &mut context.accounts.audit_verdict;
        verdict.settled_count += 1;
        (
            verdict.batch_start,
            verdict.batch_end,
            verdict.committed_hash,
            verdict.replayed_hash,
        )
    };

    msg!(
        "slash_losing_verifier: loser={} slashable={}",
        loser,
        slashable
    );

    if let (Some(index), true) = (loser_index, slashable) {
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
                index,
                batch_start,
                batch_end,
                committed_hash,
                replayed_hash,
            },
        )?;
    }

    Ok(())
}
