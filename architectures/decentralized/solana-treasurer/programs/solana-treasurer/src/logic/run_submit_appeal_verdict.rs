use anchor_lang::prelude::*;
use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;

use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VerdictStatus;
use crate::state::MAX_APPEAL_VOTERS;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSubmitAppealVerdictParams)]
pub struct RunSubmitAppealVerdictAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account()]
    pub appellate: Signer<'info>,

    #[account(
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            appellate.key().as_ref(),
        ],
        bump = appellate_participant.bump,
    )]
    pub appellate_participant: Box<Account<'info, Participant>>,

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
pub struct RunSubmitAppealVerdictParams {
    pub target: Pubkey,
    pub target_index: u64,
    pub overturn: bool,
}

pub fn run_submit_appeal_verdict_processor(
    context: Context<RunSubmitAppealVerdictAccounts>,
    params: RunSubmitAppealVerdictParams,
) -> Result<()> {
    if context.accounts.appellate_participant.bond_amount
        < context.accounts.run.bond_minimum_amount
    {
        return err!(ProgramError::BondBelowMinimum);
    }

    let appellate_key = context.accounts.appellate.key();
    let tie_breaker_size = context.accounts.run.tie_breaker_committee_size;

    let quorum = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;

        let appellate_index = coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| *client.id.signer() == appellate_key.to_bytes())
            .ok_or_else(|| error!(ProgramError::VerifierNotInEpoch))?;

        let selection =
            CommitteeSelection::from_coordinator_with_tie_breakers(coordinator, 0, tie_breaker_size)
                .map_err(|_| error!(ProgramError::NotTieBreaker))?;
        if selection.get_committee(appellate_index as u64).committee != Committee::TieBreaker {
            return err!(ProgramError::NotTieBreaker);
        }

        let target_client = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(params.target_index as usize)
            .ok_or_else(|| error!(ProgramError::TargetMismatch))?;
        if *target_client.id.signer() != params.target.to_bytes() {
            return err!(ProgramError::TargetMismatch);
        }

        let tie_breaker_nodes = selection.get_num_tie_breaker_nodes();
        (2u64 * tie_breaker_nodes).div_ceil(3).max(1)
    };

    let should_slash;
    let batch_start;
    let batch_end;
    let committed_hash;
    let replayed_hash;
    {
        let verdict = &mut context.accounts.audit_verdict;
        if verdict.target != params.target {
            return err!(ProgramError::TargetMismatch);
        }
        if verdict.status != VerdictStatus::Challenged {
            return err!(ProgramError::VerdictNotChallenged);
        }
        if verdict.appeal_voters.iter().any(|voter| voter == &appellate_key) {
            return err!(ProgramError::DuplicateAppealVerdict);
        }
        if verdict.appeal_voters.len() >= MAX_APPEAL_VOTERS {
            return err!(ProgramError::AppealVotersFull);
        }

        verdict.appeal_voters.push(appellate_key);
        if params.overturn {
            verdict.overturn_count += 1;
        } else {
            verdict.uphold_count += 1;
        }

        msg!(
            "appeal_verdict: overturn={} uphold={} quorum={}",
            verdict.overturn_count,
            verdict.uphold_count,
            quorum
        );

        batch_start = verdict.batch_start;
        batch_end = verdict.batch_end;
        committed_hash = verdict.committed_hash;
        replayed_hash = verdict.replayed_hash;

        if (verdict.overturn_count as u64) >= quorum {
            verdict.status = VerdictStatus::Overturned;
            should_slash = false;
            msg!("appeal_verdict: overturned, verifiers forfeit their bonds");
        } else if (verdict.uphold_count as u64) >= quorum {
            verdict.status = VerdictStatus::Upheld;
            should_slash = true;
            msg!("appeal_verdict: upheld, target slash finalizes");
        } else {
            should_slash = false;
        }
    }

    if should_slash {
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
                index: params.target_index,
                batch_start,
                batch_end,
                committed_hash,
                replayed_hash,
            },
        )?;
    }

    Ok(())
}
