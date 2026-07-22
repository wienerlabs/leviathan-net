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
use crate::state::MAX_VERDICT_VOTERS;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSubmitAuditVerdictParams)]
pub struct RunSubmitAuditVerdictAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account()]
    pub verifier: Signer<'info>,

    #[account(
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            verifier.key().as_ref(),
        ],
        bump = verifier_participant.bump,
    )]
    pub verifier_participant: Box<Account<'info, Participant>>,

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
        init_if_needed,
        payer = payer,
        space = AuditVerdict::space_with_discriminator(),
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump,
    )]
    pub audit_verdict: Box<Account<'info, AuditVerdict>>,

    #[account()]
    pub coordinator_program: Program<'info, PsycheSolanaCoordinator>,

    #[account()]
    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSubmitAuditVerdictParams {
    pub target: Pubkey,
    pub target_index: u64,
    pub batch_start: u64,
    pub batch_end: u64,
    pub committed_hash: [u8; 32],
    pub replayed_hash: [u8; 32],
}

pub fn run_submit_audit_verdict_processor(
    context: Context<RunSubmitAuditVerdictAccounts>,
    params: RunSubmitAuditVerdictParams,
) -> Result<()> {
    if context.accounts.verifier_participant.bond_amount
        < context.accounts.run.bond_minimum_amount
    {
        return err!(ProgramError::BondBelowMinimum);
    }

    let verifier_key = context.accounts.verifier.key();

    let (current_epoch, quorum) = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;

        let verifier_index = coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| *client.id.signer() == verifier_key.to_bytes())
            .ok_or_else(|| error!(ProgramError::VerifierNotInEpoch))?;

        let selection = CommitteeSelection::from_coordinator(coordinator, 0)
            .map_err(|_| error!(ProgramError::VerifierNotAssigned))?;
        if selection.get_committee(verifier_index as u64).committee != Committee::Verifier {
            return err!(ProgramError::VerifierNotAssigned);
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

        let verifier_nodes = selection.get_num_verifier_nodes();
        let quorum = (2u64 * verifier_nodes).div_ceil(3).max(1);
        (coordinator.progress.epoch, quorum)
    };

    let should_slash;
    {
        let verdict = &mut context.accounts.audit_verdict;
        if verdict.run == Pubkey::default() {
            verdict.bump = context.bumps.audit_verdict;
            verdict.run = context.accounts.run.key();
            verdict.target = params.target;
            verdict.reset_for_epoch(current_epoch);
        } else if verdict.epoch != current_epoch {
            verdict.reset_for_epoch(current_epoch);
        }

        if verdict.resolved {
            return err!(ProgramError::VerdictAlreadyResolved);
        }
        if verdict.voters.iter().any(|voter| voter == &verifier_key) {
            return err!(ProgramError::DuplicateVerdict);
        }
        if verdict.voters.len() >= MAX_VERDICT_VOTERS {
            return err!(ProgramError::VerdictVotersFull);
        }

        verdict.voters.push(verifier_key);
        verdict.verdict_count += 1;
        verdict.committed_hash = params.committed_hash;
        verdict.replayed_hash = params.replayed_hash;

        msg!(
            "audit_verdict: target_index={} epoch={} count={} quorum={}",
            params.target_index,
            current_epoch,
            verdict.verdict_count,
            quorum
        );

        should_slash = (verdict.verdict_count as u64) >= quorum;
        if should_slash {
            verdict.resolved = true;
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
                batch_start: params.batch_start,
                batch_end: params.batch_end,
                committed_hash: params.committed_hash,
                replayed_hash: params.replayed_hash,
            },
        )?;
    }

    Ok(())
}
