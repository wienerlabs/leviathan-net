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
use crate::state::MAX_VERDICT_VOTERS;
use crate::ProgramError;

/// The smallest verifier committee a slash may come out of.
///
/// At one seat the old `(2*1).div_ceil(3).max(1)` gave a quorum of one, so a
/// single verifier ejected whoever it named and "quorum" meant nothing
/// (wienerlabs/leviathan#15, finding 11). Two seats with a quorum of two is a
/// real, if small, agreement, and it is a configuration an operator may
/// legitimately want. One seat is not a committee, so it is refused rather than
/// silently given a quorum it cannot fail.
///
/// How large a committee should be for the sampling argument to hold is the
/// operator's call and is priced in `docs/COMMITTEE_ECONOMICS.md`; this is only
/// the floor below which the mechanism is not doing anything at all.
pub const MIN_VERIFIER_COMMITTEE: u64 = 2;

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
    let tie_breaker_size = context.accounts.run.tie_breaker_committee_size;

    let (current_epoch, current_round, quorum) = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;

        let verifier_index = coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| *client.id.signer() == verifier_key.to_bytes())
            .ok_or_else(|| error!(ProgramError::VerifierNotInEpoch))?;

        let selection =
            CommitteeSelection::from_coordinator_with_tie_breakers(coordinator, 0, tie_breaker_size)
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

        // A committee has to be big enough for agreement to mean something.
        // `verifier_nodes` rounds down, so small runs land on one seat, where
        // `.max(1)` made a single verifier a quorum of one and it could eject
        // anyone it named (wienerlabs/leviathan#15, finding 11).
        let verifier_nodes = selection.get_num_verifier_nodes();
        if verifier_nodes < MIN_VERIFIER_COMMITTEE {
            return err!(ProgramError::VerifierCommitteeTooSmall);
        }
        let quorum = (2u64 * verifier_nodes).div_ceil(3).max(2);

        // A quorum the voter list cannot hold is a verdict that can never
        // resolve, so refuse the configuration rather than stall on it
        // (finding 12).
        if quorum > MAX_VERDICT_VOTERS as u64 {
            return err!(ProgramError::QuorumExceedsVoterCapacity);
        }

        let current_round = coordinator
            .current_round()
            .ok_or_else(|| error!(ProgramError::VerifierNotAssigned))?
            .height;
        (coordinator.progress.epoch, current_round, quorum)
    };

    let should_slash;
    {
        let verdict = &mut context.accounts.audit_verdict;
        let fresh_account = verdict.run == Pubkey::default();
        if fresh_account {
            verdict.bump = context.bumps.audit_verdict;
            verdict.run = context.accounts.run.key();
            verdict.target = params.target;
            verdict.reset_for_round(current_epoch, current_round);
        } else if verdict.epoch != current_epoch || verdict.round_height != current_round {
            // One PDA per (run, target) is reused for every dispute, so taking
            // it over has to wait until the last one is finished. Overwriting an
            // unfinished dispute erased an appeal in progress, and erased the
            // voter list the losing-verifier crank walks - which let one of
            // those verifiers free the whole cohort with a single transaction
            // (finding 6).
            if !verdict.is_settled() {
                return err!(ProgramError::PreviousVerdictUnsettled);
            }
            verdict.reset_for_round(current_epoch, current_round);
        }

        if verdict.status != VerdictStatus::Voting {
            return err!(ProgramError::VerdictAlreadyResolved);
        }
        if verdict.voters.iter().any(|voter| voter == &verifier_key) {
            return err!(ProgramError::DuplicateVerdict);
        }
        if verdict.voters.len() >= MAX_VERDICT_VOTERS {
            return err!(ProgramError::VerdictVotersFull);
        }

        // The evidence is written once, by whoever votes first, and every later
        // vote has to agree with it. Letting each vote overwrite it meant the
        // verifier who completed the quorum decided what the on-chain record
        // said, which is the only thing an appeal bench or an observer can check
        // (finding 9). Two verifiers who disagree about what they replayed is
        // information, and it belongs in a rejected transaction rather than
        // silently in the last writer's favour.
        if verdict.voters.is_empty() {
            verdict.committed_hash = params.committed_hash;
            verdict.replayed_hash = params.replayed_hash;
            verdict.target_index = params.target_index;
            verdict.batch_start = params.batch_start;
            verdict.batch_end = params.batch_end;
        } else if verdict.committed_hash != params.committed_hash
            || verdict.replayed_hash != params.replayed_hash
            || verdict.target_index != params.target_index
            || verdict.batch_start != params.batch_start
            || verdict.batch_end != params.batch_end
        {
            return err!(ProgramError::EvidenceMismatch);
        }

        verdict.voters.push(verifier_key);
        verdict.verdict_count += 1;

        msg!(
            "audit_verdict: target_index={} epoch={} round={} count={} quorum={}",
            params.target_index,
            current_epoch,
            current_round,
            verdict.verdict_count,
            quorum
        );

        let quorum_reached = (verdict.verdict_count as u64) >= quorum;
        if quorum_reached {
            if context.accounts.run.challenge_window_seconds > 0 {
                verdict.status = VerdictStatus::SlashPending;
                verdict.pending_since_unix = Clock::get()?.unix_timestamp;
                should_slash = false;
                msg!("audit_verdict: quorum reached, slash pending challenge window");
            } else {
                verdict.status = VerdictStatus::Upheld;
                should_slash = true;
            }
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
                batch_start: params.batch_start,
                batch_end: params.batch_end,
                committed_hash: params.committed_hash,
                replayed_hash: params.replayed_hash,
            },
        )?;
    }

    Ok(())
}
