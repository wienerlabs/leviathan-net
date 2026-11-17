use anchor_lang::prelude::*;
use anchor_lang::solana_program::sysvar::instructions as instructions_sysvar;
use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;

use crate::state::AuditVerdict;
use crate::state::CommitmentEvidence;
use crate::state::commitment_run;
use crate::state::transaction_proves_signature;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VerdictStatus;
use crate::state::VoteEvidence;
use crate::state::MAX_VERDICT_VOTERS;
use crate::state::Landing;
use crate::state::coordinator_slashed_points;
use crate::state::committee_matches_roster;
use crate::state::conviction_landing;
use crate::state::ejection_charge;
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

    /// Writable because the vote is counted on it: an overturn would charge
    /// this bond, so it cannot leave while the vote is at stake
    /// (wienerlabs/leviathan-net#39).
    #[account(
        mut,
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

    // An account of another version is refused, not loaded: loading one
    // of another size panics (wienerlabs/leviathan-net#62).
    #[account(
        mut,
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
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

    /// CHECK: the instructions sysvar, pinned by address. It is read for the
    /// precompile instruction proving the target signed the commitment.
    #[account(address = instructions_sysvar::ID)]
    pub instructions: UncheckedAccount<'info>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSubmitAuditVerdictParams {
    pub target: Pubkey,
    pub target_index: u64,
    pub batch_start: u64,
    pub batch_end: u64,
    /// The payload the target committed to for `batch_start..=batch_end`. The
    /// same transaction has to carry an ed25519 precompile instruction with the
    /// target's p2p key signing its commitment hash.
    pub evidence: CommitmentEvidence,
    pub replayed_hash: [u8; 32],
}

pub fn run_submit_audit_verdict_processor(
    context: Context<RunSubmitAuditVerdictAccounts>,
    params: RunSubmitAuditVerdictParams,
) -> Result<()> {
    let verifier_key = context.accounts.verifier.key();

    // A later round can draw the target onto the committee judging it. Its
    // vote is no verdict on itself: seated first, it fixed the evidence the
    // rest of the round had to agree with.
    if verifier_key == params.target {
        return err!(ProgramError::VerifierIsTarget);
    }

    let (
        current_epoch,
        current_round,
        quorum,
        landing,
        ejection,
        target_p2p,
        run,
    ) = {
        let account = context.accounts.coordinator_account.load()?;

        // A verdict that turns out wrong is paid for out of this bond, at the
        // epoch boundary after an overturn. So the bond that counts is the one
        // still standing then: not what a conviction already charged, and not a
        // withdrawal already requested, which could leave before the overturn
        // reaches it (wienerlabs/leviathan-net#34).
        if context
            .accounts
            .verifier_participant
            .committed_bond(coordinator_slashed_points(&account, &verifier_key))
            < context.accounts.run.bond_minimum_amount
        {
            return err!(ProgramError::BondBelowMinimum);
        }

        let coordinator = &account.state.coordinator;

        // A seat is only a seat on the roster it was drawn over.
        if !committee_matches_roster(coordinator) {
            return err!(ProgramError::CommitteeNotDrawnForRoster);
        }

        let verifier_index = coordinator
            .epoch_state
            .clients
            .iter()
            .position(|client| *client.id.signer() == verifier_key.to_bytes())
            .ok_or_else(|| error!(ProgramError::VerifierNotInEpoch))?;

        // The round's own draw, with the tie-breaker seats it reserves, which
        // is the committee its clients train under and the audit lottery
        // assigns from (wienerlabs/leviathan#15, finding 4).
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
        let target_p2p = *target_client.id.p2p_identity();
        let run = commitment_run(coordinator);

        // A conviction is for work done in this epoch. A commitment from an
        // earlier one names work whose epoch was already charged.
        if !params.evidence.is_from_current_epoch(coordinator) {
            return err!(ProgramError::CommitmentOutsideEpoch);
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
        // Where an immediate conviction would land, decided now while the
        // coordinator is loaded. The target was just checked to be at
        // `target_index`.
        let landing =
            conviction_landing(coordinator, &params.target, params.target_index)?;
        (
            coordinator.progress.epoch,
            current_round,
            quorum,
            landing,
            ejection_charge(&account),
            target_p2p,
            run,
        )
    };

    // The committed side of the fraud proof is checked here rather than taken
    // on the voters' word. The hash is rebuilt from the batch this verdict
    // names, in this run, and the target's own key, read from the roster, has
    // to have signed it. Otherwise a quorum could record a hash the target never
    // committed to, and the record an appeal or an observer reads would be
    // fiction (wienerlabs/leviathan-net#43).
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

    let should_slash;
    let recorded;
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

        // Votes are counted per payload the target signed, and a quorum is two
        // thirds of the committee agreeing on one. The first vote used to fix
        // the evidence every later vote had to match, the replayed hash
        // included: one seat citing another of the target's payloads, all of
        // which are broadcast, vetoed the round, and honest verifiers whose
        // replays differ in the last bits never reached a quorum at all. The
        // verdict records the evidence most votes are for, so the record a
        // quorum leaves is still the one its votes agreed on (finding 9).
        let agreeing = verdict.cast_vote(
            verifier_key,
            VoteEvidence {
                committed_hash,
                evidence: params.evidence,
                target_index: params.target_index,
                batch_start: params.batch_start,
                batch_end: params.batch_end,
                replayed_hash: params.replayed_hash,
            },
        );
        context.accounts.verifier_participant.open_verdict_vote()?;

        msg!(
            "audit_verdict: target_index={} epoch={} round={} count={} recorded={} quorum={}",
            params.target_index,
            current_epoch,
            current_round,
            agreeing,
            verdict.verdict_count,
            quorum
        );

        // The quorum is the recorded evidence's, and so is what a slash
        // charges: the verdict, not the vote that happened to complete it, is
        // what an appeal and every reader of the account see.
        recorded = SlashClientParams {
            index: verdict.target_index,
            batch_start: verdict.batch_start,
            batch_end: verdict.batch_end,
            committed_hash: verdict.committed_hash,
            replayed_hash: verdict.replayed_hash,
        };
        let quorum_reached = (verdict.verdict_count as u64) >= quorum;
        if quorum_reached {
            if context.accounts.run.challenge_window_seconds > 0 {
                verdict.status = VerdictStatus::SlashPending;
                verdict.pending_since_unix = Clock::get()?.unix_timestamp;
                should_slash = false;
                msg!("audit_verdict: quorum reached, slash pending challenge window");
            } else {
                match landing {
                    Landing::Eject(_) => {
                        verdict.land_ejection(ejection);
                        should_slash = true;
                    },
                    // Reached where no ejection would be charged: the
                    // conviction stands and is recorded as due, instead of
                    // ejecting the target onto a roster nobody will charge
                    // (wienerlabs/leviathan-net#36).
                    Landing::Due => {
                        verdict.status = VerdictStatus::Due;
                        should_slash = false;
                        msg!("audit_verdict: quorum reached outside a charging epoch, conviction due");
                    },
                }
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
            recorded,
        )?;
    }

    Ok(())
}
