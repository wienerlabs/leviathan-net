use anchor_lang::prelude::*;
use anchor_lang::solana_program::sysvar::instructions as instructions_sysvar;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;

use crate::logic::check_reporter;
use crate::logic::record_slash_report;
use crate::logic::SlashRecord;
use crate::state::commitment_run;
use crate::state::conviction_landing;
use crate::state::ejection_charge;
use crate::state::transaction_proves_signature;
use crate::state::CommitmentEvidence;
use crate::state::EquivocationClaim;
use crate::state::EquivocationCommitment;
use crate::state::Landing;
use crate::state::Participant;
use crate::state::Run;
use crate::state::SlashKind;
use crate::state::SlashReport;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSlashEquivocationParams)]
pub struct RunSlashEquivocationAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    /// Anyone: the proof needs no committee and no authority. Whoever signs it
    /// is recorded as its reporter (wienerlabs/leviathan-net#72), if it
    /// committed to the proof in an earlier slot
    /// (wienerlabs/leviathan-net#92).
    #[account()]
    pub reporter: Signer<'info>,

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

    #[account()]
    pub coordinator_program: Program<'info, PsycheSolanaCoordinator>,

    /// CHECK: the instructions sysvar, pinned by address. It is read for the
    /// two precompile instructions proving the target signed both commitments.
    #[account(address = instructions_sysvar::ID)]
    pub instructions: UncheckedAccount<'info>,

    /// CHECK: the target's participant account at its own PDA, counted on if
    /// the target has one. A target with none is still convicted.
    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump,
    )]
    pub target_participant: UncheckedAccount<'info>,

    /// Who reported the equivocation, written when it ejects its target.
    #[account(
        init_if_needed,
        payer = payer,
        space = SlashReport::space_with_discriminator(),
        seeds = [
            SlashReport::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
            params.first.epoch_start.to_le_bytes().as_ref(),
        ],
        bump,
    )]
    pub slash_report: Box<Account<'info, SlashReport>>,

    /// The reporter's commitment to this proof, made in an earlier slot
    /// through `run_commit_equivocation`. Taking the proof closes it, and its
    /// rent goes to the payer (wienerlabs/leviathan-net#92).
    #[account(
        mut,
        close = payer,
        seeds = [
            EquivocationCommitment::SEEDS_PREFIX,
            run.key().as_ref(),
            reporter.key().as_ref(),
            params.commitment.as_ref(),
        ],
        bump = equivocation_commitment.bump,
    )]
    pub equivocation_commitment: Box<Account<'info, EquivocationCommitment>>,

    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSlashEquivocationParams {
    /// The target's wallet, which has to be the one at `target_index`.
    pub target: Pubkey,
    /// Where the target sits on the roster now.
    pub target_index: u64,
    /// The batch both commitments are for.
    pub batch_start: u64,
    pub batch_end: u64,
    /// The two payloads the target committed to for that batch. The same
    /// transaction has to carry an ed25519 precompile instruction for each,
    /// with the target's p2p key signing its commitment hash.
    pub first: CommitmentEvidence,
    pub second: CommitmentEvidence,
    /// The reporter's commitment to this proof, `EquivocationClaim::commitment`
    /// for its key, which it made in an earlier slot.
    pub commitment: [u8; 32],
}

/// Whether `reporter` may reveal the proof in `params` in `slot`, against its
/// commitment `committed`: the commitment is to exactly this proof and this
/// reporter, and was made in an earlier slot (wienerlabs/leviathan-net#92).
pub fn check_equivocation_reveal(
    run: &Pubkey,
    params: &RunSlashEquivocationParams,
    reporter: &Pubkey,
    committed: &EquivocationCommitment,
    slot: u64,
) -> Result<()> {
    let claim = EquivocationClaim {
        run,
        target: &params.target,
        batch_start: params.batch_start,
        batch_end: params.batch_end,
        first: &params.first,
        second: &params.second,
    };
    if claim.commitment(reporter) != committed.commitment {
        return err!(ProgramError::EquivocationNotCommitted);
    }
    if !committed.revealable_in(slot) {
        return err!(ProgramError::EquivocationCommittedTooRecently);
    }
    Ok(())
}

/// Whether `first` and `second` are two different commitments to the same
/// work: the same step, and hashes that differ once rebuilt for the same run
/// and batch. Both are checked to be from the current epoch before, so the
/// same step is the same round (wienerlabs/leviathan-net#67). A trainer signs one payload per batch it trains, so two such
/// commitments signed by its key prove it told different peers different
/// things about one piece of work (wienerlabs/leviathan-net#46).
pub fn equivocates(
    first_hash: &[u8; 32],
    second_hash: &[u8; 32],
    first_step: u32,
    second_step: u32,
) -> bool {
    first_step == second_step && first_hash != second_hash
}

/// Convicts a target of equivocation: two different payload commitments for
/// one batch, both signed by its roster p2p key.
///
/// The proof is self-proving. It needs no replay and no committee to agree,
/// so anyone may send it, and there is nothing for an appeal to contest. It
/// lands only as an ejection, charged with the epoch like any other: a target
/// that cannot be ejected into an epoch that will charge it is refused,
/// because no account records a conviction here to charge later. A peer sees
/// the second payload in the round it arrives, so the proof can land within
/// the epoch, whose final round keeps its roster through the cooldown
/// (wienerlabs/leviathan-net#40).
pub fn run_slash_equivocation_processor(
    context: Context<RunSlashEquivocationAccounts>,
    params: RunSlashEquivocationParams,
) -> Result<()> {
    check_reporter(&params.target, Some(&context.accounts.reporter.key()))?;
    // The proof is the reporter's only if it committed to it before the proof
    // was public: a copier would need a commitment from before it saw it.
    check_equivocation_reveal(
        &context.accounts.run.key(),
        &params,
        &context.accounts.reporter.key(),
        &context.accounts.equivocation_commitment,
        Clock::get()?.slot,
    )?;
    let (target_p2p, run, index, ejection) = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;
        let target = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(params.target_index as usize)
            .ok_or_else(|| error!(ProgramError::TargetMismatch))?;
        let target_key = Pubkey::new_from_array(*target.id.signer());
        if target_key != params.target {
            return err!(ProgramError::TargetMismatch);
        }
        for evidence in [&params.first, &params.second] {
            if !evidence.is_from_current_epoch(coordinator) {
                return err!(ProgramError::CommitmentOutsideEpoch);
            }
        }
        let index = match conviction_landing(
            coordinator,
            &target_key,
            params.target_index,
        )? {
            Landing::Eject(index) => index,
            Landing::Due => return err!(ProgramError::EquivocationNotLandable),
        };
        (
            *target.id.p2p_identity(),
            commitment_run(coordinator),
            index,
            ejection_charge(&account),
        )
    };

    let first_hash =
        params
            .first
            .committed_hash(&run, params.batch_start, params.batch_end);
    let second_hash = params.second.committed_hash(
        &run,
        params.batch_start,
        params.batch_end,
    );
    if !equivocates(
        &first_hash,
        &second_hash,
        params.first.step,
        params.second.step,
    ) {
        return err!(ProgramError::EquivocationNotProven);
    }
    for hash in [&first_hash, &second_hash] {
        if !transaction_proves_signature(
            &context.accounts.instructions,
            &target_p2p,
            hash,
        )? {
            return err!(ProgramError::CommitmentNotSigned);
        }
    }

    msg!(
        "slash_equivocation: target_index={} step={} batch=[{}, {}]",
        index,
        params.first.step,
        params.batch_start,
        params.batch_end
    );

    let run_account = &context.accounts.run;
    let run_signer_seeds: &[&[&[u8]]] = &[&[
        Run::SEEDS_PREFIX,
        &run_account.index.to_le_bytes(),
        &[run_account.bump],
    ]];
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
            batch_start: params.batch_start,
            batch_end: params.batch_end,
            committed_hash: first_hash,
            replayed_hash: second_hash,
        },
    )?;

    let run_key = context.accounts.run.key();
    let reporter = context.accounts.reporter.key();
    let bump = context.bumps.slash_report;
    let target_participant =
        context.accounts.target_participant.to_account_info();
    record_slash_report(
        &mut context.accounts.slash_report,
        bump,
        &target_participant,
        SlashRecord {
            run: run_key,
            target: params.target,
            ejection,
            kind: SlashKind::Equivocation,
            reporter: Some(reporter),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_different_hashes_for_one_step_equivocate() {
        assert!(equivocates(&[1; 32], &[2; 32], 7, 7));
    }

    #[test]
    fn one_commitment_twice_is_not_equivocation() {
        assert!(!equivocates(&[1; 32], &[1; 32], 7, 7));
    }

    #[test]
    fn commitments_for_different_steps_are_not_equivocation() {
        assert!(!equivocates(&[1; 32], &[2; 32], 7, 8));
    }

    fn proof_params() -> RunSlashEquivocationParams {
        RunSlashEquivocationParams {
            target: Pubkey::new_unique(),
            target_index: 3,
            batch_start: 4,
            batch_end: 5,
            first: CommitmentEvidence {
                step: 7,
                results_digest: [1; 32],
                ..CommitmentEvidence::default()
            },
            second: CommitmentEvidence {
                step: 7,
                results_digest: [2; 32],
                ..CommitmentEvidence::default()
            },
            commitment: [0; 32],
        }
    }

    fn committed(
        run: &Pubkey,
        params: &RunSlashEquivocationParams,
        reporter: &Pubkey,
        slot: u64,
    ) -> EquivocationCommitment {
        let claim = EquivocationClaim {
            run,
            target: &params.target,
            batch_start: params.batch_start,
            batch_end: params.batch_end,
            first: &params.first,
            second: &params.second,
        };
        EquivocationCommitment {
            bump: 0,
            run: *run,
            reporter: *reporter,
            commitment: claim.commitment(reporter),
            slot,
        }
    }

    #[test]
    fn a_reporter_reveals_the_proof_it_committed_to_in_an_earlier_slot() {
        let run = Pubkey::new_unique();
        let reporter = Pubkey::new_unique();
        let params = proof_params();
        let commitment = committed(&run, &params, &reporter, 10);
        assert!(check_equivocation_reveal(
            &run,
            &params,
            &reporter,
            &commitment,
            11
        )
        .is_ok());
        // Not in the slot it committed in.
        assert_eq!(
            check_equivocation_reveal(
                &run,
                &params,
                &reporter,
                &commitment,
                10
            )
            .unwrap_err(),
            error!(ProgramError::EquivocationCommittedTooRecently)
        );
    }

    #[test]
    fn a_copier_cannot_reveal_against_a_commitment_that_is_not_its_own() {
        let run = Pubkey::new_unique();
        let honest = Pubkey::new_unique();
        let copier = Pubkey::new_unique();
        let params = proof_params();
        // The honest reporter's commitment does not name the copier.
        let honest_commitment = committed(&run, &params, &honest, 10);
        assert_eq!(
            check_equivocation_reveal(
                &run,
                &params,
                &copier,
                &honest_commitment,
                11
            )
            .unwrap_err(),
            error!(ProgramError::EquivocationNotCommitted)
        );
        // A commitment to another proof does not cover this one.
        let mut other = proof_params();
        other.target = params.target;
        other.second.results_digest = [3; 32];
        let other_commitment = committed(&run, &other, &copier, 10);
        assert_eq!(
            check_equivocation_reveal(
                &run,
                &params,
                &copier,
                &other_commitment,
                11
            )
            .unwrap_err(),
            error!(ProgramError::EquivocationNotCommitted)
        );
        // Nor does one made for another run.
        let elsewhere = committed(&Pubkey::new_unique(), &params, &copier, 10);
        assert_eq!(
            check_equivocation_reveal(&run, &params, &copier, &elsewhere, 11)
                .unwrap_err(),
            error!(ProgramError::EquivocationNotCommitted)
        );
    }
}
