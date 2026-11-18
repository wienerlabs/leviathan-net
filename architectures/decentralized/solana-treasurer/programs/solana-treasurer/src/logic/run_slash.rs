use anchor_lang::prelude::*;
use anchor_lang::solana_program::sysvar::instructions as instructions_sysvar;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;

use crate::logic::check_reporter;
use crate::logic::record_slash_report;
use crate::logic::SlashRecord;
use crate::state::ejection_charge;
use crate::state::CommitmentEvidence;
use crate::state::Participant;
use crate::state::Run;
use crate::state::SlashKind;
use crate::state::SlashReport;
use crate::state::commitment_run;
use crate::state::transaction_proves_signature;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSlashParams)]
pub struct RunSlashAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

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

    /// CHECK: the target's participant account at its own PDA, counted on if
    /// the target has one. A target with none is still slashed.
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

    /// Who reported the slash, written when it ejects its target
    /// (wienerlabs/leviathan-net#72).
    #[account(
        init_if_needed,
        payer = payer,
        space = SlashReport::space_with_discriminator(),
        seeds = [
            SlashReport::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
            params.evidence.epoch_start.to_le_bytes().as_ref(),
        ],
        bump,
    )]
    pub slash_report: Box<Account<'info, SlashReport>>,

    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSlashParams {
    /// The target's wallet, which has to be the one at `index`.
    pub target: Pubkey,
    /// Whose bond the slash's bounty is credited to. None keeps it in the
    /// vault.
    pub reporter: Option<Pubkey>,
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
    check_reporter(&params.target, params.reporter.as_ref())?;
    let (target_p2p, run, ejection) = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;
        let target = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(params.index as usize)
            .ok_or_else(|| error!(ProgramError::TargetMismatch))?;
        if *target.id.signer() != params.target.to_bytes() {
            return err!(ProgramError::TargetMismatch);
        }
        if !params.evidence.is_from_current_epoch(coordinator) {
            return err!(ProgramError::CommitmentOutsideEpoch);
        }
        (
            *target.id.p2p_identity(),
            commitment_run(coordinator),
            ejection_charge(&account),
        )
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

    let run = context.accounts.run.key();
    let bump = context.bumps.slash_report;
    let target_participant =
        context.accounts.target_participant.to_account_info();
    record_slash_report(
        &mut context.accounts.slash_report,
        bump,
        &target_participant,
        SlashRecord {
            run,
            target: params.target,
            ejection,
            kind: SlashKind::AuthoritySlash,
            reporter: params.reporter,
        },
    )
}
