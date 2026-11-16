use anchor_lang::prelude::*;
use anchor_lang::solana_program::sysvar::instructions as instructions_sysvar;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::slash_client;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::SlashClientParams;

use crate::state::commitment_run;
use crate::state::conviction_landing;
use crate::state::transaction_proves_signature;
use crate::state::CommitmentEvidence;
use crate::state::Landing;
use crate::state::Run;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSlashEquivocationParams)]
pub struct RunSlashEquivocationAccounts<'info> {
    /// Anyone: the proof needs no committee and no authority.
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
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunSlashEquivocationParams {
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
    let (target_p2p, run, index) = {
        let account = context.accounts.coordinator_account.load()?;
        let coordinator = &account.state.coordinator;
        let target = coordinator
            .epoch_state
            .clients
            .iter()
            .nth(params.target_index as usize)
            .ok_or_else(|| error!(ProgramError::TargetMismatch))?;
        let target_key = Pubkey::new_from_array(*target.id.signer());
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

    Ok(())
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
}
