//! A trainer that signs two different payloads for one batch has equivocated,
//! and the two signatures prove it on their own (wienerlabs/leviathan-net#46).
//! These send the proof to the treasurer.
//!
//! A reporter commits to the proof first and reveals it in a later slot, so
//! that nobody who sees the reveal can copy it and land it as its own
//! (wienerlabs/leviathan-net#92).

use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use anyhow::bail;
use anyhow::Result;
use psyche_solana_coordinator::find_coordinator_instance;
use psyche_solana_treasurer::accounts::RunCloseEquivocationCommitmentAccounts;
use psyche_solana_treasurer::accounts::RunCommitEquivocationAccounts;
use psyche_solana_treasurer::accounts::RunSlashEquivocationAccounts;
use psyche_solana_treasurer::find_equivocation_commitment;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::find_slash_report;
use psyche_solana_treasurer::instruction::RunCloseEquivocationCommitment;
use psyche_solana_treasurer::instruction::RunCommitEquivocation;
use psyche_solana_treasurer::instruction::RunSlashEquivocation;
use psyche_solana_treasurer::logic::RunCloseEquivocationCommitmentParams;
use psyche_solana_treasurer::logic::RunCommitEquivocationParams;
use psyche_solana_treasurer::logic::RunSlashEquivocationParams;
use psyche_solana_treasurer::state::EquivocationClaim;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::system_program;
use solana_sdk::sysvar;
use solana_toolbox_endpoint::ToolboxEndpoint;

use crate::signed_commitment::SignedCommitment;

/// What `reporter` commits to before it reveals that `target` signed both
/// `first` and `second` in `run`.
pub fn equivocation_commitment(
    run: &Pubkey,
    reporter: &Pubkey,
    target: &Pubkey,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> [u8; 32] {
    EquivocationClaim {
        run,
        target,
        batch_start: first.batch_start,
        batch_end: first.batch_end,
        first: &first.evidence,
        second: &second.evidence,
    }
    .commitment(reporter)
}

/// The instruction committing `reporter` to `commitment`, a proof of
/// equivocation it will reveal in a later slot.
pub fn run_commit_equivocation_instruction(
    payer: &Pubkey,
    reporter: &Pubkey,
    run: &Pubkey,
    commitment: &[u8; 32],
) -> Instruction {
    let accounts = RunCommitEquivocationAccounts {
        payer: *payer,
        reporter: *reporter,
        run: *run,
        equivocation_commitment: find_equivocation_commitment(
            run, reporter, commitment,
        ),
        system_program: system_program::ID,
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunCommitEquivocation {
            params: RunCommitEquivocationParams {
                commitment: *commitment,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// `reporter` commits to the proof that `target` signed both `first` and
/// `second`, to reveal it in a later slot. Returns the commitment.
pub async fn process_treasurer_run_commit_equivocation(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    reporter: &Keypair,
    run: &Pubkey,
    target: &Pubkey,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> Result<[u8; 32]> {
    let commitment =
        equivocation_commitment(run, &reporter.pubkey(), target, first, second);
    let instruction = run_commit_equivocation_instruction(
        &payer.pubkey(),
        &reporter.pubkey(),
        run,
        &commitment,
    );
    endpoint
        .process_instruction_with_signers(payer, instruction, &[reporter])
        .await?;
    Ok(commitment)
}

/// `reporter` closes its commitment to `commitment`, which it will not
/// reveal, and takes its rent back.
pub async fn process_treasurer_run_close_equivocation_commitment(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    reporter: &Keypair,
    run: &Pubkey,
    commitment: &[u8; 32],
) -> Result<()> {
    let accounts = RunCloseEquivocationCommitmentAccounts {
        reporter: reporter.pubkey(),
        run: *run,
        equivocation_commitment: find_equivocation_commitment(
            run,
            &reporter.pubkey(),
            commitment,
        ),
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunCloseEquivocationCommitment {
            params: RunCloseEquivocationCommitmentParams {
                commitment: *commitment,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[reporter])
        .await?;
    Ok(())
}

/// The instruction convicting `target`, at `target_index`, of having signed
/// both `first` and `second`. It is refused unless the same transaction proves
/// both signatures; `process_treasurer_run_slash_equivocation` sends them
/// together. `reporter` signs it and is recorded as the conviction's reporter
/// (wienerlabs/leviathan-net#72). It reveals the proof `reporter` committed
/// to in an earlier slot, and closes that commitment
/// (wienerlabs/leviathan-net#92).
#[allow(clippy::too_many_arguments)]
pub fn run_slash_equivocation_instruction(
    payer: &Pubkey,
    reporter: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target: &Pubkey,
    target_index: u64,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> Instruction {
    let commitment =
        equivocation_commitment(run, reporter, target, first, second);
    let accounts = RunSlashEquivocationAccounts {
        payer: *payer,
        reporter: *reporter,
        run: *run,
        coordinator_instance: find_coordinator_instance(run_id),
        coordinator_account: *coordinator_account,
        coordinator_program: psyche_solana_coordinator::ID,
        instructions: sysvar::instructions::ID,
        target_participant: find_participant(run, target),
        slash_report: find_slash_report(
            run,
            target,
            first.evidence.epoch_start,
        ),
        equivocation_commitment: find_equivocation_commitment(
            run,
            reporter,
            &commitment,
        ),
        system_program: system_program::ID,
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSlashEquivocation {
            params: RunSlashEquivocationParams {
                target: *target,
                target_index,
                batch_start: first.batch_start,
                batch_end: first.batch_end,
                first: first.evidence,
                second: second.evidence,
                commitment,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// `reporter` convicts `target`, at `target_index`, of equivocation on
/// `first` and `second`, with the two precompile instructions that prove the
/// target signed both. It reveals the proof `reporter` committed to through
/// `process_treasurer_run_commit_equivocation` in an earlier slot.
///
/// The batch is `first`'s. A `second` signed for another batch does not prove
/// a signature over this batch's commitment, so the chain refuses it; it is
/// refused here first, as it can never convict.
#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_run_slash_equivocation(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    reporter: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target: &Pubkey,
    target_index: u64,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> Result<()> {
    if (first.batch_start, first.batch_end)
        != (second.batch_start, second.batch_end)
    {
        bail!("the two commitments are for different batches, which is not equivocation");
    }
    let instruction = run_slash_equivocation_instruction(
        &payer.pubkey(),
        &reporter.pubkey(),
        run,
        coordinator_account,
        run_id,
        target,
        target_index,
        first,
        second,
    );
    endpoint
        .process_instructions_with_signers(
            payer,
            &[
                first.proof_instruction(),
                second.proof_instruction(),
                instruction,
            ],
            &[reporter],
        )
        .await?;
    Ok(())
}
