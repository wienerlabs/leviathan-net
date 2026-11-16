//! A trainer that signs two different payloads for one batch has equivocated,
//! and the two signatures prove it on their own (wienerlabs/leviathan-net#46).
//! These send the proof to the treasurer.

use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use anyhow::bail;
use anyhow::Result;
use psyche_solana_coordinator::find_coordinator_instance;
use psyche_solana_treasurer::accounts::RunSlashEquivocationAccounts;
use psyche_solana_treasurer::instruction::RunSlashEquivocation;
use psyche_solana_treasurer::logic::RunSlashEquivocationParams;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::sysvar;
use solana_toolbox_endpoint::ToolboxEndpoint;

use crate::signed_commitment::SignedCommitment;

/// The instruction convicting the target at `target_index` of having signed
/// both `first` and `second`. It is refused unless the same transaction proves
/// both signatures; `process_treasurer_run_slash_equivocation` sends them
/// together.
pub fn run_slash_equivocation_instruction(
    reporter: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target_index: u64,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> Instruction {
    let accounts = RunSlashEquivocationAccounts {
        reporter: *reporter,
        run: *run,
        coordinator_instance: find_coordinator_instance(run_id),
        coordinator_account: *coordinator_account,
        coordinator_program: psyche_solana_coordinator::ID,
        instructions: sysvar::instructions::ID,
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSlashEquivocation {
            params: RunSlashEquivocationParams {
                target_index,
                batch_start: first.batch_start,
                batch_end: first.batch_end,
                first: first.evidence,
                second: second.evidence,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// `reporter` convicts the target at `target_index` of equivocation on
/// `first` and `second`, with the two precompile instructions that prove the
/// target signed both.
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
        &reporter.pubkey(),
        run,
        coordinator_account,
        run_id,
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
