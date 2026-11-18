use anchor_lang::prelude::*;

use crate::state::EquivocationCommitment;
use crate::state::Run;

#[derive(Accounts)]
#[instruction(params: RunCommitEquivocationParams)]
pub struct RunCommitEquivocationAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    /// Whoever will reveal the proof, and be its reporter.
    #[account()]
    pub reporter: Signer<'info>,

    #[account()]
    pub run: Box<Account<'info, Run>>,

    #[account(
        init,
        payer = payer,
        space = EquivocationCommitment::space_with_discriminator(),
        seeds = [
            EquivocationCommitment::SEEDS_PREFIX,
            run.key().as_ref(),
            reporter.key().as_ref(),
            params.commitment.as_ref(),
        ],
        bump,
    )]
    pub equivocation_commitment: Box<Account<'info, EquivocationCommitment>>,

    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunCommitEquivocationParams {
    /// `EquivocationClaim::commitment` of the proof, for this reporter.
    pub commitment: [u8; 32],
}

/// Commits `reporter` to a proof of equivocation it will reveal in a later
/// slot through `run_slash_equivocation` (wienerlabs/leviathan-net#92).
///
/// The proof is not on chain until then, only its hash with the reporter's
/// key, so nobody who sees the reveal can copy the proof and land it as its
/// own: it would need a commitment from before the proof was public.
pub fn run_commit_equivocation_processor(
    context: Context<RunCommitEquivocationAccounts>,
    params: RunCommitEquivocationParams,
) -> Result<()> {
    let slot = Clock::get()?.slot;
    let commitment = &mut context.accounts.equivocation_commitment;
    commitment.bump = context.bumps.equivocation_commitment;
    commitment.run = context.accounts.run.key();
    commitment.reporter = context.accounts.reporter.key();
    commitment.commitment = params.commitment;
    commitment.slot = slot;
    msg!(
        "commit_equivocation: reporter={} slot={}",
        commitment.reporter,
        slot
    );
    Ok(())
}
