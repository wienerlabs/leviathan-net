use anchor_lang::prelude::*;

use crate::state::EquivocationBounty;
use crate::state::Run;
use crate::state::MAX_EQUIVOCATION_BOUNTY_BPS;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSetEquivocationBountyParams)]
pub struct RunSetEquivocationBountyAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account()]
    pub main_authority: Signer<'info>,

    #[account(constraint = run.main_authority == main_authority.key())]
    pub run: Box<Account<'info, Run>>,

    #[account(
        init_if_needed,
        payer = payer,
        space = EquivocationBounty::space_with_discriminator(),
        seeds = [EquivocationBounty::SEEDS_PREFIX, run.key().as_ref()],
        bump,
    )]
    pub equivocation_bounty: Box<Account<'info, EquivocationBounty>>,

    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunSetEquivocationBountyParams {
    pub bounty_bps: u16,
}

/// Whether a run may give an equivocation's reporter `bounty_bps` of the
/// target's forfeit: less than the whole of it, so that a target reporting
/// itself from a second wallet always loses some of the charge.
pub fn check_equivocation_bounty(bounty_bps: u16) -> Result<()> {
    if bounty_bps > MAX_EQUIVOCATION_BOUNTY_BPS {
        return err!(ProgramError::EquivocationBountyTooHigh);
    }
    Ok(())
}

/// Sets the share of an equivocating target's forfeit that goes to whoever
/// reported it (wienerlabs/leviathan-net#92). It is the run's own, set with
/// its bond economics; at the charge it is also held to the run's
/// `slash_bounty_bps`.
pub fn run_set_equivocation_bounty_processor(
    context: Context<RunSetEquivocationBountyAccounts>,
    params: RunSetEquivocationBountyParams,
) -> Result<()> {
    check_equivocation_bounty(params.bounty_bps)?;
    let bounty = &mut context.accounts.equivocation_bounty;
    bounty.bump = context.bumps.equivocation_bounty;
    bounty.run = context.accounts.run.key();
    bounty.bounty_bps = params.bounty_bps;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bounty_of_the_whole_forfeit_is_refused() {
        for bps in [0, 1, 5_000, MAX_EQUIVOCATION_BOUNTY_BPS] {
            assert!(check_equivocation_bounty(bps).is_ok(), "{bps}");
        }
        for bps in [10_000, u16::MAX] {
            assert_eq!(
                check_equivocation_bounty(bps).unwrap_err(),
                error!(ProgramError::EquivocationBountyTooHigh),
                "{bps}"
            );
        }
    }
}
