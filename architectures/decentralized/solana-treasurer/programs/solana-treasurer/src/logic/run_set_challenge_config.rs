use anchor_lang::prelude::*;
use psyche_solana_coordinator::cpi::accounts::OwnerCoordinatorAccounts;
use psyche_solana_coordinator::cpi::set_tie_breaker_nodes;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;

use crate::state::Run;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunSetChallengeConfigParams)]
pub struct RunSetChallengeConfigAccounts<'info> {
    #[account()]
    pub main_authority: Signer<'info>,

    #[account(
        mut,
        constraint = run.main_authority == main_authority.key(),
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
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy)]
pub struct RunSetChallengeConfigParams {
    pub challenge_window_seconds: i64,
    /// The bench's size: how many tie-breaker seats each round reserves, from
    /// the next round to start. The coordinator carries it into every round,
    /// so the clients, the audit lottery and the treasurer draw one committee
    /// (wienerlabs/leviathan#15, finding 4). It is refused once the next
    /// round's seed is committed (`RoundSeedCommitted`); send it again a slot
    /// later.
    pub tie_breaker_committee_size: u16,
    /// How long an opened challenge may hold a conviction before the verifiers'
    /// verdict finalises anyway (wienerlabs/leviathan#15, finding 10). It may
    /// be zero only with appeals off. Switching appeals off ends the appeals in
    /// flight in the verifiers' favour: an open challenge finalises at once
    /// rather than holding the conviction, and every vote on it, for good
    /// (wienerlabs/leviathan-net#39).
    pub appeal_window_seconds: i64,
}

pub fn run_set_challenge_config_processor(
    context: Context<RunSetChallengeConfigAccounts>,
    params: RunSetChallengeConfigParams,
) -> Result<()> {
    if params.challenge_window_seconds < 0 || params.appeal_window_seconds < 0 {
        return err!(ProgramError::InvalidParameter);
    }
    // An appeal with no deadline holds the target's bond for as long as no
    // bench resolves it, and once the target or the run is gone no bench ever
    // can. With appeals on, the verdict the verifiers reached has to be able
    // to finalise eventually (wienerlabs/leviathan-net#36).
    if params.challenge_window_seconds > 0 && params.appeal_window_seconds == 0 {
        return err!(ProgramError::AppealWindowRequired);
    }

    // The coordinator draws the rounds, and the clients train what its round
    // assigns them. Kept only here, the bench size reserved seats in the
    // treasurer's draw and in nobody else's, so every audit named a batch its
    // target never trained (wienerlabs/leviathan#15, finding 4).
    let run = &context.accounts.run;
    let run_signer_seeds: &[&[&[u8]]] =
        &[&[Run::SEEDS_PREFIX, &run.index.to_le_bytes(), &[run.bump]]];
    set_tie_breaker_nodes(
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
        params.tie_breaker_committee_size,
    )?;

    let run = &mut context.accounts.run;
    run.challenge_window_seconds = params.challenge_window_seconds;
    run.tie_breaker_committee_size = params.tie_breaker_committee_size;
    run.appeal_window_seconds = params.appeal_window_seconds;
    Ok(())
}
