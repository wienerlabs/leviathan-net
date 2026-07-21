use anchor_lang::prelude::*;
use psyche_solana_authorizer::cpi::accounts::AuthorizationCreateAccounts;
use psyche_solana_authorizer::cpi::accounts::AuthorizationGrantorUpdateAccounts;
use psyche_solana_authorizer::cpi::authorization_create;
use psyche_solana_authorizer::cpi::authorization_grantor_update;
use psyche_solana_authorizer::logic::AuthorizationCreateParams;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_authorizer::program::PsycheSolanaAuthorizer;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;

use crate::state::Participant;
use crate::state::Run;
use crate::ProgramError;

#[derive(Accounts)]
pub struct ParticipantAuthorizeJoinAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    /// CHECK: the join grantee; its key seeds the participant and is passed to the authorizer
    pub user: UncheckedAccount<'info>,

    #[account(
        constraint = run.join_authority == run.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    #[account(
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            user.key().as_ref(),
        ],
        bump = participant.bump,
    )]
    pub participant: Box<Account<'info, Participant>>,

    /// CHECK: created and validated by the authorizer CPI against its own seeds
    #[account(mut)]
    pub authorization: UncheckedAccount<'info>,

    #[account()]
    pub authorizer_program: Program<'info, PsycheSolanaAuthorizer>,

    #[account()]
    pub system_program: Program<'info, System>,
}

pub fn participant_authorize_join_processor(
    context: Context<ParticipantAuthorizeJoinAccounts>,
) -> Result<()> {
    if context.accounts.participant.bond_amount < context.accounts.run.bond_minimum_amount {
        return err!(ProgramError::BondBelowMinimum);
    }

    let run = &context.accounts.run;
    let run_signer_seeds: &[&[&[u8]]] =
        &[&[Run::SEEDS_PREFIX, &run.index.to_le_bytes(), &[run.bump]]];

    authorization_create(
        CpiContext::new(
            context.accounts.authorizer_program.to_account_info(),
            AuthorizationCreateAccounts {
                payer: context.accounts.payer.to_account_info(),
                grantor: context.accounts.run.to_account_info(),
                authorization: context.accounts.authorization.to_account_info(),
                system_program: context.accounts.system_program.to_account_info(),
            },
        )
        .with_signer(run_signer_seeds),
        AuthorizationCreateParams {
            grantee: context.accounts.user.key(),
            scope: JOIN_RUN_AUTHORIZATION_SCOPE.to_vec(),
        },
    )?;

    authorization_grantor_update(
        CpiContext::new(
            context.accounts.authorizer_program.to_account_info(),
            AuthorizationGrantorUpdateAccounts {
                grantor: context.accounts.run.to_account_info(),
                authorization: context.accounts.authorization.to_account_info(),
            },
        )
        .with_signer(run_signer_seeds),
        AuthorizationGrantorUpdateParams { active: true },
    )?;

    Ok(())
}
