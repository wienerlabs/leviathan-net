use anchor_lang::prelude::*;
use psyche_core::NodeIdentity;
use psyche_solana_authorizer::cpi::accounts::AuthorizationCreateAccounts;
use psyche_solana_authorizer::cpi::accounts::AuthorizationGrantorUpdateAccounts;
use psyche_solana_authorizer::cpi::authorization_create;
use psyche_solana_authorizer::cpi::authorization_grantor_update;
use psyche_solana_authorizer::logic::AuthorizationCreateParams;
use psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams;
use psyche_solana_authorizer::program::PsycheSolanaAuthorizer;
use psyche_solana_authorizer::state::Authorization;
use psyche_solana_coordinator::cpi::accounts::JoinRunAccounts;
use psyche_solana_coordinator::cpi::join_run;
use psyche_solana_coordinator::logic::JoinRunParams;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_coordinator::program::PsycheSolanaCoordinator;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;

use crate::state::coordinator_slashed_points;
use crate::state::Participant;
use crate::state::Run;
use crate::ProgramError;

/// Admits a node to the next epoch of a run whose join authority is the run
/// itself, against the bond it has committed as of this join.
///
/// The bond used to be checked once, when the join authorization was created,
/// and the authorization then stayed active for good. A node could withdraw its
/// whole bond and keep joining, and a convicted node re-joined on a bond its
/// conviction had already forfeited (wienerlabs/leviathan-net#34). Here the
/// authorization is opened only for the coordinator's `join_run` inside this
/// instruction and closed again before it returns, so at rest it admits nobody
/// and every epoch's join passes through this check.
#[derive(Accounts)]
#[instruction(params: ParticipantJoinRunParams)]
pub struct ParticipantJoinRunAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account()]
    pub user: Signer<'info>,

    #[account(
        constraint = run.join_authority == run.key() @ ProgramError::RunNotBondGated,
        constraint = run.coordinator_instance == coordinator_instance.key(),
        constraint = run.coordinator_account == coordinator_account.key(),
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

    /// CHECK: pinned to the joining signer's own authorization by the seeds
    /// below, so a bond posted by one identity never opens the seat of another
    /// (a grantee's delegates included). Created and updated only through the
    /// authorizer CPIs, which check it against their own seeds and owner.
    #[account(
        mut,
        seeds = [
            Authorization::SEEDS_PREFIX,
            run.key().as_ref(),
            user.key().as_ref(),
            JOIN_RUN_AUTHORIZATION_SCOPE,
        ],
        bump,
        seeds::program = authorizer_program.key(),
    )]
    pub authorization: UncheckedAccount<'info>,

    #[account()]
    pub coordinator_instance: Box<Account<'info, CoordinatorInstance>>,

    #[account(mut)]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account()]
    pub authorizer_program: Program<'info, PsycheSolanaAuthorizer>,

    #[account()]
    pub coordinator_program: Program<'info, PsycheSolanaCoordinator>,

    #[account()]
    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ParticipantJoinRunParams {
    pub client_id: NodeIdentity,
}

pub fn participant_join_run_processor(
    context: Context<ParticipantJoinRunAccounts>,
    params: ParticipantJoinRunParams,
) -> Result<()> {
    let user_key = context.accounts.user.key();
    let slashed_points = coordinator_slashed_points(
        &*context.accounts.coordinator_account.load()?,
        &user_key,
    );
    let committed_bond =
        context.accounts.participant.committed_bond(slashed_points);
    let bond_minimum_amount = context.accounts.run.bond_minimum_amount;
    if committed_bond < bond_minimum_amount {
        msg!(
            "participant_join_run: committed bond {} below the run minimum {} (bond {}, unsettled slash {}, pending withdrawal {})",
            committed_bond,
            bond_minimum_amount,
            context.accounts.participant.bond_amount,
            context
                .accounts
                .participant
                .unsettled_slashed_points(slashed_points),
            context.accounts.participant.bond_withdraw_pending_amount,
        );
        return err!(ProgramError::BondBelowMinimum);
    }

    let run = &context.accounts.run;
    let run_index = run.index.to_le_bytes();
    let run_signer_seeds: &[&[&[u8]]] =
        &[&[Run::SEEDS_PREFIX, &run_index, &[run.bump]]];

    if context.accounts.authorization.data_is_empty() {
        authorization_create(
            CpiContext::new(
                context.accounts.authorizer_program.to_account_info(),
                AuthorizationCreateAccounts {
                    payer: context.accounts.payer.to_account_info(),
                    grantor: context.accounts.run.to_account_info(),
                    authorization: context
                        .accounts
                        .authorization
                        .to_account_info(),
                    system_program: context
                        .accounts
                        .system_program
                        .to_account_info(),
                },
            )
            .with_signer(run_signer_seeds),
            AuthorizationCreateParams {
                grantee: user_key,
                scope: JOIN_RUN_AUTHORIZATION_SCOPE.to_vec(),
            },
        )?;
    }

    set_join_authorization_active(&context, run_signer_seeds, true)?;

    join_run(
        CpiContext::new(
            context.accounts.coordinator_program.to_account_info(),
            JoinRunAccounts {
                user: context.accounts.user.to_account_info(),
                authorization: context.accounts.authorization.to_account_info(),
                coordinator_instance: context
                    .accounts
                    .coordinator_instance
                    .to_account_info(),
                coordinator_account: context
                    .accounts
                    .coordinator_account
                    .to_account_info(),
            },
        ),
        JoinRunParams {
            client_id: params.client_id,
        },
    )?;

    set_join_authorization_active(&context, run_signer_seeds, false)?;

    Ok(())
}

fn set_join_authorization_active(
    context: &Context<ParticipantJoinRunAccounts>,
    run_signer_seeds: &[&[&[u8]]],
    active: bool,
) -> Result<()> {
    authorization_grantor_update(
        CpiContext::new(
            context.accounts.authorizer_program.to_account_info(),
            AuthorizationGrantorUpdateAccounts {
                grantor: context.accounts.run.to_account_info(),
                authorization: context.accounts.authorization.to_account_info(),
            },
        )
        .with_signer(run_signer_seeds),
        AuthorizationGrantorUpdateParams { active },
    )
}
