use anchor_lang::prelude::*;
use anchor_spl::token::Token;
use anchor_spl::token::TokenAccount;
use anchor_spl::token::Transfer;
use anchor_spl::token::transfer;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::ProgramError;
use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::coordinator_slashed_points;
use crate::state::read_own_verdict;
use crate::state::unledgered_conviction_charge;

#[derive(Accounts)]
#[instruction(params: ParticipantClaimParams)]
pub struct ParticipantClaimAccounts<'info> {
    #[account()]
    pub user: Signer<'info>,

    #[account(
        mut,
        constraint = user_collateral.mint == run.collateral_mint,
        constraint = user_collateral.owner == user.key(),
        constraint = user_collateral.delegate == None.into(),
    )]
    pub user_collateral: Box<Account<'info, TokenAccount>>,

    #[account(
        mut,
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    #[account(
        mut,
        associated_token::mint = run.collateral_mint,
        associated_token::authority = run,
    )]
    pub run_collateral: Box<Account<'info, TokenAccount>>,

    #[account(
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            user.key().as_ref()
        ],
        bump = participant.bump
    )]
    pub participant: Box<Account<'info, Participant>>,

    #[account()]
    pub token_program: Program<'info, Token>,

    /// CHECK: the claimant's own verdict, pinned to its PDA by the seeds below
    /// and read only after this checks that it exists and belongs to this
    /// program. A conviction there that stands but has not reached the ledger
    /// still owes its charge. Last, so a client that passes it still works
    /// with a treasurer that does not read it.
    #[account(
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            user.key().as_ref(),
        ],
        bump,
    )]
    pub own_verdict: UncheckedAccount<'info>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ParticipantClaimParams {
    pub claim_earned_points: u64,
}

pub fn participant_claim_processor(
    context: Context<ParticipantClaimAccounts>,
    params: ParticipantClaimParams,
) -> Result<()> {
    let user_key = context.accounts.user.key();
    let own_verdict =
        read_own_verdict(&context.accounts.own_verdict.to_account_info())?;
    let (
        participant_earned_points,
        participant_slashed_points,
        unledgered_conviction,
    ) = {
        let coordinator_account = context.accounts.coordinator_account.load()?;
        let earned = coordinator_account
            .state
            .clients_state
            .clients
            .iter()
            .find(|client| *client.id.signer() == user_key.to_bytes())
            .map(|client| client.earned)
            .unwrap_or(0);
        (
            earned,
            coordinator_slashed_points(&coordinator_account, &user_key),
            unledgered_conviction_charge(
                &context.accounts.run,
                &coordinator_account,
                &user_key,
                own_verdict.as_ref(),
                Clock::get()?.unix_timestamp,
            ),
        )
    };

    // Nothing is taken without something left to slash, and a bond a conviction
    // has already charged is not something left: it is only still counted in
    // `bond_amount` because settlement waits for the participant to withdraw
    // (wienerlabs/leviathan-net#34). Nor is the bond a conviction that stands
    // will take once it reaches the ledger.
    if context.accounts.participant.unforfeited_bond_owing(
        participant_slashed_points,
        unledgered_conviction,
    ) < context.accounts.run.bond_minimum_amount
    {
        return err!(ProgramError::BondBelowMinimum);
    }

    let participant = &mut context.accounts.participant;
    let run = &mut context.accounts.run;

    let participant_unclaimed_earned_points =
        participant_earned_points - participant.claimed_earned_points;
    if params.claim_earned_points > participant_unclaimed_earned_points {
        return err!(ProgramError::InvalidParameter);
    }

    // We distribute 1 collateral per point and let the coordinator decide the point reward rate
    let claim_collateral_amount = params.claim_earned_points;

    participant.claimed_collateral_amount += claim_collateral_amount;
    participant.claimed_earned_points += params.claim_earned_points;

    run.total_claimed_collateral_amount += claim_collateral_amount;
    run.total_claimed_earned_points += params.claim_earned_points;

    let run_signer_seeds: &[&[&[u8]]] =
        &[&[Run::SEEDS_PREFIX, &run.index.to_le_bytes(), &[run.bump]]];
    transfer(
        CpiContext::new(
            context.accounts.token_program.to_account_info(),
            Transfer {
                from: context.accounts.run_collateral.to_account_info(),
                to: context.accounts.user_collateral.to_account_info(),
                authority: context.accounts.run.to_account_info(),
            },
        )
        .with_signer(run_signer_seeds),
        claim_collateral_amount,
    )?;

    Ok(())
}
