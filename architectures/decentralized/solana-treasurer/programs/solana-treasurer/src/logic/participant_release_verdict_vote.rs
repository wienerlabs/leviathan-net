use anchor_lang::prelude::*;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VoteRelease;

#[derive(Accounts)]
#[instruction(params: ParticipantReleaseVerdictVoteParams)]
pub struct ParticipantReleaseVerdictVoteAccounts<'info> {
    /// Anyone. When it is the voter itself, a vote on a verdict that has not
    /// reached quorum may be withdrawn while its round is still under way.
    #[account()]
    pub authority: Signer<'info>,

    #[account(
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    // An account of another version is refused, not loaded: loading one
    // of another size panics (wienerlabs/leviathan-net#62).
    #[account(
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account(
        mut,
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump = audit_verdict.bump,
    )]
    pub audit_verdict: Box<Account<'info, AuditVerdict>>,

    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            params.voter.as_ref(),
        ],
        bump = participant.bump,
    )]
    pub participant: Box<Account<'info, Participant>>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ParticipantReleaseVerdictVoteParams {
    pub target: Pubkey,
    pub voter: Pubkey,
}

/// Stops `voter`'s vote on `target`'s verdict holding its bond, once the verdict
/// says the vote can no longer cost it (wienerlabs/leviathan-net#39).
///
/// Releasing never takes anything from anyone, and only ever lets a bond leave
/// that nothing can charge any more, so anyone may send it: the verifier itself
/// before a withdrawal, or a verifier about to open a new dispute against the
/// same target, which has to wait until the last one holds nobody.
pub fn participant_release_verdict_vote_processor(
    context: Context<ParticipantReleaseVerdictVoteAccounts>,
    params: ParticipantReleaseVerdictVoteParams,
) -> Result<()> {
    let round_over = context.accounts.audit_verdict.round_over(
        &context
            .accounts
            .coordinator_account
            .load()?
            .state
            .coordinator,
    );
    let voter_consents = context.accounts.authority.key() == params.voter;

    let release = context.accounts.audit_verdict.release_vote(
        &params.voter,
        voter_consents,
        round_over,
    )?;
    context.accounts.participant.close_verdict_vote()?;

    msg!(
        "release_verdict_vote: voter={} target={} {}",
        params.voter,
        params.target,
        match release {
            VoteRelease::Withdrawn => "withdrawn",
            VoteRelease::Released => "released",
        }
    );
    Ok(())
}
