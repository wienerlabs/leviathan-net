pub mod logic;
pub mod state;

use anchor_lang::prelude::*;
use logic::*;

// The cluster a build targets is chosen here, not by patching this file at
// deploy time. A mainnet build carries a different program id, so a binary
// meant for devnet can never be pointed at mainnet by mistake.
#[cfg(not(feature = "mainnet"))]
declare_id!("9A1kc8Dr9dFJW9t1npAk7EHrADm6TAyFeVLH27CDdvv8");
#[cfg(feature = "mainnet")]
declare_id!("A6Z8jZeKi81zUaozR7X7SGXtY8EyXm1YyTeFMuFgXEkW");

pub fn find_run(index: u64) -> Pubkey {
    Pubkey::find_program_address(
        &[state::Run::SEEDS_PREFIX, index.to_le_bytes().as_ref()],
        &crate::ID,
    )
    .0
}

pub fn find_participant(run: &Pubkey, user: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(
        &[
            state::Participant::SEEDS_PREFIX,
            run.as_ref(),
            user.as_ref(),
        ],
        &crate::ID,
    )
    .0
}

pub fn find_audit_verdict(run: &Pubkey, target: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(
        &[
            state::AuditVerdict::SEEDS_PREFIX,
            run.as_ref(),
            target.as_ref(),
        ],
        &crate::ID,
    )
    .0
}

#[program]
pub mod psyche_solana_treasurer {
    use super::*;

    pub fn run_create(
        context: Context<RunCreateAccounts>,
        params: RunCreateParams,
    ) -> Result<()> {
        run_create_processor(context, params)
    }

    pub fn run_update(
        context: Context<RunUpdateAccounts>,
        params: RunUpdateParams,
    ) -> Result<()> {
        run_update_processor(context, params)
    }

    pub fn participant_create(
        context: Context<ParticipantCreateAccounts>,
        params: ParticipantCreateParams,
    ) -> Result<()> {
        participant_create_processor(context, params)
    }

    pub fn participant_claim(
        context: Context<ParticipantClaimAccounts>,
        params: ParticipantClaimParams,
    ) -> Result<()> {
        participant_claim_processor(context, params)
    }

    pub fn run_bond_config_update(
        context: Context<RunBondConfigUpdateAccounts>,
        params: RunBondConfigUpdateParams,
    ) -> Result<()> {
        run_bond_config_update_processor(context, params)
    }

    pub fn participant_bond_deposit(
        context: Context<ParticipantBondDepositAccounts>,
        params: ParticipantBondDepositParams,
    ) -> Result<()> {
        participant_bond_deposit_processor(context, params)
    }

    pub fn participant_bond_request_withdraw(
        context: Context<ParticipantBondRequestWithdrawAccounts>,
        params: ParticipantBondRequestWithdrawParams,
    ) -> Result<()> {
        participant_bond_request_withdraw_processor(context, params)
    }

    pub fn participant_bond_finalize_withdraw<'info>(
        context: Context<'_, '_, 'info, 'info, ParticipantBondFinalizeWithdrawAccounts<'info>>,
        params: ParticipantBondFinalizeWithdrawParams,
    ) -> Result<()> {
        participant_bond_finalize_withdraw_processor(context, params)
    }

    pub fn run_slash(
        context: Context<RunSlashAccounts>,
        params: RunSlashParams,
    ) -> Result<()> {
        run_slash_processor(context, params)
    }

    pub fn run_set_slash_bounty(
        context: Context<RunSetSlashBountyAccounts>,
        params: RunSetSlashBountyParams,
    ) -> Result<()> {
        run_set_slash_bounty_processor(context, params)
    }

    pub fn participant_authorize_join(
        context: Context<ParticipantAuthorizeJoinAccounts>,
    ) -> Result<()> {
        participant_authorize_join_processor(context)
    }

    pub fn run_submit_audit_verdict(
        context: Context<RunSubmitAuditVerdictAccounts>,
        params: RunSubmitAuditVerdictParams,
    ) -> Result<()> {
        run_submit_audit_verdict_processor(context, params)
    }

    pub fn run_set_challenge_config(
        context: Context<RunSetChallengeConfigAccounts>,
        params: RunSetChallengeConfigParams,
    ) -> Result<()> {
        run_set_challenge_config_processor(context, params)
    }

    pub fn run_open_challenge(
        context: Context<RunOpenChallengeAccounts>,
        params: RunOpenChallengeParams,
    ) -> Result<()> {
        run_open_challenge_processor(context, params)
    }

    pub fn run_submit_appeal_verdict(
        context: Context<RunSubmitAppealVerdictAccounts>,
        params: RunSubmitAppealVerdictParams,
    ) -> Result<()> {
        run_submit_appeal_verdict_processor(context, params)
    }

    pub fn run_finalize_slash(
        context: Context<RunFinalizeSlashAccounts>,
        params: RunFinalizeSlashParams,
    ) -> Result<()> {
        run_finalize_slash_processor(context, params)
    }

    pub fn run_slash_losing_verifier(
        context: Context<RunSlashLosingVerifierAccounts>,
        params: RunSlashLosingVerifierParams,
    ) -> Result<()> {
        run_slash_losing_verifier_processor(context, params)
    }
}

#[error_code]
pub enum ProgramError {
    #[msg("Invalid parameter")]
    InvalidParameter,

    #[msg("run_id must be 32 bytes or less")]
    RunIdInvalidLength,

    #[msg("Bond balance is insufficient for this request")]
    InsufficientBond,

    #[msg("The bond withdraw delay has not elapsed yet")]
    WithdrawDelayNotElapsed,

    #[msg("A slash bounty is configured but no reporter account was provided")]
    MissingReporter,

    #[msg("Bond is below the run minimum required to claim rewards")]
    BondBelowMinimum,

    #[msg("A run that requires a bond must also set a positive bond withdraw delay")]
    BondWindowRequired,

    #[msg("The verifier is not a participant in the current epoch")]
    VerifierNotInEpoch,

    #[msg("The signer is not an assigned verifier for this round")]
    VerifierNotAssigned,

    #[msg("The target index does not match the provided target key")]
    TargetMismatch,

    #[msg("This verifier already submitted a verdict for this target this epoch")]
    DuplicateVerdict,

    #[msg("The verdict has already resolved into a slash this epoch")]
    VerdictAlreadyResolved,

    #[msg("The verdict voter set is full")]
    VerdictVotersFull,

    #[msg("A bounty recipient does not match the corresponding verdict voter")]
    BountyRecipientMismatch,

    #[msg("The verdict is not in a slash-pending state")]
    VerdictNotPending,

    #[msg("The challenge window has already closed")]
    ChallengeWindowClosed,

    #[msg("The challenge window is still open")]
    ChallengeWindowOpen,

    #[msg("The verdict is not under challenge")]
    VerdictNotChallenged,

    #[msg("The signer is not an assigned tie-breaker for this round")]
    NotTieBreaker,

    #[msg("This tie-breaker already voted on this appeal")]
    DuplicateAppealVerdict,

    #[msg("The appeal voter set is full")]
    AppealVotersFull,

    #[msg("The verdict was not overturned")]
    VerdictNotOverturned,

    #[msg("All losing verifiers have already been settled")]
    AllLosersSettled,
}
