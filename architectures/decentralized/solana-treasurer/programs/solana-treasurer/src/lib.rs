pub mod logic;
pub mod state;

use anchor_lang::prelude::*;
use logic::*;

declare_id!("9A1kc8Dr9dFJW9t1npAk7EHrADm6TAyFeVLH27CDdvv8");

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
        context: Context<'_, '_, '_, 'info, ParticipantBondFinalizeWithdrawAccounts<'info>>,
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
}
