pub mod logic;
pub mod state;

use anchor_lang::prelude::*;
use logic::*;

// The cluster a build targets is chosen here, not by patching this file at
// deploy time. A mainnet build carries a different program id, so a binary
// meant for devnet can never be pointed at mainnet by mistake.
#[cfg(not(feature = "mainnet"))]
declare_id!("2Kg5ERG6ubuzyPmQ24axsws7V2ja2EvWp5CHMKFCrTxv");
#[cfg(feature = "mainnet")]
declare_id!("2QXAd9g31vKFGSyxZC2wcjJdCZ4bjCdzrXA95H6Ft2eU");

pub fn find_authorization(
    grantor: &Pubkey,
    grantee: &Pubkey,
    scope: &[u8],
) -> Pubkey {
    Pubkey::find_program_address(
        &[
            state::Authorization::SEEDS_PREFIX,
            grantor.as_ref(),
            grantee.as_ref(),
            scope,
        ],
        &crate::ID,
    )
    .0
}

#[program]
pub mod psyche_solana_authorizer {
    use super::*;

    pub fn authorization_create(
        context: Context<AuthorizationCreateAccounts>,
        params: AuthorizationCreateParams,
    ) -> Result<()> {
        authorization_create_processor(context, params)
    }

    pub fn authorization_grantor_update(
        context: Context<AuthorizationGrantorUpdateAccounts>,
        params: AuthorizationGrantorUpdateParams,
    ) -> Result<()> {
        authorization_grantor_update_processor(context, params)
    }

    pub fn authorization_grantee_update(
        context: Context<AuthorizationGranteeUpdateAccounts>,
        params: AuthorizationGranteeUpdateParams,
    ) -> Result<()> {
        authorization_grantee_update_processor(context, params)
    }

    pub fn authorization_close(
        context: Context<AuthorizationCloseAccounts>,
        params: AuthorizationCloseParams,
    ) -> Result<()> {
        authorization_close_processor(context, params)
    }
}

#[error_code]
pub enum ProgramError {
    #[msg("AuthorizationActive is true")]
    AuthorizationActiveIsTrue,
    #[msg("Authorization closing conditions not reached yet")]
    AuthorizationClosingConditionsNotReachedYet,
    #[msg("An authorization cannot hold more delegates than the cap allows")]
    TooManyDelegates,
}
