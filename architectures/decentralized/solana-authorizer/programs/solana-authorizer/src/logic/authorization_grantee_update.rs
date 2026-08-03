use anchor_lang::prelude::*;

use crate::ProgramError;
use crate::state::Authorization;

#[derive(Accounts)]
#[instruction(params: AuthorizationGranteeUpdateParams)]
pub struct AuthorizationGranteeUpdateAccounts<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,

    #[account()]
    pub grantee: Signer<'info>,

    #[account(
        mut,
        constraint = authorization.grantee == grantee.key(),
        realloc = Authorization::space_with_discriminator(
            authorization.scope.len(),
            match params.delegates_clear {
                true => 0,
                false => authorization.delegates.len(),
            } + params.delegates_added.len()
        ),
        realloc::payer = payer,
        realloc::zero = true,
    )]
    pub authorization: Box<Account<'info, Authorization>>,

    #[account()]
    pub system_program: Program<'info, System>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct AuthorizationGranteeUpdateParams {
    pub delegates_clear: bool,
    pub delegates_added: Vec<Pubkey>,
}

pub fn authorization_grantee_update_processor(
    context: Context<AuthorizationGranteeUpdateAccounts>,
    params: AuthorizationGranteeUpdateParams,
) -> Result<()> {
    let authorization = &mut context.accounts.authorization;
    if params.delegates_clear {
        authorization.delegates.clear();
    }

    // Bounded, because this is the grantee extending its own authorization
    // without the grantor's signature, and each key it adds passes
    // `is_valid_for` (wienerlabs/leviathan#15, finding 19).
    if authorization.delegates.len() + params.delegates_added.len()
        > Authorization::MAX_DELEGATES
    {
        return err!(ProgramError::TooManyDelegates);
    }

    authorization.delegates.extend(params.delegates_added);
    Ok(())
}
