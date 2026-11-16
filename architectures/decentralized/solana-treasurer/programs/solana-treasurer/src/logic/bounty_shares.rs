use anchor_lang::prelude::*;
use anchor_spl::token::transfer;
use anchor_spl::token::TokenAccount;
use anchor_spl::token::Transfer;

use crate::state::Run;
use crate::ProgramError;

/// Where a forfeited bond's bounty is paid from, and on whose authority.
pub struct BountyVault<'a, 'info> {
    pub token_program: &'a AccountInfo<'info>,
    pub run_collateral: &'a AccountInfo<'info>,
    pub run: &'a Account<'info, Run>,
}

/// The bounty share of a forfeit of `forfeited_amount` in `run`.
pub fn bounty_amount(run: &Run, forfeited_amount: u64) -> u64 {
    (forfeited_amount as u128 * run.slash_bounty_bps as u128 / 10_000) as u64
}

/// Splits `bounty_amount` evenly between `recipients`, paying each through
/// the account at the same position in `recipient_accounts`.
///
/// Each of those accounts has to be a token account of the run's collateral
/// owned by the recipient at its position, so whoever sends the instruction
/// chooses nothing about where the bounty goes. A share that rounds down to
/// zero pays nothing, and what does not divide evenly stays in the vault.
pub fn pay_bounty_shares<'info>(
    vault: &BountyVault<'_, 'info>,
    recipients: &[Pubkey],
    recipient_accounts: &[AccountInfo<'info>],
    bounty_amount: u64,
) -> Result<()> {
    if recipient_accounts.len() < recipients.len() {
        return err!(ProgramError::MissingReporter);
    }
    let share = match recipients.len() {
        0 => return Ok(()),
        count => bounty_amount / count as u64,
    };
    if share == 0 {
        return Ok(());
    }
    for (recipient, account) in recipients.iter().zip(recipient_accounts) {
        pay_from_vault(vault, recipient, account, share)?;
    }
    Ok(())
}

/// Pays `amount` out of the vault to `account`, which has to be a token
/// account of the run's collateral owned by `recipient`.
pub fn pay_from_vault<'info>(
    vault: &BountyVault<'_, 'info>,
    recipient: &Pubkey,
    account: &AccountInfo<'info>,
    amount: u64,
) -> Result<()> {
    let run = vault.run;
    // What `Account::<TokenAccount>::try_from` checks: the token program owns
    // the account, and its data is a token account.
    if account.owner != &anchor_spl::token::ID {
        return err!(ProgramError::BountyRecipientMismatch);
    }
    let token_account =
        TokenAccount::try_deserialize(&mut &account.try_borrow_data()?[..])
            .map_err(|_| error!(ProgramError::BountyRecipientMismatch))?;
    if token_account.owner != *recipient
        || token_account.mint != run.collateral_mint
    {
        return err!(ProgramError::BountyRecipientMismatch);
    }
    let index_bytes = run.index.to_le_bytes();
    let run_signer_seeds: &[&[&[u8]]] =
        &[&[Run::SEEDS_PREFIX, &index_bytes, &[run.bump]]];
    transfer(
        CpiContext::new(
            vault.token_program.clone(),
            Transfer {
                from: vault.run_collateral.clone(),
                to: account.clone(),
                authority: run.to_account_info(),
            },
        )
        .with_signer(run_signer_seeds),
        amount,
    )
}
