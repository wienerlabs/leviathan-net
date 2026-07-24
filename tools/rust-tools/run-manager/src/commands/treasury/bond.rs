use crate::commands::Command;
use anchor_spl::associated_token;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use clap::Args;

use crate::{SolanaBackend, instructions};
use psyche_solana_rpc::utils::native_amount_to_ui_amount;

#[derive(Debug, Clone, Args)]
#[command()]
pub struct CommandBondDeposit {
    #[clap(short, long, env)]
    pub run_id: String,
    #[clap(long, env)]
    pub treasurer_index: Option<u64>,
    #[clap(long)]
    pub amount: u64,
}

#[async_trait]
impl Command for CommandBondDeposit {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
            amount,
        } = self;

        let treasurer_index = backend
            .resolve_treasurer_index(&run_id, treasurer_index)
            .await?
            .context("Failed to resolve treasurer")?;
        let run_address = psyche_solana_treasurer::find_run(treasurer_index);
        let run_state = backend.get_treasurer_run(&run_address).await?;
        let user = backend.get_payer();
        let decimals = backend
            .get_token_mint(&run_state.collateral_mint)
            .await?
            .decimals;

        println!("Run: {run_id}");
        println!("Wallet: {user}");
        println!(
            "Bond minimum: {}",
            native_amount_to_ui_amount(run_state.bond_minimum_amount, decimals)
        );

        let participant_address =
            psyche_solana_treasurer::find_participant(&run_address, &user);
        if backend.get_balance(&participant_address).await? == 0 {
            let instruction = instructions::treasurer_participant_create(
                &backend.get_payer(),
                treasurer_index,
                &user,
            );
            let signature = backend
                .send_and_retry("Create participant PDA", &[instruction], &[])
                .await?;
            println!("Created the participant account in transaction: {signature}");
        }

        let user_collateral = associated_token::get_associated_token_address(
            &user,
            &run_state.collateral_mint,
        );
        let available = backend.get_token_account(&user_collateral).await?.amount;
        if available < amount {
            return Err(anyhow!(
                "collateral balance {} is below the requested deposit {}",
                native_amount_to_ui_amount(available, decimals),
                native_amount_to_ui_amount(amount, decimals)
            ));
        }

        let instruction = instructions::treasurer_participant_bond_deposit(
            treasurer_index,
            &run_state.collateral_mint,
            &user,
            amount,
        );
        let signature = backend
            .send_and_retry("Bond deposit", &[instruction], &[])
            .await?;
        println!(
            "Deposited {} as bond in transaction: {}",
            native_amount_to_ui_amount(amount, decimals),
            signature
        );

        let participant = backend
            .get_treasurer_participant(&participant_address)
            .await?;
        println!(
            "Bond amount now: {}",
            native_amount_to_ui_amount(participant.bond_amount, decimals)
        );
        if participant.bond_amount >= run_state.bond_minimum_amount {
            println!("This wallet now meets the run bond minimum");
        } else {
            println!("This wallet is still below the run bond minimum");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Args)]
#[command()]
pub struct CommandBondStatus {
    #[clap(short, long, env)]
    pub run_id: String,
    #[clap(long, env)]
    pub treasurer_index: Option<u64>,
}

#[async_trait]
impl Command for CommandBondStatus {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
        } = self;

        let treasurer_index = backend
            .resolve_treasurer_index(&run_id, treasurer_index)
            .await?
            .context("Failed to resolve treasurer")?;
        let run_address = psyche_solana_treasurer::find_run(treasurer_index);
        let run_state = backend.get_treasurer_run(&run_address).await?;
        let user = backend.get_payer();
        let decimals = backend
            .get_token_mint(&run_state.collateral_mint)
            .await?
            .decimals;

        println!("Run: {run_id}");
        println!("Wallet: {user}");
        println!("Collateral mint: {}", run_state.collateral_mint);
        println!(
            "Bond minimum: {}",
            native_amount_to_ui_amount(run_state.bond_minimum_amount, decimals)
        );
        println!(
            "Bond withdraw delay: {} seconds",
            run_state.bond_withdraw_delay_seconds
        );

        let participant_address =
            psyche_solana_treasurer::find_participant(&run_address, &user);
        if backend.get_balance(&participant_address).await? == 0 {
            println!("No participant account yet, run bond-deposit first");
            return Ok(());
        }

        let participant = backend
            .get_treasurer_participant(&participant_address)
            .await?;
        println!(
            "Bond amount: {}",
            native_amount_to_ui_amount(participant.bond_amount, decimals)
        );
        println!(
            "Withdraw pending: {}",
            native_amount_to_ui_amount(participant.bond_withdraw_pending_amount, decimals)
        );
        println!(
            "Withdraw requested at: {}",
            participant.bond_withdraw_requested_at
        );
        println!(
            "Settled slashed points: {}",
            participant.bond_settled_slashed_points
        );
        if participant.bond_amount >= run_state.bond_minimum_amount {
            println!("Status: meets the run bond minimum");
        } else {
            println!("Status: below the run bond minimum");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Args)]
#[command()]
pub struct CommandBondWithdrawRequest {
    #[clap(short, long, env)]
    pub run_id: String,
    #[clap(long, env)]
    pub treasurer_index: Option<u64>,
    #[clap(long)]
    pub amount: u64,
}

#[async_trait]
impl Command for CommandBondWithdrawRequest {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
            amount,
        } = self;

        let treasurer_index = backend
            .resolve_treasurer_index(&run_id, treasurer_index)
            .await?
            .context("Failed to resolve treasurer")?;
        let run_address = psyche_solana_treasurer::find_run(treasurer_index);
        let run_state = backend.get_treasurer_run(&run_address).await?;
        let user = backend.get_payer();
        let decimals = backend
            .get_token_mint(&run_state.collateral_mint)
            .await?
            .decimals;

        let instruction = instructions::treasurer_participant_bond_request_withdraw(
            treasurer_index,
            &user,
            amount,
        );
        let signature = backend
            .send_and_retry("Bond withdraw request", &[instruction], &[])
            .await?;
        println!(
            "Requested withdraw of {} in transaction: {}",
            native_amount_to_ui_amount(amount, decimals),
            signature
        );
        println!(
            "Finalize after {} seconds with bond-withdraw-finalize",
            run_state.bond_withdraw_delay_seconds
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Args)]
#[command()]
pub struct CommandBondWithdrawFinalize {
    #[clap(short, long, env)]
    pub run_id: String,
    #[clap(long, env)]
    pub treasurer_index: Option<u64>,
}

#[async_trait]
impl Command for CommandBondWithdrawFinalize {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
        } = self;

        let treasurer_index = backend
            .resolve_treasurer_index(&run_id, treasurer_index)
            .await?
            .context("Failed to resolve treasurer")?;
        let run_address = psyche_solana_treasurer::find_run(treasurer_index);
        let run_state = backend.get_treasurer_run(&run_address).await?;
        let user = backend.get_payer();
        let decimals = backend
            .get_token_mint(&run_state.collateral_mint)
            .await?
            .decimals;

        let instruction = instructions::treasurer_participant_bond_finalize_withdraw(
            treasurer_index,
            &run_state.collateral_mint,
            &run_state.coordinator_account,
            &user,
        );
        let signature = backend
            .send_and_retry("Bond withdraw finalize", &[instruction], &[])
            .await?;
        println!("Finalized the bond withdraw in transaction: {signature}");

        let participant_address =
            psyche_solana_treasurer::find_participant(&run_address, &user);
        let participant = backend
            .get_treasurer_participant(&participant_address)
            .await?;
        println!(
            "Bond amount now: {}",
            native_amount_to_ui_amount(participant.bond_amount, decimals)
        );
        Ok(())
    }
}
