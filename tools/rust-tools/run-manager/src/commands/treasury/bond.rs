use crate::commands::Command;
use anchor_spl::associated_token;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use clap::Args;

use crate::{SolanaBackend, instructions};
use anchor_client::solana_sdk::pubkey::Pubkey;
use psyche_solana_rpc::utils::native_amount_to_ui_amount;
use psyche_solana_treasurer::state::{
    AuditVerdict, Participant, Run, WithdrawalHold, coordinator_slashed_points, withdrawal_hold,
};

/// Whether `user`'s withdrawal is held, read from chain exactly the way the
/// treasurer reads it at finalize: the run's coordinator and `user`'s own audit
/// verdict (wienerlabs/leviathan-net#36).
async fn current_withdrawal_hold(
    backend: &SolanaBackend,
    run_address: &Pubkey,
    run_state: &Run,
    user: &Pubkey,
) -> Result<Option<WithdrawalHold>> {
    let coordinator_account = backend
        .get_coordinator_account(&run_state.coordinator_account)
        .await?;
    let verdict_address = psyche_solana_treasurer::find_audit_verdict(run_address, user);
    let own_verdict = match backend.get_account(&verdict_address).await? {
        Some(account) => AuditVerdict::from_pda_account(&account.owner, &account.data)
            .map_err(|error| anyhow!("Unable to decode the audit verdict: {error}"))?,
        None => None,
    };
    Ok(withdrawal_hold(
        &coordinator_account,
        own_verdict.as_ref(),
        user,
    ))
}

fn describe_withdrawal_hold(hold: WithdrawalHold) -> &'static str {
    match hold {
        WithdrawalHold::ConvictionPending => {
            "a conviction against this wallet is still pending (challenge window or appeal); it can be finalized once the conviction resolves"
        },
        WithdrawalHold::EpochNotSettled => {
            "this wallet trains in an epoch that has not been charged yet; it can be finalized once the epoch ends"
        },
    }
}

/// What the treasurer will count of this bond, read from chain: a conviction
/// the bond has not paid yet already holds part of it, and a requested
/// withdrawal is on its way out. Claims are checked against what a slash could
/// still take; joins and votes against what stays behind new work
/// (wienerlabs/leviathan-net#34).
async fn report_bond_standing(
    backend: &SolanaBackend,
    run_state: &Run,
    participant: &Participant,
    user: &Pubkey,
    decimals: u8,
) -> Result<()> {
    let coordinator_account = backend
        .get_coordinator_account(&run_state.coordinator_account)
        .await?;
    let slashed_points = coordinator_slashed_points(&coordinator_account, user);
    let unforfeited = participant.unforfeited_bond(slashed_points);
    let committed = participant.committed_bond(slashed_points);
    println!(
        "Unsettled slashed points: {}",
        participant.unsettled_slashed_points(slashed_points)
    );
    println!(
        "Unforfeited bond: {}",
        native_amount_to_ui_amount(unforfeited, decimals)
    );
    println!(
        "Committed bond: {}",
        native_amount_to_ui_amount(committed, decimals)
    );
    let open = |bond: u64| {
        if bond >= run_state.bond_minimum_amount {
            "open"
        } else {
            "closed, below the run bond minimum"
        }
    };
    println!("Claims: {}", open(unforfeited));
    println!("Joins and votes: {}", open(committed));
    Ok(())
}

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
        report_bond_standing(&backend, &run_state, &participant, &user, decimals).await?;
        if participant.bond_withdraw_pending_amount > 0 {
            match current_withdrawal_hold(&backend, &run_address, &run_state, &user).await? {
                Some(hold) => println!("Withdrawal: held, {}", describe_withdrawal_hold(hold)),
                None => println!(
                    "Withdrawal: not held; finalizable once {} seconds have passed since it was requested",
                    participant.bond_withdraw_delay_snapshot
                ),
            }
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
        report_bond_standing(&backend, &run_state, &participant, &user, decimals).await
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

        if let Some(hold) =
            current_withdrawal_hold(&backend, &run_address, &run_state, &user).await?
        {
            return Err(anyhow!(
                "The withdrawal is held: {}",
                describe_withdrawal_hold(hold)
            ));
        }

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
