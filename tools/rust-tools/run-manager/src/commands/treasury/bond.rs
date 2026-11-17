use crate::commands::Command;
use anchor_spl::associated_token;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use clap::Args;

use super::verdict_votes::free_wallet_votes;
use crate::{SolanaBackend, instructions};
use anchor_client::solana_sdk::pubkey::Pubkey;
use psyche_solana_rpc::utils::native_amount_to_ui_amount;
use anchor_client::solana_sdk::instruction::AccountMeta;
use psyche_solana_treasurer::logic::bounty_amount;
use psyche_solana_treasurer::state::{
    Participant, Run, WithdrawalHold, WithdrawalStanding, coordinator_slashed_points,
    unattributed_ledger_forfeit, unledgered_conviction_charge, withdrawal_standing,
};

/// How `user`'s withdrawal stands, read from chain exactly the way the
/// treasurer reads it at finalize: the run, its coordinator, `user`'s
/// participant account and own audit verdict, and the cluster clock
/// (wienerlabs/leviathan-net#36, #39). Also returns what the settlement would
/// forfeit to points no verdict accounts for, whose bounty is paid to a
/// reporter the withdrawal names (wienerlabs/leviathan-net#64).
async fn current_withdrawal_standing(
    backend: &SolanaBackend,
    run_address: &Pubkey,
    run_state: &Run,
    user: &Pubkey,
) -> Result<(WithdrawalStanding, u64)> {
    let coordinator_account = backend
        .get_coordinator_account(&run_state.coordinator_account)
        .await?;
    let participant = backend
        .get_treasurer_participant(&psyche_solana_treasurer::find_participant(
            run_address,
            user,
        ))
        .await?;
    let own_verdict = backend.get_own_audit_verdict(run_address, user).await?;
    let now = backend.get_clock().await?.unix_timestamp;
    let standing = withdrawal_standing(
        run_state,
        &coordinator_account,
        &participant,
        own_verdict.as_ref(),
        user,
        now,
    );
    let unattributed = unattributed_ledger_forfeit(
        &participant,
        coordinator_slashed_points(&coordinator_account, user),
        own_verdict.as_ref(),
    );
    Ok((standing, unattributed))
}

fn describe_withdrawal_hold(hold: WithdrawalHold) -> &'static str {
    match hold {
        WithdrawalHold::ConvictionPending => {
            "a conviction against this wallet is still inside its challenge window or appeal; it can be finalized once that closes"
        },
        WithdrawalHold::EpochNotSettled => {
            "this wallet trains in an epoch that has not been charged yet; it can be finalized once the epoch ends"
        },
        WithdrawalHold::VotesAtStake => {
            "this wallet voted on a verdict that can still cost it; bond-withdraw-finalize frees each vote once it can"
        },
    }
}

/// What the treasurer will count of this bond, read from chain: a conviction
/// the bond has not paid yet already holds part of it, so does a conviction of
/// this wallet's own that stands but has not reached the ledger, and a
/// requested withdrawal is on its way out. Claims are checked against what a
/// slash could still take; joins and votes against what stays behind new work
/// (wienerlabs/leviathan-net#34).
async fn report_bond_standing(
    backend: &SolanaBackend,
    run_address: &Pubkey,
    run_state: &Run,
    participant: &Participant,
    user: &Pubkey,
    decimals: u8,
) -> Result<()> {
    let coordinator_account = backend
        .get_coordinator_account(&run_state.coordinator_account)
        .await?;
    let own_verdict = backend.get_own_audit_verdict(run_address, user).await?;
    let slashed_points = coordinator_slashed_points(&coordinator_account, user);
    let unledgered = unledgered_conviction_charge(&coordinator_account, own_verdict.as_ref());
    let unforfeited = participant.unforfeited_bond_owing(slashed_points, unledgered);
    let committed = participant.committed_bond_owing(slashed_points, unledgered);
    println!(
        "Unsettled slashed points: {}",
        participant.unsettled_slashed_points(slashed_points)
    );
    println!(
        "Conviction not yet charged to the ledger: {}",
        native_amount_to_ui_amount(unledgered, decimals)
    );
    println!(
        "Unforfeited bond: {}",
        native_amount_to_ui_amount(unforfeited, decimals)
    );
    println!(
        "Committed bond: {}",
        native_amount_to_ui_amount(committed, decimals)
    );
    println!(
        "Verdict votes that can still cost the bond: {}",
        participant.open_verdict_votes
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
        report_bond_standing(
            &backend,
            &run_address,
            &run_state,
            &participant,
            &user,
            decimals,
        )
        .await?;
        if participant.bond_withdraw_pending_amount > 0 {
            match current_withdrawal_standing(&backend, &run_address, &run_state, &user).await?.0 {
                WithdrawalStanding::Held(hold) => {
                    println!("Withdrawal: held, {}", describe_withdrawal_hold(hold))
                },
                WithdrawalStanding::Free { settle_conviction } => {
                    println!(
                        "Withdrawal: not held; finalizable once {} seconds have passed since it was requested",
                        participant.bond_withdraw_delay_snapshot
                    );
                    if settle_conviction {
                        println!(
                            "Finalizing settles this wallet's conviction against its bond: it ran out its windows without reaching the ledger"
                        );
                    }
                },
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
        report_bond_standing(
            &backend,
            &run_address,
            &run_state,
            &participant,
            &user,
            decimals,
        )
        .await
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
    /// Also withdraw this wallet's votes on verdicts whose round is still under
    /// way. They stop counting toward a conviction.
    #[clap(long)]
    pub withdraw_live_votes: bool,
    /// The wallet that reported a slash no verdict accounts for, such as one
    /// the run authority made. A withdrawal that settles one pays its bounty
    /// to this wallet's collateral account, and is refused without it. It
    /// cannot be this wallet.
    #[clap(long)]
    pub reporter: Option<Pubkey>,
}

#[async_trait]
impl Command for CommandBondWithdrawFinalize {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
            withdraw_live_votes,
            reporter,
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

        // Votes this wallet cast hold the bond until each can no longer cost
        // it, so the ones that can be freed are freed first
        // (wienerlabs/leviathan-net#39). Every share of a bounty the wallet is
        // owed is credited to its bond too, as a voter, a tie-breaker or
        // either; that holds whether or not it still has a vote open, since a
        // tie-breaker's never counts, and anyone may have released a voter's
        // (wienerlabs/leviathan-net#64).
        //
        // Finding them reads every verdict in the run, which a public RPC may
        // refuse. A wallet with a vote open cannot withdraw without them, so
        // that fails the command; any other wallet is warned that a share it
        // may be owed was not credited, and withdraws all the same.
        let participant_address = psyche_solana_treasurer::find_participant(&run_address, &user);
        let open_votes = backend
            .get_treasurer_participant(&participant_address)
            .await?
            .open_verdict_votes;
        match free_wallet_votes(
            &backend,
            treasurer_index,
            &run_address,
            &run_state,
            &user,
            withdraw_live_votes,
        )
        .await
        {
            Ok(held) => {
                for (target, reason) in &held {
                    println!(
                        "A vote on the verdict against {target} still holds the bond: {reason}"
                    );
                }
            }
            Err(error) if open_votes == 0 => {
                println!(
                    "Could not read the run's verdicts to credit any bounty share this wallet is owed; withdrawing without it: {error:#}"
                );
            }
            Err(error) => return Err(error),
        }

        let (standing, unattributed_forfeit) =
            current_withdrawal_standing(&backend, &run_address, &run_state, &user).await?;
        let settle_conviction = match standing {
            WithdrawalStanding::Held(hold) => {
                return Err(anyhow!(
                    "The withdrawal is held: {}",
                    describe_withdrawal_hold(hold)
                ));
            },
            WithdrawalStanding::Free { settle_conviction } => settle_conviction,
        };

        let mut instruction = instructions::treasurer_participant_bond_finalize_withdraw(
            treasurer_index,
            &run_state.collateral_mint,
            &run_state.coordinator_account,
            &user,
        );
        // A conviction's bounty is recorded on its verdict and credited to
        // each recipient's bond by anyone, never paid here. What the ledger
        // charged that no verdict accounts for pays its bounty to a reporter
        // this withdrawal names (wienerlabs/leviathan-net#64).
        let reporter_bounty = bounty_amount(&run_state, unattributed_forfeit);
        if reporter_bounty > 0 {
            let reporter = reporter.ok_or_else(|| {
                anyhow!(
                    "This withdrawal settles {} of slashes no verdict accounts for, which owes a reporter {}; pass --reporter <WALLET>",
                    native_amount_to_ui_amount(unattributed_forfeit, decimals),
                    native_amount_to_ui_amount(reporter_bounty, decimals),
                )
            })?;
            if reporter == user {
                return Err(anyhow!(
                    "The reporter cannot be the wallet whose bond is slashed"
                ));
            }
            instruction.accounts.push(AccountMeta::new(
                associated_token::get_associated_token_address(
                    &reporter,
                    &run_state.collateral_mint,
                ),
                false,
            ));
        } else if reporter.is_some() {
            println!("No slash this withdrawal settles owes a reporter; --reporter is not used");
        }
        if settle_conviction {
            println!("Settling this wallet's conviction against its bond as part of the withdrawal");
        }
        let signature = backend
            .send_and_retry("Bond withdraw finalize", &[instruction], &[])
            .await?;
        println!("Finalized the bond withdraw in transaction: {signature}");

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
