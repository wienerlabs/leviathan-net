use crate::commands::Command;
use anchor_spl::associated_token;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use clap::Args;

use super::verdict_votes::free_wallet_votes;
use crate::{SolanaBackend, instructions};
use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_lang::AccountDeserialize;
use psyche_solana_rpc::utils::native_amount_to_ui_amount;
use psyche_solana_treasurer::state::{
    ConvictionCharge, Participant, Run, SlashReport, WithdrawalHold, WithdrawalStanding,
    coordinator_slashed_points, unledgered_conviction_charge, withdrawal_standing,
};

/// The slash report at `address` as it stands now. A step that fails may have
/// been taken first by someone else, the verifier daemon's sweep for one, and
/// this tells the two apart.
async fn fresh_slash_report(
    backend: &SolanaBackend,
    address: &Pubkey,
) -> Result<Option<SlashReport>> {
    let Some(account) = backend.get_account(address).await? else {
        return Ok(None);
    };
    let report = SlashReport::try_deserialize(&mut &account.data[..])
        .map_err(|error| anyhow!("Unable to decode the slash report {address}: {error}"))?;
    Ok(Some(report))
}

/// How `user`'s withdrawal stands, read from chain exactly the way the
/// treasurer reads it at finalize: the run, its coordinator, `user`'s
/// participant account and own audit verdict, and the cluster clock
/// (wienerlabs/leviathan-net#36, #39, #72).
async fn current_withdrawal_standing(
    backend: &SolanaBackend,
    run_address: &Pubkey,
    run_state: &Run,
    user: &Pubkey,
) -> Result<WithdrawalStanding> {
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
    Ok(standing)
}

/// Settles the slash reports that concern `user` in the run and that the
/// ledger allows now (wienerlabs/leviathan-net#72): it charges each slash that
/// ejected `user` once its epoch is charged, since its bond is held until
/// then, and credits `user`'s bond the bounty of each slash it reported.
/// Anyone may send either; the verifier daemon does both each pass, and a
/// step someone else already took is skipped. Returns how many reports still
/// hold `user`'s bond.
async fn settle_wallet_slash_reports(
    backend: &SolanaBackend,
    treasurer_index: u64,
    run_address: &Pubkey,
    run_state: &Run,
    user: &Pubkey,
) -> Result<usize> {
    let reports = backend.get_treasurer_slash_reports(run_address).await?;
    // Whether the ledger has charged each report's ejection, read once and
    // not held across the sends below: a coordinator is a large value.
    let charged: Vec<bool> = {
        let account = backend
            .get_coordinator_account(&run_state.coordinator_account)
            .await?;
        reports
            .iter()
            .map(|(_, report)| report.ejection_charged(&account.state.coordinator))
            .collect()
    };
    let mut holding = 0;
    for ((address, report), charged) in reports.into_iter().zip(charged) {
        let charge = match report.charge {
            ConvictionCharge::Ejected if report.target == *user => {
                if !charged {
                    holding += 1;
                    continue;
                }
                Some(instructions::treasurer_run_charge_slash_report(
                    treasurer_index,
                    &run_state.coordinator_account,
                    user,
                    &report.target,
                    report.charge_epoch_start,
                ))
            }
            _ => None,
        };
        if let Some(instruction) = charge {
            match backend
                .send_and_retry("Slash report charge", &[instruction], &[])
                .await
            {
                Ok(signature) => println!(
                    "Charged the slash that ejected this wallet in the epoch from {} in transaction: {signature}",
                    report.charge_epoch_start
                ),
                Err(error) => match fresh_slash_report(backend, &address).await? {
                    Some(fresh) if fresh.charge != ConvictionCharge::Ejected => println!(
                        "The slash that ejected this wallet in the epoch from {} was already charged",
                        report.charge_epoch_start
                    ),
                    _ => return Err(error),
                },
            }
            continue;
        }
        let owed = report.charge == ConvictionCharge::Charged
            && report.reporter == Some(*user)
            && report.bounty_owed > 0
            && !report.bounty_paid;
        if owed {
            let instruction = instructions::treasurer_run_pay_slash_report_bounty(
                treasurer_index,
                user,
                &report.target,
                report.charge_epoch_start,
                user,
            );
            match backend
                .send_and_retry("Slash report bounty", &[instruction], &[])
                .await
            {
                Ok(signature) => println!(
                    "Credited this wallet the bounty for reporting {} in transaction: {signature}",
                    report.target
                ),
                Err(error) => match fresh_slash_report(backend, &address).await? {
                    Some(fresh) if fresh.bounty_paid => println!(
                        "The bounty for reporting {} was already credited to this wallet",
                        report.target
                    ),
                    _ => return Err(error),
                },
            }
        }
    }
    Ok(holding)
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
        WithdrawalHold::SlashReportsUncharged => {
            "a slash that ejected this wallet has not been charged against its bond yet; bond-withdraw-finalize charges it once its epoch has been"
        }
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
    let now = backend.get_clock().await?.unix_timestamp;
    let unledgered = unledgered_conviction_charge(
        run_state,
        &coordinator_account,
        user,
        own_verdict.as_ref(),
        now,
    );
    let unforfeited = participant.unforfeited_bond_owing(slashed_points, unledgered);
    let committed = participant.committed_bond_owing(slashed_points, unledgered);
    println!(
        "Unsettled slashed points: {}",
        participant.unsettled_slashed_points(slashed_points)
    );
    println!(
        "Ejection or conviction not yet charged to the ledger: {}",
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
            match current_withdrawal_standing(&backend, &run_address, &run_state, &user).await? {
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
}

#[async_trait]
impl Command for CommandBondWithdrawFinalize {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
            withdraw_live_votes,
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

        // A slash no verdict accounts for holds the bond until it is charged,
        // and a slash this wallet reported owes it a bounty; both are settled
        // here when the ledger allows (wienerlabs/leviathan-net#72). Reading
        // the reports takes getProgramAccounts, which a public RPC may refuse:
        // a wallet a report holds cannot withdraw without them, any other is
        // warned and withdraws all the same.
        let uncharged = backend
            .get_treasurer_participant(&participant_address)
            .await?
            .uncharged_slash_reports;
        match settle_wallet_slash_reports(
            &backend,
            treasurer_index,
            &run_address,
            &run_state,
            &user,
        )
        .await
        {
            Ok(0) => {}
            Ok(holding) => println!(
                "{holding} slash(es) that ejected this wallet wait for their epoch to be charged"
            ),
            Err(error) if uncharged == 0 => println!(
                "Could not read the run's slash reports to credit any bounty this wallet is owed; withdrawing without it: {error:#}"
            ),
            Err(error) => return Err(error),
        }

        let standing =
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

        // A withdrawal names no reporter: a slash's reporter is recorded when
        // the slash is made, and credited from its report
        // (wienerlabs/leviathan-net#72).
        let instruction = instructions::treasurer_participant_bond_finalize_withdraw(
            treasurer_index,
            &run_state.collateral_mint,
            &run_state.coordinator_account,
            &user,
        );
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
