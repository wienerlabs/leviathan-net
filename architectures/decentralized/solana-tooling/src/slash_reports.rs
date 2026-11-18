//! Charging and paying slashes no audit verdict accounts for: a `run_slash`
//! or an equivocation conviction records its reporter on a report when it
//! ejects its target, the target's bond pays for it once the ledger has
//! charged the ejection, and the reporter's bond is credited the bounty
//! (wienerlabs/leviathan-net#72). Both steps are permissionless.

use anchor_lang::AccountDeserialize;
use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use anyhow::Context;
use anyhow::Result;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_treasurer::accounts::RunChargeSlashReportAccounts;
use psyche_solana_treasurer::accounts::RunPaySlashReportBountyAccounts;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::find_slash_report;
use psyche_solana_treasurer::instruction::RunChargeSlashReport;
use psyche_solana_treasurer::instruction::RunPaySlashReportBounty;
use psyche_solana_treasurer::logic::RunChargeSlashReportParams;
use psyche_solana_treasurer::logic::RunPaySlashReportBountyParams;
use psyche_solana_treasurer::state::ConvictionCharge;
use psyche_solana_treasurer::state::SlashReport;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

/// The report of the slash that ejected `target` in the epoch that started
/// at `charge_epoch_start`, if one was made.
pub async fn get_slash_report(
    endpoint: &mut ToolboxEndpoint,
    run: &Pubkey,
    target: &Pubkey,
    charge_epoch_start: u64,
) -> Result<Option<SlashReport>> {
    let address = find_slash_report(run, target, charge_epoch_start);
    let Some(account) = endpoint.get_account(&address).await? else {
        return Ok(None);
    };
    if account.owner != psyche_solana_treasurer::ID || account.data.is_empty() {
        return Ok(None);
    }
    let report = SlashReport::try_deserialize(&mut &account.data[..]).map_err(
        |error| anyhow::anyhow!("Unable to decode the slash report: {error}"),
    )?;
    Ok(Some(report))
}

/// Whether `report` still has a step anyone can take: a charge, or a bounty
/// its reporter has not been credited.
pub fn unsettled(report: &SlashReport) -> bool {
    match report.charge {
        ConvictionCharge::Ejected => true,
        ConvictionCharge::Charged => {
            report.reporter.is_some()
                && report.bounty_owed > 0
                && !report.bounty_paid
        },
        ConvictionCharge::Unlanded => false,
    }
}

/// Every report of a slash in `run` that still waits to be charged or to
/// credit its reporter, oldest epoch first.
pub async fn unsettled_slash_reports(
    endpoint: &mut ToolboxEndpoint,
    run: &Pubkey,
) -> Result<Vec<SlashReport>> {
    // A report starts with its discriminator, then its bump, then its run.
    let discriminator: &[u8] =
        <SlashReport as anchor_lang::Discriminator>::DISCRIMINATOR;
    let addresses = endpoint
        .search_addresses(
            &psyche_solana_treasurer::ID,
            Some(SlashReport::space_with_discriminator()),
            &[(0, discriminator), (8 + 1, run.as_ref())],
        )
        .await?;
    let mut reports = vec![];
    for address in addresses {
        let Some(account) = endpoint.get_account(&address).await? else {
            continue;
        };
        let report = SlashReport::try_deserialize(&mut &account.data[..])
            .map_err(|error| {
                anyhow::anyhow!(
                    "Unable to decode the slash report {address}: {error}"
                )
            })?;
        if report.run == *run && unsettled(&report) {
            reports.push(report);
        }
    }
    reports.sort_by_key(|report| (report.charge_epoch_start, report.target));
    Ok(reports)
}

/// The instruction charging the slash that ejected `target` in the epoch
/// that started at `charge_epoch_start` against its bond.
pub fn run_charge_slash_report_instruction(
    payer: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
    charge_epoch_start: u64,
) -> Instruction {
    let accounts = RunChargeSlashReportAccounts {
        payer: *payer,
        run: *run,
        coordinator_account: *coordinator_account,
        slash_report: find_slash_report(run, target, charge_epoch_start),
        target_participant: find_participant(run, target),
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunChargeSlashReport {
            params: RunChargeSlashReportParams {
                target: *target,
                charge_epoch_start,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// Charges the slash that ejected `target` in the epoch that started at
/// `charge_epoch_start` against its bond.
pub async fn process_treasurer_run_charge_slash_report(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
    charge_epoch_start: u64,
) -> Result<()> {
    let instruction = run_charge_slash_report_instruction(
        &payer.pubkey(),
        run,
        coordinator_account,
        target,
        charge_epoch_start,
    );
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}

/// The instruction crediting `reporter` the bounty of the slash that ejected
/// `target` in the epoch that started at `charge_epoch_start`.
pub fn run_pay_slash_report_bounty_instruction(
    payer: &Pubkey,
    run: &Pubkey,
    target: &Pubkey,
    charge_epoch_start: u64,
    reporter: &Pubkey,
) -> Instruction {
    let accounts = RunPaySlashReportBountyAccounts {
        payer: *payer,
        run: *run,
        slash_report: find_slash_report(run, target, charge_epoch_start),
        reporter_participant: find_participant(run, reporter),
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunPaySlashReportBounty {
            params: RunPaySlashReportBountyParams {
                target: *target,
                charge_epoch_start,
                reporter: *reporter,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// Credits the reporter the slash's report records its bounty. Whom to credit
/// is read from the report.
pub async fn process_treasurer_run_pay_slash_report_bounty(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    target: &Pubkey,
    charge_epoch_start: u64,
) -> Result<()> {
    let report = get_slash_report(endpoint, run, target, charge_epoch_start)
        .await?
        .context("no slash ejected the target in that epoch")?;
    let reporter = report.reporter.context(
        "the slash names no reporter, so its bounty stays in the vault",
    )?;
    let instruction = run_pay_slash_report_bounty_instruction(
        &payer.pubkey(),
        run,
        target,
        charge_epoch_start,
        &reporter,
    );
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}

/// Which of `reports` the ledger has charged the ejection of, as the
/// coordinator stands now. The coordinator is read by reference and nothing
/// of it is kept, so the sweep holds no copy of it across an await: the sweep
/// runs inside the verifier daemon's pass, which has to fit a test's default
/// stack (wienerlabs/leviathan-net#35).
async fn charged_ejections(
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
    reports: &[SlashReport],
) -> Result<Vec<bool>> {
    let data = endpoint
        .get_account_data(coordinator_account)
        .await?
        .context("the run's coordinator account is gone")?;
    let coordinator = &coordinator_account_from_bytes(&data)?.state.coordinator;
    Ok(reports
        .iter()
        .map(|report| report.ejection_charged(coordinator))
        .collect())
}

/// What [`settle_slash_reports`] did.
#[derive(Debug, Default)]
pub struct SlashReportSweep {
    /// The reports that were not settled when the sweep began.
    pub open: usize,
    /// Slashes charged against their target's bond.
    pub charged: usize,
    /// Reporters credited their bounty.
    pub credited: usize,
    /// Reporters owed a bounty that have no participant account to credit
    /// it to yet. A later sweep credits them once they make one.
    pub awaiting_reporter: usize,
    /// Reports a step failed on, with why. The rest of the sweep goes on
    /// without them.
    pub failures: Vec<(Pubkey, u64, anyhow::Error)>,
}

/// Takes every step a slash report in `run` still waits for and that the
/// ledger allows now: it charges a slash whose epoch has been charged, then
/// credits its reporter. A report whose epoch is still running is left for a
/// later sweep. Another sender may be settling the same report, so each step
/// reads it afresh.
pub async fn settle_slash_reports(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
) -> Result<SlashReportSweep> {
    let open = unsettled_slash_reports(endpoint, run).await?;
    let charged =
        charged_ejections(endpoint, coordinator_account, &open).await?;
    let mut sweep = SlashReportSweep {
        open: open.len(),
        ..SlashReportSweep::default()
    };
    for (report, charged) in open.into_iter().zip(charged) {
        let (target, epoch_start) = (report.target, report.charge_epoch_start);
        if report.charge == ConvictionCharge::Ejected {
            if !charged {
                continue;
            }
            match process_treasurer_run_charge_slash_report(
                endpoint,
                payer,
                run,
                coordinator_account,
                &target,
                epoch_start,
            )
            .await
            {
                Ok(()) => sweep.charged += 1,
                Err(error) => {
                    sweep.failures.push((target, epoch_start, error));
                    continue;
                },
            }
        }
        let Some(fresh) =
            get_slash_report(endpoint, run, &target, epoch_start).await?
        else {
            continue;
        };
        if !unsettled(&fresh) {
            continue;
        }
        // A reporter with no participant account cannot be credited yet, and
        // sending would only fail again every pass until it makes one. An
        // address someone only sent lamports to is no participant account
        // either.
        if let Some(reporter) = fresh.reporter {
            let address = find_participant(run, &reporter);
            let ready =
                endpoint
                    .get_account(&address)
                    .await?
                    .is_some_and(|account| {
                        account.owner == psyche_solana_treasurer::ID
                            && !account.data.is_empty()
                    });
            if !ready {
                sweep.awaiting_reporter += 1;
                continue;
            }
        }
        match process_treasurer_run_pay_slash_report_bounty(
            endpoint,
            payer,
            run,
            &target,
            epoch_start,
        )
        .await
        {
            Ok(()) => sweep.credited += 1,
            Err(error) => sweep.failures.push((target, epoch_start, error)),
        }
    }
    Ok(sweep)
}
