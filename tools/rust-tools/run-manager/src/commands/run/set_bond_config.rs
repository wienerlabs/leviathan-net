use crate::commands::Command;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use clap::Args;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;

use crate::{SolanaBackend, instructions};

/// Sets the bond a treasurer run requires, and how long a withdrawal waits so
/// that a conviction can reach the bond first. Signed by the run's main
/// authority. A non-zero minimum needs a non-zero delay
/// (wienerlabs/leviathan-net#51).
#[derive(Debug, Clone, Args)]
#[command()]
pub struct CommandSetBondConfig {
    #[clap(short, long, env)]
    pub run_id: String,
    #[clap(long, env)]
    pub treasurer_index: Option<u64>,
    /// The bond a participant must keep posted to join, vote and claim.
    #[clap(long, env)]
    pub bond_minimum_amount: u64,
    /// How long a requested withdrawal waits before it can be finalized.
    #[clap(long, env)]
    pub bond_withdraw_delay_seconds: i64,
}

#[async_trait]
impl Command for CommandSetBondConfig {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
            bond_minimum_amount,
            bond_withdraw_delay_seconds,
        } = self;

        let treasurer_index = backend
            .resolve_treasurer_index(&run_id, treasurer_index)
            .await?
            .ok_or_else(|| {
                anyhow!("run {run_id} has no treasurer run, and only a treasurer run holds a bond")
            })?;
        let instruction = instructions::treasurer_run_bond_config_update(
            treasurer_index,
            &backend.get_payer(),
            RunBondConfigUpdateParams {
                bond_minimum_amount,
                bond_withdraw_delay_seconds,
            },
        );

        let signature = backend
            .send_and_retry("Set bond config", &[instruction], &[])
            .await?;
        println!(
            "Set the bond of run {run_id} to {bond_minimum_amount}, withdrawals waiting {bond_withdraw_delay_seconds}s, with transaction {signature}"
        );

        println!("\n===== Logs =====");
        for log in backend.get_logs(&signature).await? {
            println!("{log}");
        }

        Ok(())
    }
}
