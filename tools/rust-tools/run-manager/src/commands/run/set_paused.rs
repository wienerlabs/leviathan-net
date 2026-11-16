use crate::commands::Command;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use clap::Args;
use psyche_solana_treasurer::logic::RunUpdateParams;
use std::time::Duration;

use crate::{SolanaBackend, instructions};

/// How many times a pause is sent while the round it would cancel waits for
/// its committed slot to be done, a slot apart.
const PAUSE_ATTEMPTS: usize = 8;
const SLOT_TIME: Duration = Duration::from_millis(400);

#[derive(Debug, Clone, Args)]
#[command()]
pub struct CommandSetPaused {
    #[clap(short, long, env)]
    pub run_id: String,
    #[clap(long, env)]
    pub treasurer_index: Option<u64>,
    #[clap(long, env)]
    pub resume: bool,
}

#[async_trait]
impl Command for CommandSetPaused {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            treasurer_index,
            resume,
        } = self;

        let paused = !resume;
        let main_authority = backend.get_payer();

        let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(&run_id);
        let coordinator_instance_state = backend
            .get_coordinator_instance(&coordinator_instance)
            .await?;
        let coordinator_account = coordinator_instance_state.coordinator_account;

        let instruction = if let Some(treasurer_index) = backend
            .resolve_treasurer_index(&run_id, treasurer_index)
            .await?
        {
            instructions::treasurer_run_update(
                &run_id,
                treasurer_index,
                &coordinator_account,
                &main_authority,
                RunUpdateParams {
                    metadata: None,
                    config: None,
                    model: None,
                    progress: None,
                    epoch_earning_rate_total_shared: None,
                    epoch_slashing_rate_per_client: None,
                    paused: Some(paused),
                    client_version: None,
                },
            )
        } else {
            instructions::coordinator_set_paused(
                &run_id,
                &coordinator_account,
                &main_authority,
                paused,
            )
        };

        let mut attempt = 0;
        let signature = loop {
            attempt += 1;
            let seed_committed = paused
                && backend
                    .get_coordinator_account(&coordinator_account)
                    .await?
                    .state
                    .seed_committed();
            let tick = instructions::coordinator_tick(
                &coordinator_instance,
                &coordinator_account,
                &main_authority,
            );
            let sent = backend
                .send_and_retry(
                    "Set paused",
                    &instructions::set_paused_instructions(
                        seed_committed,
                        tick,
                        instruction.clone(),
                    ),
                    &[],
                )
                .await;
            match sent {
                Ok(signature) => break signature,
                // Refused while a seed is committed: in the very slot it was
                // committed in, before the round can start; since the account
                // was read; or to a slot the sysvar no longer holds, where the
                // tick in front renews the commitment and the refused pause
                // reverts the renewal with it. A tick of its own keeps the
                // renewal, and a slot later the round can start.
                Err(error) if paused && attempt < PAUSE_ATTEMPTS => {
                    let committed = backend
                        .get_coordinator_account(&coordinator_account)
                        .await?
                        .state
                        .seed_committed();
                    if !committed {
                        return Err(error).context("Set paused");
                    }
                    println!("A round's seed is committed; ticking, then pausing a slot later ({error:#})");
                    let tick = instructions::coordinator_tick(
                        &coordinator_instance,
                        &coordinator_account,
                        &main_authority,
                    );
                    backend
                        .send_and_retry("Tick", &[tick], &[])
                        .await
                        .context("Tick before the pause")?;
                    tokio::time::sleep(SLOT_TIME).await;
                }
                Err(error) if paused => {
                    bail!(
                        "the pause was refused {PAUSE_ATTEMPTS} times while a round's seed was committed; last: {error:#}"
                    )
                }
                Err(error) => return Err(error).context("Set paused"),
            }
        };
        println!("Set pause state to {paused} on run {run_id} with transaction {signature}");

        println!("\n===== Logs =====");
        for log in backend.get_logs(&signature).await? {
            println!("{log}");
        }

        Ok(())
    }
}
