use crate::commands::Command;
use anchor_client::solana_sdk::pubkey::Pubkey;
use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use clap::Args;
use psyche_coordinator::RunState;

use psyche_solana_rpc::SolanaBackend;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::state::coordinator_slashed_points;
use psyche_solana_treasurer::state::unledgered_conviction_charge;

#[derive(Debug, Clone, Args)]
#[command()]
pub struct CommandCanJoin {
    #[clap(short, long, env)]
    pub run_id: String,
    #[clap(long, env)]
    pub authorizer: Option<Pubkey>,
    #[clap(long, env, alias = "wallet", alias = "user", value_name = "PUBKEY")]
    pub address: Pubkey,
}

#[async_trait]
impl Command for CommandCanJoin {
    async fn execute(self, backend: SolanaBackend) -> Result<()> {
        let Self {
            run_id,
            authorizer,
            address,
        } = self;

        let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(&run_id);
        let coordinator_instance_state = backend
            .get_coordinator_instance(&coordinator_instance)
            .await?;

        let join_authority = coordinator_instance_state.join_authority;
        let coordinator_account = backend
            .get_coordinator_account(&coordinator_instance_state.coordinator_account)
            .await?;
        let join_authority_account = backend.get_account(&join_authority).await?;

        if SolanaBackend::is_bond_gated_run(
            &join_authority,
            &coordinator_instance,
            join_authority_account.as_ref(),
        )? {
            // The treasurer opens the join authorization only inside its own
            // join, so there is no standing authorization to check here: what
            // decides admission is the bond as the treasurer will count it
            // (wienerlabs/leviathan-net#34).
            let run = backend.get_treasurer_run(&join_authority).await?;
            let participant_address = find_participant(&join_authority, &address);
            if backend.get_balance(&participant_address).await? == 0 {
                bail!(
                    "Run id {run_id} is gated on the bond and {address} has no participant account; post a bond with `run-manager bond-deposit` first"
                );
            }
            let participant = backend
                .get_treasurer_participant(&participant_address)
                .await?;
            let own_verdict = backend
                .get_own_audit_verdict(&join_authority, &address)
                .await?;
            let slashed_points = coordinator_slashed_points(&coordinator_account, &address);
            let now = backend.get_clock().await?.unix_timestamp;
            let unledgered = unledgered_conviction_charge(
                &run,
                &coordinator_account,
                &address,
                own_verdict.as_ref(),
                now,
            );
            let committed = participant.committed_bond_owing(slashed_points, unledgered);
            println!(
                "Bond-gated run: bond {}, unsettled slashed points {}, ejection or conviction not yet charged {}, withdrawal pending {}, committed {}, minimum {}",
                participant.bond_amount,
                participant.unsettled_slashed_points(slashed_points),
                unledgered,
                participant.bond_withdraw_pending_amount,
                committed,
                run.bond_minimum_amount,
            );
            if committed < run.bond_minimum_amount {
                bail!(
                    "The committed bond of {address} is below the minimum of run id {run_id}; the treasurer will refuse the join"
                );
            }
            println!("bond committed for run id {run_id} using pubkey {address}");
        } else {
            let authorization =
                SolanaBackend::find_join_authorization(&join_authority, authorizer);
            if backend.get_balance(&authorization).await? == 0 {
                bail!(
                    "Authorization does not exist for authorizer: {authorizer:?} (authorization address: {authorization:?}, join authority: {join_authority:?}). Authorizer must be set to grantee pubkey for permissioned runs"
                );
            }
            if !backend
                .get_authorization(&authorization)
                .await?
                .is_valid_for(
                    &join_authority,
                    &address,
                    psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE,
                )
            {
                bail!("Authorization invalid for run id {run_id} using pubkey {address}");
            }
            println!("authorization valid for run id {run_id} using pubkey {address}");
        }

        let coordinator_account_state = coordinator_account.state.coordinator;

        println!(
            "Coordinator: run_state: {}",
            coordinator_account_state.run_state
        );
        let is_paused = matches!(
            coordinator_account_state.run_state,
            RunState::Paused | RunState::Uninitialized
        );
        println!("Coordinator: is_paused: {is_paused}");

        if !is_paused {
            let client_with_our_key = coordinator_account_state
                .epoch_state
                .clients
                .iter()
                .find(|c| *c.id.signer() == address.to_bytes());
            if client_with_our_key.is_some() {
                bail!(
                    "A client with our pubkey {address} is in the current epoch, you can't join with this key!"
                );
            }
        }

        println!("✓ Can join run {run_id} with pubkey {address}");
        println!("\nTo predownload model and eval tasks before joining, run:");
        println!(
            "  psyche-solana-client predownload --run-id {run_id} --model --eval-tasks <TASKS>"
        );

        Ok(())
    }
}
