use crate::instructions::{self, coordinator_tick};
use crate::retry::{RetryError, retry_function_with_params};
use anchor_client::anchor_lang::AccountDeserialize;
use anchor_client::solana_sdk::hash::hash;
use anchor_client::solana_sdk::instruction::Instruction;
use anchor_client::solana_sdk::program_pack::Pack;
use anchor_client::{
    Client, ClientError, Cluster, Program,
    anchor_lang::system_program,
    solana_client::{
        nonblocking::pubsub_client::PubsubClient,
        rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig, RpcTransactionConfig},
        rpc_filter::{Memcmp, RpcFilterType},
        rpc_response::Response as RpcResponse,
    },
    solana_sdk::{
        account::Account,
        clock::Clock,
        commitment_config::CommitmentConfig,
        pubkey::Pubkey,
        signature::{Keypair, Signature, Signer},
    },
};
use anyhow::{Context, Result, anyhow};
use futures_util::StreamExt;
use psyche_coordinator::model::{self, Checkpoint};
use psyche_coordinator::{CommitteeProof, Coordinator, HealthChecks};
use psyche_core::IntegrationTestLogMarker;
use psyche_event_sourcing::event;
use psyche_event_sourcing::events::RpcCallType;
use psyche_watcher::{Backend as WatcherBackend, OpportunisticData};
use solana_account_decoder_client_types::{UiAccount, UiAccountEncoding};
use solana_transaction_status_client_types::UiTransactionEncoding;
use std::future::Future;
use std::{cmp::min, sync::Arc, time::Duration};
use tokio::{
    sync::{broadcast, mpsc},
    task::{JoinError, JoinSet},
    time::timeout,
};
use tracing::{error, info, trace, warn};

type ProgramCoordinator = Arc<Program<Arc<Keypair>>>;
const RPC_RETRY_ATTEMPTS: usize = 3;
const RPC_INITIAL_BACKOFF_MS: u64 = 500;
const RPC_MAX_BACKOFF_MS: u64 = 10_000;

#[derive(Clone)]
pub struct SolanaBackend {
    program_coordinators: Vec<ProgramCoordinator>,
    cluster: Cluster,
    backup_clusters: Vec<Cluster>,
    wallet: Arc<Keypair>,
}

pub struct SolanaBackendRunner {
    pub(crate) backend: SolanaBackend,
    instance: Pubkey,
    account: Pubkey,
    updates: broadcast::Receiver<Coordinator>,
    init: Option<Coordinator>,
    /// The checkpoint sends this runner started whose outcome it has not read
    /// yet. They used to be detached, so a client that exited right after its
    /// last checkpoint dropped the send with the process
    /// (wienerlabs/leviathan-net#85).
    checkpoints: JoinSet<Result<()>>,
    /// The first checkpoint send read to have failed, kept until a flush
    /// reports it.
    failed_checkpoint: Option<anyhow::Error>,
}

async fn subscribe_to_account(
    url: String,
    commitment: CommitmentConfig,
    coordinator_account: &Pubkey,
    tx: mpsc::UnboundedSender<RpcResponse<UiAccount>>,
    id: u64,
) {
    let mut retries: u64 = 0;
    loop {
        // wait a time before we try a reconnection
        let sleep_time = min(30, retries.saturating_mul(2));
        tokio::time::sleep(Duration::from_secs(sleep_time)).await;
        retries += 1;
        let Ok(sub_client) = PubsubClient::new(&url).await else {
            event!(coordinator::SolanaSubscriptionChanged {
                url: url.to_string(),
                status: SubscriptionStatus::Down,
            });
            warn!(
                integration_test_log_marker = %IntegrationTestLogMarker::SolanaSubscription,
                url = url,
                subscription_number = id,
                "Solana subscription error, could not connect to url: {url}",
            );
            continue;
        };

        let mut notifications = match sub_client
            .account_subscribe(
                coordinator_account,
                Some(RpcAccountInfoConfig {
                    encoding: Some(UiAccountEncoding::Base64Zstd),
                    commitment: Some(commitment),
                    ..Default::default()
                }),
            )
            .await
        {
            Ok((notifications, _)) => notifications,
            Err(err) => {
                error!(
                    url = url,
                    subscription_number = id,
                    error = format!("{err:#}"),
                    "Solana account subscribe error",
                );
                continue;
            }
        };

        event!(coordinator::SolanaSubscriptionChanged {
            url: url.to_string(),
            status: SubscriptionStatus::Up,
        });
        info!(
            integration_test_log_marker = %IntegrationTestLogMarker::SolanaSubscription,
            url = url,
            subscription_number = id,
            "Correctly subscribe to Solana url: {url}",
        );

        retries = 0;

        loop {
            tokio::select! {
                update = notifications.next() => {
                    match update {
                        Some(data) => {
                                if tx.send(data).is_err() {
                                    break;
                                }
                        }
                        None => {
                            event!(coordinator::SolanaSubscriptionChanged {
                                url: url.to_string(),
                                status: SubscriptionStatus::Down,
                            });
                            warn!(
                                integration_test_log_marker = %IntegrationTestLogMarker::SolanaSubscription,
                                url = url,
                                subscription_number = id,
                                "Solana subscription error, websocket closed");
                            break
                        }
                    }
                }
            }
        }
    }
}

impl SolanaBackend {
    #[allow(clippy::result_large_err)]
    pub fn new(
        cluster: Cluster,
        backup_clusters: Vec<Cluster>,
        payer: Arc<Keypair>,
        commitment: CommitmentConfig,
    ) -> Result<Self> {
        let client = Client::new_with_options(cluster.clone(), payer.clone(), commitment);

        let mut program_coordinators = vec![];
        program_coordinators.push(Arc::new(client.program(psyche_solana_coordinator::ID)?));

        let backup_program_coordinators: Result<Vec<_>, _> = backup_clusters
            .iter()
            .filter(|backup| backup.url() != cluster.url())
            .map(|cluster| {
                Client::new_with_options(cluster.clone(), payer.clone(), commitment)
                    .program(psyche_solana_coordinator::ID)
            })
            .collect();
        program_coordinators.extend(backup_program_coordinators?.into_iter().map(Arc::new));

        Ok(Self {
            program_coordinators,
            cluster,
            backup_clusters,
            wallet: payer,
        })
    }

    pub async fn start(
        self,
        run_id: String,
        coordinator_account: Pubkey,
    ) -> Result<SolanaBackendRunner> {
        let (tx_update, rx_update) = broadcast::channel(32);
        let commitment_config = self.get_commitment_config();

        let (tx_subscribe, mut rx_subscribe) = mpsc::unbounded_channel();

        let tx_subscribe_ = tx_subscribe.clone();

        let mut subscription_number = 1;
        let url = self.cluster.clone().ws_url().to_string();
        tokio::spawn(async move {
            subscribe_to_account(
                url,
                commitment_config,
                &coordinator_account,
                tx_subscribe_,
                subscription_number,
            )
            .await
        });

        for cluster in self.backup_clusters.clone() {
            subscription_number += 1;
            let tx_subscribe_ = tx_subscribe.clone();
            tokio::spawn(async move {
                subscribe_to_account(
                    cluster.ws_url().to_string().clone(),
                    commitment_config,
                    &coordinator_account,
                    tx_subscribe_,
                    subscription_number,
                )
                .await
            });
        }
        tokio::spawn(async move {
            let mut last_nonce = 0;
            while let Some(update) = rx_subscribe.recv().await {
                match update.value.data.decode() {
                    Some(data) => {
                        match psyche_solana_coordinator::coordinator_account_from_bytes(&data) {
                            Ok(account) => {
                                if account.nonce > last_nonce {
                                    trace!(
                                        nonce = account.nonce,
                                        last_nonce = last_nonce,
                                        "Coordinator account update"
                                    );
                                    if let Err(err) = tx_update.send(account.state.coordinator) {
                                        error!("Error sending coordinator update: {err:#}");
                                        break;
                                    }
                                    last_nonce = account.nonce;
                                }
                            }
                            Err(err) => error!("Error deserializing coordinator account: {err:#}"),
                        }
                    }
                    None => error!("Error decoding coordinator account"),
                }
            }
            error!("No subscriptions available");
        });

        let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(&run_id);

        info!("Coordinator account address: {}", coordinator_account);
        info!(
            "Coordinator instance address for run \"{}\": {}",
            run_id, coordinator_instance
        );

        let init = self
            .get_coordinator_account(&coordinator_account)
            .await?
            .state
            .coordinator;

        Ok(SolanaBackendRunner {
            backend: self,
            updates: rx_update,
            instance: coordinator_instance,
            account: coordinator_account,
            init: Some(init),
            checkpoints: JoinSet::new(),
            failed_checkpoint: None,
        })
    }

    pub async fn join_run(
        &self,
        coordinator_instance: Pubkey,
        coordinator_account: Pubkey,
        id: psyche_core::NodeIdentity,
        authorizer: Option<Pubkey>,
    ) -> Result<Signature> {
        let coordinator_instance_state =
            self.get_coordinator_instance(&coordinator_instance).await?;
        let join_authority = coordinator_instance_state.join_authority;
        let join_authority_account = self.get_account(&join_authority).await?;
        let instruction = if Self::is_bond_gated_run(
            &join_authority,
            &coordinator_instance,
            join_authority_account.as_ref(),
        )? {
            info!("Run {join_authority} gates joins on the bond, joining through the treasurer");
            instructions::treasurer_participant_join_run(
                &join_authority,
                &coordinator_instance,
                &coordinator_account,
                &self.get_payer(),
                id,
            )
        } else {
            let authorization = Self::find_join_authorization(&join_authority, authorizer);
            instructions::coordinator_join_run(
                &coordinator_instance,
                &coordinator_account,
                &authorization,
                id,
            )
        };
        // send() polls Solana until the tx is confirmed; we cancel early and let the
        // caller retry on the next epoch. The public devnet RPC frequently needs more
        // than a few seconds to confirm, so the deadline is env-tunable and defaults
        // to a value that survives it (LEVIATHAN_JOIN_TIMEOUT_SECS).
        let join_timeout_secs = std::env::var("LEVIATHAN_JOIN_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(30);
        event!(coordinator::RpcCallSubmitted {
            call_type: RpcCallType::Join
        });
        let result = match timeout(
            Duration::from_secs(join_timeout_secs),
            self.send_and_retry("Join run", &[instruction], &[]),
        )
        .await
        {
            Ok(Ok(signature)) => Ok(signature),
            Err(elapsed) => Err(anyhow!("join_run timeout: {elapsed}")),
            Ok(Err(error)) => Err(error),
        };
        match &result {
            Ok(_) => event!(coordinator::RpcCallResult {
                call_type: RpcCallType::Join,
                result: Ok(())
            }),
            Err(e) => event!(coordinator::RpcCallResult {
                call_type: RpcCallType::Join,
                result: Err(e.to_string())
            }),
        }
        result
    }

    pub fn send_tick(&self, coordinator_instance: Pubkey, coordinator_account: Pubkey) {
        let user = self.get_payer();
        let instruction = coordinator_tick(&coordinator_instance, &coordinator_account, &user);
        self.spawn_scheduled_send("Tick", &[instruction], &[], RpcCallType::Tick);
    }

    pub fn send_witness(
        &self,
        coordinator_instance: Pubkey,
        coordinator_account: Pubkey,
        opportunistic_data: OpportunisticData,
    ) {
        let user = self.get_payer();
        let (instruction, call_type) = match opportunistic_data {
            OpportunisticData::WitnessStep(witness, metadata) => (
                instructions::coordinator_witness(
                    &coordinator_instance,
                    &coordinator_account,
                    &user,
                    witness,
                    metadata,
                ),
                RpcCallType::Witness,
            ),
            OpportunisticData::WarmupStep(witness) => (
                instructions::coordinator_warmup_witness(
                    &coordinator_instance,
                    &coordinator_account,
                    &user,
                    witness,
                ),
                RpcCallType::WarmupWitness,
            ),
        };
        self.spawn_scheduled_send("Witness", &[instruction], &[], call_type);
    }

    pub fn send_health_check(
        &self,
        coordinator_instance: Pubkey,
        coordinator_account: Pubkey,
        id: psyche_core::NodeIdentity,
        check: CommitteeProof,
    ) {
        let user = self.get_payer();
        let instruction = instructions::coordinator_health_check(
            &coordinator_instance,
            &coordinator_account,
            &user,
            id,
            check,
        );
        self.spawn_scheduled_send(
            "Health check",
            &[instruction],
            &[],
            RpcCallType::HealthCheck,
        );
    }

    /// Records `repo` as the run's checkpoint, retried as every scheduled send
    /// is. Unlike the others it is not spawned here: the returned future is the
    /// send, and the caller keeps it, so that a client about to exit can wait
    /// for its last checkpoint to land (wienerlabs/leviathan-net#85).
    pub fn send_checkpoint(
        &self,
        coordinator_instance: Pubkey,
        coordinator_account: Pubkey,
        repo: Checkpoint,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let user = self.get_payer();
        let instruction = instructions::coordinator_checkpoint(
            &coordinator_instance,
            &coordinator_account,
            &user,
            repo,
        );
        self.scheduled_send("Checkpoint", &[instruction], &[], RpcCallType::Checkpoint)
    }

    /// Whether `join_authority`, holding `join_authority_account`, is a
    /// treasurer `Run` that gates `coordinator_instance` on its own bond.
    ///
    /// Such a run is its own join authority, and admits a node only through the
    /// treasurer's `participant_join_run`, which checks the committed bond on
    /// every join and leaves the join authorization inactive at rest. Joining it
    /// with the coordinator's `join_run` directly is refused
    /// (wienerlabs/leviathan-net#34). Any other join authority, a wallet or a
    /// treasurer run gated by an outside key, is joined directly.
    pub fn is_bond_gated_run(
        join_authority: &Pubkey,
        coordinator_instance: &Pubkey,
        join_authority_account: Option<&Account>,
    ) -> Result<bool> {
        let Some(account) = join_authority_account else {
            return Ok(false);
        };
        if account.owner != psyche_solana_treasurer::ID {
            return Ok(false);
        }
        let run = psyche_solana_treasurer::state::Run::try_deserialize(&mut account.data.as_slice())
            .map_err(|error| {
                anyhow!("join authority {join_authority} is a treasurer account but not a run: {error}")
            })?;
        Ok(run.join_authority == *join_authority && run.coordinator_instance == *coordinator_instance)
    }

    pub fn find_join_authorization(join_authority: &Pubkey, authorizer: Option<Pubkey>) -> Pubkey {
        psyche_solana_authorizer::find_authorization(
            join_authority,
            &authorizer.unwrap_or(system_program::ID),
            psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE,
        )
    }

    pub async fn get_authorization(
        &self,
        authorization: &Pubkey,
    ) -> Result<psyche_solana_authorizer::state::Authorization> {
        let data = self.get_data(authorization).await?;
        psyche_solana_authorizer::state::Authorization::try_deserialize(&mut data.as_slice())
            .map_err(|error| anyhow!("Unable to decode authorization data: {error}"))
    }

    pub async fn get_coordinator_instance(
        &self,
        coordinator_instance: &Pubkey,
    ) -> Result<psyche_solana_coordinator::CoordinatorInstance> {
        let data = self.get_data(coordinator_instance).await?;
        psyche_solana_coordinator::CoordinatorInstance::try_deserialize(&mut data.as_slice())
            .map_err(|error| anyhow!("Unable to decode coordinator instance data: {error}"))
    }

    pub async fn get_coordinator_account(
        &self,
        coordinator_account: &Pubkey,
    ) -> Result<psyche_solana_coordinator::CoordinatorAccount> {
        let data = self.get_data(coordinator_account).await?;
        psyche_solana_coordinator::coordinator_account_from_bytes(&data)
            .map_err(|error| anyhow!("Unable to decode coordinator account data: {error}"))
            .copied()
    }

    pub async fn get_treasurer_run(
        &self,
        treasurer_run: &Pubkey,
    ) -> Result<psyche_solana_treasurer::state::Run> {
        let data = self.get_data(treasurer_run).await?;
        psyche_solana_treasurer::state::Run::try_deserialize(&mut data.as_slice())
            .map_err(|error| anyhow!("Unable to decode treasurer run data: {error}"))
    }

    pub async fn get_treasurer_participant(
        &self,
        treasurer_participant: &Pubkey,
    ) -> Result<psyche_solana_treasurer::state::Participant> {
        let data = self.get_data(treasurer_participant).await?;
        psyche_solana_treasurer::state::Participant::try_deserialize(&mut data.as_slice())
            .map_err(|error| anyhow!("Unable to decode treasurer participant data: {error}"))
    }

    /// The audit verdict that names `user` as its target in `treasurer_run`,
    /// if `user` was ever disputed there.
    pub async fn get_own_audit_verdict(
        &self,
        treasurer_run: &Pubkey,
        user: &Pubkey,
    ) -> Result<Option<psyche_solana_treasurer::state::AuditVerdict>> {
        let address = psyche_solana_treasurer::find_audit_verdict(treasurer_run, user);
        match self.get_account(&address).await? {
            Some(account) => psyche_solana_treasurer::state::AuditVerdict::from_pda_account(
                &account.owner,
                &account.data,
            )
            .map_err(|error| anyhow!("Unable to decode the audit verdict: {error}")),
            None => Ok(None),
        }
    }

    /// Every audit verdict the treasurer holds for `treasurer_run`, with its
    /// address. A verdict account keeps the run at a fixed offset, right after
    /// its discriminator and bump, so the cluster filters by it.
    pub async fn get_treasurer_audit_verdicts(
        &self,
        treasurer_run: &Pubkey,
    ) -> Result<Vec<(Pubkey, psyche_solana_treasurer::state::AuditVerdict)>> {
        use anchor_client::anchor_lang::Discriminator;
        use psyche_solana_treasurer::state::AuditVerdict;
        const RUN_OFFSET: usize = 8 + 1;
        let config = RpcProgramAccountsConfig {
            filters: Some(vec![
                RpcFilterType::DataSize(AuditVerdict::space_with_discriminator() as u64),
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(0, AuditVerdict::DISCRIMINATOR)),
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(
                    RUN_OFFSET,
                    treasurer_run.as_ref(),
                )),
            ]),
            account_config: RpcAccountInfoConfig {
                encoding: Some(UiAccountEncoding::Base64),
                ..Default::default()
            },
            ..Default::default()
        };
        let accounts = self
            .rpc_with_fallback("get_treasurer_audit_verdicts", |coord| {
                let config = config.clone();
                async move {
                    coord
                        .rpc()
                        .get_program_accounts_with_config(&psyche_solana_treasurer::ID, config)
                        .await
                        .map_err(Into::into)
                }
            })
            .await
            .context(
                "Unable to list the run's audit verdicts: the RPC endpoint has to serve getProgramAccounts",
            )?;
        accounts
            .into_iter()
            .map(|(address, account)| {
                AuditVerdict::try_deserialize(&mut account.data.as_slice())
                    .map(|verdict| (address, verdict))
                    .map_err(|error| anyhow!("Unable to decode audit verdict {address}: {error}"))
            })
            .collect()
    }

    /// Every slash report in `treasurer_run`, with its address
    /// (wienerlabs/leviathan-net#72). A report keeps its run where a verdict
    /// does, so it is found the same way.
    pub async fn get_treasurer_slash_reports(
        &self,
        treasurer_run: &Pubkey,
    ) -> Result<Vec<(Pubkey, psyche_solana_treasurer::state::SlashReport)>> {
        use anchor_client::anchor_lang::Discriminator;
        use psyche_solana_treasurer::state::SlashReport;
        const RUN_OFFSET: usize = 8 + 1;
        let config = RpcProgramAccountsConfig {
            filters: Some(vec![
                RpcFilterType::DataSize(SlashReport::space_with_discriminator() as u64),
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(0, SlashReport::DISCRIMINATOR)),
                RpcFilterType::Memcmp(Memcmp::new_base58_encoded(
                    RUN_OFFSET,
                    treasurer_run.as_ref(),
                )),
            ]),
            account_config: RpcAccountInfoConfig {
                encoding: Some(UiAccountEncoding::Base64),
                ..Default::default()
            },
            ..Default::default()
        };
        let accounts = self
            .rpc_with_fallback("get_treasurer_slash_reports", |coord| {
                let config = config.clone();
                async move {
                    coord
                        .rpc()
                        .get_program_accounts_with_config(&psyche_solana_treasurer::ID, config)
                        .await
                        .map_err(Into::into)
                }
            })
            .await
            .context(
                "Unable to list the run's slash reports: the RPC endpoint has to serve getProgramAccounts",
            )?;
        accounts
            .into_iter()
            .map(|(address, account)| {
                SlashReport::try_deserialize(&mut account.data.as_slice())
                    .map(|report| (address, report))
                    .map_err(|error| anyhow!("Unable to decode slash report {address}: {error}"))
            })
            .collect()
    }

    pub async fn get_token_account(
        &self,
        token_account: &Pubkey,
    ) -> Result<anchor_spl::token::spl_token::state::Account> {
        let data = self.get_data(token_account).await?;
        anchor_spl::token::spl_token::state::Account::unpack(&data)
            .map_err(|error| anyhow!("Unable to decode token account data: {error}"))
    }

    pub async fn get_token_mint(
        &self,
        mint: &Pubkey,
    ) -> Result<anchor_spl::token::spl_token::state::Mint> {
        let data = self.get_data(mint).await?;
        anchor_spl::token::spl_token::state::Mint::unpack(&data)
            .map_err(|error| anyhow!("Unable to decode mint account data: {error}"))
    }

    pub fn compute_deterministic_treasurer_index(
        run_id: &str,
        treasurer_index: Option<u64>,
    ) -> u64 {
        treasurer_index.unwrap_or_else(|| {
            let hashed = hash(run_id.as_bytes()).to_bytes();
            u64::from_le_bytes(hashed[0..8].try_into().unwrap())
        })
    }

    pub async fn resolve_treasurer_index(
        &self,
        run_id: &str,
        treasurer_index: Option<u64>,
    ) -> Result<Option<u64>> {
        let treasurer_index = Self::compute_deterministic_treasurer_index(run_id, treasurer_index);
        let run = psyche_solana_treasurer::find_run(treasurer_index);
        let run_balance = self.get_balance(&run).await?;
        Ok(if run_balance > 0 {
            Some(treasurer_index)
        } else {
            None
        })
    }

    pub fn get_payer(&self) -> Pubkey {
        self.wallet.pubkey()
    }

    pub fn sign_message(&self, message: &[u8]) -> Vec<u8> {
        use anchor_client::solana_sdk::signature::Signer;
        self.wallet.sign_message(message).as_ref().to_vec()
    }

    pub fn get_commitment_config(&self) -> CommitmentConfig {
        self.program_coordinators[0].rpc().commitment()
    }

    pub async fn get_minimum_balance_for_rent_exemption(&self, space: usize) -> Result<u64> {
        self.rpc_with_fallback(
            "get_minimum_balance_for_rent_exemption",
            |coord| async move {
                coord
                    .rpc()
                    .get_minimum_balance_for_rent_exemption(space)
                    .await
                    .map_err(Into::into)
            },
        )
        .await
    }

    pub async fn get_balance(&self, address: &Pubkey) -> Result<u64> {
        let address = *address;
        self.rpc_with_fallback("get_balance", |coord| async move {
            coord.rpc().get_balance(&address).await.map_err(Into::into)
        })
        .await
    }

    /// The cluster's clock, read from the Clock sysvar: the time the programs
    /// see, not this machine's.
    pub async fn get_clock(&self) -> Result<Clock> {
        let clock_sysvar = anchor_client::solana_sdk::sysvar::clock::ID;
        self.get_account(&clock_sysvar)
            .await?
            .ok_or_else(|| anyhow!("the clock sysvar is not readable"))?
            .deserialize_data::<Clock>()
            .map_err(|error| anyhow!("Unable to decode the clock sysvar: {error}"))
    }

    /// The account at `address` with its owner, or `None` if nothing lives there.
    pub async fn get_account(&self, address: &Pubkey) -> Result<Option<Account>> {
        let address = *address;
        self.rpc_with_fallback("get_account", |coord| async move {
            let commitment = coord.rpc().commitment();
            coord
                .rpc()
                .get_account_with_commitment(&address, commitment)
                .await
                .map(|response| response.value)
                .map_err(Into::into)
        })
        .await
    }

    pub async fn get_data(&self, address: &Pubkey) -> Result<Vec<u8>> {
        let address = *address;
        self.rpc_with_fallback("get_data", |coord| async move {
            coord
                .rpc()
                .get_account_data(&address)
                .await
                .map_err(Into::into)
        })
        .await
    }

    pub async fn get_logs(&self, tx: &Signature) -> Result<Vec<String>> {
        let tx_sig = *tx;
        let response = self
            .rpc_with_fallback("get_logs", |coord| async move {
                coord
                    .rpc()
                    .get_transaction_with_config(
                        &tx_sig,
                        RpcTransactionConfig {
                            encoding: Some(UiTransactionEncoding::Json),
                            commitment: Some(CommitmentConfig::confirmed()),
                            max_supported_transaction_version: None,
                        },
                    )
                    .await
                    .map_err(Into::into)
            })
            .await?;
        Ok(response
            .transaction
            .meta
            .context("Transaction has no meta information")?
            .log_messages
            .unwrap_or(Vec::new()))
    }

    pub async fn send_and_retry(
        &self,
        name: &str,
        instructions: &[Instruction],
        signers: &[Arc<Keypair>],
    ) -> Result<Signature> {
        info!("Sending transaction: {name}");
        let instructions: Arc<[Instruction]> = instructions.to_vec().into();
        let signers: Arc<[Arc<Keypair>]> = signers.to_vec().into();
        let signature = self
            .rpc_with_fallback(name, |coord| {
                let instructions = instructions.clone();
                let signers = signers.clone();
                async move {
                    let mut request = coord.request();
                    for instruction in instructions.iter() {
                        request = request.instruction(instruction.clone());
                    }
                    for signer in signers.iter() {
                        request = request.signer(signer.clone());
                    }
                    request.send().await.map_err(Into::into)
                }
            })
            .await?;
        info!("Transaction success: {name}, {signature}");
        Ok(signature)
    }

    pub fn spawn_scheduled_send(
        &self,
        name: &str,
        instructions: &[Instruction],
        signers: &[Arc<Keypair>],
        call_type: RpcCallType,
    ) {
        tokio::task::spawn(self.scheduled_send(name, instructions, signers, call_type));
    }

    /// Sends `instructions` with retries, recording the call and its outcome,
    /// and logging a failure. Returns the outcome, so a caller that keeps the
    /// send can report it.
    fn scheduled_send(
        &self,
        name: &str,
        instructions: &[Instruction],
        signers: &[Arc<Keypair>],
        call_type: RpcCallType,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let backend = self.clone();
        let name = name.to_string();
        let instructions = instructions.to_vec();
        let signers = signers.to_vec();
        async move {
            event!(coordinator::RpcCallSubmitted { call_type });
            match backend.send_and_retry(&name, &instructions, &signers).await {
                Ok(_) => {
                    event!(coordinator::RpcCallResult {
                        call_type,
                        result: Ok(())
                    });
                    Ok(())
                }
                Err(err) => {
                    event!(coordinator::RpcCallResult {
                        call_type,
                        result: Err(err.to_string())
                    });
                    error!(
                        "Failed to send {} transaction after all retries: {}",
                        name, err
                    );
                    Err(err.context(format!("{name} transaction failed after all retries")))
                }
            }
        }
    }

    /// Tries the operation against each RPC in order, with max retries per RPC.
    /// Only falls back to the next RPC on retryable errors (network/timeout).
    async fn rpc_with_fallback<T, F, Fut>(&self, name: &str, f: F) -> Result<T>
    where
        F: Fn(ProgramCoordinator) -> Fut,
        Fut: Future<Output = Result<T, RetryError<ClientError>>>,
    {
        let mut last_error = None;
        for (i, coordinator) in self.program_coordinators.iter().enumerate() {
            let coordinator = coordinator.clone();
            let result = retry_function_with_params(
                name,
                || {
                    let coord = coordinator.clone();
                    f(coord)
                },
                RPC_INITIAL_BACKOFF_MS,
                1.5,
                RPC_RETRY_ATTEMPTS,
                RPC_MAX_BACKOFF_MS,
            )
            .await;

            match result {
                Ok(value) => return Ok(value),
                Err(RetryError::Retryable(e)) => {
                    warn!(
                        integration_test_log_marker = %IntegrationTestLogMarker::RpcFallback,
                        failed_rpc_index = i,
                        next_rpc_index = i + 1,
                        error = %e,
                        "RPC {i} exhausted retries, trying next: {e}",
                    );
                    last_error = Some(e);
                }
                Err(RetryError::NonRetryable(e)) => {
                    return Err(anyhow!("non-retryable RPC error: {e}"));
                }
                Err(RetryError::Fatal(e)) => {
                    return Err(anyhow!("fatal RPC error: {e}"));
                }
            }
        }

        Err(anyhow!(
            "all RPCs exhausted: {}",
            last_error.map(|e| e.to_string()).unwrap_or_default()
        ))
    }
}

#[async_trait::async_trait]
impl WatcherBackend for SolanaBackendRunner {
    async fn wait_for_new_state(&mut self) -> Result<Coordinator> {
        match self.init.take() {
            Some(init) => Ok(init),
            None => self
                .updates
                .recv()
                .await
                .map_err(|err| anyhow!("Error receiving new state: {err}")),
        }
    }

    async fn send_witness(&mut self, opportunistic_data: OpportunisticData) -> Result<()> {
        self.backend
            .send_witness(self.instance, self.account, opportunistic_data);
        Ok(())
    }

    async fn send_health_check(&mut self, checks: HealthChecks) -> Result<()> {
        for (id, proof) in checks {
            self.backend
                .send_health_check(self.instance, self.account, id, proof);
        }
        Ok(())
    }

    async fn send_checkpoint(&mut self, checkpoint: model::Checkpoint) -> Result<()> {
        self.reap_checkpoints();
        self.checkpoints.spawn(self.backend.send_checkpoint(
            self.instance,
            self.account,
            checkpoint,
        ));
        Ok(())
    }

    /// Waits for every checkpoint send still under way, and returns the first
    /// that failed, whenever it failed. A checkpoint that could not be
    /// recorded is the run's model missing from chain, so it fails the
    /// client's shutdown (wienerlabs/leviathan-net#85).
    async fn flush_checkpoints(&mut self) -> Result<()> {
        while let Some(joined) = self.checkpoints.join_next().await {
            self.note_checkpoint(joined);
        }
        match self.failed_checkpoint.take() {
            Some(failed) => Err(failed),
            None => Ok(()),
        }
    }
}

impl SolanaBackendRunner {
    /// Reads the outcome of every checkpoint send that has finished, so the
    /// set holds only those under way.
    fn reap_checkpoints(&mut self) {
        while let Some(joined) = self.checkpoints.try_join_next() {
            self.note_checkpoint(joined);
        }
    }

    /// Keeps the first failure among the checkpoint sends read so far.
    fn note_checkpoint(&mut self, joined: std::result::Result<Result<()>, JoinError>) {
        let outcome = joined
            .map_err(|error| anyhow!("the checkpoint send did not finish: {error}"))
            .and_then(|sent| sent);
        if let Err(error) = outcome {
            self.failed_checkpoint.get_or_insert(error);
        }
    }

    pub fn updates(&self) -> broadcast::Receiver<Coordinator> {
        self.updates.resubscribe()
    }
}
