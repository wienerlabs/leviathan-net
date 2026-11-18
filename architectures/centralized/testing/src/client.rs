use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Error, Result};
use psyche_centralized_client::app::App as ClientApp;
use psyche_centralized_client::app::build_app as build_client_app;
use psyche_client::NC;
use psyche_client::RunInitConfig;
use psyche_network::allowlist;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tracing::debug;

use crate::test_utils::dummy_client_app_params;

/// The training delay of a client spawned with `ClientHandle::default`.
const DEFAULT_TRAINING_DELAY_SECS: u64 = 5;

/// The stack of a client runtime's workers, as the client binary builds its
/// runtime. The client copies its ~94 KiB `Coordinator` by value as it
/// polls for new state, which takes more than the 2 MiB a test runtime's
/// workers get by default.
const CLIENT_STACK_SIZE: usize = 10 * 1024 * 1024;

fn client_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .thread_stack_size(CLIENT_STACK_SIZE)
        .build()
        .expect("failed to build a client runtime")
}

struct Client {
    inner: ClientApp,
}

impl Client {
    /// A client that trains for `training_delay_secs` a step, and keeps its
    /// checkpoints in `checkpoint_dir` if one is given.
    pub async fn new(
        server_port: u16,
        run_id: &str,
        training_delay_secs: u64,
        checkpoint_dir: Option<&Path>,
    ) -> (Self, allowlist::AllowDynamic, NC, RunInitConfig) {
        let client_app_params =
            dummy_client_app_params(server_port, run_id, training_delay_secs, checkpoint_dir);
        let (client_app, allowlist, p2p, state_options) = build_client_app(
            client_app_params.cancel,
            client_app_params.server_addr,
            None,
            client_app_params.train_args,
        )
        .await
        .unwrap();

        (Self { inner: client_app }, allowlist, p2p, state_options)
    }

    pub async fn run(
        &mut self,
        allowlist: allowlist::AllowDynamic,
        p2p: NC,
        state_options: RunInitConfig,
    ) -> Result<()> {
        debug!("spawned new client: {:?}", p2p.endpoint_addr().await);
        // The app returns once its run is over (wienerlabs/leviathan-net#82),
        // or with the error that stopped it.
        self.inner.run(allowlist, p2p, state_options).await
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct ClientHandle {
    pub client_handle: JoinHandle<Result<(), Error>>,
    /// The runtime the client is built and runs on. It is shut down when the
    /// handle is dropped, which stops the client as the end of the test's own
    /// runtime used to.
    runtime: Option<Runtime>,
}

impl ClientHandle {
    pub async fn default(server_port: u16, run_id: &str) -> Self {
        Self::spawn(
            server_port,
            run_id,
            DEFAULT_TRAINING_DELAY_SECS,
            None,
            Duration::from_millis(50),
        )
        .await
    }

    pub async fn new_with_training_delay(
        server_port: u16,
        run_id: &str,
        training_delay_secs: u64,
    ) -> Self {
        debug!("spawning new client...");
        let handle = Self::spawn(
            server_port,
            run_id,
            training_delay_secs,
            None,
            Duration::from_millis(100),
        )
        .await;
        debug!("new client spawned!");
        handle
    }

    /// A client that keeps its checkpoints in `checkpoint_dir`.
    pub async fn new_with_checkpoint_dir(
        server_port: u16,
        run_id: &str,
        checkpoint_dir: &Path,
    ) -> Self {
        Self::spawn(
            server_port,
            run_id,
            DEFAULT_TRAINING_DELAY_SECS,
            Some(checkpoint_dir.to_path_buf()),
            Duration::from_millis(50),
        )
        .await
    }

    /// Builds a client and runs it on a runtime of its own, set up as the
    /// client binary sets up its runtime, then gives it `settle` to start.
    async fn spawn(
        server_port: u16,
        run_id: &str,
        training_delay_secs: u64,
        checkpoint_dir: Option<PathBuf>,
        settle: Duration,
    ) -> Self {
        let runtime = client_runtime();
        let run_id = run_id.to_string();
        let (mut client, allowlist, p2p, state_options) = runtime
            .spawn(async move {
                Client::new(
                    server_port,
                    &run_id,
                    training_delay_secs,
                    checkpoint_dir.as_deref(),
                )
                .await
            })
            .await
            .expect("building the client panicked");
        let client_handle =
            runtime.spawn(async move { client.run(allowlist, p2p, state_options).await });
        tokio::time::sleep(settle).await;
        Self {
            client_handle,
            runtime: Some(runtime),
        }
    }
}

impl Drop for ClientHandle {
    fn drop(&mut self) {
        // Tests let go of their clients in async code, where a runtime cannot
        // be dropped and block on its shutdown.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}
