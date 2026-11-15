//! Chain state the suites read, kept off their futures.
//!
//! A `CoordinatorInstanceState` is the coordinator's zero-copy account, a few
//! hundred kilobytes. A test future holds every local that lives across an
//! await point, so a suite holding a handful of these overflowed the 2 MB
//! test-thread stack (wienerlabs/leviathan-net#35). Boxed, each one costs the
//! future a pointer.

use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use solana_sdk::pubkey::Pubkey;
use solana_toolbox_endpoint::ToolboxEndpoint;

/// The coordinator's state as it stands, on the heap.
pub async fn coordinator_state(
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
) -> Box<CoordinatorInstanceState> {
    Box::new(
        get_coordinator_account_state(endpoint, coordinator_account)
            .await
            .unwrap()
            .expect("the coordinator account exists"),
    )
}
