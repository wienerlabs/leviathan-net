//! Keeps a memnet endpoint's blockhash alive across a long setup loop.
//!
//! The program-test proxy hands out a blockhash it cached when the endpoint was
//! built, and refreshes it only when the clock is forwarded. Meanwhile
//! `ProgramTestContext` spawns a background task that registers a new blockhash
//! on a *wall-clock* interval, so the cached one falls out of the bank's queue
//! after roughly a second of real time no matter what the test is doing. Past
//! that, every transaction fails with "Blockhash not found".
//!
//! That makes the failure a function of how fast the machine is rather than of
//! anything the test did, which is why it showed up as a flake:
//! `memnet_coordinator_rewards` signs 720 transactions to seat 240 clients, and
//! at roughly a millisecond each that lands near enough to the deadline that a
//! cold cache or a busy machine pushes it over.
//!
//! Forwarding a single slot refreshes the blockhash. It moves `clock.slot` by
//! one and leaves `unix_timestamp` alone, because the proxy converts slots to
//! seconds with `slot_delta / 2` in integer arithmetic. Tests that assert on
//! elapsed time are therefore unaffected. Tests that assert on a specific
//! committee draw would see a different seed, since the coordinator derives its
//! seed from the slot, so keep the refreshes inside setup rather than sprinkling
//! them through a round.

use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use solana_toolbox_endpoint::ToolboxEndpoint;

/// How long to let the cached blockhash age before refreshing it.
///
/// Measured expiry on this harness is about 1.05 seconds, so this leaves a
/// margin of roughly two and a half times rather than the 1.5 the failing loop
/// had. The cost of being wrong in this direction is one extra slot.
const REFRESH_AFTER: Duration = Duration::from_millis(400);

pub struct BlockhashKeeper {
    refreshed_at: Instant,
}

impl Default for BlockhashKeeper {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockhashKeeper {
    pub fn new() -> Self {
        Self {
            refreshed_at: Instant::now(),
        }
    }

    /// Call once per iteration of a long loop. Refreshes only when the cached
    /// blockhash has aged enough to be worth replacing, so a short loop pays
    /// nothing and does not have its clock moved at all.
    pub async fn tick(&mut self, endpoint: &mut ToolboxEndpoint) -> Result<()> {
        if self.refreshed_at.elapsed() < REFRESH_AFTER {
            return Ok(());
        }
        endpoint.forward_clock_slot(1).await?;
        self.refreshed_at = Instant::now();
        Ok(())
    }
}
