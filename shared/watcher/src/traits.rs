use anyhow::Result;
use psyche_coordinator::{Coordinator, HealthChecks, Witness, WitnessMetadata, model};
use serde::{Deserialize, Serialize};

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpportunisticData {
    WitnessStep(Witness, WitnessMetadata),
    WarmupStep(Witness),
}

impl OpportunisticData {
    pub fn kind(&self) -> &'static str {
        match self {
            OpportunisticData::WitnessStep(..) => "witness",
            OpportunisticData::WarmupStep(..) => "warmup",
        }
    }
}

#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    /// # Cancel safety
    ///
    /// This method must be cancel safe.
    async fn wait_for_new_state(&mut self) -> Result<Coordinator>;
    async fn send_witness(&mut self, opportunistic_data: OpportunisticData) -> Result<()>;
    async fn send_health_check(&mut self, health_check: HealthChecks) -> Result<()>;
    async fn send_checkpoint(&mut self, checkpoint: model::Checkpoint) -> Result<()>;
    /// Waits until every checkpoint `send_checkpoint` was given has been
    /// recorded, or has failed for good, and returns the first failure.
    ///
    /// A backend may send a checkpoint in the background, so the call that
    /// handed it over says nothing about whether it landed. A client that is
    /// about to exit waits here, or the send dies with the process and the
    /// run's last checkpoint is never recorded (wienerlabs/leviathan-net#85).
    async fn flush_checkpoints(&mut self) -> Result<()>;
}
