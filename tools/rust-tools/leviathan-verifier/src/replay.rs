use std::collections::HashMap;
use std::sync::Mutex;

use psyche_core::BatchId;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::Trainer;
use psyche_verifier::ReplayEngine;
use psyche_verifier::ReplayError;
use tokio_util::sync::CancellationToken;

use crate::decompress_results;

#[derive(Debug, Clone)]
pub struct ReplayAssignment {
    pub target_index: u64,
    pub step: u32,
    pub batch_id: BatchId,
    pub data: Vec<BatchDataCPU>,
}

pub struct TrainerReplayEngine {
    trainer: Mutex<Option<Trainer>>,
    assignments: HashMap<u64, ReplayAssignment>,
    device: tch::Device,
}

impl TrainerReplayEngine {
    pub fn new(
        trainer: Trainer,
        assignments: Vec<ReplayAssignment>,
        device: tch::Device,
    ) -> Self {
        Self {
            trainer: Mutex::new(Some(trainer)),
            assignments: assignments
                .into_iter()
                .map(|a| (a.target_index, a))
                .collect(),
            device,
        }
    }

    pub fn into_trainer(self) -> Option<Trainer> {
        self.trainer.into_inner().unwrap_or(None)
    }

    fn recompute(&self, assignment: &ReplayAssignment) -> Result<Vec<f32>, ReplayError> {
        let mut slot = self
            .trainer
            .lock()
            .map_err(|_| ReplayError::Backend("replay trainer lock poisoned".to_string()))?;
        let trainer = slot
            .take()
            .ok_or_else(|| ReplayError::Backend("replay trainer unavailable".to_string()))?;

        let output = trainer.train(
            assignment.step,
            Batch {
                id: assignment.batch_id,
                data: BatchData::CPU(assignment.data.clone()),
            },
            None,
            true,
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        );

        let output = match output {
            Ok(output) => output,
            Err(error) => {
                return Err(ReplayError::Backend(format!("replay training failed: {error}")));
            }
        };

        let results = output.distro_results.clone();
        *slot = Some(output.trainer);
        drop(slot);

        let results = results.ok_or_else(|| {
            ReplayError::Backend(
                "replay produced no distro results, the run is not using DisTrO".to_string(),
            )
        })?;

        decompress_results(&results, self.device)
            .map_err(|e| ReplayError::Backend(format!("cannot decompress replay: {e}")))
    }
}

impl ReplayEngine for TrainerReplayEngine {
    fn replay(&self, target_index: u64) -> Result<Vec<f32>, ReplayError> {
        let assignment = self
            .assignments
            .get(&target_index)
            .ok_or(ReplayError::NotAssigned(target_index))?
            .clone();
        self.recompute(&assignment)
    }
}
