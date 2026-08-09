use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Context;
use anyhow::Result;
use psyche_core::Barrier;
use psyche_core::BatchId;
use psyche_core::CancellableBarrier;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_modeling::auto_model_for_causal_lm_from_pretrained;
use psyche_modeling::CausalLM;
use psyche_modeling::LocalTrainer;
use psyche_modeling::ParallelModels;
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

pub fn parse_assignment_key(key: &str) -> Option<(u32, BatchId)> {
    let (step_part, batch_part) = key.split_once("-batchB")?;
    let step: u32 = step_part.strip_prefix("step")?.parse().ok()?;
    let bounds = batch_part.trim_start_matches('[').trim_end_matches(']');
    let (start, end) = bounds.split_once(',')?;
    let start: u64 = start.trim().parse().ok()?;
    let end: u64 = end.trim().parse().ok()?;
    Some((step, BatchId(psyche_core::ClosedInterval::new(start, end))))
}

#[derive(Debug, Clone)]
pub struct ReplayTrainerConfig {
    pub repo_files: Vec<std::path::PathBuf>,
    pub sequence_length: usize,
    pub micro_batch_size: usize,
    pub learning_rate: f64,
    pub compression_decay: f32,
    pub compression_topk: u16,
    pub compression_chunk: u16,
    pub clip_grad_norm: Option<f32>,
    pub quantize_1bit: bool,
    pub device: tch::Device,
    /// The dtype the model is loaded and trained in. The real network trains in
    /// `BFloat16`; the verifier historically replayed in `Float` (fp32). The
    /// gap between the two is the dominant source of honest drift the tolerance
    /// band has to cover, so the calibration harness sweeps this.
    pub kind: tch::Kind,
}

pub fn build_replay_trainer(config: &ReplayTrainerConfig) -> Result<Trainer> {
    let model: Box<dyn CausalLM> = auto_model_for_causal_lm_from_pretrained(
        config.repo_files.clone(),
        Some(config.kind),
        None,
        Some(config.device),
        None,
        Some(config.sequence_length),
    )
    .context("cannot load the replay model")?;
    model.prepare_for_training();
    Ok(LocalTrainer::new(
        ParallelModels {
            models: vec![model],
            barrier: Arc::new(CancellableBarrier::new(1)) as Arc<dyn Barrier>,
            data_parallel: None,
        },
        LearningRateSchedule::Constant(ConstantLR::new(config.learning_rate, 0, 0.0)),
        OptimizerDefinition::Distro {
            clip_grad_norm: config.clip_grad_norm,
            compression_decay: config.compression_decay,
            compression_topk: config.compression_topk,
            compression_chunk: config.compression_chunk,
            quantize_1bit: config.quantize_1bit,
            weight_decay: None,
        },
        config.micro_batch_size,
        None,
        false,
    )
    .into())
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
