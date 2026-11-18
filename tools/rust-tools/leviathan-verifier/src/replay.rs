use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;
use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HttpLLMTrainingDataLocation;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_core::Barrier;
use psyche_core::BatchId;
use psyche_core::CancellableBarrier;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_data_provider::download_model_repo_sync;
use psyche_modeling::auto_model_for_causal_lm_from_pretrained;
use psyche_modeling::CausalLM;
use psyche_modeling::LocalTrainer;
use psyche_modeling::ParallelModels;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::Trainer;
use psyche_verifier::ReplayError;
use tokio_util::sync::CancellationToken;

use crate::decompress_results;

/// A piece of work a verifier can replay: the batch trained at a step of an
/// epoch, and what training it takes.
///
/// It names the work, not whoever did it. A replay of a batch is the same
/// whichever node trained it, and a node's place in the roster moves under a
/// verifier whenever someone ahead of it leaves the epoch, so an assignment
/// keyed by roster position could be judged against another node's batch and
/// convict an honest one.
#[derive(Debug, Clone)]
pub struct ReplayAssignment {
    /// When the epoch the work was trained in started (the coordinator's
    /// `epoch_state.start_timestamp`).
    pub epoch_start: u64,
    pub step: u32,
    /// The run's cold-start warmup bounds at `step`, which the client trains
    /// with (`Coordinator::get_cold_start_warmup_bounds`).
    pub warmup_lr_between: Option<(u32, u32)>,
    pub batch_id: BatchId,
    pub data: Vec<BatchDataCPU>,
}

impl ReplayAssignment {
    /// The work this assignment replays, as a payload dump names it.
    pub fn work(&self) -> DumpKey {
        DumpKey {
            epoch_start: self.epoch_start,
            step: self.step,
            batch_id: self.batch_id,
        }
    }
}

/// The work a payload dump is for: the round, named by its epoch and step,
/// and the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DumpKey {
    /// When the epoch the payload was trained in started (the coordinator's
    /// `epoch_state.start_timestamp`). A run restarted from an earlier step
    /// trains the step again in a later epoch (wienerlabs/leviathan-net#67).
    pub epoch_start: u64,
    pub step: u32,
    pub batch_id: BatchId,
}

/// Written as a payload dump's contribution key names the work, so
/// `parse_assignment_key` reads it back.
impl std::fmt::Display for DumpKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "epoch{}-step{}-batch{}",
            self.epoch_start, self.step, self.batch_id
        )
    }
}

/// The epoch whose dumps of `step` a replay reproduces, out of the dumps
/// `keys` name: `wanted` if one is named, else the only one there is. `None`
/// when no dump is of `step`, or none of the epoch named.
///
/// A run restarted from an earlier step trains that step again in a later
/// epoch, from other weights. Dumps of one step from two epochs cannot both be
/// judged against one replay, and offline there is no coordinator to say
/// which epoch is the run's first, so the caller has to name it
/// (wienerlabs/leviathan-net#67).
pub fn replay_epoch<'a>(
    keys: impl IntoIterator<Item = &'a DumpKey>,
    step: u32,
    wanted: Option<u64>,
) -> Result<Option<u64>> {
    let epochs: std::collections::BTreeSet<u64> = keys
        .into_iter()
        .filter(|key| key.step == step)
        .map(|key| key.epoch_start)
        .collect();
    if let Some(epoch) = wanted {
        return Ok(epochs.contains(&epoch).then_some(epoch));
    }
    match epochs.len() {
        0 | 1 => Ok(epochs.into_iter().next()),
        _ => Err(anyhow!(
            "the dumps of step {step} come from {} epochs, started at {:?}: name the one to replay with --epoch-start",
            epochs.len(),
            epochs
        )),
    }
}

/// The work a payload dump is for, from its contribution key
/// (`epoch1700000000-step7-batchB[12, 19]`, or `...-batchB15` for a batch of
/// one sample).
pub fn parse_assignment_key(key: &str) -> Option<DumpKey> {
    let (epoch_part, rest) = key.split_once("-step")?;
    let epoch_start = epoch_part.strip_prefix("epoch")?.parse().ok()?;
    let (step_part, batch_part) = rest.split_once("-batch")?;
    let step = step_part.parse().ok()?;
    let batch_id = batch_part.strip_prefix('B')?.parse().ok()?;
    Some(DumpKey {
        epoch_start,
        step,
        batch_id,
    })
}

/// What a replay trains with. It has to be what the run trains with: a replay
/// at another learning rate, with another optimizer or sequence length,
/// reproduces some other node's work (wienerlabs/leviathan-net#47).
#[derive(Debug, Clone)]
pub struct ReplayTrainerConfig {
    pub repo_files: Vec<PathBuf>,
    pub sequence_length: usize,
    pub micro_batch_size: usize,
    pub lr_schedule: LearningRateSchedule,
    pub optimizer: OptimizerDefinition,
    pub device: tch::Device,
    /// The dtype the model is loaded and trained in. The real network trains in
    /// `BFloat16`; the verifier historically replayed in `Float` (fp32). The
    /// gap between the two is the dominant source of honest drift the tolerance
    /// band has to cover, so the calibration harness sweeps this.
    pub kind: tch::Kind,
}

impl ReplayTrainerConfig {
    /// The replay of a run that trains `llm`: its learning-rate schedule, its
    /// optimizer and its sequence length, on the checkpoint files `repo_files`.
    pub fn for_run(
        llm: &LLM,
        repo_files: Vec<PathBuf>,
        device: tch::Device,
        kind: tch::Kind,
    ) -> Self {
        Self {
            repo_files,
            sequence_length: llm.max_seq_len as usize,
            micro_batch_size: 1,
            lr_schedule: llm.lr_schedule,
            optimizer: llm.optimizer,
            device,
            kind,
        }
    }

    /// Whether the run transmits its results as 1-bit signs, which the replay
    /// has to do too before it is compared with what a node sent.
    pub fn quantize_1bit(&self) -> bool {
        matches!(
            self.optimizer,
            OptimizerDefinition::Distro {
                quantize_1bit: true,
                ..
            }
        )
    }
}

/// The published checkpoint a cold start trains from: the Hub repository and
/// revision the run names, directly or as the repository its peers publish to.
/// A cold start turns a P2P checkpoint back into that Hub one.
pub fn published_checkpoint(llm: &LLM) -> Result<HubRepo> {
    match llm.checkpoint {
        Checkpoint::Hub(hub) | Checkpoint::P2P(hub) => Ok(hub),
        other => Err(anyhow!(
            "a replay starts from a published Hub checkpoint, the run names {other}"
        )),
    }
}

/// The files of the published checkpoint `hub`, at its revision.
pub fn checkpoint_repo_files(hub: &HubRepo) -> Result<Vec<PathBuf>> {
    download_model_repo_sync(
        &hub.repo_id.to_string(),
        hub.revision.map(|revision| revision.to_string()),
        None,
        None,
        true,
    )
    .with_context(|| format!("cannot fetch the run's checkpoint {}", hub.repo_id))
}

/// How the run's training data is laid out: the token size and shuffle of its
/// Http location. A replay reads the run's shards back from a local copy in
/// that order, and can reproduce no other kind of data location.
pub fn replay_data_layout(llm: &LLM) -> Result<(TokenSize, Shuffle)> {
    match llm.data_location {
        LLMTrainingDataLocation::Http(HttpLLMTrainingDataLocation {
            token_size_in_bytes,
            shuffle,
            ..
        }) => Ok((token_size_in_bytes, shuffle)),
        other => Err(anyhow!(
            "a replay reads local shards in Http order and cannot reproduce the run's {other:?} data location"
        )),
    }
}

/// The cold-start warmup bounds of a cold-start epoch of `llm` starting at
/// `start_step`, as `Coordinator::get_cold_start_warmup_bounds` gives them to
/// the client. The offline tools have no coordinator to ask.
pub fn cold_start_warmup_bounds(llm: &LLM, start_step: u32) -> Option<(u32, u32)> {
    (llm.cold_start_warmup_steps != 0)
        .then(|| (start_step, start_step + llm.cold_start_warmup_steps))
}

/// A run's model, as a run configuration file states it
/// (`config/leviathan/*.toml`): what the offline tools replay against.
pub fn read_run_model(path: &Path) -> Result<LLM> {
    #[derive(serde::Deserialize)]
    struct RunFile {
        model: Model,
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read run config {}", path.display()))?;
    let run: RunFile = toml::from_str(&text)
        .with_context(|| format!("cannot parse run config {}", path.display()))?;
    let Model::LLM(llm) = run.model;
    Ok(llm)
}

/// The global batch a run trains, in sequences, as its configuration states it
/// (`config/leviathan/*.toml`). A run whose batch ramps up is refused: the
/// offline tools train a fixed one.
pub fn read_run_global_batch(path: &Path) -> Result<usize> {
    #[derive(serde::Deserialize)]
    struct Batches {
        global_batch_size_start: u16,
        global_batch_size_end: u16,
    }
    #[derive(serde::Deserialize)]
    struct RunFile {
        config: Batches,
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read run config {}", path.display()))?;
    let run: RunFile = toml::from_str(&text)
        .with_context(|| format!("cannot parse run config {}", path.display()))?;
    let Batches {
        global_batch_size_start: start,
        global_batch_size_end: end,
    } = run.config;
    if start != end || end == 0 {
        return Err(anyhow!(
            "the run's global batch goes from {start} to {end}; the offline tools train a fixed, non-empty one"
        ));
    }
    Ok(end as usize)
}

/// The share of a round's nodes a run seats as verifiers, in percent, as its
/// configuration states it.
pub fn read_run_verification_percent(path: &Path) -> Result<u8> {
    #[derive(serde::Deserialize)]
    struct Committee {
        verification_percent: u8,
    }
    #[derive(serde::Deserialize)]
    struct RunFile {
        config: Committee,
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read run config {}", path.display()))?;
    let run: RunFile = toml::from_str(&text)
        .with_context(|| format!("cannot parse run config {}", path.display()))?;
    Ok(run.config.verification_percent)
}

/// A trainer to replay with, built from the checkpoint as the client builds
/// its own.
///
/// A replay trainer is reused across targets and passes, and only its error
/// feedback is zeroed between them. DisTrO's weight decay scales the weights
/// in place on every step, so with it each replay after the first would start
/// from weights the earlier ones had already decayed. A run with weight decay
/// is refused rather than replayed on drifting weights.
pub fn build_replay_trainer(config: &ReplayTrainerConfig) -> Result<Trainer> {
    if let OptimizerDefinition::Distro {
        weight_decay: Some(weight_decay),
        ..
    } = config.optimizer
    {
        if weight_decay != 0.0 {
            return Err(anyhow!(
                "the run decays its weights ({weight_decay}), which a replay trainer reused across targets cannot reproduce"
            ));
        }
    }
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
        config.lr_schedule,
        config.optimizer,
        config.micro_batch_size,
        None,
        false,
    )
    .into())
}

/// Recomputes the honest result of a piece of work.
///
/// Asked for by the work a payload names (its epoch, step and batch), never by
/// who sent it or where that node sits in the roster: the payload being judged
/// names the work it claims, so the reference it is judged against is a replay
/// of exactly that work, whatever has happened to the roster since.
pub trait ReplayWork {
    fn replay_work(&self, work: &DumpKey) -> Result<Vec<f32>, ReplayError>;
}

pub struct TrainerReplayEngine {
    trainer: Mutex<Option<Trainer>>,
    assignments: HashMap<DumpKey, ReplayAssignment>,
    device: tch::Device,
    quantize_1bit: bool,
}

impl TrainerReplayEngine {
    /// An engine replaying `assignments` with `trainer`. `quantize_1bit` is
    /// whether the run sends its results as 1-bit signs
    /// (`ReplayTrainerConfig::quantize_1bit`), so the replay is compared as
    /// what a node sends rather than what it computed.
    pub fn new(
        trainer: Trainer,
        assignments: Vec<ReplayAssignment>,
        device: tch::Device,
        quantize_1bit: bool,
    ) -> Self {
        Self {
            trainer: Mutex::new(Some(trainer)),
            assignments: assignments.into_iter().map(|a| (a.work(), a)).collect(),
            device,
            quantize_1bit,
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

        // Trained as the client trains a step with zeroed error feedback: the
        // run's warmup bounds, and no results of its own from before.
        let output = trainer.train(
            assignment.step,
            Batch {
                id: assignment.batch_id,
                data: BatchData::CPU(assignment.data.clone()),
            },
            assignment.warmup_lr_between,
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
        let results = match self.quantize_1bit {
            true => Trainer::quantize_results(&results),
            false => results,
        };

        decompress_results(&results, self.device)
            .map_err(|e| ReplayError::Backend(format!("cannot decompress replay: {e}")))
    }
}

impl ReplayWork for TrainerReplayEngine {
    fn replay_work(&self, work: &DumpKey) -> Result<Vec<f32>, ReplayError> {
        let assignment = self
            .assignments
            .get(work)
            .ok_or_else(|| ReplayError::WorkNotAssigned(work.to_string()))?
            .clone();
        self.recompute(&assignment)
    }
}
