use crate::{
    AllReduce, CausalLM, Communicator, CommunicatorId, CudaSynchronize, Distro, DistroResult,
    EosToks, Fp32GradientAccumulator, Optimizer, ReduceType, StableVariableIterator,
    keep_round_state, round_state_digest, round_state_len, unsharded_cpu_variables,
    write_round_state,
};
use anyhow::{Error, Result};
use psyche_core::{Barrier, BatchId, LearningRateSchedule, OptimizerDefinition};
use std::{
    collections::HashMap,
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use tch::{Device, Kind, Tensor};
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, trace, warn};

#[cfg(feature = "parallelism")]
use tch::CNCCL;

pub struct ParallelModels {
    pub models: Vec<Box<dyn CausalLM>>,
    pub barrier: Arc<dyn Barrier>,
    pub data_parallel: Option<Vec<DataParallel>>,
}

pub type DistroResults = Vec<DistroResult>;

#[derive(Debug, Clone)]
pub struct BatchDataCPU {
    pub input_ids: Vec<i32>,
    pub labels: Option<Vec<i32>>,
    pub position_ids: Option<Vec<i32>>,
    pub sequence_lengths: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct BatchDataGPU {
    pub input_ids: Tensor,
    pub labels: Option<Tensor>,
    pub position_ids: Option<Tensor>,
    pub sequence_lengths: Option<Vec<Vec<i32>>>,
}

#[derive(Debug)]
pub enum BatchData {
    CPU(Vec<BatchDataCPU>),
    GPU(BatchDataGPU),
}

impl BatchData {
    pub fn size(&self) -> usize {
        match self {
            BatchData::CPU(cpu) => cpu.len(),
            BatchData::GPU(gpu) => gpu.input_ids.size()[0] as usize,
        }
    }

    pub fn gpu(self, device: Device) -> BatchDataGPU {
        let gpu = match self {
            BatchData::CPU(cpu) => {
                let mut all_input_ids = Vec::with_capacity(cpu.len());
                let mut all_labels = Vec::with_capacity(cpu.len());
                let mut all_position_ids = Vec::with_capacity(cpu.len());
                let mut all_sequence_lengths = Vec::with_capacity(cpu.len());

                for data in cpu.into_iter() {
                    all_input_ids.push(data.input_ids);
                    all_labels.push(data.labels);
                    all_position_ids.push(data.position_ids);
                    all_sequence_lengths.push(data.sequence_lengths);
                }

                let input_ids = Tensor::from_slice2(&all_input_ids)
                    .to_kind(Kind::Int64)
                    .to(device);

                let labels = all_labels
                    .into_iter()
                    .collect::<Option<Vec<Vec<i32>>>>()
                    .map(|v| Tensor::from_slice2(&v).to_kind(Kind::Int64).to(device));

                let position_ids = all_position_ids
                    .into_iter()
                    .collect::<Option<Vec<Vec<i32>>>>()
                    .map(|v| Tensor::from_slice2(&v).to_kind(Kind::Int64).to(device));

                let sequence_lengths = all_sequence_lengths
                    .into_iter()
                    .collect::<Option<Vec<Vec<i32>>>>();

                BatchDataGPU {
                    input_ids,
                    labels,
                    position_ids,
                    sequence_lengths,
                }
            }
            BatchData::GPU(gpu) => gpu,
        };
        BatchDataGPU {
            input_ids: gpu.input_ids.to(device),
            labels: gpu.labels.map(|x| x.to(device)),
            position_ids: gpu.position_ids.map(|x| x.to(device)),
            sequence_lengths: gpu.sequence_lengths,
        }
    }
}

impl Clone for BatchData {
    fn clone(&self) -> Self {
        match self {
            Self::CPU(cpu) => Self::CPU(cpu.clone()),
            Self::GPU(gpu) => Self::GPU(BatchDataGPU {
                input_ids: gpu.input_ids.shallow_clone(),
                labels: gpu.labels.as_ref().map(|x| x.shallow_clone()),
                position_ids: gpu.position_ids.as_ref().map(|x| x.shallow_clone()),
                sequence_lengths: gpu.sequence_lengths.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Batch {
    pub id: BatchId,
    pub data: BatchData,
}

impl Batch {
    pub fn gpu(self, device: Device) -> Self {
        Self {
            id: self.id,
            data: BatchData::GPU(self.data.gpu(device)),
        }
    }

    pub fn pad(&mut self, world_size: usize) {
        match &mut self.data {
            BatchData::CPU(cpu_data) => {
                let current_batch_size = cpu_data.len();
                let remainder = current_batch_size % world_size;

                if remainder != 0 {
                    let padding_needed = world_size - remainder;
                    trace!(
                        "Batch size {} not divisible by world_size {}. Adding {} padding samples",
                        current_batch_size, world_size, padding_needed
                    );

                    // Get sequence length from the first sample
                    if !cpu_data.is_empty() {
                        let seq_len = cpu_data[0].input_ids.len();

                        // Create padding samples with labels set to -100
                        for _ in 0..padding_needed {
                            let padding_sample = BatchDataCPU {
                                // Use 0 as padding token ID
                                input_ids: vec![0; seq_len],
                                // Set labels to -100 so they're ignored in loss computation
                                labels: cpu_data[0].labels.as_ref().map(|_| vec![-100; seq_len]),
                                position_ids: cpu_data[0]
                                    .position_ids
                                    .as_ref()
                                    .map(|_| (0..seq_len as i32).collect()),
                                sequence_lengths: cpu_data[0]
                                    .sequence_lengths
                                    .as_ref()
                                    .map(|_| vec![seq_len as i32; 1]),
                            };
                            cpu_data.push(padding_sample);
                        }
                    }
                }
            }
            BatchData::GPU(gpu_data) => {
                let current_batch_size = gpu_data.input_ids.size()[0] as usize;
                let remainder = current_batch_size % world_size;

                if remainder != 0 {
                    let padding_needed = world_size - remainder;
                    trace!(
                        "Batch size {} not divisible by world_size {}. Adding {} padding samples",
                        current_batch_size, world_size, padding_needed
                    );

                    if current_batch_size == 0 {
                        return;
                    }

                    let seq_len = gpu_data.input_ids.size()[1];
                    let device = gpu_data.input_ids.device();

                    let padding_input_ids =
                        Tensor::zeros([padding_needed as i64, seq_len], (Kind::Int64, device));
                    gpu_data.input_ids = Tensor::cat(&[&gpu_data.input_ids, &padding_input_ids], 0);

                    if let Some(labels) = gpu_data.labels.take() {
                        let padding_labels = Tensor::full(
                            [padding_needed as i64, seq_len],
                            -100i64,
                            (Kind::Int64, device),
                        );
                        gpu_data.labels = Some(Tensor::cat(&[&labels, &padding_labels], 0));
                    }

                    if gpu_data.position_ids.is_some() {
                        let pos_row = Tensor::arange(seq_len, (Kind::Int64, device));
                        let padding_pos = pos_row.unsqueeze(0).repeat([padding_needed as i64, 1]);
                        if let Some(pos) = gpu_data.position_ids.take() {
                            gpu_data.position_ids = Some(Tensor::cat(&[&pos, &padding_pos], 0));
                        }
                    }

                    if let Some(seq_lens) = &mut gpu_data.sequence_lengths {
                        let pad_seq = vec![seq_len as i32; 1];
                        for _ in 0..padding_needed {
                            seq_lens.push(pad_seq.clone());
                        }
                    }
                }
            }
        }
    }
}

pub struct TrainOutput {
    pub batch_id: BatchId,
    pub trainer: Trainer,
    pub loss: f32,
    pub step: u32,
    pub nonce: u32,
    pub distro_results: Option<DistroResults>,
    pub cancelled: bool,
    /// The digest of the state the round was trained from, which the node
    /// commits to in the payload it signs (wienerlabs/leviathan-net#89). None
    /// from a trainer that commits to none.
    pub state_digest: Option<[u8; 32]>,
}

#[derive(Clone, Debug)]
pub struct DataParallel {
    pub id: CommunicatorId,
    pub barrier: Arc<dyn Barrier>,
    pub rank: usize,
    pub world_size: usize,
}

enum ParallelAssignment {
    Train {
        batch: Batch,
        step: u32,
        warmup_lr_between: Option<(u32, u32)>,
        zero_optim: bool,
        #[allow(unused)]
        rollback: Vec<(u32, Vec<DistroResults>)>,
        cancel_training: CancellationToken,
        prev_self_distro_results: Option<Vec<DistroResults>>,
        state_commitment: StateCommitment,
    },
    Optimize {
        distro_results: Option<Vec<DistroResults>>,
        step: u32,
        warmup_lr_between: Option<(u32, u32)>,
    },
    Forward {
        data: Tensor,
        labels: Option<Tensor>,
        position_ids: Option<Tensor>,
        sequence_lengths: Option<Vec<Vec<i32>>>,
        num_logits_to_keep: Option<i64>,
        loss_scale: Option<f64>,
    },
    Extract,
    TruncateBf16,
    ErrorFeedback,
    Load {
        weights: HashMap<String, Tensor>,
        error_feedback: Option<Vec<Tensor>>,
    },
    SetErrorFeedbackKind {
        kind: Kind,
    },
    RoundStateLen,
}

#[derive(Debug)]
enum ParallelResult {
    Train {
        loss: f32,
        nonce: u32,
        cancelled: bool,
        distro_results: Option<DistroResults>,
        state_digest: Option<[u8; 32]>,
    },
    Optimize,
    Forward {
        logits_and_loss: Option<(Option<Tensor>, Option<Tensor>)>,
    },
    Extract {
        variables: HashMap<String, Tensor>,
    },
    TruncateBf16,
    ErrorFeedback {
        buffer: Option<Vec<Tensor>>,
    },
    Load {
        error: Option<String>,
    },
    SetErrorFeedbackKind {
        error: Option<String>,
    },
    RoundStateLen {
        len: Result<Option<u64>, String>,
    },
}

#[derive(Debug)]
pub enum Trainer {
    Local(LocalTrainer),
    #[cfg(feature = "python")]
    PythonDistributed(crate::PythonDistributedTrainer),
}

impl Trainer {
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        self,
        step: u32,
        data: Batch,
        warmup_lr_between: Option<(u32, u32)>,
        zero_optim: bool,
        rollback: Vec<(u32, Vec<DistroResults>)>,
        prev_self_distro_results: Option<Vec<DistroResults>>,
        cancel_training: CancellationToken,
    ) -> Result<TrainOutput, TrainerThreadCommunicationError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.train(
                step,
                data,
                warmup_lr_between,
                zero_optim,
                rollback,
                prev_self_distro_results,
                cancel_training,
            ),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python) => python.train(
                step,
                data,
                warmup_lr_between,
                zero_optim,
                rollback,
                prev_self_distro_results,
                cancel_training,
            ),
        }
    }

    pub fn optimize(
        self,
        step: u32,
        warmup_lr_between: Option<(u32, u32)>,
        distro_results: Option<Vec<DistroResults>>,
    ) -> Result<Self, ApplyDistroResultError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer
                .optimize(step, warmup_lr_between, distro_results)
                .map(|x| x.into()),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python) => python
                .optimize(step, warmup_lr_between, distro_results)
                .map(|x| x.into()),
        }
    }

    pub fn extract(&mut self) -> Result<HashMap<String, Tensor>, TrainerThreadCommunicationError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.extract(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python) => python.extract(),
        }
    }

    pub fn truncate_bf16(&mut self) -> Result<(), TrainerThreadCommunicationError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.truncate_bf16(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python) => python.truncate_bf16(),
        }
    }

    /// A copy of each model thread's DisTrO error-feedback buffer on the CPU,
    /// None for a thread whose optimizer keeps none.
    pub fn error_feedback(
        &mut self,
    ) -> Result<Vec<Option<Vec<Tensor>>>, TrainerThreadCommunicationError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.error_feedback(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(_) => Err(TrainerThreadCommunicationError::Unsupported(
                "hand out its error feedback",
            )),
        }
    }

    /// Keeps every model thread's DisTrO error-feedback buffer in `kind`
    /// from here on, starting from zeros (wienerlabs/leviathan-net#90).
    pub fn set_error_feedback_kind(
        &mut self,
        kind: Kind,
    ) -> Result<(), TrainerThreadCommunicationError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.set_error_feedback_kind(kind),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(_) => Err(TrainerThreadCommunicationError::Unsupported(
                "move its error feedback to another dtype",
            )),
        }
    }

    /// The length of the encoding of a state this trainer trains a round
    /// from (`round_state_len`), from its model's and its optimizer's shapes
    /// and dtypes, or None if its optimizer keeps no error feedback. A
    /// verifier expects a state it fetches to be this long
    /// (wienerlabs/leviathan-net#89).
    pub fn round_state_len(&mut self) -> Result<Option<u64>, TrainerThreadCommunicationError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.round_state_len(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(_) => Err(TrainerThreadCommunicationError::Unsupported(
                "say how long its state is",
            )),
        }
    }

    /// Loads `weights`, as [`Self::extract`] returns them, into every model
    /// thread, and each thread's `error_feedback`, as [`Self::error_feedback`]
    /// returns it, into that thread's DisTrO buffer. Each thread checks the
    /// state against its model before it writes any of it, so a state that
    /// does not fit changes nothing in that thread.
    pub fn load(
        &mut self,
        weights: &HashMap<String, Tensor>,
        error_feedback: Option<&[Vec<Tensor>]>,
    ) -> Result<(), TrainerThreadCommunicationError> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.load(weights, error_feedback),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(_) => {
                Err(TrainerThreadCommunicationError::Unsupported("load a state"))
            }
        }
    }

    pub fn can_do_inference(&self) -> bool {
        match self {
            Trainer::Local(local_trainer) => local_trainer.can_do_inference(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.can_do_inference()
            }
        }
    }

    pub fn get_lr(
        lr_scheduler: &LearningRateSchedule,
        step: u32,
        warmup_lr_between: Option<(u32, u32)>,
    ) -> f64 {
        let factor = match warmup_lr_between {
            Some((start, end)) => match step >= start && step <= end {
                true => (step - start) as f64 / (end - start) as f64,
                false => 1.,
            },
            None => 1.,
        };
        lr_scheduler.get_lr(step) * factor
    }

    pub fn quantize_results(results: &DistroResults) -> DistroResults {
        results
            .iter()
            .map(|x| DistroResult {
                sparse_idx: x.sparse_idx.copy(),
                sparse_val: Distro::quantize_nozeros_tensor_to_boolean_sign(&x.sparse_val),
                xshape: x.xshape.clone(),
                totalk: x.totalk,
                stats: x.stats.clone(),
            })
            .collect()
    }
}

impl From<LocalTrainer> for Trainer {
    fn from(value: LocalTrainer) -> Self {
        Self::Local(value)
    }
}

#[derive(Debug)]
pub struct LocalTrainer {
    models: Vec<(
        flume::Sender<ParallelAssignment>,
        flume::Receiver<ParallelResult>,
    )>,
    first_model_device: Device,
    first_model_max_context_length: usize,
    barrier: Arc<dyn Barrier>,
    data_parallel: Option<Vec<DataParallel>>,
    can_do_inferences: Vec<Arc<AtomicBool>>,
    /// What each round's training commits to, and keeps, of the state it
    /// trains from (wienerlabs/leviathan-net#89).
    state_commitment: StateCommitment,
}

/// What each round's training commits to, and keeps, of the state it trains
/// from (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
#[derive(Debug, Clone)]
pub enum StateCommitment {
    /// Nothing: a trainer driven from Python, whose ranks there do not join
    /// the gathers the digest needs.
    None,
    /// The state's digest, with the state kept nowhere.
    Digest,
    /// The state's digest, with the state kept in this directory, in a file
    /// named by the digest, so the node can reveal it to an audit.
    Keep(PathBuf),
}

#[derive(Debug, Error)]
pub enum TrainerThreadCommunicationError {
    #[error("Failed to send command to trainer thread; thread has dropped RX")]
    SendCommand,

    #[error("Failed to recv result from trainer thread; thread has dropped TX")]
    RecvResult,

    #[error("Got unexpected result from trainer thread: {0}")]
    UnexpectedResult(String),

    #[error("Attempting to pad batch that's already on GPU")]
    PaddingBatch,

    #[error("This trainer cannot {0}")]
    Unsupported(&'static str),

    #[error("Could not load the trainer's state: {0}")]
    Load(String),

    #[error("Could not keep the error feedback in another dtype: {0}")]
    ErrorFeedbackKind(String),

    #[error("A model thread could not say how long its round state is: {0}")]
    RoundStateLen(String),

    #[cfg(feature = "python")]
    #[error("Python error: {0}")]
    PythonError(#[from] pyo3::PyErr),
}

impl LocalTrainer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        models: ParallelModels,
        lr_scheduler: LearningRateSchedule,
        optimizer: OptimizerDefinition,
        micro_batch_size: usize,
        stats: Option<u32>,
        grad_accum_in_fp32: bool,
    ) -> Self {
        let ParallelModels {
            models,
            barrier,
            data_parallel,
        } = models;

        assert!(!models.is_empty());
        let first_model_device = models[0].device();
        let first_model_max_context_length = models[0].max_context_length();

        let mut ret = Vec::with_capacity(models.len());

        let data_parallels = match &data_parallel {
            Some(data_parallel) => {
                assert_eq!(data_parallel.len(), models.len());
                data_parallel
                    .iter()
                    .map(|x| Some(x.clone()))
                    .collect::<Vec<_>>()
            }
            None => std::iter::repeat_n(None, models.len()).collect(),
        };

        let mut can_do_inferences = Vec::new();

        for (index, (model, data_parallel)) in models.into_iter().zip(data_parallels).enumerate() {
            let (assignment_tx, assignment_rx) = flume::unbounded();
            let (result_tx, result_rx) = flume::unbounded();
            ret.push((assignment_tx, result_rx));

            let optimizer = Optimizer::new(optimizer, model.as_ref());

            let barrier = barrier.clone();
            let data_parallel = data_parallel.clone();
            let can_do_inference = Arc::new(AtomicBool::new(false));
            can_do_inferences.push(can_do_inference.clone());

            std::thread::spawn(move || {
                Self::model_thread(
                    model,
                    assignment_rx,
                    result_tx,
                    optimizer,
                    index,
                    micro_batch_size,
                    lr_scheduler,
                    barrier,
                    stats,
                    grad_accum_in_fp32,
                    data_parallel,
                    can_do_inference,
                )
            });
        }

        Self {
            models: ret,
            first_model_device,
            first_model_max_context_length,
            barrier,
            data_parallel,
            can_do_inferences,
            state_commitment: StateCommitment::Digest,
        }
    }

    /// This trainer, committing to no state in the rounds it trains: one
    /// driven from Python, whose ranks there do not join the gathers the
    /// digest needs (wienerlabs/leviathan-net#89).
    pub fn without_state_commitment(self) -> Self {
        Self {
            state_commitment: StateCommitment::None,
            ..self
        }
    }

    /// This trainer, keeping each state it commits to in `dir`, in a file
    /// named by the state's digest (`round_state_file`), so the node can
    /// reveal it to an audit (wienerlabs/leviathan-net#89).
    pub fn keeping_round_states_in(self, dir: PathBuf) -> Self {
        Self {
            state_commitment: StateCommitment::Keep(dir),
            ..self
        }
    }

    /// The digest of the state `model` and `optimizer` train round `step`
    /// from, on the model thread at `index`, as `commitment` asks. The first
    /// thread hashes it, and keeps it where it is kept. The others of a
    /// sharded model only join the gathers it needs, and those of an
    /// unsharded one do nothing. A state that cannot be encoded is logged and
    /// committed to as none, which an audit then finds unrevealable. A state
    /// that cannot be kept is logged and still committed to: it is the state
    /// the round trains from, which the node then cannot reveal.
    fn commit_round_state(
        commitment: &StateCommitment,
        model: &dyn CausalLM,
        optimizer: &Distro,
        index: usize,
        step: u32,
    ) -> Option<[u8; 32]> {
        let keep_in = match commitment {
            StateCommitment::None => return None,
            StateCommitment::Digest => None,
            StateCommitment::Keep(dir) => Some(dir.as_path()),
        };
        if index == 0 {
            let digest = match keep_in {
                Some(dir) => keep_round_state(dir, model, optimizer).map(|kept| {
                    if let Err(err) = &kept.file {
                        error!(
                            step,
                            dir = %dir.display(),
                            "Cannot keep the state this round trains from, so it cannot be revealed: {err}"
                        );
                    }
                    kept.digest
                }),
                None => round_state_digest(model, optimizer),
            };
            return match digest {
                Ok(digest) => Some(digest),
                Err(err) => {
                    error!(
                        step,
                        "Cannot commit to the state this round trains from: {err}"
                    );
                    None
                }
            };
        }
        if model.variables().any(|variable| variable.is_sharded()) {
            if let Err(err) = write_round_state(&mut std::io::sink(), model, optimizer) {
                error!(
                    step,
                    "Cannot gather the state this round trains from: {err}"
                );
            }
        }
        None
    }

    fn forward_backward(
        model: &mut dyn CausalLM,
        inputs: Tensor,
        labels: Option<Tensor>,
        position_ids: Option<Tensor>,
        sequence_lengths: Option<Vec<Vec<i32>>>,
        barrier: &Arc<dyn Barrier>,
        loss_scale: Option<f64>,
    ) -> Result<Option<Tensor>> {
        let labels = labels.unwrap_or_else(|| inputs.copy());
        if barrier.wait().is_err() {
            return Ok(None);
        }
        let device = inputs.device();
        let (_, loss) = model.forward(
            &inputs,
            Some(&labels),
            position_ids.as_ref(),
            sequence_lengths.as_ref(),
            None,
            loss_scale,
        );
        let loss = loss.ok_or(Error::msg("No loss"))?;
        loss.backward();
        if device.is_cuda() {
            device.cuda_synchronize();
        }
        Ok(Some(loss.detach()))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        model: &mut dyn CausalLM,
        data: &Tensor,
        labels: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        barrier: &Arc<dyn Barrier>,
        num_logits_to_keep: Option<i64>,
        loss_scale: Option<f64>,
    ) -> Option<(Option<Tensor>, Option<Tensor>)> {
        let _guard = tch::no_grad_guard();
        if barrier.wait().is_err() {
            return None;
        }
        let device = model.device();
        let inputs = data.to(device);
        let labels = labels.map(|x| x.to(device));
        let device = inputs.device();
        let (logits, loss) = model.forward(
            &inputs,
            labels.as_ref(),
            position_ids,
            sequence_lengths,
            num_logits_to_keep,
            loss_scale,
        );
        if device.is_cuda() {
            device.cuda_synchronize();
        }
        Some((logits, loss.map(|x| x.detach())))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train(
        self,
        step: u32,
        data: Batch,
        warmup_lr_between: Option<(u32, u32)>,
        zero_optim: bool,
        rollback: Vec<(u32, Vec<DistroResults>)>,
        prev_self_distro_results: Option<Vec<DistroResults>>,
        cancel_training: CancellationToken,
    ) -> Result<TrainOutput, TrainerThreadCommunicationError> {
        if !rollback.is_empty() {
            error!(
                "we have not implemented getting data from previous rounds. this should be impossible to hit.. this step is {step}, rollback passed is {:?}",
                rollback.iter().map(|(step, _)| step).collect::<Vec<_>>()
            );
        }
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::Train {
                batch: data.clone(),
                step,
                warmup_lr_between,
                zero_optim,
                rollback: rollback.clone(),
                prev_self_distro_results: prev_self_distro_results.clone(),
                cancel_training: cancel_training.clone(),
                state_commitment: self.state_commitment.clone(),
            })
            .map_err(|_| TrainerThreadCommunicationError::SendCommand)?;
        }

        let mut final_loss = 0.0;
        let mut final_distro_results = None;
        let mut final_cancelled = false;
        let mut final_nonce = 0;
        let mut final_state_digest = None;
        for (_, rx) in &self.models {
            match rx
                .recv()
                .map_err(|_| TrainerThreadCommunicationError::RecvResult)?
            {
                ParallelResult::Train {
                    loss,
                    distro_results,
                    cancelled,
                    nonce,
                    state_digest,
                } => {
                    if final_distro_results.is_none() {
                        final_distro_results = distro_results;
                        final_nonce = nonce;
                    }
                    if final_state_digest.is_none() {
                        final_state_digest = state_digest;
                    }
                    final_cancelled = cancelled;
                    final_loss += loss;
                }
                weird => {
                    return Err(TrainerThreadCommunicationError::UnexpectedResult(format!(
                        "{weird:?}"
                    )));
                }
            }
        }
        final_loss /= self.models.len() as f32;
        Ok(TrainOutput {
            batch_id: data.id,
            trainer: Trainer::Local(self),
            loss: final_loss,
            step,
            distro_results: final_distro_results,
            cancelled: final_cancelled,
            nonce: final_nonce,
            state_digest: final_state_digest,
        })
    }

    pub fn optimize(
        self,
        step: u32,
        warmup_lr_between: Option<(u32, u32)>,
        results: Option<Vec<DistroResults>>,
    ) -> Result<Self, ApplyDistroResultError> {
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::Optimize {
                distro_results: results.clone(),
                step,
                warmup_lr_between,
            })
            .map_err(|_| ApplyDistroResultError::SendOptimize)?;
        }
        let start = Instant::now();
        for (_, rx) in &self.models {
            match rx.recv()? {
                ParallelResult::Optimize => {
                    trace!(
                        "ParallelResult::Optimize received in {}s",
                        (Instant::now() - start).as_secs_f32()
                    );
                }
                o => {
                    return Err(ApplyDistroResultError::ReceivedWrongResultType(format!(
                        "{o:?}"
                    )));
                }
            }
        }
        Ok(self)
    }

    pub fn extract(&mut self) -> Result<HashMap<String, Tensor>, TrainerThreadCommunicationError> {
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::Extract)
                .map_err(|_| TrainerThreadCommunicationError::SendCommand)?;
        }
        let mut extracted = HashMap::new();
        for (_, rx) in &self.models {
            match rx
                .recv()
                .map_err(|_| TrainerThreadCommunicationError::RecvResult)?
            {
                ParallelResult::Extract { variables } => {
                    if extracted.is_empty() && !variables.is_empty() {
                        extracted = variables;
                    }
                }
                result => {
                    return Err(TrainerThreadCommunicationError::UnexpectedResult(format!(
                        "{result:?}"
                    )));
                }
            }
        }
        Ok(extracted)
    }

    pub fn truncate_bf16(&mut self) -> Result<(), TrainerThreadCommunicationError> {
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::TruncateBf16)
                .map_err(|_| TrainerThreadCommunicationError::SendCommand)?;
        }
        for (_, rx) in &self.models {
            match rx
                .recv()
                .map_err(|_| TrainerThreadCommunicationError::RecvResult)?
            {
                ParallelResult::TruncateBf16 => {}
                result => {
                    return Err(TrainerThreadCommunicationError::UnexpectedResult(format!(
                        "{result:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// A copy of each model thread's DisTrO error-feedback buffer on the CPU,
    /// None for a thread whose optimizer keeps none.
    pub fn error_feedback(
        &mut self,
    ) -> Result<Vec<Option<Vec<Tensor>>>, TrainerThreadCommunicationError> {
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::ErrorFeedback)
                .map_err(|_| TrainerThreadCommunicationError::SendCommand)?;
        }
        let mut buffers = Vec::with_capacity(self.models.len());
        for (_, rx) in &self.models {
            match rx
                .recv()
                .map_err(|_| TrainerThreadCommunicationError::RecvResult)?
            {
                ParallelResult::ErrorFeedback { buffer } => buffers.push(buffer),
                result => {
                    return Err(TrainerThreadCommunicationError::UnexpectedResult(format!(
                        "{result:?}"
                    )));
                }
            }
        }
        Ok(buffers)
    }

    /// Keeps each model thread's DisTrO error-feedback buffer in `kind` from
    /// here on, starting from zeros (wienerlabs/leviathan-net#90). A thread
    /// whose optimizer keeps none, or whose model is sharded, refuses.
    pub fn set_error_feedback_kind(
        &mut self,
        kind: Kind,
    ) -> Result<(), TrainerThreadCommunicationError> {
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::SetErrorFeedbackKind { kind })
                .map_err(|_| TrainerThreadCommunicationError::SendCommand)?;
        }
        let mut first_error = None;
        for (_, rx) in &self.models {
            match rx
                .recv()
                .map_err(|_| TrainerThreadCommunicationError::RecvResult)?
            {
                ParallelResult::SetErrorFeedbackKind { error } => {
                    if first_error.is_none() {
                        first_error = error;
                    }
                }
                result => {
                    return Err(TrainerThreadCommunicationError::UnexpectedResult(format!(
                        "{result:?}"
                    )));
                }
            }
        }
        match first_error {
            Some(error) => Err(TrainerThreadCommunicationError::ErrorFeedbackKind(error)),
            None => Ok(()),
        }
    }

    /// The length of the encoding of a state this trainer trains a round
    /// from, as the first model thread works it out from its model and its
    /// optimizer (`round_state_len`), or None if the optimizer keeps no error
    /// feedback (wienerlabs/leviathan-net#89).
    pub fn round_state_len(&mut self) -> Result<Option<u64>, TrainerThreadCommunicationError> {
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::RoundStateLen)
                .map_err(|_| TrainerThreadCommunicationError::SendCommand)?;
        }
        let mut first = None;
        for (_, rx) in &self.models {
            match rx
                .recv()
                .map_err(|_| TrainerThreadCommunicationError::RecvResult)?
            {
                ParallelResult::RoundStateLen { len } => {
                    if first.is_none() {
                        first = Some(len);
                    }
                }
                result => {
                    return Err(TrainerThreadCommunicationError::UnexpectedResult(format!(
                        "{result:?}"
                    )));
                }
            }
        }
        match first {
            Some(Ok(len)) => Ok(len),
            Some(Err(error)) => Err(TrainerThreadCommunicationError::RoundStateLen(error)),
            None => Err(TrainerThreadCommunicationError::RecvResult),
        }
    }

    /// Loads `weights`, as [`Self::extract`] returns them, into every model
    /// thread, and each thread's `error_feedback`, as [`Self::error_feedback`]
    /// returns it, into that thread's DisTrO buffer. Each thread checks the
    /// state against its model before it writes any of it, so a state that
    /// does not fit changes nothing in that thread.
    pub fn load(
        &mut self,
        weights: &HashMap<String, Tensor>,
        error_feedback: Option<&[Vec<Tensor>]>,
    ) -> Result<(), TrainerThreadCommunicationError> {
        if let Some(buffers) = error_feedback {
            if buffers.len() != self.models.len() {
                return Err(TrainerThreadCommunicationError::Load(format!(
                    "{} buffers for {} model threads",
                    buffers.len(),
                    self.models.len()
                )));
            }
        }
        self.barrier.reset();
        for (index, (tx, _)) in self.models.iter().enumerate() {
            let weights = weights
                .iter()
                .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
                .collect();
            let error_feedback = error_feedback
                .map(|buffers| buffers[index].iter().map(Tensor::shallow_clone).collect());
            tx.send(ParallelAssignment::Load {
                weights,
                error_feedback,
            })
            .map_err(|_| TrainerThreadCommunicationError::SendCommand)?;
        }
        let mut failure = None;
        for (_, rx) in &self.models {
            match rx
                .recv()
                .map_err(|_| TrainerThreadCommunicationError::RecvResult)?
            {
                ParallelResult::Load { error } => failure = failure.or(error),
                result => {
                    return Err(TrainerThreadCommunicationError::UnexpectedResult(format!(
                        "{result:?}"
                    )));
                }
            }
        }
        match failure {
            Some(error) => Err(TrainerThreadCommunicationError::Load(error)),
            None => Ok(()),
        }
    }

    // todo: refactor args into a struct
    #[allow(clippy::too_many_arguments)]
    fn model_thread(
        mut model: Box<dyn CausalLM>,
        assignment: flume::Receiver<ParallelAssignment>,
        submission: flume::Sender<ParallelResult>,
        mut optimizer: Optimizer,
        index: usize,
        micro_batch_size: usize,
        lr_scheduler: LearningRateSchedule,
        barrier: Arc<dyn Barrier>,
        optim_stats_every_n_steps: Option<u32>,
        grad_accum_in_fp32: bool,
        data_parallel_def: Option<DataParallel>,
        can_do_inference: Arc<AtomicBool>,
    ) {
        #[allow(unused_mut)]
        let mut data_parallel: Option<(Arc<Communicator>, Arc<dyn Barrier>)> = None;

        #[cfg(feature = "parallelism")]
        if let Some(data_parallel_def) = data_parallel_def {
            let id = match data_parallel_def.id {
                CommunicatorId::NCCL(cstore) => cstore,
                _ => panic!("Wrong communicator type for model_thread"),
            };
            let comm = match CNCCL::new(
                id,
                data_parallel_def.rank as i64,
                data_parallel_def.world_size as i64,
                model.device(),
            ) {
                Ok(comm) => comm,
                Err(err) => {
                    error!("Error creating DP mesh: {err:#}");
                    return;
                }
            };
            data_parallel = Some((
                #[allow(clippy::arc_with_non_send_sync)]
                // TODO: analyze how we're using Arc here, is this right?
                Arc::new(Communicator::NCCL(comm)),
                data_parallel_def.barrier,
            ))
        };

        #[cfg(not(feature = "parallelism"))]
        if data_parallel_def.is_some() {
            error!("DP with parallelism feature off");
            return;
        }

        if barrier.wait().is_err() {
            error!("Incorrect model_thread boot");
            return;
        }
        model.prepare_for_training();

        let mut grad_accum: Option<Fp32GradientAccumulator> = None;
        let mut nonce = 0;
        loop {
            match assignment.recv() {
                Ok(ParallelAssignment::Train {
                    batch,
                    step,
                    warmup_lr_between,
                    zero_optim,
                    rollback: _,
                    prev_self_distro_results,
                    cancel_training,
                    state_commitment,
                }) => {
                    // this needs even more work for overlapped
                    // for (step, result) in rollback.iter().rev() {
                    //     // TODO freeze the optimizer and prevent this from modifying its internal state, methinks, right? or maybe save it and restore it later?
                    //     // we just want to have the same optimizer state wyhen we exit, save for the main operation (if not frozen. hmm)
                    //     let lr = lr_scheduler.get_lr(*step);
                    //     if optimize_step(&mut model, lr, &mut optimizer, Some(result), &barrier)
                    //         .is_break()
                    //     {
                    //         panic!("Failed to roll back.")
                    //     };
                    // }

                    debug!(batch_id=%batch.id, "model thread training on batch {}", batch.id);

                    let batch_size = batch.data.size();

                    let mut grad_accum_steps = batch_size / micro_batch_size;
                    if batch_size % micro_batch_size != 0 {
                        grad_accum_steps += 1;
                    }
                    if grad_accum_in_fp32 && grad_accum_steps != 1 && grad_accum.is_none() {
                        debug!("Allocating FP32 gradient accumulator");
                        grad_accum = Some(Fp32GradientAccumulator::new(model.as_ref()))
                    }
                    let grad_accum_divisor = grad_accum_steps as f64;

                    // collate data for batch
                    let BatchDataGPU {
                        input_ids,
                        labels,
                        position_ids,
                        sequence_lengths,
                    } = batch.data.gpu(model.device());
                    // note: torch chunk argument is total number of chunks,
                    // rust iter chunk is number of elements per chunk
                    let input_ids = input_ids.chunk(grad_accum_steps as i64, 0);
                    let labels = labels
                        .map(|x| {
                            x.chunk(grad_accum_steps as i64, 0)
                                .into_iter()
                                .map(Some)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_else(|| {
                            std::iter::from_fn(|| Some(None))
                                .take(grad_accum_steps)
                                .collect()
                        });
                    let position_ids = position_ids
                        .map(|x| {
                            x.chunk(grad_accum_steps as i64, 0)
                                .into_iter()
                                .map(Some)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_else(|| {
                            std::iter::from_fn(|| Some(None))
                                .take(grad_accum_steps)
                                .collect()
                        });
                    let sequence_lengths = sequence_lengths
                        .map(|x| {
                            x.chunks(micro_batch_size)
                                .map(|y| Some(y.to_vec()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_else(|| vec![None; grad_accum_steps]);
                    assert_eq!(input_ids.len(), grad_accum_steps);
                    assert_eq!(labels.len(), grad_accum_steps);
                    assert_eq!(position_ids.len(), grad_accum_steps);
                    assert_eq!(sequence_lengths.len(), grad_accum_steps);
                    let micro_batches =
                        itertools::izip!(input_ids, labels, position_ids, sequence_lengths);

                    if let Some(grad_accum) = &mut grad_accum {
                        grad_accum.zero_grad();
                    }

                    let mut state_digest = None;
                    let lr = Trainer::get_lr(&lr_scheduler, step, warmup_lr_between);
                    let prev_lr = match step {
                        0 => Trainer::get_lr(&lr_scheduler, 0, warmup_lr_between),
                        step => Trainer::get_lr(&lr_scheduler, step - 1, warmup_lr_between),
                    };

                    tracing::debug!(
                        lr = lr,
                        prev_lr = prev_lr,
                        step = step,
                        micro_batches = grad_accum_steps,
                        "Train begin"
                    );

                    for var in model.variables() {
                        var.zero_grad();
                    }

                    match &mut optimizer {
                        Optimizer::Torch { .. } => {
                            if zero_optim {
                                tracing::warn!("Zeroing optimizing states not supported for AdamW");
                            }
                        }
                        Optimizer::Distro { optimizer, .. } => {
                            if zero_optim {
                                optimizer.zero_optim();
                                tracing::info!("Zeroed optimizer states");
                            }
                            // The state this round trains from, committed to
                            // before the lookahead moves the weights
                            // (wienerlabs/leviathan-net#89).
                            state_digest = Self::commit_round_state(
                                &state_commitment,
                                model.as_ref(),
                                optimizer,
                                index,
                                step,
                            );
                            match &prev_self_distro_results {
                                Some(_) => optimizer.error_correction(model.as_ref(), prev_lr),
                                None => {
                                    error!(
                                        "Got DisTrO train assignment, but null previous results"
                                    );
                                    return;
                                }
                            };
                        }
                        Optimizer::Null => {}
                    };

                    let mut loss = None;
                    let mut cancelled = false;
                    for (index, (input_ids, labels, position_ids, sequence_lengths)) in
                        micro_batches.into_iter().enumerate()
                    {
                        if cancel_training.is_cancelled() {
                            cancelled = true;
                            barrier.cancel();
                            warn!("Aborting training upon request");
                            break;
                        }
                        match Self::forward_backward(
                            &mut *model,
                            input_ids,
                            labels,
                            position_ids,
                            sequence_lengths,
                            &barrier,
                            Some(grad_accum_divisor),
                        ) {
                            Ok(Some(batch_loss)) => {
                                if batch_loss.double_value(&[]).is_finite() {
                                    match loss.as_mut() {
                                        Some(loss) => *loss += batch_loss,
                                        None => {
                                            loss = Some(batch_loss);
                                        }
                                    }
                                }
                            }
                            Ok(None) => {
                                // cancelled barrier catching race to on run_state
                                cancelled = true;
                                warn!("Aborting training, run state changed");
                                break;
                            }
                            Err(err) => {
                                error!("Train error: {err:#}");
                                return;
                            }
                        }
                        if let Some(grad_accum) = &mut grad_accum {
                            grad_accum.accumulate_gradients();
                        }
                        trace!(micro_batch = index, "Finished micro batch forward/backward");
                    }
                    if let Some(grad_accum) = &mut grad_accum {
                        grad_accum.apply_accumulation();
                    }

                    // reduce grads across DP ranks
                    if let Some((dp_comm, dp_barrier)) = &data_parallel {
                        dp_barrier.wait().unwrap(); // cannot cancel dp
                        match &mut grad_accum {
                            Some(grad_accum) => grad_accum.reduce_gradients(dp_comm.clone()),
                            None => {
                                for variable in model.variables() {
                                    let mut grad = variable.local_tensor().grad();
                                    if grad.defined() {
                                        // reduce grads in fp32
                                        let mut fp32_grad = grad.to_kind(Kind::Float);
                                        fp32_grad
                                            .all_reduce(&Some(dp_comm.clone()), ReduceType::Mean);
                                        grad.copy_(&fp32_grad.to_kind(grad.kind()));
                                    }
                                }
                            }
                        }
                        if let Some(loss) = loss.as_mut() {
                            loss.all_reduce(&Some(dp_comm.clone()), ReduceType::Mean);
                        }
                        dp_barrier.wait().unwrap(); // cannot cancel dp
                    }

                    let distro_results = match cancelled {
                        false => match &mut optimizer {
                            Optimizer::Torch {
                                optimizer: _,
                                clip_grad_norm: _,
                            } => None,
                            Optimizer::Distro {
                                optimizer,
                                clip_grad_norm,
                                quantize_1bit: _,
                            } => {
                                let clipped = match clip_grad_norm {
                                    Some(clip_grad_norm) => match barrier.wait() {
                                        Ok(_) => {
                                            if *clip_grad_norm > 0. {
                                                model.clip_grad_norm(*clip_grad_norm as f64);
                                            }
                                            barrier.wait().is_ok()
                                        }
                                        Err(_) => false,
                                    },
                                    None => true,
                                };
                                if clipped {
                                    let ret = optimizer.generate(
                                        model.as_ref(),
                                        &prev_self_distro_results.unwrap_or_default(),
                                        prev_lr,
                                        lr,
                                        optim_stats_every_n_steps
                                            .map(|stats| step % stats == 0)
                                            .unwrap_or(false),
                                    );
                                    // just need results from one of the ranks
                                    match index == 0 {
                                        true => Some(ret),
                                        false => None,
                                    }
                                } else {
                                    cancelled = true;
                                    None
                                }
                            }
                            Optimizer::Null => None,
                        },
                        true => None,
                    };
                    // with Python FSDP we need to do a full forward/backward correctly before we can do inference
                    can_do_inference.store(true, Ordering::Relaxed);
                    if submission
                        .send(ParallelResult::Train {
                            loss: match loss {
                                Some(loss) => loss.try_into().unwrap_or(0.),
                                None => 0.,
                            },
                            distro_results,
                            cancelled,
                            nonce,
                            state_digest,
                        })
                        .is_err()
                    {
                        return;
                    }

                    nonce += 1;

                    // for (_, result) in rollback.iter() {
                    //     // TODO freeze the optimizer and prevent this from modifying its internal state, methinks, right? or maybe save it and restore it later?
                    //     // we just want to have the same optimizer state wyhen we exit, save for the main operation (if not frozen. hmm)
                    //     if optimize_step(&mut model, lr, &mut optimizer, Some(result), &barrier)
                    //         .is_break()
                    //     {
                    //         panic!("Failed to roll forwards.")
                    //     };
                    // }
                }
                Ok(ParallelAssignment::Optimize {
                    distro_results,
                    step,
                    warmup_lr_between,
                }) => {
                    let lr = Trainer::get_lr(&lr_scheduler, step, warmup_lr_between);
                    if optimize_step(
                        &mut model,
                        lr,
                        &mut optimizer,
                        distro_results.as_ref(),
                        &barrier,
                    )
                    .is_break()
                    {
                        return;
                    }
                    if submission.send(ParallelResult::Optimize).is_err() {
                        return;
                    }
                }
                Ok(ParallelAssignment::Forward {
                    data,
                    labels,
                    position_ids,
                    sequence_lengths,
                    num_logits_to_keep,
                    loss_scale,
                }) => {
                    if !can_do_inference.load(Ordering::Relaxed) {
                        error!("This model not set up for inference, but got inference request");
                        return;
                    }
                    let logits_and_loss = Self::forward(
                        &mut *model,
                        &data,
                        labels.as_ref(),
                        position_ids.as_ref(),
                        sequence_lengths.as_ref(),
                        &barrier,
                        num_logits_to_keep,
                        loss_scale,
                    );
                    if submission
                        .send(ParallelResult::Forward { logits_and_loss })
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(ParallelAssignment::Extract) => {
                    match unsharded_cpu_variables(model.as_ref(), model.communicator()) {
                        Ok(variables) => {
                            if submission
                                .send(ParallelResult::Extract { variables })
                                .is_err()
                            {
                                return;
                            }
                        }
                        Err(err) => {
                            error!("Unexpected error in extract: {err:#}");
                            return;
                        }
                    }
                }
                Ok(ParallelAssignment::TruncateBf16) => {
                    let _no_grad = tch::no_grad_guard();
                    for var in model.variables() {
                        let mut tensor = var.local_tensor();
                        let original_kind = tensor.kind();
                        let truncated = tensor.to_kind(Kind::BFloat16).to_kind(original_kind);
                        tensor.copy_(&truncated);
                    }
                    if submission.send(ParallelResult::TruncateBf16).is_err() {
                        return;
                    }
                }
                Ok(ParallelAssignment::ErrorFeedback) => {
                    let buffer = match &optimizer {
                        Optimizer::Distro { optimizer, .. } => Some(optimizer.error_feedback()),
                        Optimizer::Torch { .. } | Optimizer::Null => None,
                    };
                    if submission
                        .send(ParallelResult::ErrorFeedback { buffer })
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(ParallelAssignment::SetErrorFeedbackKind { kind }) => {
                    let error = match &mut optimizer {
                        Optimizer::Distro { optimizer, .. } => optimizer
                            .set_error_feedback_kind(kind)
                            .err()
                            .map(|error| error.to_string()),
                        Optimizer::Torch { .. } | Optimizer::Null => {
                            Some("the optimizer keeps no error feedback".to_string())
                        }
                    };
                    if submission
                        .send(ParallelResult::SetErrorFeedbackKind { error })
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(ParallelAssignment::RoundStateLen) => {
                    let len = match &optimizer {
                        Optimizer::Distro { optimizer, .. } => {
                            round_state_len(model.as_ref(), optimizer)
                                .map(Some)
                                .map_err(|error| error.to_string())
                        }
                        Optimizer::Torch { .. } | Optimizer::Null => Ok(None),
                    };
                    if submission
                        .send(ParallelResult::RoundStateLen { len })
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(ParallelAssignment::Load {
                    weights,
                    error_feedback,
                }) => {
                    let error = load_state(
                        model.as_ref(),
                        &mut optimizer,
                        &weights,
                        error_feedback.as_deref(),
                    )
                    .err();
                    if submission.send(ParallelResult::Load { error }).is_err() {
                        return;
                    }
                }
                Err(_) => {
                    return;
                }
            }
        }
    }

    pub fn device(&self) -> &Device {
        &self.first_model_device
    }

    pub fn data_parallel_barrier(&self) {
        if let Some(data_parallel) = &self.data_parallel {
            for dp in data_parallel {
                dp.barrier.reset();
            }
        }
    }

    pub fn data_parallel_world_size(&self) -> usize {
        self.data_parallel
            .as_ref()
            .and_then(|x| x.first().map(|y| y.world_size))
            .unwrap_or(1)
    }

    pub fn can_do_inference(&self) -> bool {
        for can in &self.can_do_inferences {
            if !can.load(Ordering::Relaxed) {
                return false;
            }
        }
        true
    }
}

#[derive(Error, Debug)]
pub enum ApplyDistroResultError {
    #[error("failed to send optimization to trainer thread - trainer thread RX is closed")]
    SendOptimize,

    #[error("failed to recv optimization result from trainer thread: {0}")]
    ReceiveResult(#[from] flume::RecvError),

    #[error("received wrong result type from trainer thread. expected Optimize, got {0:?}")]
    ReceivedWrongResultType(String),

    #[error("apply thread crashed")]
    ThreadCrashed,

    #[cfg(feature = "python")]
    #[error("Python error: {0}")]
    PythonError(#[from] pyo3::PyErr),
}

impl CausalLM for LocalTrainer {
    fn forward(
        &self,
        x: &Tensor,
        labels: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        num_logits_to_keep: Option<i64>,
        loss_scale: Option<f64>,
    ) -> (Option<Tensor>, Option<Tensor>) {
        self.barrier.reset();
        for (tx, _) in &self.models {
            tx.send(ParallelAssignment::Forward {
                data: x.shallow_clone(),
                labels: labels.map(|y| y.shallow_clone()),
                position_ids: position_ids.map(|y| y.shallow_clone()),
                sequence_lengths: sequence_lengths.cloned(),
                num_logits_to_keep,
                loss_scale,
            })
            .expect("Error getting result from forward");
        }
        let mut final_logits_and_loss = None;
        for (_, rx) in &self.models {
            match rx.recv() {
                Ok(ParallelResult::Forward { logits_and_loss }) => {
                    if final_logits_and_loss.is_none() {
                        final_logits_and_loss = logits_and_loss;
                    }
                }
                _ => panic!("Got unexpected forward result"),
            }
        }
        final_logits_and_loss.expect("No forward logits and loss")
    }

    fn bos_token_id(&self) -> Option<i64> {
        None
    }

    fn eos_token_ids(&self) -> Option<EosToks> {
        None
    }

    fn device(&self) -> tch::Device {
        self.first_model_device
    }

    fn max_context_length(&self) -> usize {
        self.first_model_max_context_length
    }

    fn variables(&self) -> StableVariableIterator {
        unimplemented!()
    }

    fn communicator(&self) -> Option<Arc<Communicator>> {
        unimplemented!()
    }

    fn prepare_for_training(&self) {}

    fn clip_grad_norm(&self, _max_grad_norm: f64) {}

    fn convert(&self, _state_dict: Option<HashMap<String, Tensor>>) -> HashMap<String, Tensor> {
        unimplemented!();
    }
}

/// Writes `weights` into `model`'s variables and `error_feedback` into
/// `optimizer`'s DisTrO buffer. Both are checked against the model before
/// either is written, so a state that does not fit changes nothing.
fn load_state(
    model: &dyn CausalLM,
    optimizer: &mut Optimizer,
    weights: &HashMap<String, Tensor>,
    error_feedback: Option<&[Tensor]>,
) -> Result<(), String> {
    let _no_grad = tch::no_grad_guard();
    let variables: Vec<_> = model.variables().collect();
    if variables.len() != weights.len() {
        return Err(format!(
            "{} weights for {} variables",
            weights.len(),
            variables.len()
        ));
    }
    for variable in &variables {
        let given = weights
            .get(variable.name())
            .ok_or_else(|| format!("no weights for {}", variable.name()))?;
        if given.size() != variable.full_tensor_shape() {
            return Err(format!(
                "{} has shape {:?}, its weights {:?}",
                variable.name(),
                variable.full_tensor_shape(),
                given.size()
            ));
        }
    }
    match (optimizer, error_feedback) {
        (_, None) => {}
        (Optimizer::Distro { optimizer, .. }, Some(buffer)) => optimizer
            .load_error_feedback(buffer)
            .map_err(|error| error.to_string())?,
        (Optimizer::Torch { .. } | Optimizer::Null, Some(_)) => {
            return Err("the optimizer keeps no error feedback".to_string());
        }
    }
    for variable in &variables {
        let given = weights[variable.name()].shallow_clone();
        let source = match variable.is_sharded() {
            true => variable.shard_other_tensor_like_me(given),
            false => given,
        };
        let mut local = variable.local_tensor();
        local.copy_(&source.to_device(local.device()).to_kind(local.kind()));
    }
    Ok(())
}

fn optimize_step(
    model: &mut Box<dyn CausalLM>,
    lr: f64,
    optimizer: &mut Optimizer,
    distro_results: Option<&Vec<Vec<DistroResult>>>,
    barrier: &Arc<dyn Barrier>,
) -> ControlFlow<()> {
    match optimizer {
        Optimizer::Torch {
            optimizer,
            clip_grad_norm,
        } => {
            optimizer.set_learning_rate(lr).unwrap();
            if let Some(clip_grad_norm) = clip_grad_norm {
                if barrier.wait().is_err() {
                    return ControlFlow::Break(());
                }
                if *clip_grad_norm > 0. {
                    model.clip_grad_norm(*clip_grad_norm as f64);
                }
                if barrier.wait().is_err() {
                    return ControlFlow::Break(());
                }
            }
            optimizer.step().unwrap();
            optimizer.zero_grad().unwrap();
        }
        Optimizer::Distro { optimizer, .. } => match distro_results {
            Some(results) => {
                if !results.is_empty() {
                    trace!("Applying {} DisTrO gradients", results.len());
                    if barrier.wait().is_err() {
                        return ControlFlow::Break(());
                    }
                    optimizer.apply(model.as_ref(), results, lr);
                    if barrier.wait().is_err() {
                        return ControlFlow::Break(());
                    }
                } else {
                    warn!("Empty DisTrO gradients, model parameters will not be updated");
                }
            }
            None => {
                error!("Got DisTrO optimizer assignment, but no results");
                return ControlFlow::Break(());
            }
        },
        Optimizer::Null => (),
    };
    ControlFlow::Continue(())
}

impl CausalLM for Trainer {
    fn forward(
        &self,
        x: &Tensor,
        labels: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        sequence_lengths: Option<&Vec<Vec<i32>>>,
        num_logits_to_keep: Option<i64>,
        loss_scale: Option<f64>,
    ) -> (Option<Tensor>, Option<Tensor>) {
        match self {
            Trainer::Local(local_trainer) => local_trainer.forward(
                x,
                labels,
                position_ids,
                sequence_lengths,
                num_logits_to_keep,
                loss_scale,
            ),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => python_distributed_trainer
                .forward(
                    x,
                    labels,
                    position_ids,
                    sequence_lengths,
                    num_logits_to_keep,
                    loss_scale,
                ),
        }
    }

    fn bos_token_id(&self) -> Option<i64> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.bos_token_id(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.bos_token_id()
            }
        }
    }

    fn eos_token_ids(&self) -> Option<EosToks> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.eos_token_ids(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.eos_token_ids()
            }
        }
    }

    fn device(&self) -> Device {
        match self {
            Trainer::Local(local_trainer) => *local_trainer.device(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.device()
            }
        }
    }

    fn max_context_length(&self) -> usize {
        match self {
            Trainer::Local(local_trainer) => local_trainer.max_context_length(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.max_context_length()
            }
        }
    }

    fn variables(&self) -> StableVariableIterator {
        match self {
            Trainer::Local(local_trainer) => local_trainer.variables(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.variables()
            }
        }
    }

    fn communicator(&self) -> Option<Arc<Communicator>> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.communicator(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.communicator()
            }
        }
    }

    fn prepare_for_training(&self) {
        match self {
            Trainer::Local(local_trainer) => local_trainer.prepare_for_training(),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.prepare_for_training()
            }
        }
    }

    fn clip_grad_norm(&self, max_grad_norm: f64) {
        match self {
            Trainer::Local(local_trainer) => local_trainer.clip_grad_norm(max_grad_norm),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.clip_grad_norm(max_grad_norm)
            }
        }
    }

    fn convert(&self, state_dict: Option<HashMap<String, Tensor>>) -> HashMap<String, Tensor> {
        match self {
            Trainer::Local(local_trainer) => local_trainer.convert(state_dict),
            #[cfg(feature = "python")]
            Trainer::PythonDistributed(python_distributed_trainer) => {
                python_distributed_trainer.convert(state_dict)
            }
        }
    }
}
