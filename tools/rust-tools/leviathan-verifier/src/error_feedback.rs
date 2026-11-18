//! What DisTrO's error feedback costs a replay audit
//! (wienerlabs/leviathan-net#48).
//!
//! A DisTrO node keeps an error-feedback buffer, the part of its update it has
//! computed but not yet sent:
//!
//! ```text
//! D_s = decay * (D_{s-1} - T_{s-1}) + lr * g_s        result_s = topk(encode(D_s))
//! ```
//!
//! `T_{s-1}` is what the node computed to send at its previous round, which its
//! buffer determines. `g_s` is its gradient on its batch, taken where its
//! weights would be once its own pending update lands (the lookahead). So
//! `D_s` is a function of public data since the buffer was last zeroed, and
//! replaying a node's round means replaying every round since then, on
//! hardware that drifts from the node's. The client never zeroes it in a run
//! without cold-start warmup steps, which is every Leviathan run.
//!
//! [`simulate`] trains a run the way its clients do (`psyche_client`'s
//! `TrainingStepMetadata`):
//! - a round at height `h` of its epoch first applies what every node sent at
//!   height `h - 2`, so a node trains one update behind, and its lookahead
//!   stands in for the update it has not applied yet;
//! - the first two rounds of an epoch apply nothing, and the two that close it
//!   train nothing and apply what is left;
//! - every round takes its step and its slice of the run's data, whether it
//!   trains or not;
//! - an epoch's first round carries no previous result, so nothing a node
//!   sent before it is taken out of its buffer there;
//! - each node trains its share of a round's batch, and every node applies the
//!   shares' results in the batch's order.
//!
//! On top of that:
//! - a reset policy zeroes every node's buffer every N training rounds;
//! - a shadow verifier, on its own device and dtype, replays node 0 from the
//!   same public data: the same batches, the same applied updates, the same
//!   resets;
//! - where the shadow starts each round from is [`ShadowStart`]: its own
//!   state, node 0's weights as a canonical checkpoint at every reset, or node
//!   0's weights, buffer and previous round's values as the node would reveal
//!   them for an audit of one round.
//!
//! Each training round records the nodes' mean loss and how far the shadow's
//! replay of node 0 landed from what node 0 sent. No batch is trained twice,
//! so the loss of a round is a loss on data the model has not yet seen.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::save_tensors_into_safetensors;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::DistroResult;
use psyche_modeling::Trainer;
use psyche_verifier::relative_l2_distance;
use tokio_util::sync::CancellationToken;

use crate::decompress_results;
use crate::replay::build_replay_trainer;
use crate::replay::ReplayTrainerConfig;

/// One node's part of a round's batch.
#[derive(Debug, Clone)]
pub struct Share {
    pub node: usize,
    pub batch: BatchId,
    pub data: Vec<BatchDataCPU>,
}

/// One round of a run, as the coordinator runs it.
#[derive(Debug, Clone)]
pub struct Round {
    pub step: u32,
    /// Its height in its epoch: 0 for the round that starts it.
    pub height: u32,
    /// What it trains: the shares of its batch, in the batch's order, which
    /// is the order every node applies their results in. None for a round that
    /// closes an epoch and only applies.
    pub shares: Option<Vec<Share>>,
}

/// Where a round falls in a run whose epochs all train the same number of
/// rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub step: u32,
    pub height: u32,
    pub trains: bool,
}

/// Where the shadow verifier starts a round from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShadowStart {
    /// Its own weights, built by applying every public update since the
    /// run's start on its own hardware, and its own buffer, which a reset
    /// zeroes as it zeroes the nodes'.
    Own,
    /// At every reset after the first, node 0's weights, written to this
    /// directory as a checkpoint of the run's model and read back, the way
    /// every node would take a canonical checkpoint at a public boundary.
    /// Between resets it keeps its own state. The directory is the
    /// simulation's: it is cleared at every reset.
    Resync(PathBuf),
    /// At every training round, node 0's weights, its buffer and the values
    /// it computed at its previous round, as they stand when the round
    /// starts: the state a node reveals for an audit of that one round. The
    /// previous round's indices and signs are public in its payload; their
    /// values are part of the reveal.
    Reveal,
}

/// What a simulation trains.
#[derive(Debug, Clone)]
pub struct Simulation {
    /// What every node trains with: the run's checkpoint, schedule and
    /// optimizer, on the nodes' device and dtype.
    pub nodes: ReplayTrainerConfig,
    /// How many nodes train together.
    pub node_count: usize,
    /// The shadow verifier's trainer: the same run, on the device and dtype the
    /// verifier replays in. None measures convergence alone.
    pub shadow: Option<ReplayTrainerConfig>,
    /// Zero every node's buffer every this many training rounds, counted from
    /// the first; 0 never does, as the client does not.
    pub reset_every: u32,
    /// Where the shadow starts each round from.
    pub shadow_start: ShadowStart,
}

/// How far a replay of node 0 landed from what node 0 sent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowDrift {
    /// The relative L2 distance between what node 0 sent and the replay, both
    /// decompressed: what the verifier's band is compared with.
    pub distance: f32,
    /// The share of the coordinates node 0 sent that the replay did not send
    /// at all, because its top-k kept other ones.
    pub churn: f32,
    /// Of the coordinates both sent, the share the replay sent with the other
    /// sign.
    pub flips: f32,
}

/// What one training round of a simulation measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepRecord {
    pub step: u32,
    /// The training rounds since the buffers were last zeroed, this one
    /// included: how many a verifier has to replay to reach it.
    pub since_reset: u32,
    /// The nodes' mean training loss on this round's batch.
    pub loss: f32,
    pub shadow: Option<ShadowDrift>,
}

/// What one training round left, as an observer of [`simulate`] sees it.
#[derive(Debug, Clone, Copy)]
pub struct Observed<'a> {
    pub record: &'a StepRecord,
    /// The round's shares, in the batch's order.
    pub shares: &'a [Share],
    /// What the node training each share sent, in the same order.
    pub sent: &'a [Vec<DistroResult>],
}

/// Whether the buffers are zeroed at the training round with index `index`,
/// counted from 0, under `reset_every`. The first always starts from a zeroed
/// buffer.
pub fn resets_at(reset_every: u32, index: u32) -> bool {
    index == 0 || (reset_every > 0 && index.is_multiple_of(reset_every))
}

/// How many training rounds a replay of the one with index `index` covers
/// under `reset_every`: back to the last reset, or to the first round if there
/// is none.
pub fn since_reset(reset_every: u32, index: u32) -> u32 {
    match reset_every {
        0 => index + 1,
        every => index % every + 1,
    }
}

/// The rounds a run goes through from `first_step` until it has trained
/// `training_rounds` of them, when each epoch trains `epoch_rounds` rounds and
/// two more close it. With `epoch_rounds` 0 the run is one epoch that never
/// closes.
pub fn schedule(first_step: u32, training_rounds: u32, epoch_rounds: u32) -> Vec<Slot> {
    let mut slots = Vec::new();
    let (mut step, mut height, mut trained) = (first_step, 0u32, 0u32);
    while trained < training_rounds {
        let trains = epoch_rounds == 0 || height < epoch_rounds;
        slots.push(Slot {
            step,
            height,
            trains,
        });
        trained += u32::from(trains);
        step += 1;
        height += 1;
        if epoch_rounds > 0 && height == epoch_rounds + 2 {
            height = 0;
        }
    }
    slots
}

/// The sequences of the global batch a round at `step` trains, when every
/// round from `first_step` on takes `global_batch` of them in turn: the
/// coordinator starts each round's data where the round before it ended,
/// whether that round trained or not, and each epoch where the last one ended.
pub fn round_batch(first_step: u32, step: u32, global_batch: usize) -> Result<BatchId> {
    if global_batch == 0 {
        return Err(anyhow!("a run with a global batch of 0 trains nothing"));
    }
    if step < first_step {
        return Err(anyhow!("step {step} comes before the first, {first_step}"));
    }
    let start = (step - first_step) as u64 * global_batch as u64;
    Ok(BatchId(ClosedInterval::new(
        start,
        start + global_batch as u64 - 1,
    )))
}

/// Checks that `nodes` nodes of a run that seats `verification_percent` of a
/// round's nodes as verifiers, and warms up from a cold start over
/// `cold_start_warmup_steps` steps, train the way [`simulate`] trains them:
/// every node trains every round, and nothing zeroes a buffer but a reset
/// policy. A verifier seat trains nothing that round and takes nothing out of
/// its buffer at the next, and a cold start zeroes every buffer at the first
/// round of each cold-start epoch; the simulation does neither.
pub fn check_cohort(
    nodes: usize,
    verification_percent: u8,
    cold_start_warmup_steps: u32,
) -> Result<()> {
    // The committee seats this many verifiers out of a round's nodes
    // (`CommitteeSelection::new`), with no tie-breakers on a training round.
    let verifiers = nodes * verification_percent as usize / 100;
    if verifiers > 0 {
        return Err(anyhow!(
            "{nodes} nodes at {verification_percent}% seat {verifiers} verifiers a round, who train nothing then; the simulation trains every node every round"
        ));
    }
    if cold_start_warmup_steps != 0 {
        return Err(anyhow!(
            "the run warms up from a cold start over {cold_start_warmup_steps} steps, which zeroes every buffer at the first round of each cold-start epoch; the simulation does not"
        ));
    }
    Ok(())
}

/// How many sequences each of `nodes` nodes trains of a global batch of
/// `global_batch`: an even share, or an error when there is none.
pub fn share_size(global_batch: usize, nodes: usize) -> Result<usize> {
    if nodes == 0 || global_batch < nodes || !global_batch.is_multiple_of(nodes) {
        return Err(anyhow!(
            "a global batch of {global_batch} sequences does not split evenly between {nodes} nodes"
        ));
    }
    Ok(global_batch / nodes)
}

/// `batch` split into `nodes` even shares in order, node `i` training the
/// `i`-th.
pub fn even_shares(batch: BatchId, data: Vec<BatchDataCPU>, nodes: usize) -> Result<Vec<Share>> {
    let size = share_size(data.len(), nodes)?;
    if batch.0.end - batch.0.start + 1 != data.len() as u64 {
        return Err(anyhow!(
            "batch {batch} holds {} sequences, not {}",
            batch.0.end - batch.0.start + 1,
            data.len()
        ));
    }
    Ok(data
        .chunks(size)
        .enumerate()
        .map(|(node, share)| {
            let start = batch.0.start + (node * size) as u64;
            Share {
                node,
                batch: BatchId(ClosedInterval::new(start, start + size as u64 - 1)),
                data: share.to_vec(),
            }
        })
        .collect())
}

/// Reads the sequences of `batch` from `provider`.
pub async fn read_batch<P: TokenizedDataProvider>(
    provider: &mut P,
    batch: BatchId,
) -> Result<Vec<BatchDataCPU>> {
    let data: Vec<BatchDataCPU> = provider
        .get_samples(batch)
        .await
        .with_context(|| format!("reading batch {batch}"))?
        .into_iter()
        .map(|sample| BatchDataCPU {
            input_ids: sample.input_ids,
            labels: sample.labels,
            position_ids: sample.position_ids,
            sequence_lengths: sample.sequence_lengths,
        })
        .collect();
    let expected = batch.0.end - batch.0.start + 1;
    if data.len() as u64 != expected {
        return Err(anyhow!(
            "batch {batch} read {} sequences, not {expected}",
            data.len()
        ));
    }
    Ok(data)
}

/// The rounds of `slots`, each training round with its global batch read from
/// `provider` in the run's order and split evenly between `nodes` nodes.
pub async fn read_rounds<P: TokenizedDataProvider>(
    provider: &mut P,
    slots: &[Slot],
    first_step: u32,
    global_batch: usize,
    nodes: usize,
) -> Result<Vec<Round>> {
    let mut rounds = Vec::with_capacity(slots.len());
    for slot in slots {
        let shares = match slot.trains {
            true => {
                let batch = round_batch(first_step, slot.step, global_batch)?;
                let data = read_batch(provider, batch).await?;
                Some(even_shares(batch, data, nodes)?)
            }
            false => None,
        };
        rounds.push(Round {
            step: slot.step,
            height: slot.height,
            shares,
        });
    }
    Ok(rounds)
}

/// Checks that `rounds` is a run the coordinator could run with `node_count`
/// nodes: epochs that start at height 0 and climb one step at a time, rounds
/// that train before the ones that close an epoch, and every node training one
/// share of every training round.
pub fn check_rounds(rounds: &[Round], node_count: usize) -> Result<()> {
    let first = rounds.first().ok_or_else(|| anyhow!("no round to train"))?;
    if first.height != 0 {
        return Err(anyhow!(
            "the first round, step {}, does not start an epoch",
            first.step
        ));
    }
    for pair in rounds.windows(2) {
        let (before, round) = (&pair[0], &pair[1]);
        if round.step != before.step + 1 {
            return Err(anyhow!("step {} follows step {}", round.step, before.step));
        }
        if round.height != 0 && round.height != before.height + 1 {
            return Err(anyhow!(
                "step {} is at height {} after height {}",
                round.step,
                round.height,
                before.height
            ));
        }
        if round.height != 0 && round.shares.is_some() && before.shares.is_none() {
            return Err(anyhow!(
                "step {} trains after a round that closed its epoch",
                round.step
            ));
        }
    }
    for round in rounds {
        let Some(shares) = &round.shares else {
            continue;
        };
        let mut nodes: Vec<usize> = shares.iter().map(|share| share.node).collect();
        nodes.sort_unstable();
        if nodes != (0..node_count).collect::<Vec<_>>() {
            return Err(anyhow!(
                "step {} gives shares to nodes {nodes:?}, not to each of {node_count} once",
                round.step
            ));
        }
    }
    Ok(())
}

/// Compares the coordinates node 0 `sent` with the `replayed` ones: the share
/// of node 0's coordinates the replay did not send at all, and, of those both
/// sent, the share with the other sign. When node 0 sent nothing, the replay
/// churned everything it sent, and nothing flipped.
pub fn compare_signs(sent: &[f32], replayed: &[f32]) -> Result<(f32, f32)> {
    if sent.len() != replayed.len() {
        return Err(anyhow!(
            "{} sent coordinates against {} replayed",
            sent.len(),
            replayed.len()
        ));
    }
    let (held, churned, flipped) = sent
        .iter()
        .zip(replayed)
        .filter(|(sent, _)| **sent != 0.0)
        .fold(
            (0u64, 0u64, 0u64),
            |(held, churned, flipped), (sent, replayed)| match *replayed == 0.0 {
                true => (held + 1, churned + 1, flipped),
                false => (
                    held + 1,
                    churned,
                    flipped + u64::from((*sent > 0.0) != (*replayed > 0.0)),
                ),
            },
        );
    if held == 0 {
        let replay_sent = replayed.iter().any(|value| *value != 0.0);
        return Ok((f32::from(u8::from(replay_sent)), 0.0));
    }
    let shared = held - churned;
    let flips = match shared {
        0 => 0.0,
        shared => flipped as f32 / shared as f32,
    };
    Ok((churned as f32 / held as f32, flips))
}

/// The nodes' mean training loss at `step`. The trainer reports a loss of 0
/// when none of its micro-batches had a finite one, so a loss that is not
/// positive and finite means the node diverged, and the round measures
/// nothing.
pub fn mean_loss(step: u32, losses: &[f32]) -> Result<f32> {
    if losses.is_empty() {
        return Err(anyhow!("no node trained step {step}"));
    }
    if let Some((node, loss)) = losses
        .iter()
        .enumerate()
        .find(|(_, loss)| !loss.is_finite() || **loss <= 0.0)
    {
        return Err(anyhow!(
            "node {node} diverged at step {step}: its training loss is {loss}"
        ));
    }
    Ok(losses.iter().sum::<f32>() / losses.len() as f32)
}

/// The bytes one node's error-feedback buffer holds: a tensor the shape of
/// each of the model's variables, in the dtype it trains in. Revealing the
/// buffer on audit sends this much, and each node keeps it.
pub fn error_feedback_bytes(config: &ReplayTrainerConfig) -> Result<u64> {
    let mut trainer = build_replay_trainer(config)?;
    let variables = trainer
        .extract()
        .map_err(|error| anyhow!("reading the model's variables: {error}"))?;
    Ok(variables
        .values()
        .map(|tensor| tensor.numel() as u64 * config.kind.elt_size_in_bytes() as u64)
        .sum())
}

/// Writes `trainer`'s weights to `dir` as a checkpoint of the model whose
/// files are `repo_files`: its configuration and tokenizer as they are, and
/// the weights as the client writes a checkpoint. Returns the checkpoint's
/// files.
fn write_checkpoint(
    trainer: &mut Trainer,
    repo_files: &[PathBuf],
    dir: &Path,
) -> Result<Vec<PathBuf>> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).with_context(|| format!("clearing {}", dir.display()))?;
    }
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut files = Vec::with_capacity(repo_files.len());
    for file in repo_files {
        let name = file
            .file_name()
            .ok_or_else(|| anyhow!("the checkpoint file {} has no name", file.display()))?;
        if name.to_string_lossy().contains(".safetensors") {
            continue;
        }
        let copy = dir.join(name);
        std::fs::copy(file, &copy).with_context(|| format!("copying {}", file.display()))?;
        files.push(copy);
    }
    let weights = trainer
        .extract()
        .map_err(|error| anyhow!("reading node 0's weights: {error}"))?;
    files.extend(
        save_tensors_into_safetensors(weights, dir.to_path_buf())
            .map_err(|error| anyhow!("writing node 0's weights: {error}"))?,
    );
    Ok(files)
}

/// What a trained round leaves: the trainer, the round's loss, and the results
/// the node computed, before any quantization.
struct Trained {
    trainer: Trainer,
    loss: f32,
    results: Vec<DistroResult>,
}

/// Trains `share` as a node trains its batch at `step`, with what it computed
/// at its previous round in this epoch, `sent_before`, to take out of its
/// buffer. A reset zeroes the buffer first, and then there is nothing to take
/// out of it.
fn train_share(
    trainer: Trainer,
    step: u32,
    share: &Share,
    reset: bool,
    sent_before: &[DistroResult],
) -> Result<Trained> {
    let sent_before = match reset || sent_before.is_empty() {
        true => vec![],
        false => vec![sent_before.to_vec()],
    };
    let output = trainer
        .train(
            step,
            Batch {
                id: share.batch,
                data: BatchData::CPU(share.data.clone()),
            },
            None,
            reset,
            vec![],
            Some(sent_before),
            CancellationToken::new(),
        )
        .map_err(|error| anyhow!("training step {step}: {error}"))?;
    let results = output
        .distro_results
        .ok_or_else(|| anyhow!("the run does not train with DisTrO"))?;
    Ok(Trained {
        trainer: output.trainer,
        loss: output.loss,
        results,
    })
}

/// What a node sends of `results`: 1-bit signs when the run quantizes.
fn transmitted(results: &[DistroResult], quantize_1bit: bool) -> Vec<DistroResult> {
    match quantize_1bit {
        true => Trainer::quantize_results(&results.to_vec()),
        false => results.to_vec(),
    }
}

/// How far `replayed` landed from what node 0 `sent`.
fn shadow_drift(sent: &[DistroResult], replayed: &[DistroResult]) -> Result<ShadowDrift> {
    let sent = decompress_results(sent, tch::Device::Cpu)?;
    let replayed = decompress_results(replayed, tch::Device::Cpu)?;
    let (churn, flips) = compare_signs(&sent, &replayed)?;
    Ok(ShadowDrift {
        distance: relative_l2_distance(&sent, &replayed)
            .map_err(|error| anyhow!("comparing the replay: {error:?}"))?,
        churn,
        flips,
    })
}

/// Hands `shadow` what `node` would reveal for an audit of its next round: its
/// weights and its error-feedback buffer. The caller hands over the values of
/// the node's previous round too; they follow from the buffer, whose top-k
/// they are, so a verifier could derive them instead.
fn reveal(node: &mut Trainer, shadow: &mut Trainer) -> Result<()> {
    let weights = node
        .extract()
        .map_err(|error| anyhow!("reading node 0's weights: {error}"))?;
    let buffer = node
        .error_feedback()
        .map_err(|error| anyhow!("reading node 0's error feedback: {error}"))?
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow!("node 0 keeps no error feedback"))?;
    shadow
        .load(&weights, Some(&buffer))
        .map_err(|error| anyhow!("handing node 0's state to the shadow: {error}"))
}

/// Applies `sent`, every node's results of a round in the batch's order, to
/// `trainer`, as the client does when it starts the round at `step`.
fn apply(trainer: Trainer, step: u32, sent: &[Vec<DistroResult>]) -> Result<Trainer> {
    trainer
        .optimize(step, None, Some(sent.to_vec()))
        .map_err(|error| anyhow!("applying results at step {step}: {error}"))
}

/// Takes the trainer out of `slot` to run `f` on it, and puts back what `f`
/// returns.
fn update(slot: &mut Option<Trainer>, f: impl FnOnce(Trainer) -> Result<Trainer>) -> Result<()> {
    let trainer = slot
        .take()
        .ok_or_else(|| anyhow!("a trainer was lost by an earlier error"))?;
    *slot = Some(f(trainer)?);
    Ok(())
}

/// Trains `rounds` under `simulation`, and returns what each training round
/// measured. `observe` sees each round as soon as it is trained.
pub fn simulate(
    simulation: &Simulation,
    rounds: Vec<Round>,
    observe: &mut dyn FnMut(Observed),
) -> Result<Vec<StepRecord>> {
    check_rounds(&rounds, simulation.node_count)?;
    let quantize_1bit = simulation.nodes.quantize_1bit();
    let mut nodes = (0..simulation.node_count)
        .map(|_| build_replay_trainer(&simulation.nodes).map(Some))
        .collect::<Result<Vec<_>>>()?;
    let mut shadow = simulation
        .shadow
        .as_ref()
        .map(build_replay_trainer)
        .transpose()?;
    // What each node computed at its previous round in this epoch, and what
    // every node sent at each height of it, in the batch's order.
    let mut computed: Vec<Vec<DistroResult>> = vec![vec![]; simulation.node_count];
    let mut shadow_computed: Vec<DistroResult> = vec![];
    let mut sent_at: HashMap<u32, Vec<Vec<DistroResult>>> = HashMap::new();
    let mut records = Vec::new();
    let mut index = 0u32;

    for round in rounds {
        // A new epoch starts its rounds afresh: no previous result to take
        // out of a buffer, and nothing pending from the last epoch, whose
        // closing rounds applied it.
        if round.height == 0 {
            computed = vec![vec![]; simulation.node_count];
            shadow_computed = vec![];
            sent_at.clear();
        }
        if round.height >= 2 {
            let sent = sent_at.remove(&(round.height - 2)).ok_or_else(|| {
                anyhow!(
                    "step {} applies height {}, which trained nothing",
                    round.step,
                    round.height - 2
                )
            })?;
            for slot in nodes.iter_mut().chain(std::iter::once(&mut shadow)) {
                if slot.is_some() {
                    update(slot, |trainer| apply(trainer, round.step, &sent))?;
                }
            }
        }
        let Some(shares) = round.shares else {
            continue;
        };
        let reset = resets_at(simulation.reset_every, index);

        match (
            &simulation.shadow_start,
            &simulation.shadow,
            shadow.as_mut(),
        ) {
            // The shadow starts the window from the weights node 0 starts it
            // from, as a canonical checkpoint would hand them out.
            (ShadowStart::Resync(dir), Some(config), Some(_)) if reset && index > 0 => {
                let node_zero = nodes[0]
                    .as_mut()
                    .ok_or_else(|| anyhow!("node 0's trainer was lost by an earlier error"))?;
                let files = write_checkpoint(node_zero, &simulation.nodes.repo_files, dir)?;
                drop(shadow.take());
                shadow = Some(build_replay_trainer(&ReplayTrainerConfig {
                    repo_files: files,
                    ..config.clone()
                })?);
                shadow_computed = vec![];
            }
            // The shadow starts the round from what node 0 reveals.
            (ShadowStart::Reveal, Some(_), Some(shadow)) => {
                let node_zero = nodes[0]
                    .as_mut()
                    .ok_or_else(|| anyhow!("node 0's trainer was lost by an earlier error"))?;
                reveal(node_zero, shadow)?;
                shadow_computed = computed[0].clone();
            }
            _ => {}
        }

        let mut losses = Vec::with_capacity(shares.len());
        let mut this_round = Vec::with_capacity(shares.len());
        let mut node_zero_sent = None;
        for share in &shares {
            let mut trained = None;
            update(&mut nodes[share.node], |trainer| {
                let done = train_share(trainer, round.step, share, reset, &computed[share.node])?;
                let trainer = done.trainer;
                trained = Some((done.loss, done.results));
                Ok(trainer)
            })?;
            let (loss, results) =
                trained.ok_or_else(|| anyhow!("node {} did not train", share.node))?;
            losses.push(loss);
            let sent = transmitted(&results, quantize_1bit);
            if share.node == 0 {
                node_zero_sent = Some(sent.clone());
            }
            this_round.push(sent);
            computed[share.node] = results;
        }

        let drift = match shadow.take() {
            Some(trainer) => {
                let share = shares
                    .iter()
                    .find(|share| share.node == 0)
                    .ok_or_else(|| anyhow!("node 0 has no share at step {}", round.step))?;
                let trained = train_share(trainer, round.step, share, reset, &shadow_computed)?;
                shadow = Some(trained.trainer);
                let replayed = transmitted(&trained.results, quantize_1bit);
                shadow_computed = trained.results;
                let sent = node_zero_sent
                    .as_ref()
                    .ok_or_else(|| anyhow!("node 0 sent nothing at step {}", round.step))?;
                Some(shadow_drift(sent, &replayed)?)
            }
            None => None,
        };

        let record = StepRecord {
            step: round.step,
            since_reset: since_reset(simulation.reset_every, index),
            loss: mean_loss(round.step, &losses)?,
            shadow: drift,
        };
        observe(Observed {
            record: &record,
            shares: &shares,
            sent: &this_round,
        });
        records.push(record);
        sent_at.insert(round.height, this_round);
        index += 1;
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(token: i32) -> BatchDataCPU {
        BatchDataCPU {
            input_ids: vec![token; 4],
            labels: None,
            position_ids: None,
            sequence_lengths: None,
        }
    }

    fn batch(start: u64, end: u64) -> BatchId {
        BatchId(ClosedInterval::new(start, end))
    }

    fn round(step: u32, height: u32, nodes: Option<&[usize]>) -> Round {
        Round {
            step,
            height,
            shares: nodes.map(|nodes| {
                nodes
                    .iter()
                    .map(|node| Share {
                        node: *node,
                        batch: batch(0, 0),
                        data: vec![sequence(0)],
                    })
                    .collect()
            }),
        }
    }

    #[test]
    fn the_first_training_round_always_starts_from_a_zeroed_buffer() {
        for every in [0, 1, 3, 10] {
            assert!(resets_at(every, 0), "every {every}");
        }
        assert!(!resets_at(0, 1));
        assert!(!resets_at(0, 1_000));
    }

    #[test]
    fn a_policy_resets_every_n_training_rounds_from_the_first() {
        let resets: Vec<u32> = (0..10).filter(|index| resets_at(3, *index)).collect();
        assert_eq!(resets, vec![0, 3, 6, 9]);
        assert!((0..5).all(|index| resets_at(1, index)));
    }

    #[test]
    fn a_replay_covers_the_rounds_back_to_the_last_reset() {
        let since: Vec<u32> = (0..7).map(|index| since_reset(3, index)).collect();
        assert_eq!(since, vec![1, 2, 3, 1, 2, 3, 1]);
        assert_eq!(since_reset(0, 0), 1);
        assert_eq!(since_reset(0, 499), 500);
        assert_eq!(since_reset(1, 499), 1);
        for index in 0..50 {
            assert_eq!(resets_at(4, index), since_reset(4, index) == 1);
        }
    }

    #[test]
    fn an_epoch_trains_its_rounds_and_two_more_close_it() {
        let slots = schedule(1, 5, 3);
        let shape: Vec<(u32, u32, bool)> = slots
            .iter()
            .map(|slot| (slot.step, slot.height, slot.trains))
            .collect();
        assert_eq!(
            shape,
            vec![
                (1, 0, true),
                (2, 1, true),
                (3, 2, true),
                (4, 3, false),
                (5, 4, false),
                (6, 0, true),
                (7, 1, true),
            ]
        );
        // One epoch that never closes trains every round.
        let endless = schedule(5, 4, 0);
        assert!(endless.iter().all(|slot| slot.trains));
        assert_eq!(
            endless.last().map(|slot| (slot.step, slot.height)),
            Some((8, 3))
        );
        assert!(schedule(1, 0, 3).is_empty());
    }

    #[test]
    fn every_round_takes_its_slice_of_the_data() {
        assert_eq!(round_batch(1, 1, 8).unwrap(), batch(0, 7));
        // The rounds that close an epoch take theirs too, so the next one
        // trains after them.
        assert_eq!(round_batch(1, 9, 8).unwrap(), batch(64, 71));
        assert!(round_batch(1, 9, 0).is_err());
        assert!(round_batch(2, 1, 8).is_err());
    }

    #[test]
    fn a_cohort_the_simulation_cannot_train_is_refused() {
        check_cohort(2, 10, 0).unwrap();
        check_cohort(9, 10, 0).unwrap();
        check_cohort(100, 0, 0).unwrap();
        let seated = check_cohort(10, 10, 0).unwrap_err().to_string();
        assert!(seated.contains("seat 1 verifiers"), "{seated}");
        assert!(check_cohort(4, 50, 0).is_err());
        let cold = check_cohort(2, 0, 3).unwrap_err().to_string();
        assert!(cold.contains("cold start over 3 steps"), "{cold}");
    }

    #[test]
    fn every_node_trains_an_even_share() {
        assert_eq!(share_size(8, 2).unwrap(), 4);
        assert_eq!(share_size(8, 8).unwrap(), 1);
        assert!(share_size(8, 3).is_err());
        assert!(share_size(8, 0).is_err());
        assert!(share_size(2, 4).is_err());
        assert!(share_size(0, 1).is_err());
    }

    #[test]
    fn a_global_batch_is_split_evenly_in_order() {
        let global: Vec<BatchDataCPU> = (0..8).map(sequence).collect();
        let shares = even_shares(batch(16, 23), global.clone(), 2).unwrap();
        assert_eq!(shares.len(), 2);
        assert_eq!((shares[0].node, shares[0].batch), (0, batch(16, 19)));
        assert_eq!((shares[1].node, shares[1].batch), (1, batch(20, 23)));
        assert_eq!(shares[0].data[3].input_ids[0], 3);
        assert_eq!(shares[1].data[0].input_ids[0], 4);
        assert!(even_shares(batch(16, 23), global.clone(), 3).is_err());
        assert!(even_shares(batch(16, 24), global, 2).is_err());
    }

    #[test]
    fn a_run_is_rounds_the_coordinator_could_run() {
        let both: &[usize] = &[1, 0];
        let good = vec![
            round(1, 0, Some(both)),
            round(2, 1, Some(both)),
            round(3, 2, None),
            round(4, 3, None),
            round(5, 0, Some(both)),
        ];
        check_rounds(&good, 2).unwrap();
        assert!(check_rounds(&[], 2).is_err());
        assert!(check_rounds(&good[1..], 2).is_err(), "starts mid-epoch");
        let mut gap = good.clone();
        gap[1].step = 3;
        assert!(check_rounds(&gap, 2).is_err(), "skips a step");
        let mut climb = good.clone();
        climb[2].height = 3;
        assert!(check_rounds(&climb, 2).is_err(), "skips a height");
        let mut late = good.clone();
        late[3].shares = round(4, 3, Some(both)).shares;
        assert!(check_rounds(&late, 2).is_err(), "trains after closing");
        let mut lone = good.clone();
        lone[0].shares = round(1, 0, Some(&[0, 0])).shares;
        assert!(check_rounds(&lone, 2).is_err(), "a node trains twice");
        assert!(check_rounds(&good, 3).is_err(), "a node trains nothing");
    }

    #[test]
    fn churn_counts_what_the_replay_dropped_and_flips_what_it_turned() {
        let sent = [1.0, -1.0, 0.0, 1.0, 1.0, 0.0];
        assert_eq!(compare_signs(&sent, &sent).unwrap(), (0.0, 0.0));
        // Of four sent: one dropped, three shared, one of those turned. What
        // the replay added where node 0 sent nothing does not count.
        let replayed = [1.0, 1.0, 1.0, 0.0, 1.0, -1.0];
        let (churn, flips) = compare_signs(&sent, &replayed).unwrap();
        assert!((churn - 0.25).abs() < 1e-6);
        assert!((flips - 1.0 / 3.0).abs() < 1e-6);
        assert!(compare_signs(&sent, &[1.0]).is_err());
    }

    #[test]
    fn signs_are_compared_not_magnitudes() {
        let sent = [0.5, -0.25, 0.125];
        let replayed = [0.4999, -0.3, -0.125];
        let (churn, flips) = compare_signs(&sent, &replayed).unwrap();
        assert_eq!(churn, 0.0);
        assert!((flips - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn nothing_sent_by_node_zero() {
        assert_eq!(compare_signs(&[0.0, 0.0], &[0.0, 0.0]).unwrap(), (0.0, 0.0));
        assert_eq!(
            compare_signs(&[0.0, 0.0], &[0.0, -1.0]).unwrap(),
            (1.0, 0.0)
        );
    }

    #[test]
    fn a_diverged_node_fails_the_round() {
        assert_eq!(mean_loss(3, &[2.0, 4.0]).unwrap(), 3.0);
        // The trainer's 0 for "no finite micro-batch loss" is divergence,
        // not a perfect fit.
        for diverged in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let error = mean_loss(3, &[2.0, diverged]).unwrap_err().to_string();
            assert!(error.contains("node 1 diverged at step 3"), "{error}");
        }
        assert!(mean_loss(3, &[]).is_err());
    }
}
