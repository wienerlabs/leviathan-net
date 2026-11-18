//! The state a node trains a round from, and the digest it commits to
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`), on the genesis
//! rehearsal's model.

mod common;

use std::sync::Arc;
use std::time::Instant;

use common::{Recipe, batch, distro, genesis_recipe, load};
use psyche_core::{
    BatchId, CancellableBarrier, ClosedInterval, ConstantLR, LearningRateSchedule,
    OptimizerDefinition,
};
use psyche_modeling::{
    Batch, BatchData, BatchDataCPU, CausalLM, Distro, LocalTrainer, ParallelModels,
    ROUND_STATE_DOMAIN, Trainer, round_state_digest, write_round_state,
};
use tch::Tensor;
use tokio_util::sync::CancellationToken;

/// A step of `distro` on the batch for `seed`: what fills the buffer.
fn step(recipe: &Recipe, model: &dyn CausalLM, distro: &mut Distro, seed: i64) {
    let tokens = batch(recipe, seed);
    let input = Tensor::from_slice(&tokens).reshape([1, tokens.len() as i64]);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.expect("a forward pass with labels has a loss")
        .backward();
    distro.generate(model, &[], 0.0, recipe.lr, false);
}

/// Reads the tensors of one section of an encoding at `at`: their names and
/// the number of bytes each holds.
fn section(bytes: &[u8], at: &mut usize) -> Vec<(String, usize)> {
    let mut read = |len: usize| {
        let slice = &bytes[*at..*at + len];
        *at += len;
        slice
    };
    let count = u32::from_le_bytes(read(4).try_into().unwrap());
    (0..count)
        .map(|_| {
            let name_len = u32::from_le_bytes(read(4).try_into().unwrap()) as usize;
            let name = String::from_utf8(read(name_len).to_vec()).unwrap();
            let _tag = read(1)[0];
            let dims = u32::from_le_bytes(read(4).try_into().unwrap());
            let shape: Vec<i64> = (0..dims)
                .map(|_| i64::from_le_bytes(read(8).try_into().unwrap()))
                .collect();
            let len = u64::from_le_bytes(read(8).try_into().unwrap()) as usize;
            let _data = read(len);
            let numel: i64 = shape.iter().product();
            assert_eq!(len % numel.max(1) as usize, 0, "{name}");
            (name, len)
        })
        .collect()
}

/// The digest is the BLAKE3 hash of the encoding, which holds every variable
/// of the model, whole, and then the buffer of each, and nothing else.
#[test]
fn the_digest_is_the_hash_of_an_encoding_of_every_variable_and_buffer() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut node = distro(&recipe, &*model);
    step(&recipe, &*model, &mut node, 1);
    let mut bytes = vec![];
    write_round_state(&mut bytes, &*model, &node).unwrap();
    assert!(bytes.starts_with(ROUND_STATE_DOMAIN));
    assert_eq!(
        round_state_digest(&*model, &node).unwrap(),
        *blake3::hash(&bytes).as_bytes()
    );

    let mut at = ROUND_STATE_DOMAIN.len();
    let weights = section(&bytes, &mut at);
    let buffers = section(&bytes, &mut at);
    assert_eq!(at, bytes.len(), "the encoding holds nothing else");
    let variables: Vec<_> = model.variables().collect();
    assert_eq!(weights.len(), variables.len());
    assert_eq!(buffers.len(), variables.len());
    for ((variable, (name, len)), (buffer_name, buffer_len)) in
        variables.iter().zip(&weights).zip(&buffers)
    {
        assert_eq!(name, variable.name());
        assert_eq!(buffer_name, &format!("{}.delta", variable.name()));
        let full: i64 = variable.full_tensor_shape().iter().product();
        let tensor = variable.logical_tensor();
        assert_eq!(*len, full as usize * tensor.kind().elt_size_in_bytes());
        assert_eq!(buffer_len, len);
    }
}

/// Two nodes in one state commit to one digest, and a node's digest follows
/// its weights and its buffer.
#[test]
fn the_digest_names_the_state_and_follows_it() {
    let recipe = genesis_recipe();
    let one = load(&recipe);
    let other = load(&recipe);
    let mut one_distro = distro(&recipe, &*one);
    let other_distro = distro(&recipe, &*other);
    let fresh = round_state_digest(&*one, &one_distro).unwrap();
    assert_eq!(fresh, round_state_digest(&*other, &other_distro).unwrap());
    assert_eq!(fresh, round_state_digest(&*one, &one_distro).unwrap());

    // A step fills the buffer: another state.
    step(&recipe, &*one, &mut one_distro, 1);
    let stepped = round_state_digest(&*one, &one_distro).unwrap();
    assert_ne!(stepped, fresh);

    // One weight moved: another state again.
    {
        let _no_grad = tch::no_grad_guard();
        let variable = one.variables().next().unwrap();
        let mut tensor = variable.logical_tensor();
        let _ = tensor.g_add_scalar_(1.0);
    }
    assert_ne!(round_state_digest(&*one, &one_distro).unwrap(), stepped);
}

/// The genesis rehearsal's DisTrO, at a constant rate.
fn trainer(recipe: &Recipe) -> Trainer {
    let model = load(recipe);
    model.prepare_for_training();
    LocalTrainer::new(
        ParallelModels {
            models: vec![model],
            barrier: Arc::new(CancellableBarrier::new(1)),
            data_parallel: None,
        },
        LearningRateSchedule::Constant(ConstantLR::new(recipe.lr, 0, 0.0)),
        OptimizerDefinition::Distro {
            clip_grad_norm: None,
            weight_decay: None,
            compression_decay: recipe.compression_decay as f32,
            compression_topk: recipe.compression_topk as u16,
            compression_chunk: recipe.compression_chunk as u16,
            quantize_1bit: recipe.quantize_1bit,
            robust_aggregation: recipe.robust_aggregation,
        },
        1,
        None,
        false,
    )
    .into()
}

/// Trains `trainer` on the batch for `seed` at `step`, taking `sent_before`
/// out of its buffer, and returns what it reports.
fn train(
    recipe: &Recipe,
    trainer: Trainer,
    step: u32,
    seed: i64,
    sent_before: Vec<Vec<psyche_modeling::DistroResult>>,
) -> psyche_modeling::TrainOutput {
    let tokens: Vec<i32> = batch(recipe, seed).iter().map(|&t| t as i32).collect();
    trainer
        .train(
            step,
            Batch {
                id: BatchId(ClosedInterval::new(0, 0)),
                data: BatchData::CPU(vec![BatchDataCPU {
                    labels: Some(tokens.clone()),
                    input_ids: tokens,
                    position_ids: None,
                    sequence_lengths: None,
                }]),
            },
            None,
            false,
            vec![],
            Some(sent_before),
            CancellationToken::new(),
        )
        .unwrap()
}

/// A trainer commits each round to the state the round trains from. Its first
/// round trains from the checkpoint and a zeroed buffer, which anyone can
/// rebuild, and the digest it reports is that state's. Its next round trains
/// from another state, and a second trainer that went through the same rounds
/// reports the same digests.
#[test]
fn a_trainer_commits_to_the_state_each_round_trains_from() {
    let recipe = genesis_recipe();
    let checkpoint = load(&recipe);
    let zeroed = distro(&recipe, &*checkpoint);
    let first_state = round_state_digest(&*checkpoint, &zeroed).unwrap();

    let mut digests = vec![];
    for _ in 0..2 {
        let first = train(&recipe, trainer(&recipe), 1, 1, vec![]);
        assert_eq!(first.state_digest, Some(first_state));
        let sent = first.distro_results.clone().unwrap();
        let second = train(&recipe, first.trainer, 2, 2, vec![sent]);
        let second_state = second.state_digest.expect("the trainer commits");
        assert_ne!(second_state, first_state);
        digests.push(second_state);
    }
    assert_eq!(digests[0], digests[1]);
}

/// What committing costs a node every round on the genesis rehearsal's model:
/// copying the state off the device and hashing it. Printed for the record.
#[test]
fn what_committing_to_a_round_costs() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut node = distro(&recipe, &*model);
    step(&recipe, &*model, &mut node, 1);
    let mut bytes = vec![];
    write_round_state(&mut bytes, &*model, &node).unwrap();
    let rounds = 20;
    let started = Instant::now();
    for _ in 0..rounds {
        round_state_digest(&*model, &node).unwrap();
    }
    let each = started.elapsed() / rounds;
    println!(
        "a round's state: {} bytes, committed to in {each:?} on the CPU",
        bytes.len()
    );
}
