//! A DisTrO optimizer hands out its error-feedback buffer and takes one in,
//! so a verifier can start a round from the buffer a node reveals
//! (wienerlabs/leviathan-net#48). On the genesis rehearsal's model, loaded the
//! way a node loads it.

mod common;

use common::{Recipe, batch, distro, genesis_recipe, load, load_in};
use psyche_modeling::{CausalLM, Distro, DistroResult, ErrorFeedbackMismatch};
use tch::{Kind, Tensor};

/// A step of `distro` on the batch for `seed`, at a freshly loaded model,
/// taking `sent_before` out of its buffer.
fn step(
    recipe: &Recipe,
    distro: &mut Distro,
    seed: i64,
    sent_before: &[Vec<DistroResult>],
    prev_lr: f64,
) -> Vec<DistroResult> {
    let model = load(recipe);
    let tokens = batch(recipe, seed);
    let input = Tensor::from_slice(&tokens).reshape([1, tokens.len() as i64]);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.expect("a forward pass with labels has a loss")
        .backward();
    distro.generate(&*model, sent_before, prev_lr, recipe.lr, false)
}

fn same(left: &[Tensor], right: &[Tensor]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.equal(right))
}

fn same_results(left: &[DistroResult], right: &[DistroResult]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.sparse_idx.equal(&right.sparse_idx) && left.sparse_val.equal(&right.sparse_val)
        })
}

fn optimizer(recipe: &Recipe) -> Distro {
    let model: Box<dyn CausalLM> = load(recipe);
    distro(recipe, &*model)
}

#[test]
fn a_buffer_handed_over_carries_the_next_round_on_exactly() {
    let recipe = genesis_recipe();
    let mut node = optimizer(&recipe);
    let sent = step(&recipe, &mut node, 1, &[], 0.0);
    let buffer = node.error_feedback();
    assert!(
        buffer
            .iter()
            .any(|tensor| tensor.abs().sum(None).double_value(&[]) > 0.0),
        "a step leaves something unsent in the buffer"
    );

    let mut verifier = optimizer(&recipe);
    verifier.load_error_feedback(&buffer).unwrap();
    assert!(same(&verifier.error_feedback(), &buffer));

    // The next round from the same weights and data: the verifier sends what
    // the node sends, and a verifier without the buffer does not.
    let sent_before = vec![sent];
    let from_node = step(&recipe, &mut node, 2, &sent_before, recipe.lr);
    let from_verifier = step(&recipe, &mut verifier, 2, &sent_before, recipe.lr);
    assert!(same_results(&from_node, &from_verifier));
    let mut without = optimizer(&recipe);
    let from_scratch = step(&recipe, &mut without, 2, &sent_before, recipe.lr);
    assert!(!same_results(&from_node, &from_scratch));
}

#[test]
fn a_buffer_that_does_not_fit_changes_nothing() {
    let recipe = genesis_recipe();
    let mut node = optimizer(&recipe);
    step(&recipe, &mut node, 1, &[], 0.0);
    let buffer = node.error_feedback();

    let mut verifier = optimizer(&recipe);
    let before = verifier.error_feedback();
    assert_eq!(
        verifier.load_error_feedback(&buffer[1..]),
        Err(ErrorFeedbackMismatch::Count {
            given: buffer.len() - 1,
            expected: buffer.len()
        })
    );
    let mut reshaped: Vec<Tensor> = buffer.iter().map(Tensor::copy).collect();
    // The last matrix, so a load that wrote before it checked would already
    // have written every buffer ahead of it.
    let matrix = reshaped
        .iter()
        .rposition(|tensor| tensor.dim() >= 2)
        .expect("the model has a matrix");
    assert!(
        matrix > 0,
        "a buffer goes ahead of the one that does not fit"
    );
    reshaped[matrix] = reshaped[matrix].flatten(0, -1);
    assert!(matches!(
        verifier.load_error_feedback(&reshaped),
        Err(ErrorFeedbackMismatch::Shape { index, .. }) if index == matrix
    ));
    assert!(same(&verifier.error_feedback(), &before));
}

/// A bf16 node's step, at a freshly loaded bf16 model, with `lr`.
fn bf16_step(recipe: &Recipe, distro: &mut Distro, seed: i64, lr: f64) -> Vec<DistroResult> {
    let model = load_in(recipe, Kind::BFloat16);
    let tokens = batch(recipe, seed);
    let input = Tensor::from_slice(&tokens).reshape([1, tokens.len() as i64]);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.expect("a forward pass with labels has a loss")
        .backward();
    distro.generate(&*model, &[], 0.0, lr, false)
}

/// The buffer after a step that adds nothing to it: a step at a learning rate
/// of 0, with nothing sent before to take out, leaves `decay` times the
/// buffer (wienerlabs/leviathan-net#90).
fn decayed_once(recipe: &Recipe, distro: &mut Distro) -> (Vec<Tensor>, Vec<Tensor>) {
    bf16_step(recipe, distro, 1, recipe.lr);
    let before = distro.error_feedback();
    assert!(
        before
            .iter()
            .any(|t| t.abs().sum(None).double_value(&[]) > 0.0),
        "a step leaves something in the buffer"
    );
    bf16_step(recipe, distro, 2, 0.0);
    (before, distro.error_feedback())
}

/// A node trains in bf16, and its buffer is kept in bf16 too. Multiplying a
/// bf16 value by the runs' 0.999 moves it by less than half the gap to the
/// next bf16 value, so the decay leaves every value as it was
/// (wienerlabs/leviathan-net#90).
#[test]
fn in_bf16_the_decay_leaves_the_buffer_as_it_was() {
    let recipe = genesis_recipe();
    assert_eq!(recipe.compression_decay, 0.999_f32 as f64);
    let model = load_in(&recipe, Kind::BFloat16);
    let mut node = distro(&recipe, &*model);
    let (before, after) = decayed_once(&recipe, &mut node);
    assert_eq!(before[0].kind(), Kind::BFloat16);
    assert!(same(&before, &after), "a bf16 buffer decayed");
}

/// Kept in fp32, the same buffer decays by the run's factor.
#[test]
fn a_buffer_kept_in_fp32_decays() {
    let recipe = genesis_recipe();
    let model = load_in(&recipe, Kind::BFloat16);
    let mut node = distro(&recipe, &*model);
    node.set_error_feedback_kind(Kind::Float).unwrap();
    let (before, after) = decayed_once(&recipe, &mut node);
    assert_eq!(before[0].kind(), Kind::Float);
    assert!(!same(&before, &after), "an fp32 buffer did not decay");
    for (before, after) in before.iter().zip(&after) {
        let expected = before * recipe.compression_decay;
        assert!(after.allclose(&expected, 1e-6, 1e-12, false));
    }
}

/// A buffer kept in fp32 is cast to the parameter's dtype only to be encoded,
/// so what the node sends is still the run's encoding, in the model's dtype.
#[test]
fn a_buffer_kept_in_fp32_still_sends_the_runs_encoding() {
    let recipe = genesis_recipe();
    let model = load_in(&recipe, Kind::BFloat16);
    let mut plain = distro(&recipe, &*model);
    let mut kept = distro(&recipe, &*model);
    kept.set_error_feedback_kind(Kind::Float).unwrap();
    let sent = bf16_step(&recipe, &mut plain, 1, recipe.lr);
    let sent_kept = bf16_step(&recipe, &mut kept, 1, recipe.lr);
    assert_eq!(Distro::check_encodings(&sent_kept), Ok(()));
    for (sent, kept) in sent.iter().zip(&sent_kept) {
        assert_eq!(kept.sparse_val.kind(), sent.sparse_val.kind());
        assert_eq!(kept.sparse_idx.size(), sent.sparse_idx.size());
        assert_eq!(kept.xshape, sent.xshape);
    }
    // From a zeroed buffer, the first step adds only the gradient, so both
    // pick the same coordinates.
    assert!(same_results(&sent, &sent_kept));
}

/// `lr` times the gradient a freshly loaded bf16 model takes on the batch for
/// `seed`, in bf16, as `generate` adds it to the buffer.
fn bf16_increment(recipe: &Recipe, seed: i64, lr: f64) -> Vec<Tensor> {
    let model = load_in(recipe, Kind::BFloat16);
    let tokens = batch(recipe, seed);
    let input = Tensor::from_slice(&tokens).reshape([1, tokens.len() as i64]);
    let (_, loss) = model.forward(&input, Some(&input), None, None, None, None);
    loss.expect("a forward pass with labels has a loss")
        .backward();
    model
        .variables()
        .map(|variable| variable.logical_tensor().grad().multiply_scalar(lr))
        .collect()
}

/// Two steps at a decay of 1, so nothing but what the second adds can move
/// the buffer: one at the run's rate, then one at 10⁻⁵ of it, with
/// the buffer kept in `kind` if given. Returns the buffer between the two,
/// after them, and what the second adds (wienerlabs/leviathan-net#90).
fn a_small_increment(kind: Option<Kind>) -> (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>) {
    let recipe = Recipe {
        compression_decay: 1.0,
        ..genesis_recipe()
    };
    let model = load_in(&recipe, Kind::BFloat16);
    let mut node = distro(&recipe, &*model);
    if let Some(kind) = kind {
        node.set_error_feedback_kind(kind).unwrap();
    }
    bf16_step(&recipe, &mut node, 1, recipe.lr);
    let before = node.error_feedback();
    // Where the two gradients are of a size, the second step adds about 10⁻⁵
    // of a value: well under half the gap between bf16 values, 2⁻⁹ to 2⁻⁸ of
    // it, and well over the gap between fp32 ones, at most 2⁻²³ of it.
    let small_lr = recipe.lr * 1e-5;
    bf16_step(&recipe, &mut node, 2, small_lr);
    let increment = bf16_increment(&recipe, 2, small_lr);
    (before, node.error_feedback(), increment)
}

/// The coordinates whose increment is far smaller than their value: below
/// 2⁻¹⁰ of it, under half the gap between bf16 values there, and above 2⁻²⁰
/// of it, over the gap between fp32 values.
fn far_smaller(before: &Tensor, increment: &Tensor) -> Tensor {
    let value = before.to_kind(Kind::Float).abs();
    let increment = increment.to_kind(Kind::Float).abs();
    increment
        .lt_tensor(&(&value * 2f64.powi(-10)))
        .logical_and(&increment.gt_tensor(&(&value * 2f64.powi(-20))))
}

/// Whether `far_smaller` picks most of the buffer's coordinates, over every
/// variable, so that what holds for them holds for the buffer at large.
fn most_of_the_buffer(masks: &[Tensor]) -> bool {
    let picked: i64 = masks
        .iter()
        .map(|mask| mask.sum(Kind::Int64).int64_value(&[]))
        .sum();
    let total: i64 = masks.iter().map(|mask| mask.numel() as i64).sum();
    2 * picked > total
}

/// In bf16, a buffer value takes nothing from an increment far smaller than
/// it, so a gradient that keeps arriving for a coordinate top-k does not pick
/// stops registering once the coordinate is large (wienerlabs/leviathan-net#90).
#[test]
fn in_bf16_an_increment_far_smaller_than_a_value_is_absorbed() {
    let (before, after, increment) = a_small_increment(None);
    assert_eq!(before[0].kind(), Kind::BFloat16);
    let masks: Vec<_> = before
        .iter()
        .zip(&increment)
        .map(|(before, increment)| far_smaller(before, increment))
        .collect();
    assert!(most_of_the_buffer(&masks), "too few coordinates to show it");
    for ((before, after), mask) in before.iter().zip(&after).zip(&masks) {
        assert!(
            after.masked_select(mask).equal(&before.masked_select(mask)),
            "a bf16 buffer value took an increment far smaller than it"
        );
    }
}

/// Kept in fp32, every one of those values takes its increment.
#[test]
fn a_buffer_kept_in_fp32_takes_an_increment_far_smaller_than_a_value() {
    let (before, after, increment) = a_small_increment(Some(Kind::Float));
    assert_eq!(before[0].kind(), Kind::Float);
    let masks: Vec<_> = before
        .iter()
        .zip(&increment)
        .map(|(before, increment)| far_smaller(before, increment))
        .collect();
    assert!(most_of_the_buffer(&masks), "too few coordinates to show it");
    for (((before, after), increment), mask) in
        before.iter().zip(&after).zip(&increment).zip(&masks)
    {
        let expected = before + increment.to_kind(Kind::Float);
        assert!(after.allclose(&expected, 1e-6, 0.0, false));
        let moved = after
            .masked_select(mask)
            .ne_tensor(&before.masked_select(mask));
        assert!(
            moved.all().int64_value(&[]) != 0,
            "an fp32 buffer value did not take its increment"
        );
    }
}

/// Moving the buffer to fp32 starts it over from zeros, each buffer in its
/// own shape and on its own device, whatever it held before.
#[test]
fn a_buffer_moved_to_fp32_starts_from_zeros_in_its_shape() {
    let recipe = genesis_recipe();
    let model = load_in(&recipe, Kind::BFloat16);
    let mut node = distro(&recipe, &*model);
    bf16_step(&recipe, &mut node, 1, recipe.lr);
    let before = node.error_feedback();
    assert!(
        before
            .iter()
            .any(|t| t.abs().sum(None).double_value(&[]) > 0.0),
        "a step leaves something in the buffer"
    );
    node.set_error_feedback_kind(Kind::Float).unwrap();
    let after = node.error_feedback();
    assert_eq!(after.len(), before.len());
    for (before, after) in before.iter().zip(&after) {
        assert_eq!(after.kind(), Kind::Float);
        assert_eq!(after.size(), before.size());
        assert_eq!(after.device(), before.device());
        assert_eq!(after.abs().sum(None).double_value(&[]), 0.0);
    }
}
