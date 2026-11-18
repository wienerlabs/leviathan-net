//! A DisTrO optimizer hands out its error-feedback buffer and takes one in,
//! so a verifier can start a round from the buffer a node reveals
//! (wienerlabs/leviathan-net#48). On the genesis rehearsal's model, loaded the
//! way a node loads it.

mod common;

use common::{Recipe, batch, distro, genesis_recipe, load};
use psyche_modeling::{CausalLM, Distro, DistroResult, ErrorFeedbackMismatch};
use tch::Tensor;

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
