//! A node that has trained nothing yet still applies its peers' results. A
//! run that seats verifiers can seat a node as one round after round from
//! its start, and it applies every round's results all the same; no backward
//! pass has reached its variables yet, so none has a gradient, and DisTrO
//! writes the results into each variable's gradient to step on them.

mod common;

use common::{batch, distro, genesis_recipe, load, peer_results};
use psyche_modeling::CausalLM;
use tch::Tensor;

/// A backward pass that leaves every variable a gradient of zeros, as a node
/// that has trained has them once its step zeroes them.
fn give_zero_gradients(model: &dyn CausalLM) {
    tch::with_grad(|| {
        for variable in model.variables() {
            variable
                .logical_tensor()
                .multiply_scalar(0.0)
                .sum(None)
                .backward();
        }
    });
    for variable in model.variables() {
        assert!(variable.logical_tensor().grad().defined());
    }
}

#[test]
fn a_node_that_has_trained_nothing_applies_its_peers_results() {
    let recipe = genesis_recipe();
    let results = peer_results(&recipe, &batch(&recipe, 1));

    let untrained = load(&recipe);
    assert!(
        untrained
            .variables()
            .all(|variable| !variable.logical_tensor().grad().defined()),
        "no backward pass has reached the untrained node's variables"
    );
    distro(&recipe, &*untrained).apply(&*untrained, &[results.clone()], recipe.lr);

    let trained = load(&recipe);
    give_zero_gradients(&*trained);
    distro(&recipe, &*trained).apply(&*trained, &[results], recipe.lr);

    let weights = |model: &dyn CausalLM| -> Vec<Tensor> {
        model
            .variables()
            .map(|variable| variable.logical_tensor().copy())
            .collect()
    };
    for (untrained, trained) in weights(&*untrained).iter().zip(&weights(&*trained)) {
        assert!(
            untrained.equal(trained),
            "the untrained node applied the results otherwise"
        );
    }
    let original = load(&recipe);
    assert!(
        weights(&*untrained)
            .iter()
            .zip(&weights(&*original))
            .any(|(applied, original)| !applied.equal(original)),
        "applying the results moved no weight"
    );
}
