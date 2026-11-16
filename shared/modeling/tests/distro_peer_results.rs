//! A peer's DisTrO results are applied only if they fit the local model
//! (wienerlabs/leviathan-net#41).
//!
//! Everything here is the production path: the genesis rehearsal config's model
//! loaded the way a node loads it, a real forward and backward pass, results
//! from `Distro::generate`, quantized as the config says, and carried through
//! the network serialization. The malformed payloads are what a trainer could
//! commit and sign: its honest results, cut or rearranged.

mod common;

use common::*;
use psyche_modeling::{CompressDCT, DistroResult, DistroResultMismatch, TransformDCT};
use tch::{Device, Kind, Tensor};

/// One result short.
fn fewer_results(honest: &[DistroResult]) -> Vec<DistroResult> {
    honest[..honest.len() - 1].to_vec()
}

/// Two variables of different shapes trade results.
fn swapped_shapes(honest: &[DistroResult]) -> Vec<DistroResult> {
    let mut results = honest.to_vec();
    let other = (1..results.len())
        .find(|i| results[*i].xshape != results[0].xshape)
        .expect("the model has variables of different shapes");
    results.swap(0, other);
    results
}

/// Every index of one variable points one past the end of its encoding, in
/// the 8-bit wire encoding a `totalk` below 256 is sent in.
fn indices_outside(honest: &[DistroResult]) -> Vec<DistroResult> {
    let mut results = honest.to_vec();
    let target = results
        .iter()
        .position(|r| r.totalk < 256)
        .expect("the model has a variable encoded with 8-bit indices");
    let result = &results[target];
    assert_eq!(result.sparse_idx.kind(), Kind::Uint8);
    let outside = Tensor::full(
        result.sparse_idx.size(),
        result.totalk,
        (Kind::Uint8, Device::Cpu),
    );
    results[target] = rebuilt(result, outside, result.sparse_val.copy());
    results
}

/// One variable carries one value fewer than it has indices.
fn counts_differ(honest: &[DistroResult]) -> Vec<DistroResult> {
    let mut results = honest.to_vec();
    let result = &results[0];
    let last = *result.sparse_val.size().last().unwrap();
    let short = result.sparse_val.narrow(-1, 0, last - 1);
    results[0] = rebuilt(result, result.sparse_idx.copy(), short);
    results
}

#[test]
fn a_peer_whose_results_do_not_fit_the_model_is_skipped() {
    let recipe = genesis_recipe();
    let honest = peer_results(&recipe, &batch(&recipe, 3));
    let baseline = applied(&recipe, std::slice::from_ref(&honest)).expect("honest results apply");

    for (name, malformed) in [
        ("fewer results than variables", fewer_results(&honest)),
        ("another variable's shape", swapped_shapes(&honest)),
        ("indices outside totalk", indices_outside(&honest)),
        ("index and value counts that differ", counts_differ(&honest)),
    ] {
        for peers in [
            vec![honest.clone(), malformed.clone()],
            vec![malformed.clone(), honest.clone()],
        ] {
            let after = applied(&recipe, &peers)
                .unwrap_or_else(|panic| panic!("a peer with {name} crashed apply: {panic}"));
            assert!(
                same(&after, &baseline),
                "a peer with {name} changed the model"
            );
        }
        let alone = applied(&recipe, &[malformed])
            .unwrap_or_else(|panic| panic!("a peer with {name} crashed apply alone: {panic}"));
        assert!(
            same(&alone, &fresh(&recipe)),
            "a lone peer with {name} moved the model"
        );
    }
}

#[test]
fn a_second_valid_peer_is_still_applied() {
    let recipe = genesis_recipe();
    let first = peer_results(&recipe, &batch(&recipe, 3));
    let second = peer_results(&recipe, &batch(&recipe, 11));
    let one = applied(&recipe, std::slice::from_ref(&first)).unwrap();
    let two = applied(&recipe, &[first, second]).unwrap();
    assert!(
        !same(&one, &two),
        "a second valid peer's results were not applied"
    );
}

/// Each malformed payload is refused for its own reason, and the honest one is
/// accepted, so the check is not merely refusing everything unusual.
#[test]
fn each_payload_that_does_not_fit_is_named_for_why() {
    let recipe = genesis_recipe();
    let honest = peer_results(&recipe, &batch(&recipe, 3));
    let (model, distro) = trained_node(&recipe);
    assert_eq!(distro.check_results(&*model, &honest), Ok(()));

    assert!(matches!(
        distro.check_results(&*model, &fewer_results(&honest)),
        Err(DistroResultMismatch::Count { .. })
    ));
    assert!(matches!(
        distro.check_results(&*model, &swapped_shapes(&honest)),
        Err(DistroResultMismatch::Shape { .. })
    ));
    assert!(matches!(
        distro.check_results(&*model, &indices_outside(&honest)),
        Err(DistroResultMismatch::Indices { .. })
    ));
    assert!(matches!(
        distro.check_results(&*model, &counts_differ(&honest)),
        Err(DistroResultMismatch::Values { .. })
    ));
}

/// Without 1-bit quantization the values travel as floating point, and a peer
/// that sends a NaN would turn every replica's parameters to NaN through the
/// sign step. It is skipped like any other payload that does not fit.
#[test]
fn a_peer_with_non_finite_values_is_skipped() {
    let recipe = genesis_recipe();
    let honest = peer_results_quantized(&recipe, &batch(&recipe, 3), false);
    let mut poisoned = honest.clone();
    let target = &poisoned[0];
    let nan = target.sparse_val.full_like(f64::NAN);
    poisoned[0] = rebuilt(target, target.sparse_idx.copy(), nan);

    let (model, distro) = trained_node(&recipe);
    assert_eq!(distro.check_results(&*model, &honest), Ok(()));
    assert!(matches!(
        distro.check_results(&*model, &poisoned),
        Err(DistroResultMismatch::Values { .. })
    ));

    let baseline = applied(&recipe, std::slice::from_ref(&honest)).unwrap();
    let after = applied(&recipe, &[honest, poisoned]).unwrap();
    assert!(same(&after, &baseline), "a NaN peer changed the model");
}

/// The encoding the check expects for each variable is exactly what `encode`
/// and `compress` produce for it, on every variable of the real model.
#[test]
fn the_expected_encoding_is_the_real_one() {
    let recipe = genesis_recipe();
    let model = load(&recipe);
    let mut transform = TransformDCT::new(model.variables(), recipe.compression_chunk);
    for variable in model.variables() {
        let shape = variable.logical_tensor().size();
        let expected = transform
            .encoded_shape(&shape)
            .unwrap_or_else(|| panic!("{} has no encoding", variable.name()));
        let encoded =
            transform.encode(&Tensor::zeros(shape.as_slice(), (Kind::Float, Device::Cpu)));
        assert_eq!(encoded.size(), expected, "{}", variable.name());
        let (_, _, xshape, totalk) = CompressDCT::compress(&encoded, recipe.compression_topk);
        assert_eq!(xshape, expected, "{}", variable.name());
        let flat_last = match expected.len() {
            n if n > 2 => expected[n - 2] * expected[n - 1],
            n => expected[n - 1],
        };
        assert_eq!(totalk, flat_last, "{}", variable.name());
    }
}

/// The largest magnitude a value may have, read in fp32 for a variable of
/// `xshape`, mirroring what the check derives from fp32's range.
fn safe_bound(result: &DistroResult) -> f32 {
    let numel: i64 = result.xshape.iter().product();
    (f32::MAX / numel as f32).sqrt() / 2.0
}

/// A finite value can still overflow once `apply` sums it through the DCT
/// decode or squares it under robust aggregation. A peer whose values reach
/// past fp32's safe range for the variable is skipped; one at that range is
/// applied and leaves every parameter finite.
#[test]
fn values_beyond_the_safe_range_are_skipped_and_values_within_it_stay_finite() {
    let recipe = genesis_recipe();
    let honest = peer_results_quantized(&recipe, &batch(&recipe, 3), false);
    let target = &honest[0];
    let bound = safe_bound(target);

    let mut beyond = honest.clone();
    beyond[0] = rebuilt(
        target,
        target.sparse_idx.copy(),
        target.sparse_val.full_like(2.0 * bound as f64),
    );
    let mut at_bound = honest.clone();
    at_bound[0] = rebuilt(
        target,
        target.sparse_idx.copy(),
        target
            .sparse_val
            .to_kind(Kind::Float)
            .full_like(bound as f64),
    );

    let (model, distro) = trained_node(&recipe);
    assert!(matches!(
        distro.check_results(&*model, &beyond),
        Err(DistroResultMismatch::Values { .. })
    ));
    assert_eq!(distro.check_results(&*model, &at_bound), Ok(()));

    // Under either rule a run can choose: the mean sums them through the
    // decode, robust aggregation also squares them to measure each peer.
    for robust_aggregation in [false, true] {
        let after = applied(
            &recipe.aggregating(robust_aggregation),
            &[honest.clone(), at_bound.clone()],
        )
        .unwrap();
        assert!(
            after
                .iter()
                .all(|parameter| bool::try_from(parameter.isfinite().all()).unwrap()),
            "values at the safe bound left a parameter non-finite (robust: {robust_aggregation})"
        );
    }
}
