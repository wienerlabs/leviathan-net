//! Robust aggregation squares every value of a variable when it measures a
//! peer's distance, so the safe range the check enforces has to hold there as
//! well (wienerlabs/leviathan-net#41). Its own test binary, because it turns
//! robust aggregation on for the whole process.

mod common;

use common::*;
use tch::Kind;

#[test]
fn values_at_the_safe_bound_stay_finite_under_robust_aggregation() {
    // Set before any `apply` runs in this process; nothing else here reads it.
    unsafe { std::env::set_var("LEVIATHAN_ROBUST_AGG", "1") };

    let recipe = genesis_recipe();
    let honest = peer_results_quantized(&recipe, &batch(&recipe, 3), false);
    let second = peer_results_quantized(&recipe, &batch(&recipe, 11), false);
    let mut at_bound = honest.clone();
    for (index, result) in honest.iter().enumerate() {
        let numel: i64 = result.xshape.iter().product();
        let bound = (f32::MAX / numel as f32).sqrt() / 2.0;
        at_bound[index] = rebuilt(
            result,
            result.sparse_idx.copy(),
            result
                .sparse_val
                .to_kind(Kind::Float)
                .full_like(bound as f64),
        );
    }

    let (model, distro) = trained_node(&recipe);
    assert_eq!(distro.check_results(&*model, &at_bound), Ok(()));
    let after = applied(&recipe, &[honest, second, at_bound]).unwrap();
    assert!(
        after
            .iter()
            .all(|parameter| bool::try_from(parameter.isfinite().all()).unwrap()),
        "values at the safe bound left a parameter non-finite under robust aggregation"
    );
}
