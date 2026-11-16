//! How a node combines its peers' results is the run's rule, read from chain,
//! and nothing else: it used to be whatever `LEVIATHAN_ROBUST_AGG` said in the
//! process a node ran in, so two honest nodes of one run could apply different
//! updates to the same results and silently hold different models
//! (wienerlabs/leviathan-net#60). Its own test binary, because it sets that
//! variable for the whole process to show it is ignored.

mod common;

use common::*;

#[test]
fn the_runs_rule_alone_decides_how_peers_are_combined() {
    // Set before any `apply` runs in this process, to show it decides nothing.
    unsafe { std::env::set_var("LEVIATHAN_ROBUST_AGG", "1") };

    let genesis = genesis_recipe();
    assert!(
        genesis.robust_aggregation,
        "the genesis rehearsal aggregates robustly"
    );
    let peers = [
        peer_results(&genesis, &batch(&genesis, 3)),
        peer_results(&genesis, &batch(&genesis, 11)),
        peer_results(&genesis, &batch(&genesis, 17)),
    ];

    // Two nodes of a run that aggregates robustly apply the same update.
    let robust = applied(&genesis.aggregating(true), &peers).unwrap();
    let robust_again = applied(&genesis.aggregating(true), &peers).unwrap();
    assert!(same(&robust, &robust_again));

    // Two nodes of a run that takes the mean apply the same update too, the
    // environment saying otherwise notwithstanding.
    let mean = applied(&genesis.aggregating(false), &peers).unwrap();
    let mean_again = applied(&genesis.aggregating(false), &peers).unwrap();
    assert!(same(&mean, &mean_again));

    // And the rule is what tells them apart: robust aggregation clips each
    // peer's deviation, so it is not the mean even over honest peers.
    assert!(
        !same(&robust, &mean),
        "the two rules applied the same update, so the run's rule decided nothing"
    );
}
