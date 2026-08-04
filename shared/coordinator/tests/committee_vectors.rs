//! Vectors the web port of the committee lottery is checked against.
//!
//! `leviathan-web/src/data/committee.ts` reimplements the swap-or-not shuffle
//! so the dashboard can show who drew which role without running a node. A port
//! that is subtly wrong still produces roles, just the wrong ones, so it is
//! checked against this file rather than trusted.
//!
//! Regenerate with `UPDATE_COMMITTEE_VECTORS=1 cargo test -p psyche-coordinator
//! --test committee_vectors`, then copy it to the web repo.

use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Case {
    tie_breaker_nodes: usize,
    witness_nodes: usize,
    verification_percent: u8,
    total_nodes: usize,
    /// A string because these overflow the double a JSON number becomes in the
    /// browser, and a seed that silently loses its low bits shuffles into a
    /// different committee than the chain did.
    random_seed: String,
    verifier_nodes: u64,
    assignments: Vec<Assignment>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Assignment {
    index: u64,
    position: u64,
    role: String,
    witness: bool,
}

fn role(committee: Committee) -> String {
    match committee {
        Committee::TieBreaker => "Tie breaker",
        Committee::Verifier => "Verifier",
        Committee::Trainer => "Trainer",
    }
    .to_string()
}

fn cases() -> Vec<Case> {
    // Seeds are the ones live devnet rounds actually drew, plus edge shapes:
    // a committee with no verifiers, one where every free node verifies, and
    // the single-node run that made a quorum of one possible before
    // wienerlabs/leviathan#15 finding 11.
    let shapes: &[(usize, usize, u8, usize, u64)] = &[
        (0, 0, 0, 1, 11714962589530486504),
        (0, 0, 50, 6, 1924996481851443602),
        (3, 4, 50, 12, 855779524057918524),
        (2, 8, 100, 20, 14660810609320035160),
        (0, 32, 33, 64, 17594232420418697730),
        (5, 0, 67, 33, 15982172444493427426),
        (1, 1, 10, 2, 0),
    ];

    shapes
        .iter()
        .map(|&(tie, witness, percent, total, seed)| {
            let selection =
                CommitteeSelection::new(tie, witness, percent, total, seed).unwrap();
            let assignments = (0..total as u64)
                .map(|index| {
                    let committee = selection.get_committee(index);
                    Assignment {
                        index,
                        position: committee.position,
                        role: role(committee.committee),
                        witness: selection.get_witness(index).witness.into(),
                    }
                })
                .collect();
            Case {
                tie_breaker_nodes: tie,
                witness_nodes: witness,
                verification_percent: percent,
                total_nodes: total,
                random_seed: seed.to_string(),
                verifier_nodes: selection.get_num_verifier_nodes(),
                assignments,
            }
        })
        .collect()
}

#[test]
fn vectors_match_the_committed_file() {
    let actual = cases();

    if std::env::var("UPDATE_COMMITTEE_VECTORS").is_ok() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/committee_vectors.json");
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&actual).unwrap()),
        )
        .unwrap();
        println!("wrote {path}");
        return;
    }

    let expected: Vec<Case> =
        serde_json::from_str(include_str!("committee_vectors.json"))
            .expect("committee_vectors.json is not valid JSON");
    assert_eq!(
        actual, expected,
        "the shuffle changed, so the web port in leviathan-web is now wrong"
    );
}
