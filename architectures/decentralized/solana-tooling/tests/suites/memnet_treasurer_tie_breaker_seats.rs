//! A run with an appeals court reserves tie-breaker seats in every round, and
//! every party has to draw the same committee from it: the clients, which
//! train the batches their own draw assigns them, the audit lottery, which
//! hands verifiers a target and a batch to replay, and the treasurer, which
//! takes the votes. The clients draw with the round's `tie_breaker_tasks`, so
//! the round has to carry the run's count.

use psyche_coordinator::assign_data_for_state;
use psyche_coordinator::select_audits_for_current_round;
use psyche_coordinator::Committee;
use psyche_coordinator::CommitteeSelection;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_challenge_config;

use super::appeals_harness::*;
use super::chain_state::coordinator_state;

/// The round the appeals run trains reserves the run's tie-breaker seats, so
/// every audit the lottery hands out names a batch its target was assigned,
/// and trains, and no seat the treasurer counts as a tie-breaker is given
/// data to train.
#[tokio::test]
pub async fn every_party_draws_the_round_with_the_runs_tie_breaker_seats() {
    let mut h = setup("Leviathan tie-breaker seats", 171).await;
    let coordinator =
        coordinator_state(&mut h.endpoint, &h.coordinator_account)
            .await
            .coordinator;
    let round = coordinator.current_round().unwrap();
    assert_eq!(
        round.tie_breaker_tasks, TIE_BREAKER_SIZE,
        "the round reserves the run's tie-breaker seats"
    );

    // What the clients train: the batches their own draw assigns.
    let clients_draw = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
    let trained = assign_data_for_state(&coordinator, &clients_draw);
    for (index, client) in coordinator.epoch_state.clients.iter().enumerate() {
        if clients_draw.get_committee(index as u64).committee == Committee::TieBreaker {
            assert!(
                !trained.values().any(|trainer| *trainer == client.id),
                "a tie-breaker seat was given data to train"
            );
        }
    }

    // What the audit lottery hands the verifiers.
    let audits = audits_now(&coordinator);
    assert!(!audits.is_empty(), "the round draws audits");
    for audit in &audits {
        assert_eq!(
            trained.get(&audit.batch_id),
            Some(&audit.target),
            "an audit names a batch its target never trained"
        );
    }
}

/// A new count is a new draw, so it takes effect from the next round: the
/// round under way keeps the seats its clients drew.
#[tokio::test]
pub async fn a_new_tie_breaker_count_takes_effect_from_the_next_round() {
    let mut h = setup("Leviathan tie-breaker change", 172).await;
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_set_challenge_config(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        CHALLENGE_WINDOW,
        1,
        APPEAL_WINDOW,
    )
    .await
    .unwrap();
    let now = coordinator_state(&mut h.endpoint, &h.coordinator_account)
        .await
        .coordinator;
    assert_eq!(now.current_round().unwrap().tie_breaker_tasks, TIE_BREAKER_SIZE);
    h.advance_to_next_round().await;
    let next = coordinator_state(&mut h.endpoint, &h.coordinator_account)
        .await
        .coordinator;
    assert_eq!(next.current_round().unwrap().tie_breaker_tasks, 1);
}

fn audits_now(
    coordinator: &psyche_coordinator::Coordinator,
) -> Vec<psyche_coordinator::AuditAssignment> {
    select_audits_for_current_round(coordinator).unwrap()
}
