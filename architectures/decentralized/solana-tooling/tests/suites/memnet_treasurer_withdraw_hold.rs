//! A bond cannot leave while it may still owe a conviction
//! (wienerlabs/leviathan-net#36).
//!
//! A slash reaches the coordinator ledger only at the epoch boundary, so a node
//! training in an epoch may still be charged for it until that epoch ends.

use psyche_coordinator::RunState;
use psyche_solana_coordinator::ProgramError as CoordinatorProgramError;
use psyche_solana_treasurer::ProgramError;

use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::BOND;
use super::bonded_run::MAX_TRANSITIONS;
use super::bonded_run::NODES;
use super::bonded_run::SLASHING_RATE;
use super::bonded_run::SUBJECT;
use super::bonded_run::WITHDRAW_DELAY;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

#[tokio::test]
pub async fn a_node_cannot_leave_while_its_epoch_is_unsettled() {
    let mut harness =
        Harness::new(360, "Leviathan withdraw hold epoch", JoinGate::Bond)
            .await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;

    // The node asks for its whole bond back as soon as it is training, and the
    // delay elapses while the epoch is still running.
    harness.request_withdraw(SUBJECT, BOND).await;
    harness
        .endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    assert_eq!(
        refusal_code(harness.try_finalize_withdraw(SUBJECT).await),
        treasurer_refusal(ProgramError::EpochNotSettled),
        "a node left with its bond in the middle of an epoch it could still be slashed for"
    );

    // Convicted in that same epoch, it pays for it once the epoch is charged.
    harness.convict(SUBJECT).await;
    harness.finish_epoch().await;
    let before = harness.collateral(SUBJECT).await;
    harness.finalize_withdraw(SUBJECT).await;
    assert_eq!(
        harness.collateral(SUBJECT).await - before,
        BOND - SLASHING_RATE
    );
}

/// An honest node that asks for its bond while it trains is held through every
/// phase of the epoch, because a slash it earns in any of them is charged only
/// when the epoch ends, and then gets all of it back without training again.
#[tokio::test]
pub async fn an_honest_node_leaves_once_its_epoch_is_charged() {
    let mut harness =
        Harness::new(361, "Leviathan withdraw hold honest", JoinGate::Bond)
            .await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.request_withdraw(SUBJECT, BOND).await;
    harness
        .endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();

    for phase in [
        RunState::Warmup,
        RunState::RoundTrain,
        RunState::RoundWitness,
        RunState::Cooldown,
    ] {
        harness.advance_to(phase).await;
        assert!(harness.roster().await.contains(&SUBJECT));
        assert_eq!(
            refusal_code(harness.try_finalize_withdraw(SUBJECT).await),
            treasurer_refusal(ProgramError::EpochNotSettled),
            "the node left during {phase}, before its epoch was charged"
        );
    }

    harness.finish_epoch().await;
    assert_eq!(
        harness.state().await.coordinator.run_state,
        RunState::WaitingForMembers
    );
    let before = harness.collateral(SUBJECT).await;
    harness.finalize_withdraw(SUBJECT).await;
    assert_eq!(harness.collateral(SUBJECT).await - before, BOND);
}

/// A node that is not training in the current epoch is not held: one that sat
/// the epoch out leaves while the others are still mid-round.
#[tokio::test]
pub async fn a_node_outside_the_epoch_is_not_held() {
    let mut harness =
        Harness::new(362, "Leviathan withdraw hold outside", JoinGate::Bond)
            .await;
    for node in 0..NODES {
        if node != SUBJECT {
            harness.join(node).await.unwrap();
        }
    }
    harness.advance_to(RunState::RoundTrain).await;
    assert!(!harness.roster().await.contains(&SUBJECT));

    harness.request_withdraw(SUBJECT, BOND).await;
    harness
        .endpoint
        .forward_clock_unix_timestamp(WITHDRAW_DELAY as u64)
        .await
        .unwrap();
    let before = harness.collateral(SUBJECT).await;
    harness.finalize_withdraw(SUBJECT).await;
    assert_eq!(harness.collateral(SUBJECT).await - before, BOND);
}

/// From `RoundTrain`, `RoundWitness` and `Cooldown` every way out of the epoch
/// passes through the charge, so an ejection there is charged
/// (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn an_ejection_where_its_charge_is_certain_is_charged() {
    for (index, phase) in [
        RunState::RoundTrain,
        RunState::RoundWitness,
        RunState::Cooldown,
    ]
    .into_iter()
    .enumerate()
    {
        let mut harness = Harness::new(
            363 + index as u64,
            &format!("Leviathan charged ejection {index}"),
            JoinGate::Bond,
        )
        .await;
        for node in 0..NODES {
            harness.join(node).await.unwrap();
        }
        harness.advance_to(phase).await;
        harness.convict(SUBJECT).await;
        harness.finish_epoch().await;
        assert_eq!(
            harness.ledger(SUBJECT).await.1,
            SLASHING_RATE,
            "an ejection in {phase} was not charged"
        );
    }
}

/// A pause asked for mid-epoch waits for the epoch to end, through the
/// charge, so it does not shed an ejection made before it
/// (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn a_pause_after_an_ejection_still_charges_it() {
    let mut harness =
        Harness::new(366, "Leviathan paused after ejection", JoinGate::Bond)
            .await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::RoundTrain).await;
    harness.convict(SUBJECT).await;
    harness.pause().await;
    for _ in 0..MAX_TRANSITIONS {
        if harness.state().await.coordinator.run_state == RunState::Paused {
            break;
        }
        harness.step().await;
    }
    assert_eq!(
        harness.state().await.coordinator.run_state,
        RunState::Paused
    );
    assert_eq!(harness.ledger(SUBJECT).await.1, SLASHING_RATE);
}

/// A warmup that is paused goes straight to `Paused` with no charge, so no
/// ejection lands in a warmup at all (wienerlabs/leviathan-net#36).
#[tokio::test]
pub async fn no_ejection_lands_in_a_warmup() {
    let mut harness =
        Harness::new(367, "Leviathan warmup ejection", JoinGate::Bond).await;
    for node in 0..NODES {
        harness.join(node).await.unwrap();
    }
    harness.advance_to(RunState::Warmup).await;
    assert_eq!(
        refusal_code(harness.try_convict(SUBJECT).await),
        u32::from(CoordinatorProgramError::CoordinatorErrorSlashOutsideEpoch),
    );
}
