//! A fraud in an epoch's final round has until that epoch is charged to be
//! convicted: the round's witness phase and the epoch's cooldown. That window
//! is a run parameter the operators size against a verifier's measured audit
//! latency (wienerlabs/leviathan-net#40).

use leviathan_indexer::live::read_run;
use leviathan_indexer::verdicts::VerdictLedger;
use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_challenge_config;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::signer::Signer;

use super::appeals_harness::*;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::replayed_digest;

/// The harness run with no challenge window: a quorum convicts at once, so a
/// conviction is charged with the epoch it is reached in.
async fn convicting_at_once(run_id: &str, index: u64) -> AppealsHarness {
    let mut h = setup(run_id, index).await;
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_set_challenge_config(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        0,
        0,
        0,
    )
    .await
    .unwrap();
    h
}

/// The run authority slashes `client`, at `index`, on what it signed in the
/// round under way: one client leaving the roster this round.
async fn authority_slashes(h: &mut AppealsHarness, client: usize, index: u64) {
    let commitment = h.signed_payload_of(client).await;
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_slash(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        index,
        &commitment,
        replayed_digest(),
    )
    .await
    .unwrap();
}

/// The final round's committee convicts a trainer of that round during the
/// epoch's cooldown, and the epoch's own charge reaches the target. The
/// indexer reads the window it had off the run's config. Once the epoch is
/// over, a fraud from it can no longer be convicted.
#[tokio::test]
pub async fn a_final_round_fraud_is_convicted_before_its_epoch_is_charged() {
    let mut h = setup("Leviathan final round audit", 140).await;
    // A quorum convicts at once, so the conviction is charged with this epoch.
    let main_authority = h.main_authority.insecure_clone();
    process_treasurer_run_set_challenge_config(
        &mut h.endpoint,
        &h.payer,
        &main_authority,
        &h.run,
        0,
        0,
        0,
    )
    .await
    .unwrap();

    h.advance_to_cooldown_with_witnesses().await;
    assert_eq!(h.run_state().await, RunState::Cooldown);
    let epoch = h.epoch().await;

    // The window this round had is what the indexer exports: the round's
    // witness phase and the epoch's cooldown, read from the run on chain.
    let config = h.config().await;
    let run_id = h.run_id.clone();
    let observation = read_run(&mut h.endpoint, None, &run_id)
        .await
        .unwrap()
        .observe(&mut VerdictLedger::default());
    assert_eq!(
        observation.final_round_audit_window_seconds,
        config.round_witness_time + config.cooldown_time,
    );

    // The committee the final round drew is still in force in the cooldown.
    let committee = h.committee_now().await;
    let (target, target_index) = committee.trainers[0];
    let (late, _) = committee.trainers[1];
    let target_key = h.clients[target].pubkey();
    let late_key = h.clients[late].pubkey();
    // Both signed for the batches they trained in the final round.
    let target_commitment = h.signed_payload_of(target).await;
    let late_commitment = h.signed_payload_of(late).await;
    for (voter, _) in committee.verifiers.iter().take(committee.verifier_quorum)
    {
        h.convict_on(*voter, target, target_index, &target_commitment)
            .await
            .unwrap();
    }
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Upheld);
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected)
    );
    assert_eq!(
        h.run_state().await,
        RunState::Cooldown,
        "convicted inside the window"
    );

    // The epoch's own charge reaches the target.
    h.finish_epoch_with_witnesses().await;
    assert!(h.epoch().await > epoch);
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    assert_eq!(h.slashed_of(&late_key).await, 0);

    // Past the window the same fraud cannot be convicted: the next epoch's
    // committee refuses a commitment from the epoch already charged.
    h.rejoin_all_and_start_epoch().await;
    let committee = h.committee_now().await;
    let late_index = h.roster_index_of(&late_key).await;
    // Neither the late trainer nor the convicted target, whose bond the
    // conviction already charged below the run's minimum.
    let (voter, _) = *committee
        .verifiers
        .iter()
        .find(|(client_idx, _)| *client_idx != late && *client_idx != target)
        .expect("a bonded verifier other than the two trainers");
    assert_eq!(
        refusal_code(
            h.convict_on(voter, late, late_index, &late_commitment)
                .await
        ),
        treasurer_refusal(ProgramError::CommitmentOutsideEpoch),
    );
}

/// A fraud in the final round is convicted in the round's own witness phase,
/// the first part of its window, and the epoch's charge reaches it.
#[tokio::test]
pub async fn a_final_round_fraud_is_convicted_in_its_witness_phase() {
    let mut h = convicting_at_once("Leviathan final witness", 141).await;
    h.advance_to_final_round_witness().await;
    let committee = h.committee_now().await;
    let (target, target_index) = committee.trainers[0];
    let target_key = h.clients[target].pubkey();
    for (voter, _) in committee.verifiers.iter().take(committee.verifier_quorum)
    {
        h.convict_by(*voter, target, target_index).await.unwrap();
    }
    assert_eq!(h.run_state().await, RunState::RoundWitness);
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Upheld);
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
}

/// One client leaving the final round, here ejected by the run authority, no
/// longer shuts the cooldown: the round keeps its roster through it, so its
/// committee still convicts another fraud of the round there. Compacted at
/// the round's end, the roster no longer matched the round's draw and every
/// vote in the cooldown was refused.
#[tokio::test]
pub async fn a_client_leaving_the_final_round_does_not_shut_the_cooldown() {
    let mut h = convicting_at_once("Leviathan final leaver", 142).await;
    h.advance_to_final_round_witness().await;
    let committee = h.committee_now().await;
    let (leaver, leaver_index) = committee.trainers[0];
    let (target, target_index) = committee.trainers[1];
    let leaver_key = h.clients[leaver].pubkey();
    let target_key = h.clients[target].pubkey();
    let target_commitment = h.signed_payload_of(target).await;
    authority_slashes(&mut h, leaver, leaver_index).await;

    h.advance_to_cooldown_with_witnesses().await;
    // The leaver is still on the roster, at the index the round drew it at.
    assert_eq!(
        h.client_state(&leaver_key).await,
        Some(ClientState::Ejected)
    );
    assert_eq!(h.roster_index_of(&target_key).await, target_index);
    for (voter, _) in committee.verifiers.iter().take(committee.verifier_quorum)
    {
        h.convict_on(*voter, target, target_index, &target_commitment)
            .await
            .unwrap();
    }
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Upheld);

    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.slashed_of(&leaver_key).await, SLASHING_RATE);
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    assert_eq!(
        h.client_state(&leaver_key).await,
        None,
        "compacted at the cooldown's end"
    );
}

/// A committee that convicts a target the run authority already ejected this
/// epoch records the conviction as due: the coordinator would refuse a second
/// ejection, and that refusal used to undo the vote that reached the quorum.
/// The epoch charges the authority's ejection, and the target's withdrawal
/// settles the committee's conviction against the bond.
#[tokio::test]
pub async fn a_conviction_of_a_target_already_ejected_is_due() {
    let mut h = convicting_at_once("Leviathan final ejected", 143).await;
    h.advance_to_final_round_witness().await;
    let committee = h.committee_now().await;
    let (target, target_index) = committee.trainers[0];
    let target_key = h.clients[target].pubkey();
    let target_commitment = h.signed_payload_of(target).await;
    authority_slashes(&mut h, target, target_index).await;
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected)
    );

    let voters: Vec<usize> = committee
        .verifiers
        .iter()
        .take(committee.verifier_quorum)
        .map(|(voter, _)| *voter)
        .collect();
    for voter in &voters {
        h.convict_on(*voter, target, target_index, &target_commitment)
            .await
            .unwrap();
    }
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Due);

    h.finish_epoch_with_witnesses().await;
    // The ledger charged the authority's ejection once.
    assert_eq!(h.slashed_of(&target_key).await, SLASHING_RATE);
    h.request_withdraw(target, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    let before = h.collateral_of(target).await;
    // The bounty goes to the voters the verdict names, in its order.
    h.finalize_withdraw_paying(target, &voters).await.unwrap();
    // Both convictions, each charged once.
    assert_eq!(
        h.collateral_of(target).await - before,
        BOND - 2 * SLASHING_RATE
    );
    assert_eq!(h.verdict_status_of(target).await, VerdictStatus::Upheld);
}

/// A round that gathers no witness ends the epoch with every client
/// withdrawn, and the coordinator would refuse to eject any of them. The
/// round's committee still convicts a fraud in it during the cooldown: the
/// conviction is recorded as due, and the target's withdrawal settles it.
/// Ejecting a withdrawn target used to undo the vote that reached the quorum.
#[tokio::test]
pub async fn a_round_that_loses_its_witnesses_still_convicts_through_the_bond()
{
    let mut h = convicting_at_once("Leviathan final no witness", 144).await;
    let target = h.target_client_idx;
    let target_key = h.target_key;
    assert_eq!(h.run_state().await, RunState::RoundTrain);
    // Nobody witnesses the round.
    let config = h.config().await;
    h.tick_after(config.max_round_train_time).await;
    h.tick_after(config.round_witness_time).await;
    assert_eq!(h.run_state().await, RunState::Cooldown);
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Withdrawn)
    );

    // The committee in force is drawn without a bench, as the run now has none.
    let committee = h.committee_now().await;
    let voters: Vec<usize> = committee
        .verifiers
        .iter()
        .take(committee.verifier_quorum)
        .map(|(voter, _)| *voter)
        .collect();
    let target_index = h.target_index;
    let target_commitment = h.target_commitment;
    for voter in &voters {
        h.convict_on(*voter, target, target_index, &target_commitment)
            .await
            .unwrap();
    }
    assert_eq!(h.verdict_status().await, VerdictStatus::Due);

    h.finish_epoch_with_witnesses().await;
    // Nobody was ejected, so the ledger charged nobody.
    assert_eq!(h.slashed_of(&target_key).await, 0);
    h.request_withdraw(target, BOND).await;
    h.endpoint.forward_clock_unix_timestamp(100).await.unwrap();
    let before = h.collateral_of(target).await;
    h.finalize_withdraw_paying(target, &voters).await.unwrap();
    assert_eq!(h.collateral_of(target).await - before, BOND - SLASHING_RATE);
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
}
