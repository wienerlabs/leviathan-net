//! A vote for another payload than the one a verdict's quorum convicted on.
//!
//! Votes are counted per payload the target signed, and a quorum is two
//! thirds of one committee agreeing on one. A vote citing another payload
//! does not stop that quorum, and it was not part of it: it is no seat of the
//! bounty when the conviction stands, the crank charges it nothing when the
//! conviction is overturned, and nothing the verdict comes to holds its
//! voter's bond.

use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_finalize_slash;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_tooling::verdict_votes::run_pay_verdict_bounty_instruction;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::signer::Signer;

use super::appeals_harness::*;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::payload_digest;
use super::signed_payload::HONEST_PAYLOAD;

/// Eight clients and a two-seat bench leave six free seats, and half of them
/// seat a committee of three: room for a dissent beside a quorum of two.
const COMMITTEE_OF_THREE: RunSize = RunSize {
    clients: 8,
    verification_percent: 50,
    unbonded: 0,
};

fn bounty() -> u64 {
    SLASHING_RATE * BOUNTY_BPS as u64 / 10_000
}

/// Another payload `commitment`'s signer signed for the same work: a genuine
/// commitment, as any broadcast one is, that the conviction is not about.
fn another_payload(
    h: &AppealsHarness,
    signer: usize,
    commitment: &SignedCommitment,
) -> SignedCommitment {
    SignedCommitment::sign(
        &h.p2ps[signer],
        commitment.run,
        commitment.header(),
        payload_digest(HONEST_PAYLOAD),
    )
}

/// The first verifier votes on another payload of the target, and the other
/// two on the one the conviction is about. Returns the dissenting verifier and
/// the two that formed the quorum.
async fn convict_past_a_dissent(h: &mut AppealsHarness) -> (usize, [usize; 2]) {
    let (target, target_index) = (h.target_client_idx, h.target_index);
    let commitment = h.target_commitment;
    let committee = h.committee_now().await;
    assert_eq!(committee.verifier_quorum, 2);
    let verifiers: Vec<usize> = committee
        .verifiers
        .iter()
        .map(|(client_idx, _)| *client_idx)
        .collect();
    assert!(
        verifiers.len() >= 3,
        "a dissent and a quorum need three seats"
    );
    let (dissenter, quorum) = (verifiers[0], [verifiers[1], verifiers[2]]);

    let other = another_payload(h, target, &commitment);
    h.convict_on(dissenter, target, target_index, &other)
        .await
        .unwrap();
    for voter in quorum {
        h.convict_on(voter, target, target_index, &commitment)
            .await
            .unwrap();
    }

    let target_key = h.target_key;
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(verdict.committed_hash, commitment.committed_hash());
    assert_eq!(verdict.verdict_count, 2);
    assert!(verdict.dissents(0));
    assert!(!verdict.dissents(1) && !verdict.dissents(2));
    assert_eq!(verdict.status, VerdictStatus::SlashPending);
    (dissenter, quorum)
}

/// An overturned conviction charges the verifiers whose quorum convicted, and
/// passes the dissenting vote uncharged. The dissent held nothing to begin
/// with: anyone may release it as soon as the quorum is reached, while the
/// quorum's own votes stay held until the appeal is decided.
#[tokio::test]
pub async fn an_overturn_charges_only_the_votes_that_convicted() {
    let mut h =
        setup_sized("Leviathan dissent overturn", 168, COMMITTEE_OF_THREE)
            .await;
    let target_key = h.target_key;
    let (dissenter, quorum) = convict_past_a_dissent(&mut h).await;

    h.release_vote(None, &target_key, dissenter).await.unwrap();
    assert_eq!(h.participant_of(dissenter).await.open_verdict_votes, 0);
    assert_eq!(
        refusal_code(h.release_vote(None, &target_key, quorum[0]).await),
        treasurer_refusal(ProgramError::VoteStillAtStake),
    );

    let target_wallet = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(
        &mut h.endpoint,
        &h.payer,
        &target_wallet,
        &h.run,
    )
    .await
    .unwrap();
    for appellate in h.bench() {
        h.cast_appeal(appellate, true).await.unwrap();
    }
    assert_eq!(h.verdict_status().await, VerdictStatus::Overturned);

    for position in 0..3 {
        h.crank_loser(position).await.unwrap();
    }
    let dissent = h.participant_of(dissenter).await;
    assert_eq!(dissent.bond_amount, BOND, "the dissenting vote was charged");
    assert_eq!(dissent.open_verdict_votes, 0);
    for voter in quorum {
        let loser = h.participant_of(voter).await;
        assert_eq!(loser.bond_amount, BOND - SLASHING_RATE);
        assert_eq!(loser.open_verdict_votes, 0);
    }
    let verdict = h.verdict_of(&target_key).await;
    assert_eq!(
        verdict.bounty_owed,
        2 * bounty(),
        "the bench is owed the forfeits of the two votes that convicted"
    );
    assert!(verdict.all_losers_settled());
}

/// A conviction that stands owes its bounty to the verifiers whose quorum
/// convicted, seat by seat, and not to a vote on another payload: the
/// dissenting voter is no seat, and its bond is credited nothing.
#[tokio::test]
pub async fn the_bounty_pays_only_the_votes_that_convicted() {
    let mut h =
        setup_sized("Leviathan dissent bounty", 169, COMMITTEE_OF_THREE)
            .await;
    let (target_key, target_index) = (h.target_key, h.target_index);
    let (dissenter, quorum) = convict_past_a_dissent(&mut h).await;

    // Nobody challenges, so the conviction stands once its window closes.
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    process_treasurer_run_finalize_slash(
        &mut h.endpoint,
        &h.payer,
        &h.run,
        &h.coordinator_account,
        &h.run_id,
        &target_key,
        target_index,
    )
    .await
    .unwrap();
    assert_eq!(h.verdict_status().await, VerdictStatus::Upheld);
    let seats: Vec<_> = quorum.iter().map(|i| h.clients[*i].pubkey()).collect();
    assert_eq!(h.verdict_of(&target_key).await.bounty_recipients(), seats);

    h.finish_epoch_with_witnesses().await;
    h.charge_conviction_of(&target_key).await.unwrap();
    let mut before = vec![];
    for voter in [dissenter, quorum[0], quorum[1]] {
        before.push(h.holdings_of(voter).await);
    }
    // Named at any seat, the dissenting voter is refused: it has none.
    let dissenter_key = h.clients[dissenter].pubkey();
    for seat in 0..3 {
        let payer = h.payer.insecure_clone();
        let naming_the_dissent = run_pay_verdict_bounty_instruction(
            &payer.pubkey(),
            &h.run,
            &target_key,
            seat,
            &dissenter_key,
        );
        assert_eq!(
            refusal_code(
                h.endpoint
                    .process_instruction(&payer, naming_the_dissent)
                    .await
            ),
            treasurer_refusal(ProgramError::InvalidParameter),
        );
    }
    h.pay_every_seat_of(&target_key).await;
    let share = bounty() / 2;
    assert_eq!(h.holdings_of(dissenter).await, before[0]);
    assert_eq!(h.holdings_of(quorum[0]).await - before[1], share);
    assert_eq!(h.holdings_of(quorum[1]).await - before[2], share);
}
