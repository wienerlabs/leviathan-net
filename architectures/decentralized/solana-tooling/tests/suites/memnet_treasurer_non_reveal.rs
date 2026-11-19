//! A target that does not reveal the state its payload commits to
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! Not revealing costs what fraud costs, so verifiers vote it as they vote
//! fraud, once the run's reveal window has passed in a witness phase. The
//! verdict records that its votes are for a state not revealed, and counts
//! them apart from votes on a replay. Whether a target revealed cannot be
//! proven either way, so an overturned non-reveal costs its voters nothing.

use psyche_coordinator::RunState;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_open_challenge;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_reveal_window;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_submit_non_reveal_verdict;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_treasurer::find_reveal_window;
use psyche_solana_treasurer::state::RevealWindow;
use psyche_solana_treasurer::state::VerdictKind;
use psyche_solana_treasurer::state::VerdictStatus;
use psyche_solana_treasurer::ProgramError;

use super::appeals_harness::*;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::identity;
use super::signed_payload::payload_digest;
use super::signed_payload::FRAUD_PAYLOAD;

/// Shorter than the harness's round witness time of ten seconds.
const REVEAL_SECONDS: u64 = 4;

impl AppealsHarness {
    async fn set_reveal_window(&mut self, seconds: u64) -> anyhow::Result<()> {
        let main_authority = self.main_authority.insecure_clone();
        process_treasurer_run_set_reveal_window(
            &mut self.endpoint,
            &self.payer,
            &main_authority,
            &self.run,
            &self.coordinator_account,
            seconds,
        )
        .await
    }

    /// `voter` votes that the target did not reveal the state `commitment`
    /// commits to.
    async fn vote_not_revealed(
        &mut self,
        voter: usize,
        commitment: &SignedCommitment,
    ) -> anyhow::Result<()> {
        let verifier = self.clients[voter].insecure_clone();
        let (target_key, target_index) = (self.target_key, self.target_index);
        self.endpoint.forward_clock_slot(1).await.unwrap();
        process_treasurer_run_submit_non_reveal_verdict(
            &mut self.endpoint,
            &self.payer,
            &verifier,
            &self.run,
            &self.coordinator_account,
            &self.run_id,
            &target_key,
            target_index,
            commitment,
        )
        .await
    }

    /// The verifiers whose votes make a quorum.
    fn quorum_seats(&self) -> Vec<usize> {
        self.verifiers
            .iter()
            .take(self.verifier_quorum as usize)
            .map(|(client_idx, _)| *client_idx)
            .collect()
    }

    /// Runs the round under way on into its witness phase, where its
    /// committee is still the one in force.
    async fn advance_to_witness_phase(&mut self) {
        let step = self.coordinator_state().await.coordinator.progress.step;
        self.advance_with_witnesses(|coordinator| {
            coordinator.run_state == RunState::RoundWitness
        })
        .await;
        assert_eq!(
            self.coordinator_state().await.coordinator.progress.step,
            step,
            "the witness phase is the round the target trained in"
        );
    }

    /// A quorum votes that the target did not reveal, once the window has
    /// passed in the round's witness phase.
    async fn convict_of_not_revealing(&mut self) {
        self.set_reveal_window(REVEAL_SECONDS).await.unwrap();
        self.advance_to_witness_phase().await;
        self.endpoint
            .forward_clock_unix_timestamp(REVEAL_SECONDS)
            .await
            .unwrap();
        let commitment = self.target_commitment;
        for voter in self.quorum_seats() {
            self.vote_not_revealed(voter, &commitment).await.unwrap();
        }
        let verdict = self.verdict_of(&self.target_key.clone()).await;
        assert_eq!(verdict.status, VerdictStatus::SlashPending);
        assert_eq!(verdict.kind, VerdictKind::NonReveal);
        assert_eq!(verdict.replayed_hash, [0; 32]);
        assert_eq!(verdict.committed_hash, commitment.committed_hash());
        assert_eq!(verdict.verdict_count as u64, self.verifier_quorum);
    }
}

/// A run's main authority sets the window, which has to close within a
/// round's witness phase; nobody else sets it.
#[tokio::test]
pub async fn a_reveal_window_has_to_close_within_the_witness_phase() {
    let mut h = setup("Leviathan reveal window", 171).await;
    let witness_time = h.config().await.round_witness_time;
    for seconds in [0, witness_time, witness_time + 1] {
        assert_eq!(
            refusal_code(h.set_reveal_window(seconds).await),
            treasurer_refusal(ProgramError::RevealWindowInvalid),
            "{seconds} seconds"
        );
    }
    let stranger = h.clients[0].insecure_clone();
    assert!(process_treasurer_run_set_reveal_window(
        &mut h.endpoint,
        &h.payer,
        &stranger,
        &h.run,
        &h.coordinator_account,
        REVEAL_SECONDS,
    )
    .await
    .is_err());

    h.set_reveal_window(REVEAL_SECONDS).await.unwrap();
    h.set_reveal_window(witness_time - 1).await.unwrap();
    let account = h
        .endpoint
        .get_account(&find_reveal_window(&h.run))
        .await
        .unwrap()
        .unwrap();
    let window = RevealWindow::from_pda_account(&account.owner, &account.data)
        .unwrap()
        .unwrap();
    assert_eq!(
        (window.run, window.reveal_seconds),
        (h.run, witness_time - 1)
    );
}

/// A non-reveal is voted only once the run's window has passed in a witness
/// phase, and only for a payload that commits to a state: not in a run with
/// no window, not while the round trains, not as the witness phase opens.
/// Once it has passed, a quorum convicts, and the verdict records a
/// non-reveal with nothing replayed.
#[tokio::test]
pub async fn a_non_reveal_is_voted_once_its_window_has_passed() {
    let mut h = setup("Leviathan non-reveal window", 172).await;
    let voter = h.quorum_seats()[0];
    let commitment = h.target_commitment;
    assert_eq!(h.run_state().await, RunState::RoundTrain);
    assert_eq!(
        refusal_code(h.vote_not_revealed(voter, &commitment).await),
        treasurer_refusal(ProgramError::RevealWindowUnset),
    );
    h.set_reveal_window(REVEAL_SECONDS).await.unwrap();
    assert_eq!(
        refusal_code(h.vote_not_revealed(voter, &commitment).await),
        treasurer_refusal(ProgramError::RevealWindowOpen),
        "the round still trains"
    );

    h.advance_to_witness_phase().await;
    assert_eq!(
        refusal_code(h.vote_not_revealed(voter, &commitment).await),
        treasurer_refusal(ProgramError::RevealWindowOpen),
        "the witness phase has just opened"
    );
    h.endpoint
        .forward_clock_unix_timestamp(REVEAL_SECONDS)
        .await
        .unwrap();

    // A payload that commits to no state has none to reveal.
    let target = h.target_client_idx;
    let stateless = SignedCommitment::for_round(
        &h.coordinator_state().await.coordinator,
        &identity(&h.clients[target], &h.p2ps[target]),
        &h.p2ps[target],
        payload_digest(FRAUD_PAYLOAD),
        None,
    )
    .unwrap();
    assert_eq!(
        refusal_code(h.vote_not_revealed(voter, &stateless).await),
        treasurer_refusal(ProgramError::NonRevealWithoutState),
    );

    for voter in h.quorum_seats() {
        h.vote_not_revealed(voter, &commitment).await.unwrap();
    }
    let verdict = h.verdict_of(&h.target_key.clone()).await;
    assert_eq!(verdict.status, VerdictStatus::SlashPending);
    assert_eq!(verdict.kind, VerdictKind::NonReveal);
    assert_eq!(verdict.replayed_hash, [0; 32]);
    assert_eq!(verdict.committed_hash, commitment.committed_hash());
}

/// A non-reveal the bench overturns costs its voters nothing: the target
/// served its state on appeal, which proves nothing about whether it served
/// the verifiers. The crank passes every vote uncharged, and the bench is
/// owed nothing.
#[tokio::test]
pub async fn an_overturned_non_reveal_costs_its_voters_nothing() {
    let mut h = setup("Leviathan non-reveal overturn", 173).await;
    h.convict_of_not_revealing().await;
    let voters = h.quorum_seats();

    let target = h.clients[h.target_client_idx].insecure_clone();
    process_treasurer_run_open_challenge(
        &mut h.endpoint,
        &h.payer,
        &target,
        &h.run,
    )
    .await
    .unwrap();
    for appellate in h.bench() {
        h.cast_appeal(appellate, true).await.unwrap();
    }
    assert_eq!(h.verdict_status().await, VerdictStatus::Overturned);

    for position in 0..voters.len() as u16 {
        h.crank_loser(position).await.unwrap();
    }
    for voter in voters {
        let participant = h.participant_of(voter).await;
        assert_eq!(participant.bond_amount, BOND, "a voter was charged");
        assert_eq!(participant.open_verdict_votes, 0);
    }
    let verdict = h.verdict_of(&h.target_key.clone()).await;
    assert_eq!(verdict.bounty_owed, 0, "nobody's forfeit owes the bench");
    assert!(verdict.all_losers_settled());
}

/// A non-reveal nobody challenges stands, and charges the target what fraud
/// costs.
#[tokio::test]
pub async fn a_non_reveal_that_stands_costs_what_fraud_costs() {
    let mut h = setup("Leviathan non-reveal stands", 174).await;
    h.convict_of_not_revealing().await;
    let (target_key, target_index) = (h.target_key, h.target_index);
    h.endpoint
        .forward_clock_unix_timestamp((CHALLENGE_WINDOW + 5) as u64)
        .await
        .unwrap();
    psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_finalize_slash(
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
    h.finish_epoch_with_witnesses().await;
    h.charge_conviction_of(&target_key).await.unwrap();
    let target = h.participant_of(h.target_client_idx).await;
    assert_eq!(target.bond_amount, BOND - SLASHING_RATE);
}
