use anchor_lang::prelude::*;

use psyche_coordinator::Coordinator;

use crate::state::CommitmentEvidence;
use crate::ProgramError;

pub const MAX_VERDICT_VOTERS: usize = 64;
pub const MAX_APPEAL_VOTERS: usize = 64;

// `released_voters` keeps one bit per voter, and `bench_paid` one per
// appeal voter.
const _: () = assert!(MAX_VERDICT_VOTERS <= u64::BITS as usize);
const _: () = assert!(MAX_APPEAL_VOTERS <= u64::BITS as usize);

/// How a vote stopped holding its voter's bond.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteRelease {
    /// The verdict had not reached quorum, and the vote was taken off it. It
    /// no longer counts toward a quorum, so it can never be part of one.
    Withdrawn,
    /// The conviction stands, so the vote was right and can no longer cost its
    /// voter.
    Released,
}

#[derive(
    AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug, Default,
)]
pub enum VerdictStatus {
    #[default]
    Voting,
    SlashPending,
    Challenged,
    Upheld,
    Overturned,
    /// The conviction stands, but could not eject the target into an epoch
    /// that will charge it when it was reached: it was ruled between epochs,
    /// in a warmup, or with the target off the roster. It lands through
    /// `run_finalize_slash` once the target is in such an epoch, or is charged
    /// against the bond when the target withdraws (wienerlabs/leviathan-net#36).
    Due,
}

#[account]
#[derive(Debug)]
pub struct AuditVerdict {
    pub bump: u8,
    pub run: Pubkey,
    pub target: Pubkey,
    pub epoch: u16,
    /// The round whose committee cast the votes in `voters`.
    ///
    /// A verifier seat is drawn per round, from that round's seed. Without this
    /// the verdict accumulated across rounds and pooled votes from committees
    /// that were never drawn together, which turns "two thirds of one committee"
    /// into "two thirds of the union over an epoch" - reachable by waiting
    /// (wienerlabs/leviathan#15, finding 3).
    pub round_height: u32,
    pub status: VerdictStatus,
    pub verdict_count: u16,
    pub committed_hash: [u8; 32],
    pub replayed_hash: [u8; 32],
    pub target_index: u64,
    pub batch_start: u64,
    pub batch_end: u64,
    /// The payload the target committed to, which `committed_hash` is the
    /// commitment hash of. With it anyone can rebuild that hash from this
    /// account alone and tell which step to replay (wienerlabs/leviathan-net#43).
    pub evidence: CommitmentEvidence,
    pub pending_since_unix: i64,
    /// When the challenge was opened, so the appeal can be given a deadline of
    /// its own. Without one a free challenge parked a correct conviction for
    /// good (finding 10).
    pub challenged_since_unix: i64,
    pub challenger: Pubkey,
    pub overturn_count: u16,
    pub uphold_count: u16,
    pub settled_count: u16,
    /// One bit per entry of `voters`: set once that vote can no longer cost its
    /// voter, because the conviction stands and the vote was released, or the
    /// verdict was overturned and the crank charged it. Each vote is counted on
    /// its voter's `Participant` until then, so the bit is what stops a vote
    /// being closed twice (wienerlabs/leviathan-net#39).
    pub released_voters: u64,
    /// What the losing verifiers' forfeits owe the bench, in total, added to
    /// as the crank charges each loser. Once every loser is charged, each
    /// appeal voter's even share is credited to its bond through
    /// `run_pay_bench_bounty`, so charging a loser never waits on a payout
    /// (wienerlabs/leviathan-net#39).
    pub bench_bounty_owed: u64,
    /// One bit per entry of `appeal_voters`: set once that seat was credited
    /// its share of `bench_bounty_owed`.
    pub bench_paid: u64,
    pub voters: Vec<Pubkey>,
    pub appeal_voters: Vec<Pubkey>,
}

impl AuditVerdict {
    pub const SEEDS_PREFIX: &'static [u8] = b"AuditVerdict";

    /// Who shares the bounty out of the target's forfeited bond: the
    /// tie-breakers when an appeal upheld the conviction, otherwise the
    /// verifiers whose quorum convicted it.
    pub fn bounty_recipients(&self) -> &[Pubkey] {
        let stands =
            matches!(self.status, VerdictStatus::Upheld | VerdictStatus::Due);
        if stands && !self.appeal_voters.is_empty() {
            &self.appeal_voters
        } else {
            &self.voters
        }
    }

    /// The verdict held by the account at a participant's verdict PDA, given
    /// that account's owner and data. An empty account, or one the treasurer
    /// does not own, is no verdict.
    pub fn from_pda_account(
        owner: &Pubkey,
        data: &[u8],
    ) -> Result<Option<AuditVerdict>> {
        if data.is_empty() || owner != &crate::ID {
            return Ok(None);
        }
        Ok(Some(AuditVerdict::try_deserialize(&mut &data[..])?))
    }

    pub fn space_with_discriminator() -> usize {
        8 + 1 // bump
            + 32 // run
            + 32 // target
            + 2 // epoch
            + 4 // round_height
            + 1 // status
            + 2 // verdict_count
            + 32 // committed_hash
            + 32 // replayed_hash
            + 8 // target_index
            + 8 // batch_start
            + 8 // batch_end
            + CommitmentEvidence::SPACE // evidence
            + 8 // pending_since_unix
            + 8 // challenged_since_unix
            + 32 // challenger
            + 2 // overturn_count
            + 2 // uphold_count
            + 2 // settled_count
            + 8 // released_voters
            + 8 // bench_bounty_owed
            + 8 // bench_paid
            + 4 + MAX_VERDICT_VOTERS * 32 // voters
            + 4 + MAX_APPEAL_VOTERS * 32 // appeal_voters
    }

    /// Whether a fresh dispute may take this account over.
    ///
    /// One `AuditVerdict` PDA per `(run, target)` is reused forever, so a new
    /// vote lands on whatever the last dispute left behind. Overwriting an
    /// unfinished one erased an appeal in progress, and erased the record the
    /// losing-verifier crank walks - so every verifier the crank had not yet
    /// reached escaped its forfeit, for the price of one transaction paid by one
    /// of the people escaping (wienerlabs/leviathan#15, finding 6).
    ///
    /// A dispute that is still `Voting` when its round ends never convicted
    /// anyone, so it lapses - which is how votes stop carrying from one
    /// committee's draw to the next (finding 3). What must not be overwritten is
    /// a conviction waiting to finalise, an appeal in flight, or an overturn
    /// whose losing verifiers have not all been settled.
    ///
    /// Nor a vote still counted on its voter's `Participant`: the reset erases
    /// the record that vote is closed through, and the count would hold the
    /// voter's bond for good. A lapsed dispute's votes are withdrawn, and a
    /// standing conviction's released, first; both may be done by anyone
    /// (wienerlabs/leviathan-net#39).
    pub fn is_settled(&self) -> bool {
        match self.status {
            VerdictStatus::Voting => self.voters.is_empty(),
            VerdictStatus::Upheld => self.all_votes_released(),
            VerdictStatus::Overturned => {
                self.all_votes_released() && self.bench_settled()
            },
            VerdictStatus::SlashPending
            | VerdictStatus::Challenged
            | VerdictStatus::Due => false,
        }
    }

    fn voter_bit(position: usize) -> u64 {
        1u64 << position
    }

    /// Whether the vote at `position` in `voters` no longer holds its voter.
    pub fn vote_released(&self, position: usize) -> bool {
        self.released_voters & Self::voter_bit(position) != 0
    }

    /// Whether every vote on this verdict no longer holds its voter.
    pub fn all_votes_released(&self) -> bool {
        (0..self.voters.len()).all(|position| self.vote_released(position))
    }

    /// Stops `voter`'s vote holding its bond, if the verdict says it can no
    /// longer cost it. The caller closes the vote on the voter's `Participant`.
    ///
    /// - A conviction that stands (`Upheld`, `Due`) can no longer be
    ///   overturned, so its votes were right. Each is released once.
    /// - A verdict still `Voting` may yet reach quorum. Its voter may withdraw
    ///   its vote at any time (`voter_consents`), and anyone may once the
    ///   dispute's round is over (`round_over`). Either way the vote comes off
    ///   the verdict, so it never counts toward a later quorum: a withdrawn
    ///   vote cannot be called back into a conviction, even if the round comes
    ///   back.
    /// - A conviction inside its challenge window or appeal may still be
    ///   overturned, and an overturned one is settled only by the
    ///   losing-verifier crank, which charges the vote as it closes it.
    pub fn release_vote(
        &mut self,
        voter: &Pubkey,
        voter_consents: bool,
        round_over: bool,
    ) -> Result<VoteRelease> {
        let position = self
            .voters
            .iter()
            .position(|cast| cast == voter)
            .ok_or_else(|| error!(ProgramError::NotAVoter))?;
        match self.status {
            VerdictStatus::Voting => {
                if !voter_consents && !round_over {
                    return err!(ProgramError::VoteStillAtStake);
                }
                // Nothing on a verdict that has not convicted is released, so
                // no bit is shifted by taking the vote out.
                self.voters.remove(position);
                self.verdict_count = self.voters.len() as u16;
                Ok(VoteRelease::Withdrawn)
            },
            VerdictStatus::Upheld | VerdictStatus::Due => {
                if self.vote_released(position) {
                    return err!(ProgramError::VoteAlreadyReleased);
                }
                self.released_voters |= Self::voter_bit(position);
                Ok(VoteRelease::Released)
            },
            VerdictStatus::Overturned if self.vote_released(position) => {
                err!(ProgramError::VoteAlreadyReleased)
            },
            VerdictStatus::SlashPending
            | VerdictStatus::Challenged
            | VerdictStatus::Overturned => err!(ProgramError::VoteStillAtStake),
        }
    }

    /// Whether the round this dispute opened in is over.
    ///
    /// Votes are only taken in that round: one landing in another takes the
    /// account over for a new dispute instead. So once the coordinator has
    /// moved past it, a dispute that never reached quorum has lapsed.
    ///
    /// `current_round` reads a fixed array at an index the coordinator keeps
    /// inside it, so it is never `None` in practice. Were it, the round would
    /// count as over: a vote then cannot be cast either, and a withdrawn vote
    /// never comes back into a count.
    pub fn round_over(&self, coordinator: &Coordinator) -> bool {
        coordinator.progress.epoch != self.epoch
            || coordinator.current_round().map(|round| round.height)
                != Some(self.round_height)
    }

    /// The votes `release_vote` would free right now, in an order they can be
    /// released in one after another. `consenting_voter` is the voter, if any,
    /// that signs its own release.
    pub fn releasable_votes(
        &self,
        consenting_voter: Option<&Pubkey>,
        round_over: bool,
    ) -> Vec<Pubkey> {
        let mut simulated = self.clone();
        self.voters
            .iter()
            .filter(|voter| {
                simulated
                    .release_vote(
                        voter,
                        consenting_voter == Some(*voter),
                        round_over,
                    )
                    .is_ok()
            })
            .copied()
            .collect()
    }

    /// Records that the crank charged the losing vote at `position`, and what
    /// that forfeit owes the bench.
    pub fn settle_losing_vote(&mut self, position: usize, bench_bounty: u64) {
        self.released_voters |= Self::voter_bit(position);
        self.settled_count += 1;
        self.bench_bounty_owed += bench_bounty;
    }

    /// Whether the crank has charged every losing vote, so what the bench is
    /// owed is final.
    pub fn all_losers_settled(&self) -> bool {
        self.settled_count as usize >= self.voters.len()
    }

    /// What each appeal voter is owed out of the losers' forfeits. What does
    /// not divide evenly stays in the vault.
    pub fn bench_share(&self) -> u64 {
        match self.appeal_voters.len() {
            0 => 0,
            seats => self.bench_bounty_owed / seats as u64,
        }
    }

    /// Whether the seat at `seat` in `appeal_voters` was paid its share.
    pub fn bench_seat_paid(&self, seat: usize) -> bool {
        self.bench_paid & Self::voter_bit(seat) != 0
    }

    /// Whether nothing is owed to the bench any more: no share to pay, or
    /// every seat paid.
    pub fn bench_settled(&self) -> bool {
        self.bench_share() == 0
            || (0..self.appeal_voters.len())
                .all(|seat| self.bench_seat_paid(seat))
    }

    /// Marks the seat at `seat` paid and returns its share, if the overturn is
    /// fully charged and the seat is owed one.
    pub fn pay_bench_seat(&mut self, seat: usize) -> Result<u64> {
        if self.status != VerdictStatus::Overturned {
            return err!(ProgramError::VerdictNotOverturned);
        }
        if !self.all_losers_settled() {
            return err!(ProgramError::LosersNotAllSettled);
        }
        if seat >= self.appeal_voters.len() {
            return err!(ProgramError::InvalidParameter);
        }
        if self.bench_seat_paid(seat) {
            return err!(ProgramError::BenchSeatPaid);
        }
        let share = self.bench_share();
        if share == 0 {
            return err!(ProgramError::NothingOwedToBench);
        }
        self.bench_paid |= Self::voter_bit(seat);
        Ok(share)
    }

    /// Clears the account for a dispute opened in `round_height` of `epoch`.
    ///
    /// Callers must check [`AuditVerdict::is_settled`] first.
    pub fn reset_for_round(&mut self, epoch: u16, round_height: u32) {
        self.epoch = epoch;
        self.round_height = round_height;
        self.status = VerdictStatus::Voting;
        self.verdict_count = 0;
        self.committed_hash = [0u8; 32];
        self.replayed_hash = [0u8; 32];
        self.target_index = 0;
        self.batch_start = 0;
        self.batch_end = 0;
        self.evidence = CommitmentEvidence::default();
        self.pending_since_unix = 0;
        self.challenged_since_unix = 0;
        self.challenger = Pubkey::default();
        self.overturn_count = 0;
        self.uphold_count = 0;
        self.settled_count = 0;
        self.released_voters = 0;
        self.bench_bounty_owed = 0;
        self.bench_paid = 0;
        self.voters.clear();
        self.appeal_voters.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(status: VerdictStatus, voters: &[Pubkey]) -> AuditVerdict {
        AuditVerdict {
            bump: 0,
            run: Pubkey::new_unique(),
            target: Pubkey::new_unique(),
            epoch: 1,
            round_height: 2,
            status,
            verdict_count: voters.len() as u16,
            committed_hash: [0u8; 32],
            replayed_hash: [0u8; 32],
            target_index: 0,
            batch_start: 0,
            batch_end: 0,
            evidence: CommitmentEvidence::default(),
            pending_since_unix: 0,
            challenged_since_unix: 0,
            challenger: Pubkey::default(),
            overturn_count: 0,
            uphold_count: 0,
            settled_count: 0,
            released_voters: 0,
            bench_bounty_owed: 0,
            bench_paid: 0,
            voters: voters.to_vec(),
            appeal_voters: vec![],
        }
    }

    fn voters(count: usize) -> Vec<Pubkey> {
        (0..count).map(|_| Pubkey::new_unique()).collect()
    }

    #[test]
    fn a_standing_conviction_releases_each_vote_once() {
        for status in [VerdictStatus::Upheld, VerdictStatus::Due] {
            let cast = voters(3);
            let mut v = verdict(status, &cast);
            assert_eq!(
                v.release_vote(&cast[1], false, false).unwrap(),
                VoteRelease::Released,
            );
            assert!(v.vote_released(1));
            assert!(!v.vote_released(0) && !v.vote_released(2));
            assert_eq!(
                v.release_vote(&cast[1], true, true).unwrap_err(),
                error!(ProgramError::VoteAlreadyReleased),
            );
            // The vote stays on the record the bounty is paid from.
            assert_eq!(v.voters, cast);
            assert_eq!(v.verdict_count, 3);
        }
    }

    #[test]
    fn a_vote_that_may_still_lose_is_not_released() {
        for status in [
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
            VerdictStatus::Overturned,
        ] {
            let cast = voters(2);
            let mut v = verdict(status, &cast);
            assert_eq!(
                v.release_vote(&cast[0], true, true).unwrap_err(),
                error!(ProgramError::VoteStillAtStake),
            );
            assert_eq!(v.released_voters, 0);
        }
    }

    #[test]
    fn a_losing_vote_the_crank_charged_is_already_released() {
        let cast = voters(2);
        let mut v = verdict(VerdictStatus::Overturned, &cast);
        v.settle_losing_vote(0, 0);
        assert_eq!(v.settled_count, 1);
        assert_eq!(
            v.release_vote(&cast[0], true, true).unwrap_err(),
            error!(ProgramError::VoteAlreadyReleased),
        );
        assert_eq!(
            v.release_vote(&cast[1], true, true).unwrap_err(),
            error!(ProgramError::VoteStillAtStake),
        );
    }

    #[test]
    fn a_vote_before_quorum_is_withdrawn_by_its_voter_at_any_time() {
        let cast = voters(3);
        let mut v = verdict(VerdictStatus::Voting, &cast);
        assert_eq!(
            v.release_vote(&cast[1], true, false).unwrap(),
            VoteRelease::Withdrawn,
        );
        assert_eq!(v.voters, vec![cast[0], cast[2]]);
        assert_eq!(v.verdict_count, 2);
        assert_eq!(v.released_voters, 0);
        assert_eq!(
            v.release_vote(&cast[1], true, false).unwrap_err(),
            error!(ProgramError::NotAVoter),
        );
    }

    #[test]
    fn a_vote_before_quorum_is_withdrawn_by_anyone_once_its_round_is_over() {
        let cast = voters(2);
        let mut v = verdict(VerdictStatus::Voting, &cast);
        assert_eq!(
            v.release_vote(&cast[0], false, false).unwrap_err(),
            error!(ProgramError::VoteStillAtStake),
        );
        assert_eq!(v.voters, cast);
        assert_eq!(
            v.release_vote(&cast[0], false, true).unwrap(),
            VoteRelease::Withdrawn,
        );
        assert_eq!(
            v.release_vote(&cast[1], false, true).unwrap(),
            VoteRelease::Withdrawn,
        );
        assert!(v.voters.is_empty());
        assert_eq!(v.verdict_count, 0);
    }

    #[test]
    fn a_key_that_did_not_vote_is_refused() {
        for status in [
            VerdictStatus::Voting,
            VerdictStatus::Upheld,
            VerdictStatus::Overturned,
        ] {
            let mut v = verdict(status, &voters(2));
            assert_eq!(
                v.release_vote(&Pubkey::new_unique(), true, true)
                    .unwrap_err(),
                error!(ProgramError::NotAVoter),
            );
        }
    }

    #[test]
    fn a_dispute_is_settled_only_once_no_vote_holds_a_bond() {
        let cast = voters(2);

        let mut lapsed = verdict(VerdictStatus::Voting, &cast);
        assert!(!lapsed.is_settled());
        lapsed.release_vote(&cast[0], false, true).unwrap();
        assert!(!lapsed.is_settled());
        lapsed.release_vote(&cast[1], false, true).unwrap();
        assert!(lapsed.is_settled());

        let mut upheld = verdict(VerdictStatus::Upheld, &cast);
        assert!(!upheld.is_settled());
        upheld.release_vote(&cast[0], false, false).unwrap();
        assert!(!upheld.is_settled());
        upheld.release_vote(&cast[1], false, false).unwrap();
        assert!(upheld.is_settled());

        let mut overturned = verdict(VerdictStatus::Overturned, &cast);
        assert!(!overturned.is_settled());
        overturned.settle_losing_vote(0, 0);
        assert!(!overturned.is_settled());
        overturned.settle_losing_vote(1, 0);
        assert!(overturned.is_settled());

        for status in [
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
            VerdictStatus::Due,
        ] {
            let mut open = verdict(status, &cast);
            open.released_voters = u64::MAX;
            assert!(!open.is_settled());
        }
    }

    #[test]
    fn the_bench_is_paid_once_per_seat_after_every_loser_is_charged() {
        let losers = voters(2);
        let bench = voters(3);
        let mut v = verdict(VerdictStatus::Overturned, &losers);
        v.appeal_voters = bench.clone();
        v.settle_losing_vote(0, 50);
        assert_eq!(
            v.pay_bench_seat(0).unwrap_err(),
            error!(ProgramError::LosersNotAllSettled),
        );
        v.settle_losing_vote(1, 50);
        assert_eq!(v.bench_bounty_owed, 100);
        // 100 does not divide by three: each seat is owed 33.
        assert_eq!(v.bench_share(), 33);
        assert!(!v.is_settled(), "the bench is still owed");
        assert_eq!(v.pay_bench_seat(1).unwrap(), 33);
        assert_eq!(
            v.pay_bench_seat(1).unwrap_err(),
            error!(ProgramError::BenchSeatPaid),
        );
        assert_eq!(
            v.pay_bench_seat(3).unwrap_err(),
            error!(ProgramError::InvalidParameter),
        );
        v.pay_bench_seat(0).unwrap();
        assert!(!v.is_settled());
        v.pay_bench_seat(2).unwrap();
        assert!(v.bench_settled());
        assert!(v.is_settled());
    }

    #[test]
    fn a_bench_owed_nothing_needs_no_payout() {
        let losers = voters(1);
        let mut v = verdict(VerdictStatus::Overturned, &losers);
        v.appeal_voters = voters(2);
        v.settle_losing_vote(0, 1);
        assert_eq!(v.bench_share(), 0);
        assert!(v.is_settled());
        assert_eq!(
            v.pay_bench_seat(0).unwrap_err(),
            error!(ProgramError::NothingOwedToBench),
        );
    }

    #[test]
    fn only_an_overturn_pays_the_bench() {
        for status in [
            VerdictStatus::Voting,
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
            VerdictStatus::Upheld,
            VerdictStatus::Due,
        ] {
            let mut v = verdict(status, &[]);
            v.appeal_voters = voters(1);
            v.bench_bounty_owed = 10;
            assert_eq!(
                v.pay_bench_seat(0).unwrap_err(),
                error!(ProgramError::VerdictNotOverturned),
            );
        }
    }

    #[test]
    fn a_full_voter_list_has_a_bit_for_every_vote() {
        let cast = voters(MAX_VERDICT_VOTERS);
        let mut v = verdict(VerdictStatus::Upheld, &cast);
        for voter in &cast {
            assert!(!v.is_settled());
            v.release_vote(voter, false, false).unwrap();
        }
        assert_eq!(v.released_voters, u64::MAX);
        assert!(v.is_settled());
    }

    #[test]
    fn the_releasable_votes_are_those_a_release_would_free() {
        let cast = voters(3);
        let live = verdict(VerdictStatus::Voting, &cast);
        assert!(live.releasable_votes(None, false).is_empty());
        assert_eq!(live.releasable_votes(Some(&cast[2]), false), vec![cast[2]]);
        assert_eq!(live.releasable_votes(None, true), cast);

        let mut upheld = verdict(VerdictStatus::Upheld, &cast);
        upheld.release_vote(&cast[0], false, false).unwrap();
        assert_eq!(
            upheld.releasable_votes(None, false),
            vec![cast[1], cast[2]]
        );

        for status in [
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
            VerdictStatus::Overturned,
        ] {
            let open = verdict(status, &cast);
            assert!(open.releasable_votes(Some(&cast[0]), true).is_empty());
        }
    }

    /// A coordinator in `epoch` whose current round is `height`.
    fn coordinator_at(epoch: u16, height: u32) -> Box<Coordinator> {
        let mut coordinator =
            Box::new(<Coordinator as bytemuck::Zeroable>::zeroed());
        coordinator.progress.epoch = epoch;
        let head = coordinator.epoch_state.rounds_head as usize;
        coordinator.epoch_state.rounds[head].height = height;
        coordinator
    }

    #[test]
    fn a_round_is_over_once_the_epoch_or_the_height_moves() {
        // The verdict opened in epoch 1, round 2.
        let v = verdict(VerdictStatus::Voting, &voters(1));
        assert!(!v.round_over(&coordinator_at(1, 2)));
        assert!(v.round_over(&coordinator_at(1, 3)));
        // Heights restart each epoch: the same height in a later epoch is
        // another round.
        assert!(v.round_over(&coordinator_at(2, 2)));
        assert!(v.round_over(&coordinator_at(2, 0)));
    }

    #[test]
    fn a_reset_clears_the_released_votes() {
        let cast = voters(2);
        let mut v = verdict(VerdictStatus::Upheld, &cast);
        v.release_vote(&cast[0], false, false).unwrap();
        v.release_vote(&cast[1], false, false).unwrap();
        v.bench_bounty_owed = 7;
        v.bench_paid = 1;
        v.reset_for_round(2, 0);
        assert_eq!(v.released_voters, 0);
        assert_eq!((v.bench_bounty_owed, v.bench_paid), (0, 0));
        assert!(v.voters.is_empty());
        assert_eq!(v.status, VerdictStatus::Voting);
    }
}
