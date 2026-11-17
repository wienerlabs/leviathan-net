use anchor_lang::prelude::*;

use psyche_coordinator::Coordinator;

use crate::state::CommitmentEvidence;
use crate::ProgramError;

pub const MAX_VERDICT_VOTERS: usize = 64;
pub const MAX_APPEAL_VOTERS: usize = 64;

// `released_voters` keeps one bit per voter, and `bounty_paid` one per
// bounty recipient: the voters or the appeal voters.
const _: () = assert!(MAX_VERDICT_VOTERS <= u64::BITS as usize);
const _: () = assert!(MAX_APPEAL_VOTERS <= u64::BITS as usize);

/// What a vote convicts on: the payload the target signed, where it sits, and
/// what the voter's own replay of it hashed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoteEvidence {
    pub committed_hash: [u8; 32],
    pub evidence: CommitmentEvidence,
    pub target_index: u64,
    pub batch_start: u64,
    pub batch_end: u64,
    pub replayed_hash: [u8; 32],
}

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

/// How far a conviction has reached the target's bond.
///
/// A conviction is paid for out of the target's forfeit, and its bounty out of
/// that. The bounty used to be paid at the target's own withdrawal, to whoever
/// the verdict account named at that moment: by then a later dispute could
/// have taken the account over, and an overturned verdict named its losers.
/// Now the conviction's own charge is taken from the bond as soon as the
/// ledger has made it, and its bounty recorded here, so the record it is paid
/// from never outlives the conviction (wienerlabs/leviathan-net#64).
#[derive(
    AnchorSerialize,
    AnchorDeserialize,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Debug,
    Default,
)]
pub enum ConvictionCharge {
    /// The conviction has not reached the ledger or the bond, or there is none.
    #[default]
    Unlanded,
    /// The conviction ejected the target in the epoch that started at
    /// `charge_epoch_start`, whose end charges `charge_rate` to the target's
    /// ledger. Once it has, `run_charge_conviction` takes it from the bond.
    Ejected,
    /// The target's bond has paid for the conviction, and `bounty_owed` holds
    /// what that owes its recipients.
    Charged,
}

/// What the ledger charges an ejection landed now: the epoch it lands in, by
/// the time it started, and that epoch's slashing rate.
///
/// The epoch is named by its start rather than its number: the run authority
/// can rewrite `progress` while the run is halted, but no two epochs start in
/// the same second (wienerlabs/leviathan-net#67). The rate is the one the epoch
/// charges at its end, because the coordinator only promotes new rates when an
/// epoch starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EjectionCharge {
    pub epoch_start: u64,
    pub rate: u64,
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
    /// How many votes in `voters` convict on the evidence this verdict
    /// records, the votes a quorum is counted in.
    pub verdict_count: u16,
    pub committed_hash: [u8; 32],
    /// One voter's replay of the payload: that of the vote that put the
    /// recorded evidence in the lead. Verifiers replay within a tolerance band
    /// of each other, not bit for bit, so no single replayed hash is the
    /// committee's, and votes are not asked to agree on it.
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
    /// What the losing side's forfeit owes the winning side's seats, the
    /// `bounty_recipients`. An overturn owes the bench out of the losing
    /// verifiers' forfeits, added to as the crank charges each one
    /// (wienerlabs/leviathan-net#39). A conviction that stands owes its voters,
    /// or the bench that upheld it, out of the target's forfeit, recorded when
    /// that is charged (wienerlabs/leviathan-net#64). Each seat's even share is
    /// credited to its bond through `run_pay_verdict_bounty`, so no charge ever
    /// waits on a payout.
    pub bounty_owed: u64,
    /// One bit per entry of `bounty_recipients`: set once that seat was
    /// credited its share of `bounty_owed`.
    pub bounty_paid: u64,
    /// How far a conviction that stands has reached the target's bond.
    pub charge: ConvictionCharge,
    /// With `charge` `Ejected`: the start of the epoch whose end charges it.
    pub charge_epoch_start: u64,
    /// With `charge` `Ejected`: what that epoch charges the target's ledger.
    pub charge_rate: u64,
    /// Whether the bench reached its quorum to uphold the conviction. Only
    /// then did it rule, and only then is the conviction's bounty its. An
    /// appeal that ran out its window is finalised as the verifiers reached it,
    /// so its bounty is theirs, however many tie-breakers voted before the
    /// deadline (wienerlabs/leviathan-net#64).
    pub appeal_upheld: bool,
    pub voters: Vec<Pubkey>,
    pub appeal_voters: Vec<Pubkey>,
    /// The commitment hash each vote in `voters` convicts on, position for
    /// position.
    ///
    /// A quorum is two thirds of one committee agreeing on one payload the
    /// target signed, so votes are counted per commitment, and the verdict
    /// records the evidence with the most votes. The first vote used to fix
    /// the evidence for every vote after it. Every payload is broadcast, so
    /// one verifier seat, or the target voting on itself, could cite another
    /// payload the target signed, and each honest vote after it was refused:
    /// one seat vetoed what two thirds agreed on. Requiring the replayed hash
    /// to match as well refused honest verifiers whose replays differ in the
    /// last bits, which is what the tolerance band is for.
    ///
    /// A vote for evidence the verdict does not record (`dissents`) convicted
    /// nobody: it is no seat of the bounty, the crank charges it nothing on an
    /// overturn, and it is released once the verdict resolves.
    ///
    /// Last in the account, so a decoder of the previous layout still reads
    /// every field before it.
    pub vote_commitments: Vec<[u8; 32]>,
}

impl AuditVerdict {
    pub const SEEDS_PREFIX: &'static [u8] = b"AuditVerdict";

    /// Who shares `bounty_owed`, by seat:
    ///
    /// - an overturn pays the bench out of the losing verifiers' forfeits;
    /// - a conviction that stands pays the bench that upheld it, if the bench
    ///   reached its quorum to uphold, and otherwise the verifiers whose quorum
    ///   convicted the target, out of the target's forfeit;
    /// - a dispute still open owes nobody anything.
    ///
    /// An overturned verdict used to name its losers here, so a target cleared
    /// on appeal paid them the bounty of any other conviction it carried
    /// (wienerlabs/leviathan-net#64). A vote for other evidence than the
    /// conviction records was not part of the quorum that convicted, so it is
    /// no seat here.
    pub fn bounty_recipients(&self) -> Vec<Pubkey> {
        match self.status {
            VerdictStatus::Overturned => self.appeal_voters.clone(),
            VerdictStatus::Upheld | VerdictStatus::Due
                if self.appeal_upheld =>
            {
                self.appeal_voters.clone()
            },
            VerdictStatus::Upheld | VerdictStatus::Due => self
                .voters
                .iter()
                .enumerate()
                .filter(|(position, _)| !self.dissents(*position))
                .map(|(_, voter)| *voter)
                .collect(),
            VerdictStatus::Voting
            | VerdictStatus::SlashPending
            | VerdictStatus::Challenged => vec![],
        }
    }

    /// How many votes in `voters` convict on `committed_hash`.
    pub fn votes_for(&self, committed_hash: &[u8; 32]) -> u16 {
        self.vote_commitments
            .iter()
            .filter(|cast| *cast == committed_hash)
            .count() as u16
    }

    /// Whether the vote at `position` in `voters` convicts on other evidence
    /// than this verdict records.
    pub fn dissents(&self, position: usize) -> bool {
        self.vote_commitments.get(position) != Some(&self.committed_hash)
    }

    /// Counts `voter`'s vote for `vote` and returns how many votes now convict
    /// on its evidence.
    ///
    /// The verdict records the evidence with the most votes, the earliest of a
    /// tie, so a vote that completes a quorum is always for the evidence
    /// recorded: one seat cannot hold a vote to the evidence it chose. A
    /// recorded evidence nobody votes for any more, after withdrawals, gives
    /// way to the next vote cast.
    pub fn cast_vote(&mut self, voter: Pubkey, vote: VoteEvidence) -> u16 {
        let leading = self.votes_for(&self.committed_hash);
        self.voters.push(voter);
        self.vote_commitments.push(vote.committed_hash);
        let agreeing = self.votes_for(&vote.committed_hash);
        let takes_the_lead =
            vote.committed_hash != self.committed_hash && agreeing > leading;
        if leading == 0 || takes_the_lead {
            self.committed_hash = vote.committed_hash;
            self.evidence = vote.evidence;
            self.target_index = vote.target_index;
            self.batch_start = vote.batch_start;
            self.batch_end = vote.batch_end;
            self.replayed_hash = vote.replayed_hash;
        }
        self.verdict_count = self.votes_for(&self.committed_hash);
        agreeing
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
            + 8 // bounty_owed
            + 8 // bounty_paid
            + 1 // charge
            + 8 // charge_epoch_start
            + 8 // charge_rate
            + 1 // appeal_upheld
            + 4 + MAX_VERDICT_VOTERS * 32 // voters
            + 4 + MAX_APPEAL_VOTERS * 32 // appeal_voters
            + 4 + MAX_VERDICT_VOTERS * 32 // vote_commitments
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
    ///
    /// Nor a conviction whose bounty is still owed: the reset erases who it is
    /// owed to. A conviction that stands is charged, and every seat credited,
    /// first, all of which anyone may do once the ledger has charged it, so a
    /// later dispute can never redirect the bounty and the target can never
    /// hold its verdict open (wienerlabs/leviathan-net#64).
    pub fn is_settled(&self) -> bool {
        match self.status {
            VerdictStatus::Voting => self.voters.is_empty(),
            VerdictStatus::Upheld => {
                self.all_votes_released()
                    && self.charge == ConvictionCharge::Charged
                    && self.bounty_settled()
            },
            VerdictStatus::Overturned => {
                self.all_votes_released() && self.bounty_settled()
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
                self.vote_commitments.remove(position);
                self.verdict_count = self.votes_for(&self.committed_hash);
                Ok(VoteRelease::Withdrawn)
            },
            // A vote for other evidence than a resolved verdict records was
            // not part of its quorum, so nothing the verdict comes to can cost
            // it.
            VerdictStatus::SlashPending
            | VerdictStatus::Challenged
            | VerdictStatus::Overturned
                if self.dissents(position) =>
            {
                if self.vote_released(position) {
                    return err!(ProgramError::VoteAlreadyReleased);
                }
                self.released_voters |= Self::voter_bit(position);
                Ok(VoteRelease::Released)
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
        self.bounty_owed += bench_bounty;
    }

    /// Records that the crank passed the vote at `position`, one for other
    /// evidence than the overturned conviction, without charging it.
    pub fn settle_dissenting_vote(&mut self, position: usize) {
        self.released_voters |= Self::voter_bit(position);
        self.settled_count += 1;
    }

    /// Records that the conviction stands and ejected the target where the
    /// ledger will charge it as `ejection` says.
    pub fn land_ejection(&mut self, ejection: EjectionCharge) {
        self.status = VerdictStatus::Upheld;
        self.charge = ConvictionCharge::Ejected;
        self.charge_epoch_start = ejection.epoch_start;
        self.charge_rate = ejection.rate;
    }

    /// Whether this is a conviction that stands, ejected its target, and has
    /// not been charged against the target's bond yet.
    pub fn awaits_charge(&self) -> bool {
        self.status == VerdictStatus::Upheld
            && self.charge == ConvictionCharge::Ejected
    }

    /// Whether the ledger has charged the ejection this conviction landed:
    /// the epoch it landed in is over. Every way out of an epoch that can
    /// eject passes through its charge (`Coordinator::ejections_charged`), so
    /// an epoch that no longer has charges pending, or a later one, has made
    /// it.
    pub fn ejection_charged(&self, coordinator: &Coordinator) -> bool {
        !(coordinator.charges_pending()
            && coordinator.epoch_state.start_timestamp
                == self.charge_epoch_start)
    }

    /// Records that the target's bond paid for the conviction, and what that
    /// forfeit owes the conviction's recipients.
    pub fn record_charge(&mut self, bounty: u64) {
        self.charge = ConvictionCharge::Charged;
        self.bounty_owed = bounty;
    }

    /// Records a conviction that never reached the ledger as upheld, charged
    /// at the target's withdrawal, owing `bounty` (wienerlabs/leviathan-net#36).
    pub fn settle_unlanded(&mut self, bounty: u64) {
        self.status = VerdictStatus::Upheld;
        self.record_charge(bounty);
    }

    /// Whether the crank has charged every losing vote, so what the bench is
    /// owed is final.
    pub fn all_losers_settled(&self) -> bool {
        self.settled_count as usize >= self.voters.len()
    }

    /// Whether `bounty_owed` is final, so its seats can be paid: an overturn
    /// once every loser is charged, a conviction that stands once the target
    /// is.
    pub fn bounty_final(&self) -> Result<()> {
        match self.status {
            VerdictStatus::Overturned if self.all_losers_settled() => Ok(()),
            VerdictStatus::Overturned => {
                err!(ProgramError::LosersNotAllSettled)
            },
            VerdictStatus::Upheld
                if self.charge == ConvictionCharge::Charged =>
            {
                Ok(())
            },
            VerdictStatus::Upheld | VerdictStatus::Due => {
                err!(ProgramError::ConvictionNotCharged)
            },
            VerdictStatus::Voting
            | VerdictStatus::SlashPending
            | VerdictStatus::Challenged => {
                err!(ProgramError::VerdictOwesNoBounty)
            },
        }
    }

    /// What each recipient is owed out of `bounty_owed`. What does not divide
    /// evenly stays in the vault.
    pub fn bounty_share(&self) -> u64 {
        match self.bounty_recipients().len() {
            0 => 0,
            seats => self.bounty_owed / seats as u64,
        }
    }

    /// Whether the seat at `seat` in `bounty_recipients` was paid its share.
    pub fn bounty_seat_paid(&self, seat: usize) -> bool {
        self.bounty_paid & Self::voter_bit(seat) != 0
    }

    /// Whether nothing is owed to any seat any more: no share to pay, or
    /// every seat paid.
    pub fn bounty_settled(&self) -> bool {
        self.bounty_share() == 0
            || (0..self.bounty_recipients().len())
                .all(|seat| self.bounty_seat_paid(seat))
    }

    /// Marks the seat at `seat` paid and returns its share, if what the verdict
    /// owes is final and the seat is owed one.
    pub fn pay_bounty_seat(&mut self, seat: usize) -> Result<u64> {
        self.bounty_final()?;
        if seat >= self.bounty_recipients().len() {
            return err!(ProgramError::InvalidParameter);
        }
        if self.bounty_seat_paid(seat) {
            return err!(ProgramError::BountySeatPaid);
        }
        let share = self.bounty_share();
        if share == 0 {
            return err!(ProgramError::NothingOwedToSeats);
        }
        self.bounty_paid |= Self::voter_bit(seat);
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
        self.bounty_owed = 0;
        self.bounty_paid = 0;
        self.charge = ConvictionCharge::Unlanded;
        self.charge_epoch_start = 0;
        self.charge_rate = 0;
        self.appeal_upheld = false;
        self.voters.clear();
        self.appeal_voters.clear();
        self.vote_commitments.clear();
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
            bounty_owed: 0,
            bounty_paid: 0,
            charge: ConvictionCharge::Unlanded,
            charge_epoch_start: 0,
            charge_rate: 0,
            appeal_upheld: false,
            voters: voters.to_vec(),
            appeal_voters: vec![],
            // Every vote for the evidence recorded, as a quorum casts them.
            vote_commitments: vec![[0u8; 32]; voters.len()],
        }
    }

    /// A conviction that stands and whose charge the target's bond paid,
    /// owing `bounty` to its recipients.
    fn charged(voters: &[Pubkey], bounty: u64) -> AuditVerdict {
        let mut v = verdict(VerdictStatus::Upheld, voters);
        v.record_charge(bounty);
        v
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

        let mut upheld = charged(&cast, 0);
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
            v.pay_bounty_seat(0).unwrap_err(),
            error!(ProgramError::LosersNotAllSettled),
        );
        v.settle_losing_vote(1, 50);
        assert_eq!(v.bounty_owed, 100);
        // 100 does not divide by three: each seat is owed 33.
        assert_eq!(v.bounty_share(), 33);
        assert!(!v.is_settled(), "the bench is still owed");
        assert_eq!(v.pay_bounty_seat(1).unwrap(), 33);
        assert_eq!(
            v.pay_bounty_seat(1).unwrap_err(),
            error!(ProgramError::BountySeatPaid),
        );
        assert_eq!(
            v.pay_bounty_seat(3).unwrap_err(),
            error!(ProgramError::InvalidParameter),
        );
        v.pay_bounty_seat(0).unwrap();
        assert!(!v.is_settled());
        v.pay_bounty_seat(2).unwrap();
        assert!(v.bounty_settled());
        assert!(v.is_settled());
    }

    #[test]
    fn a_bench_owed_nothing_needs_no_payout() {
        let losers = voters(1);
        let mut v = verdict(VerdictStatus::Overturned, &losers);
        v.appeal_voters = voters(2);
        v.settle_losing_vote(0, 1);
        assert_eq!(v.bounty_share(), 0);
        assert!(v.is_settled());
        assert_eq!(
            v.pay_bounty_seat(0).unwrap_err(),
            error!(ProgramError::NothingOwedToSeats),
        );
    }

    #[test]
    fn an_overturn_names_its_bench_and_never_its_losers() {
        let losers = voters(2);
        let bench = voters(3);
        let mut v = verdict(VerdictStatus::Overturned, &losers);
        v.appeal_voters = bench.clone();
        assert_eq!(v.bounty_recipients(), bench.as_slice());
        for loser in &losers {
            assert!(!v.bounty_recipients().contains(loser));
        }
    }

    #[test]
    fn a_dispute_still_open_owes_nobody() {
        for status in [
            VerdictStatus::Voting,
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
        ] {
            let mut v = verdict(status, &voters(2));
            v.appeal_voters = voters(1);
            v.bounty_owed = 10;
            assert!(v.bounty_recipients().is_empty());
            assert_eq!(v.bounty_share(), 0);
            assert_eq!(
                v.pay_bounty_seat(0).unwrap_err(),
                error!(ProgramError::VerdictOwesNoBounty),
            );
        }
    }

    #[test]
    fn a_conviction_pays_nobody_before_the_target_is_charged() {
        let cast = voters(2);
        let mut ejected = verdict(VerdictStatus::Voting, &cast);
        ejected.land_ejection(EjectionCharge {
            epoch_start: 1_000,
            rate: 200,
        });
        let due = verdict(VerdictStatus::Due, &cast);
        for mut v in [ejected, due] {
            v.bounty_owed = 10;
            assert_eq!(v.bounty_recipients(), cast.as_slice());
            assert_eq!(
                v.pay_bounty_seat(0).unwrap_err(),
                error!(ProgramError::ConvictionNotCharged),
            );
        }
    }

    #[test]
    fn a_charged_conviction_pays_its_voters_once_per_seat() {
        let cast = voters(3);
        let mut v = charged(&cast, 100);
        assert_eq!(v.bounty_recipients(), cast.as_slice());
        assert_eq!(v.bounty_share(), 33);
        assert_eq!(v.pay_bounty_seat(2).unwrap(), 33);
        assert_eq!(
            v.pay_bounty_seat(2).unwrap_err(),
            error!(ProgramError::BountySeatPaid),
        );
        assert_eq!(
            v.pay_bounty_seat(3).unwrap_err(),
            error!(ProgramError::InvalidParameter),
        );
        v.pay_bounty_seat(0).unwrap();
        v.pay_bounty_seat(1).unwrap();
        assert!(v.bounty_settled());
    }

    #[test]
    fn an_upheld_appeal_pays_its_bench_out_of_the_target() {
        let cast = voters(2);
        let bench = voters(4);
        let mut v = charged(&cast, 100);
        v.appeal_voters = bench.clone();
        v.appeal_upheld = true;
        assert_eq!(v.bounty_recipients(), bench.as_slice());
        assert_eq!(v.bounty_share(), 25);
        for seat in 0..bench.len() {
            assert_eq!(v.pay_bounty_seat(seat).unwrap(), 25);
        }
        assert!(v.bounty_settled());
    }

    #[test]
    fn an_appeal_that_ran_out_its_window_pays_the_verifiers() {
        // Two tie-breakers voted before the deadline, short of a quorum: the
        // bench never ruled, and the verifiers' conviction stands as theirs.
        let cast = voters(2);
        let mut v = charged(&cast, 100);
        v.appeal_voters = voters(2);
        v.overturn_count = 1;
        v.uphold_count = 1;
        assert_eq!(v.bounty_recipients(), cast.as_slice());
        assert_eq!(v.pay_bounty_seat(0).unwrap(), 50);
        // Due, the same.
        let mut due = verdict(VerdictStatus::Due, &cast);
        due.appeal_voters = voters(1);
        assert_eq!(due.bounty_recipients(), cast.as_slice());
    }

    #[test]
    fn a_standing_conviction_is_settled_once_charged_and_credited() {
        let cast = voters(2);
        let mut v = verdict(VerdictStatus::Voting, &cast);
        v.land_ejection(EjectionCharge {
            epoch_start: 1_000,
            rate: 200,
        });
        assert_eq!(v.status, VerdictStatus::Upheld);
        assert_eq!(v.charge, ConvictionCharge::Ejected);
        assert_eq!((v.charge_epoch_start, v.charge_rate), (1_000, 200));
        v.release_vote(&cast[0], false, false).unwrap();
        v.release_vote(&cast[1], false, false).unwrap();
        assert!(!v.is_settled(), "the target has not been charged yet");
        v.record_charge(100);
        assert!(!v.is_settled(), "the voters have not been credited yet");
        v.pay_bounty_seat(0).unwrap();
        assert!(!v.is_settled());
        v.pay_bounty_seat(1).unwrap();
        assert!(v.is_settled());
    }

    #[test]
    fn a_conviction_owing_nothing_is_settled_once_charged() {
        let cast = voters(2);
        let mut v = charged(&cast, 1);
        v.release_vote(&cast[0], false, false).unwrap();
        v.release_vote(&cast[1], false, false).unwrap();
        assert_eq!(v.bounty_share(), 0);
        assert!(v.is_settled());
    }

    #[test]
    fn an_unlanded_conviction_settled_at_withdrawal_is_upheld_and_owed() {
        let cast = voters(2);
        let mut v = verdict(VerdictStatus::Due, &cast);
        v.settle_unlanded(80);
        assert_eq!(v.status, VerdictStatus::Upheld);
        assert_eq!(v.charge, ConvictionCharge::Charged);
        assert_eq!(v.bounty_owed, 80);
        assert_eq!(v.pay_bounty_seat(0).unwrap(), 40);
    }

    #[test]
    fn a_full_voter_list_has_a_bit_for_every_vote() {
        let cast = voters(MAX_VERDICT_VOTERS);
        let mut v = charged(&cast, 0);
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
    fn an_ejection_is_charged_once_its_epoch_has_none_pending() {
        use psyche_coordinator::RunState;

        let mut v = verdict(VerdictStatus::Voting, &voters(2));
        v.land_ejection(EjectionCharge {
            epoch_start: 1_000,
            rate: 200,
        });
        let at = |run_state: RunState, start: u64| {
            let mut coordinator =
                Box::new(<Coordinator as bytemuck::Zeroable>::zeroed());
            coordinator.run_state = run_state;
            coordinator.epoch_state.start_timestamp = start;
            coordinator
        };
        // Its own epoch, still running or cooling down: not charged yet.
        for state in [
            RunState::RoundTrain,
            RunState::RoundWitness,
            RunState::Cooldown,
        ] {
            assert!(!v.ejection_charged(&at(state, 1_000)), "{state}");
        }
        // Its epoch ended, through the charge, into any state that does not
        // charge: its roster is still on the account, but charged.
        for state in [
            RunState::WaitingForMembers,
            RunState::Paused,
            RunState::Finished,
            RunState::Uninitialized,
        ] {
            assert!(v.ejection_charged(&at(state, 1_000)), "{state}");
        }
        // A later epoch, from its warmup on.
        for state in [
            RunState::Warmup,
            RunState::RoundTrain,
            RunState::RoundWitness,
            RunState::Cooldown,
        ] {
            assert!(v.ejection_charged(&at(state, 1_100)), "{state}");
        }
    }

    #[test]
    fn a_full_verdict_fills_its_account_exactly() {
        let mut v = charged(&voters(MAX_VERDICT_VOTERS), u64::MAX);
        v.appeal_voters = voters(MAX_APPEAL_VOTERS);
        v.appeal_upheld = true;
        v.charge_epoch_start = u64::MAX;
        v.charge_rate = u64::MAX;
        let mut data = vec![];
        v.try_serialize(&mut data).unwrap();
        assert_eq!(data.len(), AuditVerdict::space_with_discriminator());
        let back = AuditVerdict::try_deserialize(&mut &data[..]).unwrap();
        assert_eq!(back.charge, ConvictionCharge::Charged);
        assert_eq!(back.bounty_owed, u64::MAX);
        assert_eq!(back.voters, v.voters);
        assert_eq!(back.appeal_voters, v.appeal_voters);
    }

    #[test]
    fn a_reset_clears_the_released_votes_and_the_charge() {
        let cast = voters(2);
        let mut v = verdict(VerdictStatus::Voting, &cast);
        v.land_ejection(EjectionCharge {
            epoch_start: 1_000,
            rate: 200,
        });
        v.release_vote(&cast[0], false, false).unwrap();
        v.release_vote(&cast[1], false, false).unwrap();
        v.record_charge(7);
        v.bounty_paid = 1;
        v.appeal_upheld = true;
        v.reset_for_round(2, 0);
        assert_eq!(v.released_voters, 0);
        assert_eq!((v.bounty_owed, v.bounty_paid), (0, 0));
        assert_eq!(v.charge, ConvictionCharge::Unlanded);
        assert_eq!((v.charge_epoch_start, v.charge_rate), (0, 0));
        assert!(!v.appeal_upheld);
        assert!(v.voters.is_empty());
        assert_eq!(v.status, VerdictStatus::Voting);
    }

    /// A vote for the payload with `digest` as its commitment hash, replayed
    /// to `replayed`.
    fn vote(digest: u8, replayed: u8) -> VoteEvidence {
        VoteEvidence {
            committed_hash: [digest; 32],
            evidence: CommitmentEvidence::default(),
            target_index: 3,
            batch_start: digest as u64,
            batch_end: digest as u64,
            replayed_hash: [replayed; 32],
        }
    }

    fn fresh() -> AuditVerdict {
        let mut v = verdict(VerdictStatus::Voting, &[]);
        v.reset_for_round(1, 2);
        v
    }

    /// Votes are counted per payload, so a vote citing another payload the
    /// target signed does not stop the others reaching a quorum on theirs,
    /// and the verdict records the evidence the most votes are for.
    #[test]
    fn votes_are_counted_per_evidence_and_the_most_voted_is_recorded() {
        let cast = voters(4);
        let mut v = fresh();
        // The first vote cites another payload.
        assert_eq!(v.cast_vote(cast[0], vote(9, 1)), 1);
        assert_eq!(v.committed_hash, [9; 32]);
        // The honest votes count among themselves, not against it.
        assert_eq!(v.cast_vote(cast[1], vote(5, 2)), 1);
        assert_eq!(v.committed_hash, [9; 32], "a tie keeps the earliest");
        assert_eq!(v.cast_vote(cast[2], vote(5, 3)), 2);
        assert_eq!(
            v.committed_hash, [5; 32],
            "the evidence with more votes is recorded"
        );
        assert_eq!((v.batch_start, v.batch_end), (5, 5));
        assert_eq!(v.verdict_count, 2);
        // The replay recorded is the vote that took the lead, and a later
        // vote for the same evidence does not rewrite it.
        assert_eq!(v.replayed_hash, [3; 32]);
        assert_eq!(v.cast_vote(cast[3], vote(5, 4)), 3);
        assert_eq!(v.replayed_hash, [3; 32]);
        assert_eq!(v.verdict_count, 3);
        assert_eq!(v.voters, cast);
        assert!(v.dissents(0));
        assert!((1..4).all(|position| !v.dissents(position)));
    }

    /// Replays agree within the band, not bit for bit: two verifiers who
    /// replayed the same payload to different hashes still vote together.
    #[test]
    fn votes_on_one_payload_agree_whatever_their_replays_hashed_to() {
        let cast = voters(2);
        let mut v = fresh();
        assert_eq!(v.cast_vote(cast[0], vote(5, 1)), 1);
        assert_eq!(v.cast_vote(cast[1], vote(5, 2)), 2);
        assert_eq!(v.verdict_count, 2);
        assert_eq!(v.replayed_hash, [1; 32]);
    }

    /// A withdrawn vote leaves both lists together, and recorded evidence
    /// that nobody votes for any more gives way to the next vote.
    #[test]
    fn a_withdrawn_vote_leaves_its_evidence_with_it() {
        let cast = voters(3);
        let mut v = fresh();
        v.cast_vote(cast[0], vote(9, 1));
        v.cast_vote(cast[1], vote(5, 1));
        v.release_vote(&cast[0], true, false).unwrap();
        assert_eq!(v.voters, vec![cast[1]]);
        assert_eq!(v.vote_commitments, vec![[5; 32]]);
        assert_eq!(
            v.verdict_count, 0,
            "nobody votes for the recorded evidence"
        );
        assert_eq!(v.cast_vote(cast[2], vote(7, 1)), 1);
        assert_eq!(v.committed_hash, [7; 32]);
        assert_eq!(v.verdict_count, 1);
    }

    /// A vote for other evidence than a resolved verdict records convicted
    /// nobody: it is no seat of the conviction's bounty, and it is released
    /// whatever the verdict comes to, once.
    #[test]
    fn a_dissenting_vote_is_neither_paid_nor_held() {
        let cast = voters(3);
        for status in [
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
            VerdictStatus::Overturned,
            VerdictStatus::Upheld,
            VerdictStatus::Due,
        ] {
            let mut v = fresh();
            v.cast_vote(cast[0], vote(9, 1));
            v.cast_vote(cast[1], vote(5, 1));
            v.cast_vote(cast[2], vote(5, 1));
            v.status = status;
            let stands =
                matches!(status, VerdictStatus::Upheld | VerdictStatus::Due);
            if stands {
                assert_eq!(v.bounty_recipients(), vec![cast[1], cast[2]]);
            }
            assert_eq!(
                v.release_vote(&cast[0], false, false).unwrap(),
                VoteRelease::Released,
                "{status:?}"
            );
            assert!(v.release_vote(&cast[0], false, false).is_err());
            // The votes that convicted stay held until the verdict says
            // otherwise.
            assert_eq!(
                v.release_vote(&cast[1], false, false).is_err(),
                !stands,
                "{status:?}"
            );
        }
    }

    /// The crank passes a dissenting vote without charging it, whether or not
    /// it was released first, and counts it toward settling the overturn.
    #[test]
    fn the_crank_passes_a_dissenting_vote_uncharged() {
        let cast = voters(3);
        let mut v = fresh();
        v.cast_vote(cast[0], vote(5, 1));
        v.cast_vote(cast[1], vote(9, 1));
        v.cast_vote(cast[2], vote(5, 1));
        v.status = VerdictStatus::Overturned;
        v.settle_losing_vote(0, 40);
        v.release_vote(&cast[1], false, false).unwrap();
        v.settle_dissenting_vote(1);
        v.settle_losing_vote(2, 40);
        assert!(v.all_losers_settled());
        assert!(v.all_votes_released());
        assert_eq!(v.bounty_owed, 80);
    }

    #[test]
    fn a_reset_clears_the_vote_commitments() {
        let mut v = fresh();
        v.cast_vote(Pubkey::new_unique(), vote(5, 1));
        v.reset_for_round(2, 0);
        assert!(v.vote_commitments.is_empty());
        assert_eq!(v.committed_hash, [0; 32]);
    }
}
