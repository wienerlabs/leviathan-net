use anchor_lang::prelude::*;

pub const MAX_VERDICT_VOTERS: usize = 64;
pub const MAX_APPEAL_VOTERS: usize = 64;

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
    pub pending_since_unix: i64,
    /// When the challenge was opened, so the appeal can be given a deadline of
    /// its own. Without one a free challenge parked a correct conviction for
    /// good (finding 10).
    pub challenged_since_unix: i64,
    pub challenger: Pubkey,
    pub overturn_count: u16,
    pub uphold_count: u16,
    pub settled_count: u16,
    pub voters: Vec<Pubkey>,
    pub appeal_voters: Vec<Pubkey>,
}

impl AuditVerdict {
    pub const SEEDS_PREFIX: &'static [u8] = b"AuditVerdict";

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
            + 8 // pending_since_unix
            + 8 // challenged_since_unix
            + 32 // challenger
            + 2 // overturn_count
            + 2 // uphold_count
            + 2 // settled_count
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
    /// anyone and leaves nobody owing anything, so it simply lapses - which is
    /// how votes stop carrying from one committee's draw to the next
    /// (finding 3). What must not be overwritten is a conviction waiting to
    /// finalise, an appeal in flight, or an overturn whose losing verifiers have
    /// not all been settled.
    pub fn is_settled(&self) -> bool {
        match self.status {
            VerdictStatus::Voting | VerdictStatus::Upheld => true,
            VerdictStatus::Overturned => {
                self.settled_count as usize >= self.voters.len()
            },
            VerdictStatus::SlashPending | VerdictStatus::Challenged => false,
        }
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
        self.pending_since_unix = 0;
        self.challenged_since_unix = 0;
        self.challenger = Pubkey::default();
        self.overturn_count = 0;
        self.uphold_count = 0;
        self.settled_count = 0;
        self.voters.clear();
        self.appeal_voters.clear();
    }
}
