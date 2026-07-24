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
    pub status: VerdictStatus,
    pub verdict_count: u16,
    pub committed_hash: [u8; 32],
    pub replayed_hash: [u8; 32],
    pub target_index: u64,
    pub batch_start: u64,
    pub batch_end: u64,
    pub pending_since_unix: i64,
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
        8 + 1
            + 32
            + 32
            + 2
            + 1
            + 2
            + 32
            + 32
            + 8
            + 8
            + 8
            + 8
            + 32
            + 2
            + 2
            + 2
            + 4
            + MAX_VERDICT_VOTERS * 32
            + 4
            + MAX_APPEAL_VOTERS * 32
    }

    pub fn reset_for_epoch(&mut self, epoch: u16) {
        self.epoch = epoch;
        self.status = VerdictStatus::Voting;
        self.verdict_count = 0;
        self.committed_hash = [0u8; 32];
        self.replayed_hash = [0u8; 32];
        self.target_index = 0;
        self.batch_start = 0;
        self.batch_end = 0;
        self.pending_since_unix = 0;
        self.challenger = Pubkey::default();
        self.overturn_count = 0;
        self.uphold_count = 0;
        self.settled_count = 0;
        self.voters.clear();
        self.appeal_voters.clear();
    }
}
