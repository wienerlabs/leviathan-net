use anchor_lang::prelude::*;

pub const MAX_VERDICT_VOTERS: usize = 64;

#[account]
#[derive(Debug)]
pub struct AuditVerdict {
    pub bump: u8,
    pub run: Pubkey,
    pub target: Pubkey,
    pub epoch: u16,
    pub verdict_count: u16,
    pub resolved: bool,
    pub committed_hash: [u8; 32],
    pub replayed_hash: [u8; 32],
    pub voters: Vec<Pubkey>,
}

impl AuditVerdict {
    pub const SEEDS_PREFIX: &'static [u8] = b"AuditVerdict";

    pub fn space_with_discriminator() -> usize {
        8 + 1 + 32 + 32 + 2 + 2 + 1 + 32 + 32 + 4 + MAX_VERDICT_VOTERS * 32
    }

    pub fn reset_for_epoch(&mut self, epoch: u16) {
        self.epoch = epoch;
        self.verdict_count = 0;
        self.resolved = false;
        self.voters.clear();
        self.committed_hash = [0u8; 32];
        self.replayed_hash = [0u8; 32];
    }
}
