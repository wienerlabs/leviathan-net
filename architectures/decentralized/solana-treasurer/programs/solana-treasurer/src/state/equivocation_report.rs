use anchor_lang::prelude::*;
use anchor_lang::solana_program::hash::hashv;

use crate::state::CommitmentEvidence;
use crate::state::Run;

/// What a reporter commits to before it reveals a proof of equivocation:
/// the proof and its own key, hashed (wienerlabs/leviathan-net#92).
///
/// A proof of equivocation is public as soon as it is sent: both payloads and
/// both signatures are in the transaction. Whoever saw it could copy it, put
/// itself in as the reporter, and land it first, taking the bounty from
/// whoever found the fraud. A proof is now taken only from a reporter that
/// committed to it in an earlier slot. A copier sees nothing but this hash
/// until the proof is revealed, and by then it is too late to commit.
///
/// The target's position on the roster is left out, since it can shift
/// between the commitment and the reveal; the target's key names it.
#[derive(Clone, Copy)]
pub struct EquivocationClaim<'a> {
    pub run: &'a Pubkey,
    pub target: &'a Pubkey,
    pub batch_start: u64,
    pub batch_end: u64,
    pub first: &'a CommitmentEvidence,
    pub second: &'a CommitmentEvidence,
}

impl EquivocationClaim<'_> {
    const DOMAIN: &'static [u8] = b"leviathan-equivocation-report-v1";

    /// The commitment `reporter` makes to this claim.
    pub fn commitment(&self, reporter: &Pubkey) -> [u8; 32] {
        let first = borsh_bytes(self.first);
        let second = borsh_bytes(self.second);
        hashv(&[
            Self::DOMAIN,
            self.run.as_ref(),
            self.target.as_ref(),
            &self.batch_start.to_le_bytes(),
            &self.batch_end.to_le_bytes(),
            &first,
            &second,
            reporter.as_ref(),
        ])
        .to_bytes()
    }
}

fn borsh_bytes(evidence: &CommitmentEvidence) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(CommitmentEvidence::SPACE);
    // Writing to a Vec cannot fail.
    evidence
        .serialize(&mut bytes)
        .expect("serializing to memory");
    bytes
}

/// A reporter's commitment to a proof of equivocation it has not revealed
/// yet, and the slot it was made in. Revealing the proof closes it.
#[account()]
#[derive(Debug)]
pub struct EquivocationCommitment {
    pub bump: u8,
    pub run: Pubkey,
    pub reporter: Pubkey,
    pub commitment: [u8; 32],
    pub slot: u64,
}

impl EquivocationCommitment {
    pub const SEEDS_PREFIX: &'static [u8] = b"EquivocationCommitment";

    /// Counted against the Borsh layout.
    pub fn space_with_discriminator() -> usize {
        8 + 1 // bump
            + 32 // run
            + 32 // reporter
            + 32 // commitment
            + 8 // slot
    }

    /// Whether a proof may be revealed against this commitment in `slot`: the
    /// commitment was made in an earlier one, so it was on chain before
    /// anyone could have seen the proof.
    pub fn revealable_in(&self, slot: u64) -> bool {
        self.slot < slot
    }
}

/// The most a run may give the reporter of an equivocation out of its
/// target's forfeit: less than all of it, so that a target reporting itself
/// from a second wallet always loses some of the charge
/// (wienerlabs/leviathan-net#92).
pub const MAX_EQUIVOCATION_BOUNTY_BPS: u16 = 9_999;

/// The share of an equivocating target's forfeit a run gives the reporter,
/// set by its main authority.
///
/// It is kept apart from `slash_bounty_bps`, which may be the whole forfeit:
/// an equivocation's reporter can be the target itself, from a second wallet
/// the chain cannot tell from an honest reporter. Kept in an account of its
/// own rather than in `Run`, whose length tells the dashboard's decoders its
/// generations apart.
#[account()]
#[derive(Debug)]
pub struct EquivocationBounty {
    pub bump: u8,
    pub run: Pubkey,
    pub bounty_bps: u16,
}

impl EquivocationBounty {
    pub const SEEDS_PREFIX: &'static [u8] = b"EquivocationBounty";

    /// Counted against the Borsh layout.
    pub fn space_with_discriminator() -> usize {
        8 + 1 // bump
            + 32 // run
            + 2 // bounty_bps
    }

    /// The bounty a run sets, read from the account at its PDA. A run that
    /// set none gives an equivocation's reporter nothing.
    pub fn from_pda_account(
        owner: &Pubkey,
        data: &[u8],
    ) -> Result<Option<EquivocationBounty>> {
        if data.is_empty() || owner != &crate::ID {
            return Ok(None);
        }
        Ok(Some(EquivocationBounty::try_deserialize(&mut &data[..])?))
    }
}

/// The share of an equivocating target's forfeit that goes to its reporter in
/// `run`: what the run set for equivocation, or nothing if it set none, and
/// never more than a slash's bounty in the run. Always less than the whole
/// forfeit, so reporting itself never makes equivocating free for a target.
pub fn equivocation_bounty_bps(
    run: &Run,
    bounty: Option<&EquivocationBounty>,
) -> u16 {
    bounty
        .map_or(0, |bounty| bounty.bounty_bps)
        .min(run.slash_bounty_bps)
        .min(MAX_EQUIVOCATION_BOUNTY_BPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(step: u32, digest: u8) -> CommitmentEvidence {
        CommitmentEvidence {
            step,
            results_digest: [digest; 32],
            ..CommitmentEvidence::default()
        }
    }

    fn run(slash_bounty_bps: u16) -> Run {
        Run {
            bump: 0,
            index: 0,
            main_authority: Pubkey::default(),
            join_authority: Pubkey::default(),
            coordinator_account: Pubkey::default(),
            coordinator_instance: Pubkey::default(),
            collateral_mint: Pubkey::default(),
            total_claimed_collateral_amount: 0,
            total_claimed_earned_points: 0,
            total_bonded_amount: 0,
            bond_minimum_amount: 0,
            bond_withdraw_delay_seconds: 0,
            slash_bounty_bps,
            challenge_window_seconds: 0,
            tie_breaker_committee_size: 0,
            appeal_window_seconds: 0,
        }
    }

    fn bounty(bounty_bps: u16) -> EquivocationBounty {
        EquivocationBounty {
            bump: 0,
            run: Pubkey::default(),
            bounty_bps,
        }
    }

    #[test]
    fn a_commitment_binds_its_reporter_and_every_part_of_the_proof() {
        let run = Pubkey::new_unique();
        let target = Pubkey::new_unique();
        let (first, second) = (evidence(7, 1), evidence(7, 2));
        let claim = EquivocationClaim {
            run: &run,
            target: &target,
            batch_start: 4,
            batch_end: 5,
            first: &first,
            second: &second,
        };
        let reporter = Pubkey::new_unique();
        let committed = claim.commitment(&reporter);
        assert_eq!(committed, claim.commitment(&reporter));
        // Another reporter, a copier, commits to something else.
        assert_ne!(committed, claim.commitment(&Pubkey::new_unique()));
        // And so does any other proof.
        let other_run = Pubkey::new_unique();
        let other_target = Pubkey::new_unique();
        let third = evidence(7, 3);
        for other in [
            EquivocationClaim {
                run: &other_run,
                ..claim
            },
            EquivocationClaim {
                target: &other_target,
                ..claim
            },
            EquivocationClaim {
                batch_start: 3,
                ..claim
            },
            EquivocationClaim {
                batch_end: 6,
                ..claim
            },
            EquivocationClaim {
                first: &third,
                ..claim
            },
            EquivocationClaim {
                second: &third,
                ..claim
            },
            // The two payloads swapped.
            EquivocationClaim {
                first: &second,
                second: &first,
                ..claim
            },
        ] {
            assert_ne!(other.commitment(&reporter), committed);
        }
    }

    #[test]
    fn a_proof_is_revealed_only_after_the_slot_it_was_committed_in() {
        let commitment = EquivocationCommitment {
            bump: 0,
            run: Pubkey::default(),
            reporter: Pubkey::default(),
            commitment: [0; 32],
            slot: 100,
        };
        assert!(!commitment.revealable_in(99));
        assert!(!commitment.revealable_in(100));
        assert!(commitment.revealable_in(101));
    }

    #[test]
    fn a_run_that_set_no_equivocation_bounty_pays_none() {
        assert_eq!(equivocation_bounty_bps(&run(10_000), None), 0);
    }

    #[test]
    fn the_equivocation_bounty_never_exceeds_the_slash_bounty_or_the_whole() {
        assert_eq!(
            equivocation_bounty_bps(&run(5_000), Some(&bounty(2_000))),
            2_000
        );
        assert_eq!(
            equivocation_bounty_bps(&run(1_000), Some(&bounty(2_000))),
            1_000
        );
        // An account written outside the setter cannot make it free either.
        assert_eq!(
            equivocation_bounty_bps(&run(10_000), Some(&bounty(10_000))),
            MAX_EQUIVOCATION_BOUNTY_BPS
        );
    }

    #[test]
    fn the_accounts_fill_their_space_exactly() {
        let commitment = EquivocationCommitment {
            bump: 1,
            run: Pubkey::new_unique(),
            reporter: Pubkey::new_unique(),
            commitment: [9; 32],
            slot: u64::MAX,
        };
        let mut data = vec![];
        commitment.try_serialize(&mut data).unwrap();
        assert_eq!(
            data.len(),
            EquivocationCommitment::space_with_discriminator()
        );
        let mut data = vec![];
        bounty(MAX_EQUIVOCATION_BOUNTY_BPS)
            .try_serialize(&mut data)
            .unwrap();
        assert_eq!(data.len(), EquivocationBounty::space_with_discriminator());
    }
}
