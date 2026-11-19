//! The disputes a run's verdict accounts show, counted as they move.
//!
//! The treasurer keeps one `AuditVerdict` account per (run, target) and reuses
//! it for every dispute against that target, so an account shows only the
//! latest dispute. A dispute is named by its target and the round whose
//! committee voted on it. The ledger counts each one once as it reaches a
//! conviction, an overturn, or lapses, however many polls see it there.
//!
//! A dispute can also be replaced between two polls. The treasurer lets a new
//! dispute take an account over only once the old one is settled
//! (`AuditVerdict::is_settled`), so what the account last showed says how the
//! old one ended:
//!
//! - last seen `Voting`, it lapsed. It could only have been convicted instead
//!   by reaching a quorum, which ejects the target, and an ejected target
//!   cannot be accused again until it rejoins in a later epoch. No epoch
//!   passes between two polls.
//! - last seen `SlashPending` or `Challenged`, its conviction was counted, but
//!   whether an appeal overturned it was not seen. That is counted on its own,
//!   so a missed overturn is reported rather than lost.
//! - last seen `Upheld`, `Due` or `Overturned`, its outcome was already seen.
//!
//! The first look only learns what the accounts show. It counts nothing, so a
//! counter starts at zero and a restarted indexer does not report old disputes
//! as new ones: a counter that restarts above zero reads to Prometheus either
//! as no increase or as a fresh one (wienerlabs/leviathan-net#57).

use std::collections::HashMap;
use std::collections::HashSet;

use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::VerdictStatus;

/// Every status a verdict can be in, in the order [`VerdictTally::by_status`]
/// counts them.
pub const VERDICT_STATUSES: [VerdictStatus; 6] = [
    VerdictStatus::Voting,
    VerdictStatus::SlashPending,
    VerdictStatus::Challenged,
    VerdictStatus::Upheld,
    VerdictStatus::Overturned,
    VerdictStatus::Due,
];

/// The name a status is exported under.
pub fn status_label(status: VerdictStatus) -> &'static str {
    match status {
        VerdictStatus::Voting => "voting",
        VerdictStatus::SlashPending => "slash_pending",
        VerdictStatus::Challenged => "challenged",
        VerdictStatus::Upheld => "upheld",
        VerdictStatus::Overturned => "overturned",
        VerdictStatus::Due => "due",
    }
}

/// Where the run stands: its epoch and the height of its current round, the
/// two numbers the treasurer checks a vote against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundPosition {
    pub epoch: u16,
    pub round_height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Dispute {
    target: [u8; 32],
    epoch: u16,
    round_height: u32,
}

impl Dispute {
    fn of(verdict: &AuditVerdict) -> Self {
        Self {
            target: verdict.target.to_bytes(),
            epoch: verdict.epoch,
            round_height: verdict.round_height,
        }
    }

    /// A dispute still `Voting` once its round is behind the run can take no
    /// more votes (`run_submit_audit_verdict` only counts a vote in the
    /// verdict's own round), so it lapsed without a quorum. Compared strictly,
    /// so a verdict read after the coordinator, from a round the coordinator
    /// read had not reached yet, is never taken for lapsed.
    fn is_behind(&self, now: RoundPosition) -> bool {
        (self.epoch, self.round_height) < (now.epoch, now.round_height)
    }
}

/// What one look at a run's verdict accounts found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerdictTally {
    /// How many verdict accounts are in each of [`VERDICT_STATUSES`].
    pub by_status: [u64; 6],
    /// Disputes seen reaching a committee's quorum since the last look. A run
    /// authority's own slash writes no verdict account and is not among them;
    /// it shows as an ejected client and in the slashed points.
    pub new_convictions: u64,
    /// Convictions seen overturned on appeal since the last look: the target
    /// was innocent, and the chain says so.
    pub new_overturned: u64,
    /// Disputes seen lapsing since the last look: at least one bonded
    /// verifier voted fraud, with the target's signed commitment, and the
    /// quorum never came.
    pub new_lapsed: u64,
    /// Convictions whose account was handed to a new dispute before the
    /// ledger saw whether an appeal overturned them.
    pub new_unseen_outcomes: u64,
}

/// What the ledger knows of a run's disputes: what each account showed at the
/// last look, and the disputes already counted.
#[derive(Debug, Default)]
pub struct VerdictLedger {
    started: bool,
    last_seen: HashMap<[u8; 32], (Dispute, VerdictStatus)>,
    convicted: HashSet<Dispute>,
    overturned: HashSet<Dispute>,
    lapsed: HashSet<Dispute>,
}

impl VerdictLedger {
    /// Tallies `verdicts`, the run's verdict accounts as they stand at `now`.
    pub fn observe(&mut self, now: RoundPosition, verdicts: &[AuditVerdict]) -> VerdictTally {
        let counting = self.started;
        self.started = true;
        let mut tally = VerdictTally::default();
        let mut seen = HashSet::with_capacity(verdicts.len());

        for verdict in verdicts {
            let position = VERDICT_STATUSES
                .iter()
                .position(|status| *status == verdict.status)
                .expect("VERDICT_STATUSES lists every status");
            tally.by_status[position] += 1;

            let dispute = Dispute::of(verdict);
            seen.insert(dispute);
            if let Some((replaced, status)) = self.last_seen.get(&dispute.target).copied() {
                if replaced != dispute {
                    self.end_unseen(replaced, status, &mut tally);
                }
            }
            self.last_seen
                .insert(dispute.target, (dispute, verdict.status));

            let count =
                |set: &mut HashSet<Dispute>| -> u64 { u64::from(set.insert(dispute) && counting) };
            match verdict.status {
                VerdictStatus::Voting => {
                    if dispute.is_behind(now) {
                        tally.new_lapsed += count(&mut self.lapsed);
                    }
                }
                VerdictStatus::Overturned => {
                    tally.new_convictions += count(&mut self.convicted);
                    tally.new_overturned += count(&mut self.overturned);
                }
                VerdictStatus::SlashPending
                | VerdictStatus::Challenged
                | VerdictStatus::Upheld
                | VerdictStatus::Due => {
                    tally.new_convictions += count(&mut self.convicted);
                }
            }
        }

        // A dispute an account no longer shows was overwritten by a later one
        // and can never be shown again, so forgetting it keeps the ledger as
        // small as the run's verdict accounts.
        for set in [&mut self.convicted, &mut self.overturned, &mut self.lapsed] {
            set.retain(|dispute| seen.contains(dispute));
        }
        tally
    }

    /// Accounts for `replaced`, last seen in `status`, which a new dispute took
    /// its account over from between two looks.
    fn end_unseen(&mut self, replaced: Dispute, status: VerdictStatus, tally: &mut VerdictTally) {
        match status {
            VerdictStatus::Voting => {
                if !self.lapsed.contains(&replaced) {
                    tally.new_lapsed += 1;
                }
            }
            VerdictStatus::SlashPending | VerdictStatus::Challenged => {
                tally.new_unseen_outcomes += 1;
            }
            VerdictStatus::Upheld | VerdictStatus::Due | VerdictStatus::Overturned => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anchor_lang::prelude::Pubkey;
    use psyche_solana_treasurer::state::CommitmentEvidence;
    use psyche_solana_treasurer::state::VerdictKind;

    fn verdict(target: u8, epoch: u16, round_height: u32, status: VerdictStatus) -> AuditVerdict {
        AuditVerdict {
            bump: 0,
            run: Pubkey::new_from_array([9; 32]),
            target: Pubkey::new_from_array([target; 32]),
            epoch,
            round_height,
            status,
            verdict_count: 0,
            committed_hash: [0; 32],
            replayed_hash: [0; 32],
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
            charge: Default::default(),
            charge_epoch_start: 0,
            charge_rate: 0,
            appeal_upheld: false,
            voters: vec![],
            appeal_voters: vec![],
            vote_commitments: vec![],
            kind: VerdictKind::Replay,
        }
    }

    const NOW: RoundPosition = RoundPosition {
        epoch: 2,
        round_height: 5,
    };

    /// A ledger that has had its first look, at a run with no disputes yet.
    fn watching() -> VerdictLedger {
        let mut ledger = VerdictLedger::default();
        assert_eq!(ledger.observe(NOW, &[]), VerdictTally::default());
        ledger
    }

    #[test]
    fn every_status_is_counted_and_named() {
        let verdicts: Vec<AuditVerdict> = VERDICT_STATUSES
            .iter()
            .enumerate()
            .map(|(i, status)| verdict(i as u8, NOW.epoch, NOW.round_height, *status))
            .collect();
        let tally = VerdictLedger::default().observe(NOW, &verdicts);
        assert_eq!(tally.by_status, [1; 6]);
        let labels: Vec<&str> = VERDICT_STATUSES.iter().map(|s| status_label(*s)).collect();
        assert_eq!(
            labels,
            [
                "voting",
                "slash_pending",
                "challenged",
                "upheld",
                "overturned",
                "due"
            ]
        );
    }

    #[test]
    fn the_first_look_learns_what_is_there_and_counts_nothing() {
        let verdicts = [
            verdict(1, 2, 3, VerdictStatus::Overturned),
            verdict(2, 2, 4, VerdictStatus::SlashPending),
            verdict(3, 1, 9, VerdictStatus::Voting),
        ];
        let mut ledger = VerdictLedger::default();
        let first = ledger.observe(NOW, &verdicts);
        assert_eq!(
            first,
            VerdictTally {
                by_status: [1, 1, 0, 0, 1, 0],
                ..VerdictTally::default()
            }
        );
        // What it learned is not counted later either.
        let second = ledger.observe(NOW, &verdicts);
        assert_eq!(second, first);
        // But what moves after it is.
        let moved = [
            verdict(1, 2, 3, VerdictStatus::Overturned),
            verdict(2, 2, 4, VerdictStatus::Challenged),
            verdict(3, 1, 9, VerdictStatus::Voting),
            verdict(4, 2, 5, VerdictStatus::SlashPending),
        ];
        let third = ledger.observe(NOW, &moved);
        assert_eq!(
            (
                third.new_convictions,
                third.new_overturned,
                third.new_lapsed
            ),
            (1, 0, 0)
        );
    }

    #[test]
    fn a_conviction_is_counted_once_however_long_it_stands() {
        let mut ledger = watching();
        let pending = [verdict(1, 2, 5, VerdictStatus::SlashPending)];
        assert_eq!(ledger.observe(NOW, &pending).new_convictions, 1);
        assert_eq!(ledger.observe(NOW, &pending).new_convictions, 0);
        // The same dispute moving through its appeal is still one conviction.
        let challenged = [verdict(1, 2, 5, VerdictStatus::Challenged)];
        assert_eq!(ledger.observe(NOW, &challenged).new_convictions, 0);
        let upheld = [verdict(1, 2, 5, VerdictStatus::Upheld)];
        let tally = ledger.observe(NOW, &upheld);
        assert_eq!((tally.new_convictions, tally.new_overturned), (0, 0));
    }

    #[test]
    fn an_overturn_counts_once_as_a_conviction_and_once_as_overturned() {
        let mut ledger = watching();
        ledger.observe(NOW, &[verdict(1, 2, 5, VerdictStatus::Challenged)]);
        let overturned = [verdict(1, 2, 5, VerdictStatus::Overturned)];
        let tally = ledger.observe(NOW, &overturned);
        assert_eq!((tally.new_convictions, tally.new_overturned), (0, 1));
        assert_eq!(ledger.observe(NOW, &overturned).new_overturned, 0);
    }

    #[test]
    fn a_vote_lapses_only_once_its_round_is_behind_the_run() {
        let mut ledger = watching();
        let open = [verdict(1, 2, 5, VerdictStatus::Voting)];
        assert_eq!(ledger.observe(NOW, &open).new_lapsed, 0);
        let next_round = RoundPosition {
            epoch: 2,
            round_height: 6,
        };
        assert_eq!(ledger.observe(next_round, &open).new_lapsed, 1);
        assert_eq!(ledger.observe(next_round, &open).new_lapsed, 0);
        // A later epoch puts every earlier round behind it.
        let mut ledger = watching();
        let from_last_epoch = [verdict(1, 1, 40, VerdictStatus::Voting)];
        ledger.observe(
            RoundPosition {
                epoch: 1,
                round_height: 40,
            },
            &from_last_epoch,
        );
        assert_eq!(ledger.observe(NOW, &from_last_epoch).new_lapsed, 1);
    }

    #[test]
    fn a_verdict_ahead_of_the_coordinator_read_is_not_lapsed() {
        let ahead = [verdict(1, 2, 6, VerdictStatus::Voting)];
        assert_eq!(watching().observe(NOW, &ahead).new_lapsed, 0);
    }

    #[test]
    fn a_vote_replaced_before_it_was_seen_lapsing_is_counted_as_a_lapse() {
        let mut ledger = watching();
        ledger.observe(NOW, &[verdict(1, 2, 5, VerdictStatus::Voting)]);
        // Before the next look, the round ended and a new dispute took the
        // account over in the next one.
        let later = RoundPosition {
            epoch: 2,
            round_height: 6,
        };
        let replaced = ledger.observe(later, &[verdict(1, 2, 6, VerdictStatus::Voting)]);
        assert_eq!(replaced.new_lapsed, 1);
        // A vote already counted as lapsed is not counted again when replaced.
        let mut ledger = watching();
        ledger.observe(NOW, &[verdict(1, 2, 4, VerdictStatus::Voting)]);
        let again = ledger.observe(NOW, &[verdict(1, 2, 5, VerdictStatus::SlashPending)]);
        assert_eq!((again.new_lapsed, again.new_convictions), (0, 1));
    }

    #[test]
    fn a_conviction_replaced_before_its_appeal_was_seen_is_an_unseen_outcome() {
        for last_seen in [VerdictStatus::SlashPending, VerdictStatus::Challenged] {
            let mut ledger = watching();
            ledger.observe(NOW, &[verdict(1, 2, 5, last_seen)]);
            let later = RoundPosition {
                epoch: 3,
                round_height: 1,
            };
            let replaced = ledger.observe(later, &[verdict(1, 3, 1, VerdictStatus::Voting)]);
            assert_eq!(replaced.new_unseen_outcomes, 1, "{last_seen:?}");
            assert_eq!((replaced.new_lapsed, replaced.new_overturned), (0, 0));
        }
        // An outcome already seen is not unseen.
        for last_seen in [
            VerdictStatus::Upheld,
            VerdictStatus::Due,
            VerdictStatus::Overturned,
        ] {
            let mut ledger = watching();
            ledger.observe(NOW, &[verdict(1, 2, 5, last_seen)]);
            let replaced = ledger.observe(
                RoundPosition {
                    epoch: 3,
                    round_height: 1,
                },
                &[verdict(1, 3, 1, VerdictStatus::Voting)],
            );
            assert_eq!(
                (replaced.new_unseen_outcomes, replaced.new_lapsed),
                (0, 0),
                "{last_seen:?}"
            );
        }
    }

    #[test]
    fn a_new_dispute_on_the_same_account_is_counted_again() {
        let mut ledger = watching();
        assert_eq!(
            ledger
                .observe(NOW, &[verdict(1, 2, 5, VerdictStatus::Upheld)])
                .new_convictions,
            1
        );
        let later = RoundPosition {
            epoch: 3,
            round_height: 2,
        };
        assert_eq!(
            ledger
                .observe(later, &[verdict(1, 3, 2, VerdictStatus::SlashPending)])
                .new_convictions,
            1
        );
        // The overwritten dispute is forgotten.
        assert_eq!(ledger.convicted.len(), 1);
    }
}
