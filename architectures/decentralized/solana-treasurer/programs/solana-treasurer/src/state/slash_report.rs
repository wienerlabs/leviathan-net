use anchor_lang::prelude::*;
use psyche_coordinator::Coordinator;

use crate::state::ConvictionCharge;
use crate::state::EjectionCharge;

/// Who convicted the target a report records.
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
pub enum SlashKind {
    /// The run authority, through `run_slash`.
    #[default]
    AuthoritySlash,
    /// Anyone, through `run_slash_equivocation`.
    Equivocation,
}

/// What a slash that just ejected its target records about itself.
pub struct SlashRecord {
    pub run: Pubkey,
    pub target: Pubkey,
    pub ejection: EjectionCharge,
    pub kind: SlashKind,
    pub reporter: Option<Pubkey>,
}

/// A slash no audit verdict accounts for, recorded when it ejects its target:
/// who reported it and what the ledger charges for it
/// (wienerlabs/leviathan-net#72).
///
/// A `run_slash` and an equivocation conviction eject their target like a
/// verdict does, and the ledger charges the ejection once, at the end of the
/// epoch. Nothing used to record who reported them, so the bounty on that
/// charge went to whatever token account the target itself named when it
/// left, a second wallet of its own included. Now the reporter is recorded
/// here, when the conviction is made, and the bounty is credited to its bond.
///
/// A client is ejected at most once an epoch, so the target and the start of
/// the epoch it was ejected in name one report, and one charge.
#[account()]
#[derive(Debug)]
pub struct SlashReport {
    pub bump: u8,
    /// At offset 9, as in `AuditVerdict`, so one filter finds a run's reports.
    pub run: Pubkey,
    pub target: Pubkey,
    /// The epoch the slash ejected its target in, and what the ledger charges
    /// the ejection at that epoch's end.
    pub charge_epoch_start: u64,
    pub charge_rate: u64,
    pub kind: SlashKind,
    /// Whether the target had a participant account when the report was
    /// made, and so counts it until it is charged.
    pub counted_on_target: bool,
    pub charge: ConvictionCharge,
    /// What the target's forfeit owes the reporter, once it is charged.
    pub bounty_owed: u64,
    pub bounty_paid: bool,
    /// Whose bond the bounty is credited to. None leaves it in the vault.
    /// Last, because a None is shorter than a key.
    pub reporter: Option<Pubkey>,
}

impl SlashReport {
    pub const SEEDS_PREFIX: &'static [u8] = b"SlashReport";

    /// Counted against the Borsh layout, with room for a reporter.
    pub fn space_with_discriminator() -> usize {
        8 + 1 // bump
            + 32 // run
            + 32 // target
            + 8 // charge_epoch_start
            + 8 // charge_rate
            + 1 // kind
            + 1 // counted_on_target
            + 1 // charge
            + 8 // bounty_owed
            + 1 // bounty_paid
            + 1 + 32 // reporter
    }

    /// A report account that nothing has written yet.
    pub fn is_fresh(&self) -> bool {
        self.run == Pubkey::default()
    }

    /// Records `record`, a slash that ejected its target where its ejection
    /// says the ledger will charge it, counted on the target's participant
    /// account if `counted_on_target`.
    pub fn record(
        &mut self,
        bump: u8,
        record: SlashRecord,
        counted_on_target: bool,
    ) {
        self.bump = bump;
        self.run = record.run;
        self.target = record.target;
        self.charge_epoch_start = record.ejection.epoch_start;
        self.charge_rate = record.ejection.rate;
        self.kind = record.kind;
        self.counted_on_target = counted_on_target;
        self.charge = ConvictionCharge::Ejected;
        self.bounty_owed = 0;
        self.bounty_paid = false;
        self.reporter = record.reporter;
    }

    /// Whether the ledger has charged the ejection this report records.
    pub fn ejection_charged(&self, coordinator: &Coordinator) -> bool {
        crate::state::epoch_charged(coordinator, self.charge_epoch_start)
    }

    /// Records that the target's bond paid `forfeit` for the slash. A reporter
    /// is owed `bounty` of it; with none, the vault keeps all of it.
    pub fn record_charge(&mut self, bounty: u64) {
        self.charge = ConvictionCharge::Charged;
        self.bounty_owed = match self.reporter {
            Some(_) => bounty,
            None => 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anchor_lang::AccountSerialize;

    fn report(reporter: Option<Pubkey>) -> SlashReport {
        let mut report = SlashReport {
            bump: 0,
            run: Pubkey::default(),
            target: Pubkey::default(),
            charge_epoch_start: 0,
            charge_rate: 0,
            kind: SlashKind::default(),
            counted_on_target: false,
            charge: ConvictionCharge::default(),
            bounty_owed: 0,
            bounty_paid: false,
            reporter: None,
        };
        assert!(report.is_fresh());
        report.record(
            254,
            SlashRecord {
                run: Pubkey::new_unique(),
                target: Pubkey::new_unique(),
                ejection: EjectionCharge {
                    epoch_start: 1_700_000_000,
                    rate: 200,
                },
                kind: SlashKind::Equivocation,
                reporter,
            },
            true,
        );
        report
    }

    #[test]
    fn a_full_report_fills_its_account_exactly() {
        let mut data = vec![];
        report(Some(Pubkey::new_unique()))
            .try_serialize(&mut data)
            .unwrap();
        assert_eq!(data.len(), SlashReport::space_with_discriminator());
        // A report naming nobody fits too, with room to spare.
        let mut shorter = vec![];
        report(None).try_serialize(&mut shorter).unwrap();
        assert_eq!(shorter.len(), SlashReport::space_with_discriminator() - 32);
    }

    #[test]
    fn the_run_sits_where_a_verdict_keeps_it() {
        let made = report(None);
        let mut data = vec![];
        made.try_serialize(&mut data).unwrap();
        assert_eq!(&data[9..41], made.run.as_ref());
    }

    #[test]
    fn a_recorded_report_awaits_its_charge() {
        let made = report(Some(Pubkey::new_unique()));
        assert!(!made.is_fresh());
        assert_eq!(made.charge, ConvictionCharge::Ejected);
        assert_eq!(made.charge_rate, 200);
        assert_eq!(made.bounty_owed, 0);
        assert!(!made.bounty_paid);
    }

    #[test]
    fn a_charged_report_owes_its_reporter_and_one_naming_nobody_owes_nothing() {
        let mut named = report(Some(Pubkey::new_unique()));
        named.record_charge(100);
        assert_eq!(named.charge, ConvictionCharge::Charged);
        assert_eq!(named.bounty_owed, 100);

        let mut nobody = report(None);
        nobody.record_charge(100);
        assert_eq!(nobody.charge, ConvictionCharge::Charged);
        assert_eq!(nobody.bounty_owed, 0);
    }
}
