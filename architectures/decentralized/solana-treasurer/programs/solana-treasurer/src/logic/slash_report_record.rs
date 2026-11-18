use anchor_lang::prelude::*;

use crate::state::Participant;
pub use crate::state::SlashRecord;
use crate::state::SlashReport;
use crate::ProgramError;

/// Refuses a reporter that is the target itself. A slash's bounty is paid
/// out of the target's own forfeit, so the target naming itself would take
/// part of it back (wienerlabs/leviathan-net#72).
pub fn check_reporter(
    target: &Pubkey,
    reporter: Option<&Pubkey>,
) -> Result<()> {
    match reporter {
        Some(reporter) if reporter == target => {
            err!(ProgramError::ReporterIsTarget)
        },
        _ => Ok(()),
    }
}

/// Writes `record` to `report`, a report nothing has written before, and
/// counts it on the target's participant account at `target_participant` if
/// the target has one, so its bond is held until the report is charged.
///
/// The coordinator ejects a client at most once an epoch, and a report is
/// named by its target and the epoch it was ejected in, so a slash that
/// ejected its target always finds its report unwritten.
pub fn record_slash_report<'info>(
    report: &mut SlashReport,
    bump: u8,
    target_participant: &AccountInfo<'info>,
    record: SlashRecord,
) -> Result<()> {
    if !report.is_fresh() {
        return err!(ProgramError::InvalidParameter);
    }
    let mut participant = {
        let data = target_participant.try_borrow_data()?;
        Participant::from_pda_account(target_participant.owner, &data)?
    };
    let counted_on_target = match participant.as_mut() {
        Some(participant) => {
            participant.open_slash_report()?;
            let mut data = target_participant.try_borrow_mut_data()?;
            participant.try_serialize(&mut &mut data[..])?;
            true
        },
        None => false,
    };
    report.record(bump, record, counted_on_target);
    Ok(())
}
