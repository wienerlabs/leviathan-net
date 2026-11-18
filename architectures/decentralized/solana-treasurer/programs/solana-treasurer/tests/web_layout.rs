//! Account lengths the web dashboard keys its Borsh decoders on.
//!
//! Four generations of `Run` and three of `AuditVerdict` are live on devnet at
//! once, and length is the only thing on chain that tells them apart. The
//! dashboard picks a decoder by length, so a size changing here silently moves
//! every account of that generation into the wrong branch, where a short
//! account gets read with the long layout and invents a bond floor rather than
//! failing.

use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::Participant;
use psyche_solana_treasurer::state::Run;
use psyche_solana_treasurer::state::SlashReport;

/// Mirrors `RUN_LAYOUT_BY_SIZE` in `leviathan-web/src/data/protocol.ts`.
#[test]
fn run_is_the_length_the_web_decoder_expects() {
    assert_eq!(Run::space_with_discriminator(), 237);
}

/// Mirrors the participant length check in the same file, which knows 65.
/// Two generations followed it, each appending a u32 at the end:
///
/// - 69 adds `open_verdict_votes`, after `bond_withdraw_delay_snapshot`,
///   counting the votes that still hold the bond (wienerlabs/leviathan-net#39).
/// - 73, this one, adds `uncharged_slash_reports`, counting the slashes no
///   verdict accounts for that still hold the bond until they are charged
///   (wienerlabs/leviathan-net#72).
///
/// The fields before each are where they were, so a shorter decoder reads a
/// longer account correctly except for the counts. The length check has to
/// learn 73.
#[test]
fn participant_is_the_length_the_web_decoder_expects() {
    assert_eq!(Participant::space_with_discriminator(), 73);
}

/// A new account the dashboard does not decode yet: a slash no verdict
/// accounts for, with the reporter it records (wienerlabs/leviathan-net#72).
/// Its run sits at offset 9, as in `AuditVerdict`, so one filter finds both.
/// It is allocated at this length, with room for a reporter. One naming no
/// reporter serializes 32 bytes shorter, and the rest of the account is zero.
#[test]
fn slash_report_is_the_length_it_is_allocated_at() {
    assert_eq!(SlashReport::space_with_discriminator(), 134);
}

/// Mirrors `VERDICT_LAYOUT_BY_SIZE` in the same file, which stops at 4328
/// (`appealsWithDeadline`). Five generations followed it:
///
/// - 4368 records the payload the target signed: `evidence` (step u32,
///   trainer nonce u32, results digest 32 bytes) sits between `batch_end` and
///   `pending_since_unix` (wienerlabs/leviathan-net#43).
/// - 4392 adds three u64 fields between `settled_count` and `voters`
///   (wienerlabs/leviathan-net#39): `released_voters` (one bit per voter),
///   `bench_bounty_owed` and `bench_paid` (one bit per appeal voter). Read
///   with the 4368 layout, they land in the voter list's length and every
///   voter after it is misread.
/// - 4400 puts the epoch the payload was trained in (`epoch_start`, u64) at
///   the head of `evidence`, before the step (wienerlabs/leviathan-net#67).
///   Every field after it moves 8 bytes.
/// - 4418 records how far a conviction has reached the target's
///   bond, and whether the bench upheld it (wienerlabs/leviathan-net#64). The
///   two bounty fields keep their place under new names, `bounty_owed` and
///   `bounty_paid`, and are followed by `charge` (one byte: 0 unlanded,
///   1 ejected, 2 charged), `charge_epoch_start` (u64), `charge_rate` (u64)
///   and `appeal_upheld` (one byte), before `voters`. The voter lists move 18
///   bytes.
/// - 6470 appends `vote_commitments`: the commitment hash each vote
///   convicts on, one 32-byte entry per voter, so votes are counted per payload
///   rather than refused when the first vote cited another. It comes after
///   `appeal_voters`, so every field a 4418 decoder reads is where it was.
/// - 6503, this one, ends `evidence` with the state the target trained the
///   payload from (wienerlabs/leviathan-net#89): `state_committed` (one byte)
///   and `state_digest` (32 bytes, zeros when not committed), before
///   `pending_since_unix`. Every field after the evidence moves 33 bytes.
///
/// A decoder has to learn 6503 before it reads accounts from the deployment
/// carrying it.
#[test]
fn audit_verdict_is_the_length_the_web_decoder_expects() {
    assert_eq!(AuditVerdict::space_with_discriminator(), 6503);
}
