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

/// Mirrors `RUN_LAYOUT_BY_SIZE` in `leviathan-web/src/data/protocol.ts`.
#[test]
fn run_is_the_length_the_web_decoder_expects() {
    assert_eq!(Run::space_with_discriminator(), 237);
}

/// Mirrors the participant length check in the same file, which knows 65.
/// This generation is 4 bytes longer: `open_verdict_votes` (u32) follows
/// `bond_withdraw_delay_snapshot` at the end, counting the votes that still
/// hold the bond (wienerlabs/leviathan-net#39). The fields before it are where
/// they were, so a 65-byte decoder reads a 69-byte account correctly except
/// for the count; the length check has to learn 69.
#[test]
fn participant_is_the_length_the_web_decoder_expects() {
    assert_eq!(Participant::space_with_discriminator(), 69);
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
/// - 6470, this one, appends `vote_commitments`: the commitment hash each vote
///   convicts on, one 32-byte entry per voter, so votes are counted per payload
///   rather than refused when the first vote cited another. It comes after
///   `appeal_voters`, so every field a 4418 decoder reads is where it was.
///
/// A decoder has to learn 6470 before it reads accounts from the deployment
/// carrying it.
#[test]
fn audit_verdict_is_the_length_the_web_decoder_expects() {
    assert_eq!(AuditVerdict::space_with_discriminator(), 6470);
}
