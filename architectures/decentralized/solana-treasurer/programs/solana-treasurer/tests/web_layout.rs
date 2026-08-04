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

/// Mirrors the participant length check in the same file.
#[test]
fn participant_is_the_length_the_web_decoder_expects() {
    assert_eq!(Participant::space_with_discriminator(), 65);
}

/// Mirrors `VERDICT_LAYOUT_BY_SIZE` in the same file.
#[test]
fn audit_verdict_is_the_length_the_web_decoder_expects() {
    assert_eq!(AuditVerdict::space_with_discriminator(), 4328);
}
