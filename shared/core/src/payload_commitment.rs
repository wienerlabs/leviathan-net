use crate::{sha256, sha256v};

/// Domain tag for the hash a trainer signs over its DisTrO payload, so no
/// other message the same key signs can stand in for it. The version moved to
/// 2 when the epoch joined the header, so a v1 signature is never read as one
/// over a v2 header (wienerlabs/leviathan-net#67), and to 3 when the state the
/// trainer trained from joined it (wienerlabs/leviathan-net#89).
pub const PAYLOAD_COMMITMENT_DOMAIN: &[u8] = b"leviathan/payload-commitment/v3";

/// The run a payload commitment is bound to: the sha256 of the run id.
///
/// The same p2p key may train in more than one run. A signature made in one run
/// must not convict the key in another, so the run is part of what it signs.
pub fn payload_commitment_run(run_id: &str) -> [u8; 32] {
    sha256(run_id.as_bytes())
}

/// The hash a trainer signs for the payload it trained on
/// `batch_start..=batch_end` at `step` of the epoch that started at
/// `epoch_start`, in the run `run` (`payload_commitment_run`).
///
/// It has two layers so a program can check a conviction against it. The
/// payload itself is megabytes and never fits a transaction, so its bytes go
/// into `results_digest` (psyche-network's `results_digest`). The fields that
/// say which work the payload is - the epoch, the step, the nonce and the
/// batch - sit in a fixed-width header that anyone can rebuild from a few dozen
/// bytes, after the run it was trained in. A signature over this hash is then
/// a signature over that step and that batch, which is what lets the treasurer
/// refuse a conviction citing something the target never committed to
/// (wienerlabs/leviathan-net#43).
///
/// The step alone does not say which round it was. A run restarted from an
/// earlier step trains that step again, and a run id freed and taken again
/// starts again at step 1. The epoch's start, the coordinator's
/// `epoch_state.start_timestamp`, does. Steps only rise within an epoch. A
/// restart needs the run paused, and a paused run resumes only through a new
/// epoch. The coordinator starts an epoch only in a later second than the
/// waiting it began from, which itself follows the epoch before. A run id taken
/// again starts its epochs later still. So, on a cluster clock that never runs
/// back, no two epochs of a run id share a start, and a signature over one
/// round's work can never be read as one over another's
/// (wienerlabs/leviathan-net#67).
///
/// It also binds `state_digest`, the digest of the state the trainer trained
/// the payload from: its weights and its error-feedback buffer as the round
/// began. A verifier that audits the round later replays it from that state,
/// which the trainer reveals and which has to be the one it signed here
/// (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`). A trainer that
/// committed to no state, as one training through Python does, binds `None`,
/// which no state digest can stand in for.
#[allow(clippy::too_many_arguments)]
pub fn payload_commitment_hash(
    run: &[u8; 32],
    epoch_start: u64,
    step: u32,
    trainer_nonce: u32,
    batch_start: u64,
    batch_end: u64,
    results_digest: &[u8; 32],
    state_digest: Option<&[u8; 32]>,
) -> [u8; 32] {
    let (state_tag, state): (&[u8], &[u8]) = match state_digest {
        Some(digest) => (&[1], digest),
        None => (&[0], &[]),
    };
    sha256v(&[
        PAYLOAD_COMMITMENT_DOMAIN,
        run,
        &epoch_start.to_be_bytes(),
        &step.to_be_bytes(),
        &trainer_nonce.to_be_bytes(),
        &batch_start.to_be_bytes(),
        &batch_end.to_be_bytes(),
        results_digest,
        state_tag,
        state,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPOCH_START: u64 = 1_700_000_000;
    const STEP: u32 = 7;
    const NONCE: u32 = 3;
    const START: u64 = 16;
    const END: u64 = 23;

    fn digest() -> [u8; 32] {
        sha256(b"results")
    }

    fn run() -> [u8; 32] {
        payload_commitment_run("a run")
    }

    fn hash(
        epoch_start: u64,
        step: u32,
        nonce: u32,
        start: u64,
        end: u64,
        digest: &[u8; 32],
    ) -> [u8; 32] {
        payload_commitment_hash(
            &run(),
            epoch_start,
            step,
            nonce,
            start,
            end,
            digest,
            Some(&state()),
        )
    }

    fn state() -> [u8; 32] {
        sha256(b"the state trained from")
    }

    /// The layout is the domain tag, the run, the big-endian header, the
    /// results' digest, and the state's digest behind a tag that says whether
    /// there is one. A program, the client and the verifier all rebuild it, so
    /// it is pinned against the concatenation itself.
    #[test]
    fn the_hash_is_over_the_domain_run_header_and_digests() {
        assert_eq!(
            PAYLOAD_COMMITMENT_DOMAIN,
            b"leviathan/payload-commitment/v3"
        );
        let mut bytes = PAYLOAD_COMMITMENT_DOMAIN.to_vec();
        bytes.extend_from_slice(&sha256(b"a run"));
        bytes.extend_from_slice(&EPOCH_START.to_be_bytes());
        bytes.extend_from_slice(&STEP.to_be_bytes());
        bytes.extend_from_slice(&NONCE.to_be_bytes());
        bytes.extend_from_slice(&START.to_be_bytes());
        bytes.extend_from_slice(&END.to_be_bytes());
        bytes.extend_from_slice(&digest());
        let mut with_state = bytes.clone();
        with_state.push(1);
        with_state.extend_from_slice(&state());
        assert_eq!(
            hash(EPOCH_START, STEP, NONCE, START, END, &digest()),
            sha256(&with_state)
        );
        let mut without_state = bytes;
        without_state.push(0);
        assert_eq!(
            payload_commitment_hash(
                &run(),
                EPOCH_START,
                STEP,
                NONCE,
                START,
                END,
                &digest(),
                None
            ),
            sha256(&without_state)
        );
    }

    /// The state a trainer trained from is part of what it signs: another
    /// state, or none, is another commitment (wienerlabs/leviathan-net#89).
    #[test]
    fn the_state_changes_the_hash() {
        let base = hash(EPOCH_START, STEP, NONCE, START, END, &digest());
        let other_state = sha256(b"another state");
        for other in [Some(&other_state), None] {
            assert_ne!(
                base,
                payload_commitment_hash(
                    &run(),
                    EPOCH_START,
                    STEP,
                    NONCE,
                    START,
                    END,
                    &digest(),
                    other
                )
            );
        }
        // A missing state is not the state of all zeros.
        assert_ne!(
            payload_commitment_hash(
                &run(),
                EPOCH_START,
                STEP,
                NONCE,
                START,
                END,
                &digest(),
                None
            ),
            payload_commitment_hash(
                &run(),
                EPOCH_START,
                STEP,
                NONCE,
                START,
                END,
                &digest(),
                Some(&[0; 32])
            )
        );
    }

    #[test]
    fn every_field_changes_the_hash() {
        let base = hash(EPOCH_START, STEP, NONCE, START, END, &digest());
        for other in [
            hash(EPOCH_START, STEP + 1, NONCE, START, END, &digest()),
            hash(EPOCH_START, STEP, NONCE + 1, START, END, &digest()),
            hash(EPOCH_START, STEP, NONCE, START + 1, END, &digest()),
            hash(EPOCH_START, STEP, NONCE, START, END + 1, &digest()),
            hash(
                EPOCH_START,
                STEP,
                NONCE,
                START,
                END,
                &sha256(b"other results"),
            ),
        ] {
            assert_ne!(base, other);
        }
    }

    /// The same work trained again after a restart from an earlier step, in
    /// an epoch that started later, is another commitment
    /// (wienerlabs/leviathan-net#67).
    #[test]
    fn the_epoch_changes_the_hash() {
        assert_ne!(
            hash(EPOCH_START, STEP, NONCE, START, END, &digest()),
            hash(EPOCH_START + 1, STEP, NONCE, START, END, &digest())
        );
    }

    /// The same payload signed in another run is another commitment.
    #[test]
    fn the_run_changes_the_hash() {
        assert_ne!(
            hash(EPOCH_START, STEP, NONCE, START, END, &digest()),
            payload_commitment_hash(
                &payload_commitment_run("another run"),
                EPOCH_START,
                STEP,
                NONCE,
                START,
                END,
                &digest(),
                Some(&state())
            )
        );
    }
}
