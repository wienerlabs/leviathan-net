use crate::{sha256, sha256v};

/// Domain tag for the hash a trainer signs over its DisTrO payload, so no
/// other message the same key signs can stand in for it.
pub const PAYLOAD_COMMITMENT_DOMAIN: &[u8] = b"leviathan/payload-commitment/v1";

/// The run a payload commitment is bound to: the sha256 of the run id.
///
/// The same p2p key may train in more than one run. A signature made in one run
/// must not convict the key in another, so the run is part of what it signs.
pub fn payload_commitment_run(run_id: &str) -> [u8; 32] {
    sha256(run_id.as_bytes())
}

/// The hash a trainer signs for the payload it trained on
/// `batch_start..=batch_end` at `step` in the run `run`
/// (`payload_commitment_run`).
///
/// It has two layers so a program can check a conviction against it. The
/// payload itself is megabytes and never fits a transaction, so its bytes go
/// into `results_digest` (psyche-network's `results_digest`). The fields that
/// say which work the payload is - the step, the nonce and the batch - sit in a
/// fixed-width header that anyone can rebuild from a few dozen bytes, after the
/// run it was trained in. A
/// signature over this hash is then a signature over that step and that batch,
/// which is what lets the treasurer refuse a conviction citing something the
/// target never committed to (wienerlabs/leviathan-net#43).
pub fn payload_commitment_hash(
    run: &[u8; 32],
    step: u32,
    trainer_nonce: u32,
    batch_start: u64,
    batch_end: u64,
    results_digest: &[u8; 32],
) -> [u8; 32] {
    sha256v(&[
        PAYLOAD_COMMITMENT_DOMAIN,
        run,
        &step.to_be_bytes(),
        &trainer_nonce.to_be_bytes(),
        &batch_start.to_be_bytes(),
        &batch_end.to_be_bytes(),
        results_digest,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The layout is the domain tag, the run, the big-endian header and the
    /// digest. A program, the client and the verifier all rebuild it, so it is
    /// pinned against the concatenation itself.
    #[test]
    fn the_hash_is_over_the_domain_run_header_and_digest() {
        let mut bytes = PAYLOAD_COMMITMENT_DOMAIN.to_vec();
        bytes.extend_from_slice(&sha256(b"a run"));
        bytes.extend_from_slice(&STEP.to_be_bytes());
        bytes.extend_from_slice(&NONCE.to_be_bytes());
        bytes.extend_from_slice(&START.to_be_bytes());
        bytes.extend_from_slice(&END.to_be_bytes());
        bytes.extend_from_slice(&digest());
        assert_eq!(
            payload_commitment_hash(&run(), STEP, NONCE, START, END, &digest()),
            sha256(&bytes)
        );
    }

    #[test]
    fn every_field_changes_the_hash() {
        let base = payload_commitment_hash(&run(), STEP, NONCE, START, END, &digest());
        assert_ne!(
            base,
            payload_commitment_hash(&run(), STEP + 1, NONCE, START, END, &digest())
        );
        assert_ne!(
            base,
            payload_commitment_hash(&run(), STEP, NONCE + 1, START, END, &digest())
        );
        assert_ne!(
            base,
            payload_commitment_hash(&run(), STEP, NONCE, START + 1, END, &digest())
        );
        assert_ne!(
            base,
            payload_commitment_hash(&run(), STEP, NONCE, START, END + 1, &digest())
        );
        assert_ne!(
            base,
            payload_commitment_hash(&run(), STEP, NONCE, START, END, &sha256(b"other results"))
        );
    }

    /// The same payload signed in another run is another commitment.
    #[test]
    fn the_run_changes_the_hash() {
        assert_ne!(
            payload_commitment_hash(&run(), STEP, NONCE, START, END, &digest()),
            payload_commitment_hash(
                &payload_commitment_run("another run"),
                STEP,
                NONCE,
                START,
                END,
                &digest()
            )
        );
    }
}
