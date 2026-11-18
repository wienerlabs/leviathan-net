use anchor_lang::prelude::*;
use anchor_lang::solana_program::ed25519_program;
use anchor_lang::solana_program::sysvar::instructions::load_instruction_at_checked;
use psyche_coordinator::Coordinator;
use psyche_core::payload_commitment_hash;
use psyche_core::payload_commitment_run;

/// The payload a conviction is for, as the target committed to it.
///
/// The chain cannot recompute a gradient, so whether the payload was honest
/// remains the verifiers' claim. Whether the target committed to it does not
/// have to be a claim. The target signed the payload hash with its p2p key, and
/// that hash is rebuilt here from the epoch, the step, the nonce, the batch the
/// verdict names, and the digest of the payload's results. A conviction carries
/// the target's signature over that hash. It therefore names work the target
/// cannot disown, for exactly the round and batch it records
/// (wienerlabs/leviathan-net#43, #67).
#[derive(
    AnchorSerialize,
    AnchorDeserialize,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
)]
pub struct CommitmentEvidence {
    /// When the epoch the payload was trained in started: the coordinator's
    /// `epoch_state.start_timestamp` then. With the step, it names the round
    /// (wienerlabs/leviathan-net#67).
    pub epoch_start: u64,
    pub step: u32,
    pub trainer_nonce: u32,
    pub results_digest: [u8; 32],
    /// Whether the target committed to the state it trained the payload from
    /// (wienerlabs/leviathan-net#89).
    pub state_committed: bool,
    /// That state's digest, which the target's signature binds. All zeros,
    /// and not part of the hash, when it committed to none. A fixed width, so
    /// every field after the evidence sits where it does either way.
    pub state_digest: [u8; 32],
}

impl CommitmentEvidence {
    pub const SPACE: usize = 8 + 4 + 4 + 32 + 1 + 32;

    /// The state digest the target committed to, if it committed to one.
    pub fn state(&self) -> Option<&[u8; 32]> {
        self.state_committed.then_some(&self.state_digest)
    }

    /// `state_committed` and `state_digest` for `state`, the digest a payload
    /// names, if any: an absent one is all zeros.
    pub fn state_fields(state: Option<[u8; 32]>) -> (bool, [u8; 32]) {
        (state.is_some(), state.unwrap_or([0; 32]))
    }

    /// The hash the target signed, if this is a payload it committed to for
    /// `batch_start..=batch_end` in the run `run` (`commitment_run`).
    pub fn committed_hash(
        &self,
        run: &[u8; 32],
        batch_start: u64,
        batch_end: u64,
    ) -> [u8; 32] {
        payload_commitment_hash(
            run,
            self.epoch_start,
            self.step,
            self.trainer_nonce,
            batch_start,
            batch_end,
            &self.results_digest,
            self.state(),
        )
    }

    /// Whether the payload was trained in the epoch the coordinator is in, so
    /// a conviction cannot reach back for work an earlier epoch already settled.
    ///
    /// The step range alone does not say so. A run restarted from an earlier
    /// step trains that step again in a new epoch, so a step signed before the
    /// restart is back inside the range. The epoch it names has to be this one
    /// too (wienerlabs/leviathan-net#67).
    pub fn is_from_current_epoch(&self, coordinator: &Coordinator) -> bool {
        self.epoch_start == coordinator.epoch_state.start_timestamp
            && coordinator.epoch_state.start_step <= self.step
            && self.step <= coordinator.progress.step
    }
}

/// The run the coordinator trains, as payload commitments are bound to it. A
/// signature a target made in another run convicts it of nothing here.
pub fn commitment_run(coordinator: &Coordinator) -> [u8; 32] {
    payload_commitment_run(&String::from(&coordinator.run_id))
}

/// Byte layout of an ed25519 precompile instruction: a signature count and a
/// padding byte, then one fixed-size offsets entry per signature.
const SIGNATURE_OFFSETS_START: usize = 2;
const SIGNATURE_OFFSETS_SIZE: usize = 14;
const PUBKEY_SIZE: usize = 32;
const SIGNATURE_SIZE: usize = 64;
const MESSAGE_SIZE: usize = 32;
/// The instruction index an offsets entry uses for "this instruction".
const THIS_INSTRUCTION: u16 = u16::MAX;

fn u16_at(data: &[u8], at: usize) -> Option<u16> {
    let bytes = data.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn slice_at(data: &[u8], offset: u16, size: usize) -> Option<&[u8]> {
    let start = offset as usize;
    data.get(start..start.checked_add(size)?)
}

/// Whether the data of one ed25519 precompile instruction carries a signature
/// by `signer` over exactly `message`.
///
/// Only an entry whose key, signature and message all sit in the precompile
/// instruction itself counts. That way the bytes read here are the same bytes
/// the runtime verified. An entry pointing into another instruction could
/// otherwise have the signature checked over one thing while this function
/// reads another.
pub fn ed25519_data_proves(
    data: &[u8],
    signer: &[u8; 32],
    message: &[u8; 32],
) -> bool {
    let Some(&count) = data.first() else {
        return false;
    };
    (0..count as usize).any(|entry| {
        entry_proves(data, entry, signer, message).unwrap_or(false)
    })
}

fn entry_proves(
    data: &[u8],
    entry: usize,
    signer: &[u8; 32],
    message: &[u8; 32],
) -> Option<bool> {
    let at = SIGNATURE_OFFSETS_START
        .checked_add(entry.checked_mul(SIGNATURE_OFFSETS_SIZE)?)?;
    let signature_offset = u16_at(data, at)?;
    let signature_index = u16_at(data, at + 2)?;
    let pubkey_offset = u16_at(data, at + 4)?;
    let pubkey_index = u16_at(data, at + 6)?;
    let message_offset = u16_at(data, at + 8)?;
    let message_size = u16_at(data, at + 10)?;
    let message_index = u16_at(data, at + 12)?;

    if signature_index != THIS_INSTRUCTION
        || pubkey_index != THIS_INSTRUCTION
        || message_index != THIS_INSTRUCTION
        || message_size as usize != MESSAGE_SIZE
    {
        return Some(false);
    }
    // The signature has to be there for the runtime to have verified it.
    slice_at(data, signature_offset, SIGNATURE_SIZE)?;
    Some(
        slice_at(data, pubkey_offset, PUBKEY_SIZE)? == signer
            && slice_at(data, message_offset, MESSAGE_SIZE)? == message,
    )
}

/// Whether the transaction executing this instruction proves `signer` signed
/// `message`, through an ed25519 precompile instruction the runtime verified
/// before any program ran.
pub fn transaction_proves_signature(
    instructions_sysvar: &AccountInfo,
    signer: &[u8; 32],
    message: &[u8; 32],
) -> Result<bool> {
    let count = {
        let data = instructions_sysvar.try_borrow_data()?;
        u16_at(&data, 0).ok_or(ProgramError::InvalidAccountData)?
    };
    for index in 0..count as usize {
        let instruction =
            load_instruction_at_checked(index, instructions_sysvar)?;
        if instruction.program_id == ed25519_program::ID
            && ed25519_data_proves(&instruction.data, signer, message)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Entry<'a> {
        signer: [u8; 32],
        message: &'a [u8],
        indices: [u16; 3],
    }

    /// Precompile instruction data laid out as the runtime reads it: the
    /// count, the offsets entries, then each entry's key, signature and
    /// message. The signature bytes are not checked here; the runtime checks
    /// them before the program runs, and the memnet suite covers that.
    fn instruction_data(entries: &[Entry]) -> Vec<u8> {
        let body_start =
            SIGNATURE_OFFSETS_START + entries.len() * SIGNATURE_OFFSETS_SIZE;
        let mut offsets = vec![entries.len() as u8, 0];
        let mut body = Vec::new();
        for entry in entries {
            let pubkey_offset = (body_start + body.len()) as u16;
            body.extend_from_slice(&entry.signer);
            let signature_offset = (body_start + body.len()) as u16;
            body.extend_from_slice(&[5u8; SIGNATURE_SIZE]);
            let message_offset = (body_start + body.len()) as u16;
            body.extend_from_slice(entry.message);
            for field in [
                signature_offset,
                entry.indices[0],
                pubkey_offset,
                entry.indices[1],
                message_offset,
                entry.message.len() as u16,
                entry.indices[2],
            ] {
                offsets.extend_from_slice(&field.to_le_bytes());
            }
        }
        offsets.extend_from_slice(&body);
        offsets
    }

    const SIGNER: [u8; 32] = [7; 32];
    const OTHER: [u8; 32] = [8; 32];
    const MESSAGE: [u8; 32] = [9; 32];
    const HERE: [u16; 3] = [THIS_INSTRUCTION; 3];

    fn genuine() -> Entry<'static> {
        Entry {
            signer: SIGNER,
            message: &MESSAGE,
            indices: HERE,
        }
    }

    #[test]
    fn an_entry_naming_the_signer_and_message_proves_them() {
        let data = instruction_data(&[genuine()]);
        assert!(ed25519_data_proves(&data, &SIGNER, &MESSAGE));
    }

    #[test]
    fn another_signer_or_message_is_not_proven() {
        let data = instruction_data(&[genuine()]);
        assert!(!ed25519_data_proves(&data, &OTHER, &MESSAGE));
        assert!(!ed25519_data_proves(&data, &SIGNER, &OTHER));
    }

    /// Bytes read from another instruction are not the bytes this one was
    /// checked over.
    #[test]
    fn an_entry_reading_another_instruction_proves_nothing() {
        for slot in 0..3 {
            let mut indices = HERE;
            indices[slot] = 0;
            let data = instruction_data(&[Entry {
                indices,
                ..genuine()
            }]);
            assert!(!ed25519_data_proves(&data, &SIGNER, &MESSAGE));
        }
    }

    /// A signature over a longer message that starts with the hash is a
    /// signature over something else.
    #[test]
    fn a_message_longer_than_the_hash_proves_nothing() {
        let mut longer = MESSAGE.to_vec();
        longer.push(0);
        let data = instruction_data(&[Entry {
            message: &longer,
            ..genuine()
        }]);
        assert!(!ed25519_data_proves(&data, &SIGNER, &MESSAGE));
    }

    #[test]
    fn truncated_or_empty_data_proves_nothing() {
        let data = instruction_data(&[genuine()]);
        for len in 0..data.len() {
            assert!(!ed25519_data_proves(&data[..len], &SIGNER, &MESSAGE));
        }
        assert!(!ed25519_data_proves(
            &instruction_data(&[]),
            &SIGNER,
            &MESSAGE
        ));
    }

    /// Any entry may carry the proof, not only the first.
    #[test]
    fn a_later_entry_can_carry_the_proof() {
        let data = instruction_data(&[
            Entry {
                signer: OTHER,
                ..genuine()
            },
            genuine(),
        ]);
        assert!(ed25519_data_proves(&data, &SIGNER, &MESSAGE));
        assert!(ed25519_data_proves(&data, &OTHER, &MESSAGE));
    }

    #[test]
    fn the_committed_hash_is_the_payload_commitment_for_the_batch_and_run() {
        let evidence = CommitmentEvidence {
            epoch_start: 1_700_000_000,
            step: 4,
            trainer_nonce: 1,
            results_digest: [3; 32],
            state_committed: true,
            state_digest: [5; 32],
        };
        let run = payload_commitment_run("a run");
        assert_eq!(
            evidence.committed_hash(&run, 8, 15),
            payload_commitment_hash(
                &run,
                1_700_000_000,
                4,
                1,
                8,
                15,
                &[3; 32],
                Some(&[5; 32])
            )
        );
        // The state the target trained from is part of what it signed
        // (wienerlabs/leviathan-net#89).
        for (state_committed, state_digest) in
            [(true, [6; 32]), (false, [0; 32])]
        {
            assert_ne!(
                evidence.committed_hash(&run, 8, 15),
                CommitmentEvidence {
                    state_committed,
                    state_digest,
                    ..evidence
                }
                .committed_hash(&run, 8, 15),
                "{state_committed} {state_digest:?}"
            );
        }
        let stateless = CommitmentEvidence {
            state_committed: false,
            state_digest: [0; 32],
            ..evidence
        };
        assert_eq!(stateless.state(), None);
        assert_eq!(
            stateless.committed_hash(&run, 8, 15),
            payload_commitment_hash(
                &run,
                1_700_000_000,
                4,
                1,
                8,
                15,
                &[3; 32],
                None
            )
        );
        assert_ne!(
            evidence.committed_hash(&run, 8, 15),
            CommitmentEvidence {
                epoch_start: 1_700_000_090,
                ..evidence
            }
            .committed_hash(&run, 8, 15)
        );
        assert_ne!(
            evidence.committed_hash(&run, 8, 15),
            evidence.committed_hash(&run, 8, 16)
        );
        assert_ne!(
            evidence.committed_hash(&run, 8, 15),
            evidence.committed_hash(
                &payload_commitment_run("another run"),
                8,
                15
            )
        );
    }
}
