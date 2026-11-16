//! Where a round's seed comes from (wienerlabs/leviathan-net#62).
//!
//! A round's seed decides its committee, so nobody may choose it. It used to be
//! the hash of the clock and slot of whatever transaction started the round,
//! both known in advance, so its sender could pick the slot whose committee
//! suited it.
//!
//! Now the first transaction to find a round due, its phase over, commits to
//! its own slot, whose bank hash nobody knows until that slot is done. The
//! round starts on that hash, read back from the `SlotHashes` sysvar by a later
//! transaction, and in between nothing that shapes the round is taken: no
//! witness, no health check, no pause. Whoever learns the seed first can no
//! longer act on it through the coordinator.
//!
//! What is left is not open to a client or a ticker as such:
//!
//! - The leader of the committed slot builds the block whose hash is the seed.
//!   It can try as many variants of its block as it can build in the slot and
//!   keep the one whose committee suits it, so it chooses that round's seed.
//!   A validator commits the rounds whose first due tick lands in its slots:
//!   about its share of stake, and somewhat more if it also ticks, since it
//!   orders its own block first.
//! - Ejections are not frozen. The vote that reaches a conviction's quorum, or
//!   the permissionless crank that finalises it, ejects a client, and an
//!   ejection landed before the round starts drops that client from the
//!   committee. Whoever holds a conviction's last step can land it before or
//!   after the round starts, once the seed is known: one choice between two
//!   committees per conviction. Freezing it would refuse a vote that
//!   convicts, to remove a choice far smaller than the leader's. In a run
//!   without the treasurer, the run's authority can eject any client at any
//!   time, so it can choose the committee this way too; in a bonded run that
//!   authority is the treasurer's run, which ejects only on a conviction.
//!
//! The first is the limit of randomness drawn from the chain itself, which a
//! VRF or a beacon would remove. The second would go with a committee drawn
//! from the roster as it stood at the commitment.

use psyche_core::sha256v;

/// Separates round seeds from every other hash of a slot hash.
pub const ROUND_SEED_DOMAIN: &[u8] = b"leviathan round seed v1";

const SLOT_SIZE: usize = std::mem::size_of::<u64>();
const HASH_SIZE: usize = 32;
const ENTRY_SIZE: usize = SLOT_SIZE + HASH_SIZE;

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at.checked_add(SLOT_SIZE)?)
        .and_then(|field| field.try_into().ok())
        .map(u64::from_le_bytes)
}

/// `SlotHashes` data that is not what the runtime keeps: shorter than its own
/// count says, or holding no slot at all, which a cluster past its first slot
/// never does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnreadableSlotHashes;

/// The hash of `slot` in the raw data of the `SlotHashes` sysvar, or `None`
/// once the sysvar no longer holds it. The data is a little-endian `u64` count
/// followed by that many `(slot, hash)` entries, newest first. It is searched
/// in place: at 20 kB the sysvar is too large to deserialize in a program.
pub fn slot_hash(
    slot_hashes: &[u8],
    slot: u64,
) -> Result<Option<[u8; HASH_SIZE]>, UnreadableSlotHashes> {
    let count = read_u64(slot_hashes, 0)
        .and_then(|count| usize::try_from(count).ok())
        .filter(|count| *count > 0)
        .ok_or(UnreadableSlotHashes)?;
    let end = count
        .checked_mul(ENTRY_SIZE)
        .and_then(|length| length.checked_add(SLOT_SIZE))
        .ok_or(UnreadableSlotHashes)?;
    let entries = slot_hashes
        .get(SLOT_SIZE..end)
        .ok_or(UnreadableSlotHashes)?;
    let (mut low, mut high) = (0, count);
    while low < high {
        let middle = low + (high - low) / 2;
        let entry = middle * ENTRY_SIZE;
        let entry_slot =
            read_u64(entries, entry).ok_or(UnreadableSlotHashes)?;
        if entry_slot == slot {
            return entries
                .get(entry + SLOT_SIZE..entry + ENTRY_SIZE)
                .and_then(|hash| hash.try_into().ok())
                .map(Some)
                .ok_or(UnreadableSlotHashes);
        }
        // Newest first: a later slot sits before the one wanted.
        if entry_slot > slot {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(None)
}

/// The seed of the round of `run` (its run id's bytes) due at `step`,
/// committed to at `slot`, whose bank hash is `slot_hash`. The run and the
/// step keep two runs, or two rounds, committed in one slot from drawing one
/// committee.
pub fn round_seed(
    run: &[u8],
    step: u32,
    slot: u64,
    slot_hash: &[u8; HASH_SIZE],
) -> u64 {
    let digest = sha256v(&[
        ROUND_SEED_DOMAIN,
        run,
        &step.to_le_bytes(),
        &slot.to_le_bytes(),
        slot_hash,
    ]);
    u64::from_le_bytes(
        digest[..SLOT_SIZE]
            .try_into()
            .expect("a sha256 digest is longer than a u64"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SlotHashes` data laid out as the runtime lays it out.
    fn sysvar(entries: &[(u64, [u8; HASH_SIZE])]) -> Vec<u8> {
        let mut data = (entries.len() as u64).to_le_bytes().to_vec();
        for (slot, hash) in entries {
            data.extend_from_slice(&slot.to_le_bytes());
            data.extend_from_slice(hash);
        }
        data
    }

    #[test]
    fn every_held_slot_is_found_and_no_other() {
        // Newest first, with skipped slots in between, as a real cluster has.
        let entries: Vec<(u64, [u8; HASH_SIZE])> =
            [900u64, 899, 897, 890, 512, 3]
                .iter()
                .map(|slot| (*slot, [*slot as u8; HASH_SIZE]))
                .collect();
        let data = sysvar(&entries);
        for (slot, hash) in &entries {
            assert_eq!(slot_hash(&data, *slot), Ok(Some(*hash)), "slot {slot}");
        }
        for missing in [0, 2, 4, 898, 901, u64::MAX] {
            assert_eq!(slot_hash(&data, missing), Ok(None), "slot {missing}");
        }
    }

    #[test]
    fn data_that_is_not_a_slot_hashes_sysvar_is_unreadable() {
        let mut data = sysvar(&[(7, [7; HASH_SIZE]), (6, [6; HASH_SIZE])]);
        data.truncate(data.len() - 1);
        assert_eq!(slot_hash(&data, 6), Err(UnreadableSlotHashes));
        assert_eq!(slot_hash(&[], 6), Err(UnreadableSlotHashes));
        // Holding no slot is not a sysvar a running cluster has.
        assert_eq!(slot_hash(&sysvar(&[]), 6), Err(UnreadableSlotHashes));
        // A count that would overflow is refused, not wrapped.
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(&[0; ENTRY_SIZE]);
        assert_eq!(slot_hash(&huge, 0), Err(UnreadableSlotHashes));
    }

    #[test]
    fn a_seed_depends_on_the_run_the_step_the_slot_and_its_hash() {
        let seed = round_seed(b"run", 4, 10, &[1; HASH_SIZE]);
        assert_eq!(seed, round_seed(b"run", 4, 10, &[1; HASH_SIZE]));
        assert_ne!(seed, round_seed(b"other", 4, 10, &[1; HASH_SIZE]));
        assert_ne!(seed, round_seed(b"run", 5, 10, &[1; HASH_SIZE]));
        assert_ne!(seed, round_seed(b"run", 4, 11, &[1; HASH_SIZE]));
        assert_ne!(seed, round_seed(b"run", 4, 10, &[2; HASH_SIZE]));
    }
}
