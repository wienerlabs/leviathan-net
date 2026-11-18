use crate::{Finished, TrainingResult, fetch_data::BatchIdSet};

use psyche_coordinator::{
    Commitment, CommitteeProof, CommitteeSelection, WitnessBloom, WitnessProof,
};
use psyche_core::{BatchId, NodeIdentity};
use psyche_modeling::DistroResult;
use psyche_network::TransmittableDistroResult;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};
use thiserror::Error;

use super::types::PayloadState;

pub struct RoundState {
    pub height: u32,
    pub step: u32,
    /// When the epoch this round belongs to started (the coordinator's
    /// `epoch_state.start_timestamp`), which every payload of the round names.
    pub epoch_start: u64,
    pub sent_witness: bool,
    pub sent_finished: bool,
    pub downloads: Arc<Mutex<HashMap<psyche_network::Hash, PayloadState>>>,
    #[allow(clippy::type_complexity)]
    pub results: HashMap<BatchId, Vec<(NodeIdentity, (Commitment, TrainingResult))>>,
    pub clients_finished: HashMap<NodeIdentity, Finished>,
    pub data_assignments: BTreeMap<BatchId, NodeIdentity>,
    pub blooms: Arc<Mutex<Option<(WitnessBloom, WitnessBloom)>>>,
    pub broadcasts: Vec<[u8; 32]>,
    pub committee_info: Option<(CommitteeProof, WitnessProof, CommitteeSelection)>,
    pub batch_ids_not_yet_trained_on: Arc<Mutex<Option<BatchIdSet>>>,
    pub self_distro_results: Vec<Vec<DistroResult>>,
}

impl RoundState {
    pub fn new() -> Self {
        Self {
            height: 0,
            step: 0,
            epoch_start: 0,
            sent_witness: false,
            sent_finished: false,
            downloads: Arc::new(Mutex::new(HashMap::new())),
            results: HashMap::new(),
            broadcasts: Vec::new(),
            clients_finished: HashMap::new(),
            data_assignments: BTreeMap::new(),
            blooms: Arc::new(Mutex::new(None)),
            committee_info: None,
            batch_ids_not_yet_trained_on: Arc::new(Mutex::new(None)),
            self_distro_results: vec![],
        }
    }

    pub fn distro_result_blob_downloaded(&self, hash: &psyche_network::Hash) -> bool {
        self.downloads.lock().unwrap().contains_key(hash)
    }

    /// Checks that `payload`, downloaded in this round for a broadcast of
    /// `batch_id`, names the work it was broadcast for: this round's epoch and
    /// step, and that batch.
    ///
    /// The sender signs the header together with the results, so a header
    /// naming other work is signed too and passes the hash check. Kept, it
    /// would be a record of work outside this round: one that convicts nobody
    /// (the treasurer refuses a commitment from outside the current epoch),
    /// and that the verifier never replays, for a payload everyone trained on.
    /// So a cheater could escape audit by mislabelling its own payload. It is
    /// dropped instead (wienerlabs/leviathan-net#67).
    pub fn check_payload_header(
        &self,
        payload: &TransmittableDistroResult,
        batch_id: BatchId,
    ) -> Result<(), MislabelledPayload> {
        if payload.epoch_start != self.epoch_start {
            return Err(MislabelledPayload::Epoch {
                payload: payload.epoch_start,
                round: self.epoch_start,
            });
        }
        if payload.step != self.step {
            return Err(MislabelledPayload::Step {
                payload: payload.step,
                round: self.step,
            });
        }
        if payload.batch_id != batch_id {
            return Err(MislabelledPayload::Batch {
                payload: payload.batch_id,
                broadcast: batch_id,
            });
        }
        Ok(())
    }
}

/// How a downloaded payload's header differs from the work it was broadcast
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MislabelledPayload {
    #[error("it names the epoch that started at {payload}, the round's started at {round}")]
    Epoch { payload: u64, round: u64 },
    #[error("it names step {payload}, the round is step {round}")]
    Step { payload: u32, round: u32 },
    #[error("it names batch {payload}, it was broadcast for batch {broadcast}")]
    Batch {
        payload: BatchId,
        broadcast: BatchId,
    },
}

impl Default for RoundState {
    fn default() -> Self {
        RoundState::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use psyche_core::ClosedInterval;

    const EPOCH_START: u64 = 1_700_000_000;
    const STEP: u32 = 9;

    fn batch(start: u64) -> BatchId {
        BatchId(ClosedInterval::new(start, start + 3))
    }

    fn round() -> RoundState {
        RoundState {
            epoch_start: EPOCH_START,
            step: STEP,
            ..RoundState::new()
        }
    }

    fn payload(epoch_start: u64, step: u32, batch_id: BatchId) -> TransmittableDistroResult {
        TransmittableDistroResult {
            epoch_start,
            step,
            trainer_nonce: 1,
            batch_id,
            distro_results: vec![],
            state_digest: Some(psyche_core::sha256(b"the state trained from")),
        }
    }

    #[test]
    fn a_payload_naming_its_round_and_batch_is_accepted() {
        assert_eq!(
            round().check_payload_header(&payload(EPOCH_START, STEP, batch(4)), batch(4)),
            Ok(())
        );
    }

    /// A payload that names another epoch, step or batch than the one it was
    /// broadcast for would be kept as a record of work outside the round,
    /// which convicts nobody, so it is dropped (wienerlabs/leviathan-net#67).
    #[test]
    fn a_payload_naming_other_work_is_refused() {
        let round = round();
        assert_eq!(
            round.check_payload_header(&payload(EPOCH_START - 90, STEP, batch(4)), batch(4)),
            Err(MislabelledPayload::Epoch {
                payload: EPOCH_START - 90,
                round: EPOCH_START,
            })
        );
        assert_eq!(
            round.check_payload_header(&payload(EPOCH_START, 0, batch(4)), batch(4)),
            Err(MislabelledPayload::Step {
                payload: 0,
                round: STEP,
            })
        );
        assert_eq!(
            round.check_payload_header(&payload(EPOCH_START, STEP, batch(8)), batch(4)),
            Err(MislabelledPayload::Batch {
                payload: batch(8),
                broadcast: batch(4),
            })
        );
    }
}
