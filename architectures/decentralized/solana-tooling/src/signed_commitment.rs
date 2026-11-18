use std::path::Path;

use anyhow::anyhow;
use anyhow::Result;
use psyche_coordinator::assign_data_for_state;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::Coordinator;
use psyche_core::sha256;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_core::NodeIdentity;
use psyche_solana_treasurer::state::commitment_run;
use psyche_solana_treasurer::state::committee_matches_roster;
use psyche_solana_treasurer::state::CommitmentEvidence;
use solana_sdk::ed25519_program;
use solana_sdk::instruction::Instruction;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

/// Byte layout of an ed25519 precompile instruction carrying one signature:
/// the count and a padding byte, one offsets entry, then the data it points at.
const SIGNATURE_OFFSETS_START: usize = 2;
const SIGNATURE_OFFSETS_SIZE: usize = 14;
const DATA_START: usize = SIGNATURE_OFFSETS_START + SIGNATURE_OFFSETS_SIZE;
const PUBKEY_SIZE: usize = 32;
const SIGNATURE_SIZE: usize = 64;
/// The instruction index an offsets entry uses for "this instruction".
const THIS_INSTRUCTION: u16 = u16::MAX;

/// An ed25519 precompile instruction proving `signer` signed `message`, with
/// the key, the signature and the message all inside the instruction. That is
/// the only form the treasurer accepts. The runtime verifies the signature
/// before any program runs, and the whole transaction fails if it does not
/// hold.
pub fn ed25519_signature_instruction(
    signer: &[u8; 32],
    signature: &[u8; 64],
    message: &[u8],
) -> Instruction {
    let pubkey_offset = DATA_START;
    let signature_offset = pubkey_offset + PUBKEY_SIZE;
    let message_offset = signature_offset + SIGNATURE_SIZE;
    let mut data = Vec::with_capacity(message_offset + message.len());
    data.extend_from_slice(&[1, 0]);
    for field in [
        signature_offset as u16,
        THIS_INSTRUCTION,
        pubkey_offset as u16,
        THIS_INSTRUCTION,
        message_offset as u16,
        message.len() as u16,
        THIS_INSTRUCTION,
    ] {
        data.extend_from_slice(&field.to_le_bytes());
    }
    data.extend_from_slice(signer);
    data.extend_from_slice(signature);
    data.extend_from_slice(message);
    Instruction {
        program_id: ed25519_program::ID,
        accounts: vec![],
        data,
    }
}

/// The results digest of the payload dump at `path`. The digest a payload
/// commitment takes of its results is the sha256 of their wire encoding,
/// which is exactly the bytes a dump holds (psyche-network's
/// `results_digest`).
pub fn payload_dump_digest(path: &Path) -> Result<[u8; 32]> {
    let bytes = std::fs::read(path)
        .map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;
    Ok(sha256(&bytes))
}

/// The step of the round the coordinator is in: the one being trained, or in
/// cooldown and between epochs the last one trained. Each round of an epoch
/// trains one step, starting from the epoch's first.
pub fn trained_step(coordinator: &Coordinator) -> Result<u32> {
    let round = coordinator
        .current_round()
        .ok_or_else(|| anyhow!("the epoch has no round"))?;
    Ok(coordinator.epoch_state.start_step + round.height)
}

/// Which work a payload is: the round, named by when its epoch started and
/// its step, the trainer's nonce, and the batch (wienerlabs/leviathan-net#43,
/// #67), and the state the trainer trained it from, if it committed to one
/// (wienerlabs/leviathan-net#89).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadHeader {
    pub epoch_start: u64,
    pub step: u32,
    pub trainer_nonce: u32,
    pub batch_id: BatchId,
    pub state_digest: Option<[u8; 32]>,
}

/// Which work a payload of `node` from the round the coordinator is in is:
/// the epoch and the step trained there, the trainer's nonce, and the batch
/// the coordinator assigned `node` in that round. The committee is the
/// round's own draw, with the tie-breaker seats it reserves, as the clients
/// and the treasurer draw it. The chain does not hold the state `node` trained
/// from, so the header commits to `state_digest`, which the caller names.
pub fn payload_header(
    coordinator: &Coordinator,
    node: &NodeIdentity,
    state_digest: Option<[u8; 32]>,
) -> Result<PayloadHeader> {
    // Batches are assigned by the committee drawn for the round, over the
    // roster as it stood then. In a warmup, or once the roster has moved since
    // the round was drawn, no batch is assigned to anyone in it.
    if !committee_matches_roster(coordinator) {
        return Err(anyhow!(
            "no round has been drawn over the roster as it stands ({})",
            coordinator.run_state
        ));
    }
    let selection = CommitteeSelection::from_coordinator(coordinator, 0)
        .map_err(|err| anyhow!("the round has no committee: {err:?}"))?;
    let batch = assign_data_for_state(coordinator, &selection)
        .into_iter()
        .find_map(|(batch, trainer)| (trainer == *node).then_some(batch))
        .ok_or_else(|| anyhow!("{node} trains no batch this round"))?;
    let step = trained_step(coordinator)?;
    Ok(PayloadHeader {
        epoch_start: coordinator.epoch_state.start_timestamp,
        step,
        // The trainer counts the steps it has trained; a node that joined at
        // the start of the epoch has trained one per round before this.
        trainer_nonce: step - coordinator.epoch_state.start_step,
        batch_id: batch,
        state_digest,
    })
}

/// A payload commitment with its committer's signature: what a conviction is
/// made on (wienerlabs/leviathan-net#43).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedCommitment {
    /// The run the commitment was made in (`commitment_run`).
    pub run: [u8; 32],
    pub batch_start: u64,
    pub batch_end: u64,
    pub evidence: CommitmentEvidence,
    /// The committer's p2p key, which the coordinator roster holds.
    pub signer: [u8; 32],
    pub signature: [u8; 64],
}

impl SignedCommitment {
    /// `p2p` commits to the payload whose results digest is `results_digest`,
    /// for the work `header` names, in the run `run`. It signs the commitment
    /// hash the way a client signs the payload it broadcasts.
    pub fn sign(
        p2p: &Keypair,
        run: [u8; 32],
        header: PayloadHeader,
        results_digest: [u8; 32],
    ) -> SignedCommitment {
        let (state_committed, state_digest) =
            CommitmentEvidence::state_fields(header.state_digest);
        let evidence = CommitmentEvidence {
            epoch_start: header.epoch_start,
            step: header.step,
            trainer_nonce: header.trainer_nonce,
            results_digest,
            state_committed,
            state_digest,
        };
        let (batch_start, batch_end) =
            (header.batch_id.0.start, header.batch_id.0.end);
        let committed_hash =
            evidence.committed_hash(&run, batch_start, batch_end);
        SignedCommitment {
            run,
            batch_start,
            batch_end,
            evidence,
            signer: p2p.pubkey().to_bytes(),
            signature: p2p.sign_message(&committed_hash).into(),
        }
    }

    /// What `node` signed, with `p2p`, the key it joined with, for its payload
    /// of the round the coordinator is in, whose results digest is
    /// `results_digest`, trained from the state `state_digest` names, if it
    /// committed to one (wienerlabs/leviathan-net#89).
    pub fn for_round(
        coordinator: &Coordinator,
        node: &NodeIdentity,
        p2p: &Keypair,
        results_digest: [u8; 32],
        state_digest: Option<[u8; 32]>,
    ) -> Result<SignedCommitment> {
        if node.p2p_identity() != &p2p.pubkey().to_bytes() {
            return Err(anyhow!("{node} did not join with this p2p key"));
        }
        let header = payload_header(coordinator, node, state_digest)?;
        Ok(SignedCommitment::sign(
            p2p,
            commitment_run(coordinator),
            header,
            results_digest,
        ))
    }

    /// The work this commitment is to.
    pub fn header(&self) -> PayloadHeader {
        PayloadHeader {
            epoch_start: self.evidence.epoch_start,
            step: self.evidence.step,
            trainer_nonce: self.evidence.trainer_nonce,
            batch_id: BatchId(ClosedInterval::new(
                self.batch_start,
                self.batch_end,
            )),
            state_digest: self.evidence.state().copied(),
        }
    }

    /// The commitment hash the signature is over.
    pub fn committed_hash(&self) -> [u8; 32] {
        self.evidence.committed_hash(
            &self.run,
            self.batch_start,
            self.batch_end,
        )
    }

    /// The precompile instruction that proves the signature to the treasurer.
    pub fn proof_instruction(&self) -> Instruction {
        ed25519_signature_instruction(
            &self.signer,
            &self.signature,
            &self.committed_hash(),
        )
    }
}
