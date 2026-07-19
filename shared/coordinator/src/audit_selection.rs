use crate::{
    Client, Committee, CommitteeSelection, Coordinator, CoordinatorError, assign_data_for_state,
};

use psyche_core::{BatchId, NodeIdentity, sha256v};
use std::collections::BTreeMap;

pub const AUDIT_SALT: &str = "audit";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuditAssignment {
    pub verifier: NodeIdentity,
    pub verifier_index: u64,
    pub target: NodeIdentity,
    pub batch_id: BatchId,
}

pub fn select_audits(
    committee_selection: &CommitteeSelection,
    clients: &[Client],
    assignments: &BTreeMap<BatchId, NodeIdentity>,
) -> Vec<AuditAssignment> {
    if assignments.is_empty() {
        return Vec::new();
    }
    let targets: Vec<(BatchId, NodeIdentity)> = assignments
        .iter()
        .map(|(batch_id, target)| (*batch_id, *target))
        .collect();
    let seed = committee_selection.get_seed();
    let mut audits = Vec::new();
    for (index, client) in clients.iter().enumerate() {
        let proof = committee_selection.get_committee(index as u64);
        if !matches!(proof.committee, Committee::Verifier) {
            continue;
        }
        let digest = sha256v(&[
            &seed,
            AUDIT_SALT.as_bytes(),
            &(index as u64).to_le_bytes(),
        ]);
        let mut pick_bytes = [0u8; 8];
        pick_bytes.copy_from_slice(&digest[..8]);
        let pick = (u64::from_le_bytes(pick_bytes) % targets.len() as u64) as usize;
        let (batch_id, target) = targets[pick];
        audits.push(AuditAssignment {
            verifier: client.id,
            verifier_index: index as u64,
            target,
            batch_id,
        });
    }
    audits
}

pub fn select_audits_for_current_round(
    coordinator: &Coordinator,
) -> Result<Vec<AuditAssignment>, CoordinatorError> {
    let committee_selection = CommitteeSelection::from_coordinator(coordinator, 0)?;
    let assignments = assign_data_for_state(coordinator, &committee_selection);
    Ok(select_audits(
        &committee_selection,
        &coordinator.epoch_state.clients,
        &assignments,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClientState, get_batch_ids_for_node};
    use bytemuck::Zeroable;
    use psyche_core::FixedVec;

    fn create_test_coordinator(
        num_nodes: usize,
        global_batch_size: u16,
        verification_percent: u8,
        random_seed: u64,
    ) -> Coordinator {
        let clients: Vec<_> = (0..num_nodes)
            .map(|i| {
                let mut key = [0u8; 32];
                key[0] = i as u8;
                Client {
                    id: NodeIdentity::from_single_key(key),
                    state: ClientState::Healthy,
                    exited_height: 0,
                }
            })
            .collect();

        let mut coordinator = Coordinator::zeroed();
        coordinator.config.total_steps = 10;
        coordinator.config.global_batch_size_start = global_batch_size;
        coordinator.config.global_batch_size_end = global_batch_size;
        coordinator.config.verification_percent = verification_percent;
        coordinator.epoch_state.clients = FixedVec::from_iter(clients);

        let round = coordinator.current_round_mut().unwrap();
        round.clients_len = num_nodes as u16;
        round.random_seed = random_seed;

        coordinator
    }

    #[test]
    fn test_audits_are_deterministic_and_disjoint() {
        let coordinator = create_test_coordinator(20, 160, 20, 777);
        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        assert_eq!(selection.get_num_verifier_nodes(), 4);

        let assignments = assign_data_for_state(&coordinator, &selection);
        assert_eq!(assignments.len(), 16);

        let audits_first = select_audits(
            &selection,
            &coordinator.epoch_state.clients,
            &assignments,
        );
        let audits_second = select_audits(
            &selection,
            &coordinator.epoch_state.clients,
            &assignments,
        );
        assert_eq!(audits_first, audits_second);
        assert_eq!(audits_first.len(), 4);

        for audit in &audits_first {
            assert_ne!(audit.verifier, audit.target);
            let verifier_committee = selection
                .get_committee(audit.verifier_index)
                .committee;
            assert!(matches!(verifier_committee, Committee::Verifier));
            let target_batches = get_batch_ids_for_node(&assignments, &audit.target);
            assert!(target_batches.contains(&audit.batch_id));
        }
    }

    #[test]
    fn test_zero_verification_percent_yields_no_audits() {
        let coordinator = create_test_coordinator(20, 160, 0, 777);
        let audits = select_audits_for_current_round(&coordinator).unwrap();
        assert!(audits.is_empty());
    }

    #[test]
    fn test_seed_changes_assignments() {
        let coordinator_one = create_test_coordinator(20, 160, 20, 777);
        let coordinator_two = create_test_coordinator(20, 160, 20, 778);
        let audits_one = select_audits_for_current_round(&coordinator_one).unwrap();
        let audits_two = select_audits_for_current_round(&coordinator_two).unwrap();
        assert_eq!(audits_one.len(), 4);
        assert_eq!(audits_two.len(), 4);
        assert_ne!(audits_one, audits_two);
    }

    #[test]
    fn test_audit_pressure_tracks_verifier_share() {
        let coordinator = create_test_coordinator(100, 400, 10, 424242);
        let selection = CommitteeSelection::from_coordinator(&coordinator, 0).unwrap();
        assert_eq!(selection.get_num_verifier_nodes(), 10);
        assert_eq!(selection.get_num_trainer_nodes(), 90);

        let audits = select_audits_for_current_round(&coordinator).unwrap();
        assert_eq!(audits.len(), 10);

        let mut audited_targets: Vec<_> =
            audits.iter().map(|audit| audit.target).collect();
        audited_targets.dedup();
        assert!(!audited_targets.is_empty());
    }
}
