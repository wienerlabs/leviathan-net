use psyche_coordinator::Coordinator;
use psyche_solana_coordinator::Client;
use serde::Serialize;

pub const DEFAULT_LEADERBOARD_SIZE: usize = 16;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ClientEntry {
    pub signer: String,
    pub earned: u64,
    pub slashed: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RunTelemetry {
    pub run_id: String,
    pub run_state: String,
    pub epoch: u16,
    pub step: u32,
    pub registered_clients: usize,
    pub active_clients: usize,
    pub total_earned: u64,
    pub total_slashed: u64,
    pub convicted_clients: usize,
    pub verification_percent: u8,
    pub audit_probability: f64,
    pub expected_rounds_to_catch: Option<f64>,
    pub leaderboard: Vec<ClientEntry>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn compute_telemetry(
    coordinator: &Coordinator,
    on_chain_clients: &[Client],
    run_id: &str,
    leaderboard_size: usize,
) -> RunTelemetry {
    let total_earned = on_chain_clients.iter().map(|c| c.earned).sum();
    let total_slashed = on_chain_clients.iter().map(|c| c.slashed).sum();
    let convicted_clients = on_chain_clients.iter().filter(|c| c.slashed > 0).count();

    let mut leaderboard: Vec<ClientEntry> = on_chain_clients
        .iter()
        .map(|c| ClientEntry {
            signer: hex(c.id.signer()),
            earned: c.earned,
            slashed: c.slashed,
        })
        .collect();
    leaderboard.sort_by(|a, b| b.earned.cmp(&a.earned).then(a.signer.cmp(&b.signer)));
    leaderboard.truncate(leaderboard_size);

    let verification_percent = coordinator.config.verification_percent;
    let audit_probability = verification_percent as f64 / 100.0;
    let expected_rounds_to_catch = if audit_probability > 0.0 {
        Some(1.0 / audit_probability)
    } else {
        None
    };

    RunTelemetry {
        run_id: run_id.to_string(),
        run_state: coordinator.run_state.to_string(),
        epoch: coordinator.progress.epoch,
        step: coordinator.progress.step,
        registered_clients: on_chain_clients.len(),
        active_clients: coordinator.epoch_state.clients.len(),
        total_earned,
        total_slashed,
        convicted_clients,
        verification_percent,
        audit_probability,
        expected_rounds_to_catch,
        leaderboard,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use psyche_core::NodeIdentity;

    fn client(seed: u8, earned: u64, slashed: u64) -> Client {
        let mut c = Client::zeroed();
        let mut key = [0u8; 32];
        key[0] = seed;
        c.id = NodeIdentity::new(key, Default::default());
        c.earned = earned;
        c.slashed = slashed;
        c
    }

    #[test]
    fn telemetry_aggregates_earnings_and_convictions() {
        let mut coordinator = Coordinator::zeroed();
        coordinator.config.verification_percent = 10;
        coordinator.progress.epoch = 3;
        coordinator.progress.step = 42;
        let clients = vec![
            client(1, 100, 0),
            client(2, 50, 0),
            client(3, 0, 200),
        ];
        let t = compute_telemetry(&coordinator, &clients, "run-x", DEFAULT_LEADERBOARD_SIZE);
        assert_eq!(t.run_id, "run-x");
        assert_eq!(t.epoch, 3);
        assert_eq!(t.step, 42);
        assert_eq!(t.registered_clients, 3);
        assert_eq!(t.total_earned, 150);
        assert_eq!(t.total_slashed, 200);
        assert_eq!(t.convicted_clients, 1);
        assert_eq!(t.verification_percent, 10);
        assert!((t.audit_probability - 0.1).abs() < 1e-9);
        assert_eq!(t.expected_rounds_to_catch, Some(10.0));
    }

    #[test]
    fn leaderboard_is_ranked_and_capped() {
        let coordinator = Coordinator::zeroed();
        let clients: Vec<Client> = (0..20).map(|i| client(i as u8, (i as u64) * 10, 0)).collect();
        let t = compute_telemetry(&coordinator, &clients, "run", 5);
        assert_eq!(t.leaderboard.len(), 5);
        assert_eq!(t.leaderboard[0].earned, 190);
        assert!(t.leaderboard[0].earned >= t.leaderboard[1].earned);
    }

    #[test]
    fn zero_verification_has_no_catch_estimate() {
        let coordinator = Coordinator::zeroed();
        let t = compute_telemetry(&coordinator, &[], "run", DEFAULT_LEADERBOARD_SIZE);
        assert_eq!(t.verification_percent, 0);
        assert_eq!(t.expected_rounds_to_catch, None);
        assert_eq!(t.registered_clients, 0);
    }

    #[test]
    fn telemetry_serializes_to_json() {
        let coordinator = Coordinator::zeroed();
        let t = compute_telemetry(&coordinator, &[client(1, 5, 0)], "run", DEFAULT_LEADERBOARD_SIZE);
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("\"run_id\":\"run\""));
        assert!(json.contains("\"total_earned\":5"));
    }
}
