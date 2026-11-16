//! Whether cheating pays in a run, from what the run puts on chain.

use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_solana_coordinator::clients_state::ClientsEpochRates;
use serde::Serialize;

/// What a cheat stands to win and to lose, in the run's collateral units
/// (points and collateral base units are the same thing by the run's own
/// convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunEconomics {
    /// The most one cheat can take: a client's share of an epoch's earnings.
    pub reward: f64,
    /// The bond every earning client keeps posted, which caps what a
    /// conviction can collect.
    pub bond: f64,
    /// What a conviction charges.
    pub slash_when_caught: f64,
}

impl RunEconomics {
    pub fn is_well_formed(&self) -> bool {
        [self.reward, self.bond, self.slash_when_caught]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct SecurityAssessment {
    pub audit_probability: f64,
    pub reward: f64,
    pub bond: f64,
    pub slash_when_caught: f64,
    pub break_even_penalty: Option<f64>,
    pub effective_penalty: f64,
    pub expected_fraud_value: f64,
    pub economically_secure: bool,
}

/// The expected value of a cheat audited with probability `audit_probability`:
/// `(1 - p) * reward - p * min(slash, bond)`. The run is secure when that is
/// not positive, which is when the penalty meets the break-even penalty
/// `reward * (1 - p) / p`.
pub fn assess_security(audit_probability: f64, economics: &RunEconomics) -> SecurityAssessment {
    let p = audit_probability.clamp(0.0, 1.0);
    let break_even_penalty = if p > 0.0 {
        let value = economics.reward * (1.0 - p) / p;
        value.is_finite().then_some(value)
    } else {
        None
    };
    let effective_penalty = economics.slash_when_caught.min(economics.bond);
    let expected_fraud_value = (1.0 - p) * economics.reward - p * effective_penalty;
    SecurityAssessment {
        audit_probability: p,
        reward: economics.reward,
        bond: economics.bond,
        slash_when_caught: economics.slash_when_caught,
        break_even_penalty,
        effective_penalty,
        expected_fraud_value,
        economically_secure: p > 0.0 && expected_fraud_value <= 0.0,
    }
}

/// The run's audit probability: the share of each round's free seats that
/// `verification_percent` turns into verifiers.
pub fn audit_probability(coordinator: &Coordinator) -> f64 {
    (coordinator.config.verification_percent as f64 / 100.0).clamp(0.0, 1.0)
}

/// Whether the run is in an epoch: from warmup through cooldown, the states
/// in which it has a roster that trains.
pub fn is_training(coordinator: &Coordinator) -> bool {
    use psyche_coordinator::RunState;
    matches!(
        coordinator.run_state,
        RunState::Warmup | RunState::RoundTrain | RunState::RoundWitness | RunState::Cooldown
    )
}

/// The clients of this epoch's roster still healthy in it.
pub fn healthy_clients(coordinator: &Coordinator) -> usize {
    coordinator
        .epoch_state
        .clients
        .iter()
        .filter(|client| client.state == ClientState::Healthy)
        .count()
}

/// A client's share of the epoch's earnings. The coordinator splits
/// `earning_rate_total_shared` evenly over the clients that finish the epoch
/// (`instance_state.rs`), so in an epoch the share is the rate over the
/// clients still healthy in it. Between epochs it is the rate over the
/// smallest roster an epoch starts with, `min_clients`.
pub fn reward_per_client(coordinator: &Coordinator, rates: &ClientsEpochRates) -> u64 {
    let roster = if is_training(coordinator) {
        healthy_clients(coordinator)
    } else {
        coordinator.config.min_clients as usize
    };
    rates
        .earning_rate_total_shared
        .checked_div(roster as u64)
        .unwrap_or(rates.earning_rate_total_shared)
}

/// The epoch rates that apply now: the epoch under way's, or between epochs,
/// the next one's. The run authority sets only the next epoch's rates, and the
/// coordinator copies them into the current ones on the tick that starts the
/// next warmup (`instance_state.rs`), so between epochs the current rates are
/// the last epoch's, or all zero before the first.
pub fn rates_in_force(
    coordinator: &Coordinator,
    current: ClientsEpochRates,
    future: ClientsEpochRates,
) -> ClientsEpochRates {
    if is_training(coordinator) {
        current
    } else {
        future
    }
}

/// What a cheat stands to win and lose in a treasurer-managed run, all of it
/// read from chain: the epoch rates in force ([`rates_in_force`]) and the
/// treasurer run's bond minimum.
///
/// The reward is a whole epoch's share. A cheat is audited with probability at
/// least `p` and can take at most that share, so the expected value
/// [`assess_security`] reports bounds a cheater's from above: a verdict of
/// secure holds however the cheater spreads its cheating over an epoch.
pub fn chain_economics(
    coordinator: &Coordinator,
    rates: &ClientsEpochRates,
    bond_minimum_amount: u64,
) -> RunEconomics {
    RunEconomics {
        reward: reward_per_client(coordinator, rates) as f64,
        bond: bond_minimum_amount as f64,
        slash_when_caught: rates.slashing_rate_per_client as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use psyche_coordinator::RunState;
    use psyche_core::NodeIdentity;

    fn rates(earning: u64, slashing: u64) -> ClientsEpochRates {
        ClientsEpochRates {
            earning_rate_total_shared: earning,
            slashing_rate_per_client: slashing,
        }
    }

    fn coordinator_with_roster(run_state: RunState, states: &[ClientState]) -> Coordinator {
        let mut coordinator = Coordinator::zeroed();
        coordinator.run_state = run_state;
        coordinator.config.min_clients = 2;
        for (seed, state) in states.iter().enumerate() {
            let mut key = [0u8; 32];
            key[0] = seed as u8 + 1;
            let mut client =
                psyche_coordinator::Client::new(NodeIdentity::new(key, Default::default()));
            client.state = *state;
            coordinator.epoch_state.clients.push(client).unwrap();
        }
        coordinator
    }

    #[test]
    fn secure_when_penalty_meets_break_even() {
        let econ = RunEconomics {
            reward: 1000.0,
            bond: 100_000.0,
            slash_when_caught: 9000.0,
        };
        let s = assess_security(0.1, &econ);
        assert_eq!(s.break_even_penalty, Some(9000.0));
        assert_eq!(s.effective_penalty, 9000.0);
        assert!(s.expected_fraud_value <= 0.0);
        assert!(s.economically_secure);
    }

    #[test]
    fn insecure_when_slash_is_too_small() {
        let econ = RunEconomics {
            reward: 1000.0,
            bond: 100_000.0,
            slash_when_caught: 1000.0,
        };
        let s = assess_security(0.1, &econ);
        assert!(s.expected_fraud_value > 0.0);
        assert!(!s.economically_secure);
    }

    #[test]
    fn bond_caps_the_effective_penalty() {
        let econ = RunEconomics {
            reward: 1000.0,
            bond: 500.0,
            slash_when_caught: 1_000_000.0,
        };
        let s = assess_security(0.1, &econ);
        assert_eq!(s.effective_penalty, 500.0);
        assert!(!s.economically_secure);
    }

    #[test]
    fn zero_audit_probability_is_never_secure() {
        let econ = RunEconomics {
            reward: 1000.0,
            bond: 1_000_000.0,
            slash_when_caught: 1_000_000.0,
        };
        let s = assess_security(0.0, &econ);
        assert_eq!(s.break_even_penalty, None);
        assert!(!s.economically_secure);
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"economically_secure\":false"));
    }

    #[test]
    fn overflow_break_even_serializes_as_absent() {
        let econ = RunEconomics {
            reward: f64::MAX,
            bond: f64::MAX,
            slash_when_caught: f64::MAX,
        };
        let s = assess_security(1e-300, &econ);
        assert_eq!(s.break_even_penalty, None);
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"break_even_penalty\":null"));
    }

    #[test]
    fn non_finite_or_negative_economics_are_rejected() {
        let well_formed = RunEconomics {
            reward: 1000.0,
            bond: 500.0,
            slash_when_caught: 100.0,
        };
        assert!(well_formed.is_well_formed());
        for broken in [
            RunEconomics {
                reward: f64::INFINITY,
                ..well_formed
            },
            RunEconomics {
                reward: f64::NAN,
                ..well_formed
            },
            RunEconomics {
                bond: -1.0,
                ..well_formed
            },
        ] {
            assert!(!broken.is_well_formed(), "{broken:?}");
        }
    }

    #[test]
    fn in_an_epoch_the_share_is_split_over_the_healthy_roster() {
        let coordinator = coordinator_with_roster(
            RunState::RoundTrain,
            &[
                ClientState::Healthy,
                ClientState::Healthy,
                ClientState::Healthy,
                ClientState::Dropped,
            ],
        );
        assert_eq!(healthy_clients(&coordinator), 3);
        assert_eq!(reward_per_client(&coordinator, &rates(900, 0)), 300);
    }

    #[test]
    fn between_epochs_the_share_is_split_over_the_smallest_roster() {
        let coordinator = coordinator_with_roster(
            RunState::WaitingForMembers,
            &[
                ClientState::Healthy,
                ClientState::Healthy,
                ClientState::Healthy,
            ],
        );
        assert!(!is_training(&coordinator));
        assert_eq!(reward_per_client(&coordinator, &rates(900, 0)), 450);
    }

    #[test]
    fn an_epoch_with_nobody_healthy_left_shares_nothing_out() {
        let coordinator = coordinator_with_roster(RunState::RoundTrain, &[ClientState::Ejected]);
        assert_eq!(reward_per_client(&coordinator, &rates(900, 0)), 900);
    }

    #[test]
    fn the_economics_are_the_runs_own_rates_and_bond() {
        let mut coordinator = coordinator_with_roster(
            RunState::RoundTrain,
            &[ClientState::Healthy, ClientState::Healthy],
        );
        coordinator.config.verification_percent = 10;
        let economics = chain_economics(&coordinator, &rates(1000, 200), 500);
        assert_eq!(
            economics,
            RunEconomics {
                reward: 500.0,
                bond: 500.0,
                slash_when_caught: 200.0,
            }
        );
        // The genesis rehearsal's economics: a 500 share against a slash of
        // 200 audited one time in ten pays the cheater 430 per epoch.
        let assessment = assess_security(audit_probability(&coordinator), &economics);
        assert!((assessment.audit_probability - 0.1).abs() < 1e-12);
        assert!((assessment.expected_fraud_value - 430.0).abs() < 1e-9);
        assert!(!assessment.economically_secure);
    }

    #[test]
    fn between_epochs_the_next_epochs_rates_are_in_force() {
        let current = rates(1000, 200);
        let next = rates(3000, 900);
        for (state, expected) in [
            (RunState::WaitingForMembers, 3000),
            (RunState::Paused, 3000),
            (RunState::Warmup, 1000),
            (RunState::RoundTrain, 1000),
            (RunState::Cooldown, 1000),
        ] {
            let coordinator = coordinator_with_roster(state, &[]);
            assert_eq!(
                rates_in_force(&coordinator, current, next).earning_rate_total_shared,
                expected,
                "{state:?}"
            );
        }
    }

    #[test]
    fn the_training_states_are_warmup_through_cooldown() {
        for (state, training) in [
            (RunState::Uninitialized, false),
            (RunState::WaitingForMembers, false),
            (RunState::Warmup, true),
            (RunState::RoundTrain, true),
            (RunState::RoundWitness, true),
            (RunState::Cooldown, true),
            (RunState::Finished, false),
            (RunState::Paused, false),
        ] {
            let mut coordinator = Coordinator::zeroed();
            coordinator.run_state = state;
            assert_eq!(is_training(&coordinator), training, "{state:?}");
        }
    }
}
