//! The series `leviathan-indexer --metrics-listen` exports, one per run.
//!
//! Every value is read from chain. Values the chain keeps a running total of,
//! such as points earned or collateral claimed, are gauges here, since the
//! chain owns the count; the counters count what this indexer saw happen.

use std::sync::atomic::AtomicU64;

use leviathan_telemetry::labels::RunLabels;
use leviathan_telemetry::labels::RunStateLabels;
use leviathan_telemetry::labels::RunStatusLabels;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

use crate::observe::RunObservation;
use crate::observe::TreasurerObservation;
use crate::observe::CLIENT_STATES;
use crate::verdicts::status_label;
use crate::verdicts::VERDICT_STATUSES;

type Value = Gauge<f64, AtomicU64>;
type RunGauge = Family<RunLabels, Value>;
type RunCounter = Family<RunLabels, Counter>;
/// How a treasurer series takes its value from an observation.
type TreasurerValue = fn(&TreasurerObservation) -> f64;

fn gauge(registry: &mut Registry, name: &str, help: &str) -> RunGauge {
    let family = RunGauge::default();
    registry.register(name, help, family.clone());
    family
}

fn counter(registry: &mut Registry, name: &str, help: &str) -> RunCounter {
    let family = RunCounter::default();
    registry.register(name, help, family.clone());
    family
}

fn flag(value: bool) -> f64 {
    if value {
        1.0
    } else {
        0.0
    }
}

/// The series every run has.
struct RunSeries {
    run_state: RunGauge,
    training: RunGauge,
    epoch: RunGauge,
    step: RunGauge,
    registered_clients: RunGauge,
    active_clients: RunGauge,
    epoch_roster_size: RunGauge,
    epoch_clients: Family<RunStateLabels, Value>,
    min_clients: RunGauge,
    audit_probability: RunGauge,
    earned_points: RunGauge,
    slashed_points: RunGauge,
    convicted_clients: RunGauge,
    treasurer_managed: RunGauge,
}

/// The series only a treasurer-managed run has: its economics, its vault, and
/// its verdicts.
struct TreasurerSeries {
    gauges: Vec<(RunGauge, TreasurerValue)>,
    break_even_penalty: RunGauge,
    verdicts: Family<RunStatusLabels, Value>,
    convictions: RunCounter,
    overturned_convictions: RunCounter,
    lapsed_disputes: RunCounter,
    unseen_outcomes: RunCounter,
}

/// How fresh the indexer's view of the chain is.
struct IndexerSeries {
    snapshot_slot: RunGauge,
    slot_lag: RunGauge,
    snapshot_timestamp: RunGauge,
    last_poll_success: RunGauge,
    poll_errors: RunCounter,
}

pub struct IndexerMetrics {
    run: RunSeries,
    treasurer: TreasurerSeries,
    indexer: IndexerSeries,
}

impl RunSeries {
    fn register(registry: &mut Registry) -> Self {
        let epoch_clients = Family::<RunStateLabels, Value>::default();
        registry.register(
            "leviathan_run_epoch_clients",
            "Clients of this epoch's roster, including those that exited it, by state",
            epoch_clients.clone(),
        );
        Self {
            run_state: gauge(
                registry,
                "leviathan_run_state",
                "The coordinator's RunState, by its discriminant",
            ),
            training: gauge(
                registry,
                "leviathan_run_training",
                "1 from warmup through cooldown, when the run has a roster that trains",
            ),
            epoch: gauge(registry, "leviathan_run_epoch", "The run's epoch"),
            step: gauge(registry, "leviathan_run_step", "The run's step"),
            registered_clients: gauge(
                registry,
                "leviathan_run_registered_clients",
                "Every client the coordinator ever registered for the run",
            ),
            active_clients: gauge(
                registry,
                "leviathan_run_active_clients",
                "Clients of this epoch's roster still healthy in it",
            ),
            epoch_roster_size: gauge(
                registry,
                "leviathan_run_epoch_roster_size",
                "Clients this epoch started with: the roster and those that exited it",
            ),
            epoch_clients,
            min_clients: gauge(
                registry,
                "leviathan_run_min_clients",
                "The smallest roster an epoch starts with",
            ),
            audit_probability: gauge(
                registry,
                "leviathan_run_audit_probability",
                "verification_percent as a probability",
            ),
            earned_points: gauge(
                registry,
                "leviathan_run_earned_points",
                "Points the coordinator credited to clients, all time",
            ),
            slashed_points: gauge(
                registry,
                "leviathan_run_slashed_points",
                "Points the coordinator charged against clients, all time",
            ),
            convicted_clients: gauge(
                registry,
                "leviathan_run_convicted_clients",
                "Clients the coordinator ever charged a slash",
            ),
            treasurer_managed: gauge(
                registry,
                "leviathan_run_treasurer_managed",
                "1 when a treasurer run holds the run's bonds and rewards",
            ),
        }
    }

    fn observe(&self, labels: &RunLabels, observation: &RunObservation) {
        let set = |family: &RunGauge, value: f64| {
            family.get_or_create(labels).set(value);
        };
        set(&self.run_state, observation.run_state as f64);
        set(&self.training, flag(observation.training));
        set(&self.epoch, observation.epoch as f64);
        set(&self.step, observation.step as f64);
        set(
            &self.registered_clients,
            observation.registered_clients as f64,
        );
        set(&self.active_clients, observation.active_clients() as f64);
        set(
            &self.epoch_roster_size,
            observation.epoch_roster_size() as f64,
        );
        for ((_, state), count) in CLIENT_STATES.iter().zip(observation.epoch_clients) {
            self.epoch_clients
                .get_or_create(&RunStateLabels::new(&labels.run_id, state))
                .set(count as f64);
        }
        set(&self.min_clients, observation.min_clients as f64);
        set(&self.audit_probability, observation.audit_probability);
        set(&self.earned_points, observation.earned_points as f64);
        set(&self.slashed_points, observation.slashed_points as f64);
        set(
            &self.convicted_clients,
            observation.convicted_clients as f64,
        );
        set(
            &self.treasurer_managed,
            flag(observation.treasurer.is_some()),
        );
    }
}

impl TreasurerSeries {
    fn register(registry: &mut Registry) -> Self {
        let mut series =
            |name: &str, help: &str, value: TreasurerValue| (gauge(registry, name, help), value);
        let gauges = vec![
            series(
                "leviathan_run_verifier_committee_size",
                "Verifier seats a round draws over the roster as it stands",
                |t| t.verifier_committee_size as f64,
            ),
            series(
                "leviathan_run_committee_drawn",
                "1 when a committee stands over the roster, the condition the treasurer takes votes under",
                |t| flag(t.committee_drawn),
            ),
            series(
                "leviathan_run_min_verifier_committee",
                "The smallest committee the treasurer takes a verdict from",
                |t| t.min_verifier_committee as f64,
            ),
            series(
                "leviathan_run_epoch_earning_rate",
                "Points the epoch in force shares out among the clients that finish it",
                |t| t.epoch_earning_rate as f64,
            ),
            series(
                "leviathan_run_epoch_slashing_rate",
                "Points a conviction charges in the epoch in force",
                |t| t.epoch_slashing_rate as f64,
            ),
            series(
                "leviathan_run_bond_minimum",
                "The bond an earning client keeps posted",
                |t| t.bond_minimum as f64,
            ),
            series(
                "leviathan_run_reward_per_client",
                "A client's share of the epoch's earnings",
                |t| t.security.reward,
            ),
            series(
                "leviathan_run_effective_penalty",
                "What a conviction collects: the slash, capped at the bond",
                |t| t.security.effective_penalty,
            ),
            series(
                "leviathan_run_expected_fraud_value",
                "Expected value of a cheat at the run's audit probability",
                |t| t.security.expected_fraud_value,
            ),
            series(
                "leviathan_run_economically_secure",
                "1 when a cheat loses money in expectation",
                |t| flag(t.security.economically_secure),
            ),
            series(
                "leviathan_run_vault_balance",
                "Collateral the treasurer run's vault holds",
                |t| t.vault_balance as f64,
            ),
            series(
                "leviathan_run_bonded_amount",
                "Bonds posted into the vault and not paid out",
                |t| t.bonded_amount as f64,
            ),
            series(
                "leviathan_run_unclaimed_earnings",
                "Points credited to clients and not yet claimed; each is claimable once its holder keeps the bond minimum posted",
                |t| t.unclaimed_earnings as f64,
            ),
            series(
                "leviathan_run_claimed_collateral",
                "Collateral clients claimed against their points, all time",
                |t| t.claimed_collateral as f64,
            ),
            series(
                "leviathan_run_bond_shortfall",
                "How far the vault falls short of the bonds posted into it",
                |t| t.bond_shortfall as f64,
            ),
            series(
                "leviathan_run_obligation_shortfall",
                "How far the vault falls short of the bonds plus every point credited and not yet claimed",
                |t| t.obligation_shortfall as f64,
            ),
        ];
        let verdicts = Family::<RunStatusLabels, Value>::default();
        registry.register(
            "leviathan_run_verdicts",
            "The run's verdict accounts, by the status of the dispute each shows",
            verdicts.clone(),
        );
        Self {
            gauges,
            break_even_penalty: gauge(
                registry,
                "leviathan_run_break_even_penalty",
                "The penalty at which a cheat breaks even; absent when nothing is audited",
            ),
            verdicts,
            convictions: counter(
                registry,
                "leviathan_run_convictions",
                "Disputes this indexer saw reach a committee's quorum; a run authority's own slash writes no verdict and is not counted",
            ),
            overturned_convictions: counter(
                registry,
                "leviathan_run_overturned_convictions",
                "Convictions this indexer saw overturned on appeal: an innocent target convicted",
            ),
            lapsed_disputes: counter(
                registry,
                "leviathan_run_lapsed_disputes",
                "Fraud votes this indexer saw lapse without a quorum",
            ),
            unseen_outcomes: counter(
                registry,
                "leviathan_run_unseen_dispute_outcomes",
                "Convictions whose verdict account a new dispute took over before this indexer saw whether an appeal overturned them",
            ),
        }
    }

    fn observe(&self, labels: &RunLabels, observation: Option<&TreasurerObservation>) {
        let Some(treasurer) = observation else {
            for (family, _) in &self.gauges {
                family.remove(labels);
            }
            self.break_even_penalty.remove(labels);
            for status in VERDICT_STATUSES {
                self.verdicts
                    .remove(&RunStatusLabels::new(&labels.run_id, status_label(status)));
            }
            return;
        };
        for (family, value) in &self.gauges {
            family.get_or_create(labels).set(value(treasurer));
        }
        match treasurer.security.break_even_penalty {
            Some(penalty) => {
                self.break_even_penalty.get_or_create(labels).set(penalty);
            }
            None => {
                self.break_even_penalty.remove(labels);
            }
        }
        for (status, count) in VERDICT_STATUSES.iter().zip(treasurer.verdicts.by_status) {
            self.verdicts
                .get_or_create(&RunStatusLabels::new(&labels.run_id, status_label(*status)))
                .set(count as f64);
        }
        let tally = treasurer.verdicts;
        self.convictions
            .get_or_create(labels)
            .inc_by(tally.new_convictions);
        self.overturned_convictions
            .get_or_create(labels)
            .inc_by(tally.new_overturned);
        self.lapsed_disputes
            .get_or_create(labels)
            .inc_by(tally.new_lapsed);
        self.unseen_outcomes
            .get_or_create(labels)
            .inc_by(tally.new_unseen_outcomes);
    }
}

impl IndexerSeries {
    fn register(registry: &mut Registry) -> Self {
        Self {
            snapshot_slot: gauge(
                registry,
                "leviathan_indexer_snapshot_slot",
                "The slot the run's accounts were last read at",
            ),
            slot_lag: gauge(
                registry,
                "leviathan_indexer_slot_lag",
                "Slots the last snapshot trailed the RPC's newest slot",
            ),
            snapshot_timestamp: gauge(
                registry,
                "leviathan_indexer_snapshot_timestamp_seconds",
                "The cluster's clock at the last snapshot's slot",
            ),
            last_poll_success: gauge(
                registry,
                "leviathan_indexer_last_poll_success",
                "1 when the last poll of the run read the chain",
            ),
            poll_errors: counter(
                registry,
                "leviathan_indexer_poll_errors",
                "Polls of the run that failed to read the chain",
            ),
        }
    }
}

impl IndexerMetrics {
    pub fn register(registry: &mut Registry) -> Self {
        Self {
            run: RunSeries::register(registry),
            treasurer: TreasurerSeries::register(registry),
            indexer: IndexerSeries::register(registry),
        }
    }

    /// Exports what one poll of `run_id` read.
    pub fn observe(&self, run_id: &str, observation: &RunObservation) {
        let labels = RunLabels::new(run_id);
        self.run.observe(&labels, observation);
        self.treasurer
            .observe(&labels, observation.treasurer.as_ref());
        let indexer = &self.indexer;
        let clock = observation.clock;
        indexer
            .snapshot_slot
            .get_or_create(&labels)
            .set(clock.slot as f64);
        indexer
            .slot_lag
            .get_or_create(&labels)
            .set(clock.slot_lag() as f64);
        indexer
            .snapshot_timestamp
            .get_or_create(&labels)
            .set(clock.unix_timestamp as f64);
        indexer.last_poll_success.get_or_create(&labels).set(1.0);
        let _ = indexer.poll_errors.get_or_create(&labels);
    }

    /// Records a poll of `run_id` that could not read the chain. The last
    /// snapshot's series stay, and so does its slot and time, so the staleness
    /// shows.
    pub fn poll_failed(&self, run_id: &str) {
        let labels = RunLabels::new(run_id);
        self.indexer.poll_errors.get_or_create(&labels).inc();
        self.indexer
            .last_poll_success
            .get_or_create(&labels)
            .set(0.0);
    }
}
