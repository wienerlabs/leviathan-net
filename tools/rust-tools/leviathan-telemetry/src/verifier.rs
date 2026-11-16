use std::collections::VecDeque;
use std::ops::AddAssign;
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use std::time::Instant;

use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

use crate::labels::RunLabels;

/// What one audit pass of the verifier daemon judged.
///
/// A contribution judged for good is counted once: within band, convicted,
/// withheld, already on chain, or malformed, it is settled and not judged
/// again. A contribution whose submission failed is not settled, so the next
/// pass judges it, and counts it, again until the chain takes it. That makes
/// `submission_failures` a count of attempts, which is what an alert on a
/// verifier that cannot reach the chain wants.
///
/// An equivocation proof is judged the same way. Whatever becomes of it is
/// counted in the same outcome counters as a fraud proof, so the kill switch
/// covers it too (wienerlabs/leviathan-net#46).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassReport {
    /// Contributions replayed or compared and found within the band.
    pub cleared: u64,
    /// Contributions found outside the band: a fraud proof this verifier
    /// holds. Each is submitted, withheld, already on chain, failed, or
    /// against a committer this session already convicted.
    pub fraud_found: u64,
    /// Committers found to have signed two different payloads for one batch:
    /// a proof of equivocation this verifier holds, which needs no replay.
    /// Each is submitted, withheld, already on chain, or failed.
    pub equivocations: u64,
    /// Fraud proofs put on chain: a committee vote, a slash, or a proof of
    /// equivocation.
    pub submitted: u64,
    /// Fraud proofs the chain refused because it already holds them: the
    /// target convicted in this epoch, or this verifier's vote in the round.
    pub already_on_chain: u64,
    /// Attempts to put a fraud proof on chain that failed for any other
    /// reason. The proof is retried on the next pass. A proof of equivocation
    /// its epoch can no longer charge, because the epoch stopped charging or
    /// its committer withdrew or left the roster, is counted here once and
    /// dropped.
    pub submission_failures: u64,
    /// Fraud proofs held back because the daemon runs in dry-run mode.
    pub withheld: u64,
    /// Signed payloads that could not be judged: they do not decode, or do not
    /// fit the model the reference was replayed on. Also equivocation records
    /// that prove nothing, which a node writes only after checking them, so
    /// one that does not hold was damaged or tampered with on disk.
    pub malformed: u64,
    /// Contributions this verifier could not establish a reference for: its
    /// reference dump does not read, or the replay failed. Retried.
    pub unreplayable: u64,
}

impl PassReport {
    /// What the pass convicted or would have convicted: what it submitted,
    /// and what a dry run held back.
    pub fn convictions(&self) -> u64 {
        self.submitted + self.withheld
    }
}

impl AddAssign for PassReport {
    fn add_assign(&mut self, other: Self) {
        self.cleared += other.cleared;
        self.fraud_found += other.fraud_found;
        self.equivocations += other.equivocations;
        self.submitted += other.submitted;
        self.already_on_chain += other.already_on_chain;
        self.submission_failures += other.submission_failures;
        self.withheld += other.withheld;
        self.malformed += other.malformed;
        self.unreplayable += other.unreplayable;
    }
}

type Count = fn(&PassReport) -> u64;

/// Every counter a pass feeds, its name and help, and the field it counts.
const PASS_COUNTERS: [(&str, &str, Count); 9] = [
    (
        "leviathan_verifier_contributions_cleared",
        "Contributions this verifier judged within the tolerance band",
        |r| r.cleared,
    ),
    (
        "leviathan_verifier_fraud_found",
        "Contributions this verifier judged outside the band, counted again while a failed submission is retried",
        |r| r.fraud_found,
    ),
    (
        "leviathan_verifier_equivocations_found",
        "Committers this verifier found signing two different payloads for one batch, counted again while a failed submission is retried",
        |r| r.equivocations,
    ),
    (
        "leviathan_verifier_convictions_submitted",
        "Fraud proofs this verifier put on chain, as a committee vote, a slash or a proof of equivocation",
        |r| r.submitted,
    ),
    (
        "leviathan_verifier_already_on_chain",
        "Fraud proofs the chain refused because it already holds a conviction of the target or this verifier's vote",
        |r| r.already_on_chain,
    ),
    (
        "leviathan_verifier_submission_failures",
        "Attempts to put a fraud proof on chain that failed for any other reason; retried every pass, or counted once for a proof of equivocation the chain can no longer charge",
        |r| r.submission_failures,
    ),
    (
        "leviathan_verifier_convictions_withheld",
        "Fraud proofs held back because the daemon runs in dry-run mode",
        |r| r.withheld,
    ),
    (
        "leviathan_verifier_contributions_malformed",
        "Signed payloads that could not be judged, as they do not decode or do not fit the replayed model, and equivocation records that prove nothing",
        |r| r.malformed,
    ),
    (
        "leviathan_verifier_contributions_unreplayable",
        "Contributions this verifier could not establish a reference for; retried every pass",
        |r| r.unreplayable,
    ),
];

/// The counters `leviathan-verifier-daemon` exports, one series per run.
///
/// `submission_failures` is the daemon's side of the uncaught-fraud kill
/// switch: a fraud proof it holds that never reached the chain, so the chain
/// alone cannot show it.
pub struct VerifierMetrics {
    pass: Vec<(Family<RunLabels, Counter>, Count)>,
    pass_errors: Family<RunLabels, Counter>,
    slowest_pass: Family<RunLabels, Seconds>,
    poll_interval: Family<RunLabels, Seconds>,
}

type Seconds = Gauge<f64, AtomicU64>;

impl VerifierMetrics {
    pub fn register(registry: &mut Registry) -> Self {
        let pass = PASS_COUNTERS
            .iter()
            .map(|(name, help, count)| {
                let family = Family::<RunLabels, Counter>::default();
                registry.register(*name, *help, family.clone());
                (family, *count)
            })
            .collect();
        let pass_errors = Family::<RunLabels, Counter>::default();
        registry.register(
            "leviathan_verifier_pass_errors",
            "Audit passes that failed before judging anything",
            pass_errors.clone(),
        );
        let slowest_pass = Family::<RunLabels, Seconds>::default();
        registry.register(
            "leviathan_verifier_slowest_pass_seconds",
            "The slowest of this verifier's audit passes in the last hour, preparing the replay and submitting included",
            slowest_pass.clone(),
        );
        let poll_interval = Family::<RunLabels, Seconds>::default();
        registry.register(
            "leviathan_verifier_poll_interval_seconds",
            "How long this verifier waits between audit passes",
            poll_interval.clone(),
        );
        Self {
            pass,
            pass_errors,
            slowest_pass,
            poll_interval,
        }
    }

    /// Records the slowest pass over `run_id` in the last hour, as
    /// [`SlowestPass`] keeps it, and how long the daemon waits between passes.
    /// Together they bound how late this verifier can vote on a fraud: a
    /// payload that lands just after a pass started waits out that pass, the
    /// interval, and the next pass. The final round of an epoch has to leave
    /// at least that long (wienerlabs/leviathan-net#40).
    pub fn record_timing(&self, run_id: &str, pass: Duration, poll_interval: Duration) {
        let labels = RunLabels::new(run_id);
        self.slowest_pass
            .get_or_create(&labels)
            .set(pass.as_secs_f64());
        self.poll_interval
            .get_or_create(&labels)
            .set(poll_interval.as_secs_f64());
    }

    /// Counts what one pass over `run_id` judged.
    pub fn record(&self, run_id: &str, report: &PassReport) {
        let labels = RunLabels::new(run_id);
        for (family, count) in &self.pass {
            family.get_or_create(&labels).inc_by(count(report));
        }
        // Touched so the series exists from the first pass on: a rule taking
        // `increase()` over it needs a zero to rise from.
        let _ = self.pass_errors.get_or_create(&labels);
    }

    /// Counts a pass over `run_id` that failed before it judged anything.
    pub fn pass_failed(&self, run_id: &str) {
        let labels = RunLabels::new(run_id);
        self.pass_errors.get_or_create(&labels).inc();
        // A daemon whose every pass fails still exports the series a rule on
        // its submissions reads, at zero.
        for (family, _) in &self.pass {
            let _ = family.get_or_create(&labels);
        }
    }
}

/// The slowest audit pass within a trailing window, kept by the daemon itself.
///
/// A gauge of the last pass alone could be overwritten by a fast pass before a
/// scrape reads a slow one, and the slow one is what bounds how late a vote can
/// come. Kept here, a slow pass stays in the series for the whole window
/// however often Prometheus scrapes.
pub struct SlowestPass {
    window: Duration,
    passes: VecDeque<(Instant, Duration)>,
}

impl SlowestPass {
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            passes: VecDeque::new(),
        }
    }

    /// Records a pass that took `pass` and ended at `now`, and returns the
    /// slowest pass that ended within the window before `now`.
    pub fn record(&mut self, now: Instant, pass: Duration) -> Duration {
        self.passes.push_back((now, pass));
        while self
            .passes
            .front()
            .is_some_and(|(ended, _)| now.saturating_duration_since(*ended) > self.window)
        {
            self.passes.pop_front();
        }
        self.passes
            .iter()
            .map(|(_, took)| *took)
            .max()
            .unwrap_or(pass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exposition::render;

    #[test]
    fn a_pass_is_counted_per_run_in_every_counter() {
        let mut registry = Registry::default();
        let metrics = VerifierMetrics::register(&mut registry);
        let report = PassReport {
            cleared: 4,
            fraud_found: 6,
            equivocations: 7,
            submitted: 1,
            already_on_chain: 1,
            submission_failures: 2,
            withheld: 0,
            malformed: 3,
            unreplayable: 5,
        };
        metrics.record("run-a", &report);
        metrics.record("run-a", &report);
        metrics.pass_failed("run-b");

        let scrape = render(&registry).unwrap();
        for line in [
            r#"leviathan_verifier_contributions_cleared_total{run_id="run-a"} 8"#,
            r#"leviathan_verifier_fraud_found_total{run_id="run-a"} 12"#,
            r#"leviathan_verifier_equivocations_found_total{run_id="run-a"} 14"#,
            r#"leviathan_verifier_convictions_submitted_total{run_id="run-a"} 2"#,
            r#"leviathan_verifier_already_on_chain_total{run_id="run-a"} 2"#,
            r#"leviathan_verifier_submission_failures_total{run_id="run-a"} 4"#,
            r#"leviathan_verifier_convictions_withheld_total{run_id="run-a"} 0"#,
            r#"leviathan_verifier_contributions_malformed_total{run_id="run-a"} 6"#,
            r#"leviathan_verifier_contributions_unreplayable_total{run_id="run-a"} 10"#,
            r#"leviathan_verifier_pass_errors_total{run_id="run-a"} 0"#,
            r#"leviathan_verifier_submission_failures_total{run_id="run-b"} 0"#,
            r#"leviathan_verifier_pass_errors_total{run_id="run-b"} 1"#,
        ] {
            assert!(scrape.contains(line), "missing {line} in\n{scrape}");
        }
    }

    #[test]
    fn reports_add_up_and_count_what_a_dry_run_held_back() {
        let mut total = PassReport::default();
        let first = PassReport {
            cleared: 1,
            fraud_found: 2,
            submitted: 1,
            already_on_chain: 1,
            malformed: 1,
            ..PassReport::default()
        };
        let second = PassReport {
            fraud_found: 1,
            equivocations: 2,
            withheld: 1,
            submission_failures: 1,
            unreplayable: 2,
            ..PassReport::default()
        };
        total += first;
        total += second;
        assert_eq!(
            total,
            PassReport {
                cleared: 1,
                fraud_found: 3,
                equivocations: 2,
                submitted: 1,
                already_on_chain: 1,
                submission_failures: 1,
                withheld: 1,
                malformed: 1,
                unreplayable: 2,
            }
        );
        assert_eq!(total.convictions(), 2);
    }

    #[test]
    fn a_pass_and_the_wait_before_the_next_are_exported_in_seconds() {
        let mut registry = Registry::default();
        let metrics = VerifierMetrics::register(&mut registry);
        metrics.record_timing(
            "run-a",
            std::time::Duration::from_millis(2_500),
            std::time::Duration::from_secs(8),
        );
        // The gauge holds what it was last given: `SlowestPass` decides what
        // that is.
        metrics.record_timing(
            "run-a",
            std::time::Duration::from_millis(1_250),
            std::time::Duration::from_secs(8),
        );
        let scrape = render(&registry).unwrap();
        for line in [
            r#"leviathan_verifier_slowest_pass_seconds{run_id="run-a"} 1.25"#,
            r#"leviathan_verifier_poll_interval_seconds{run_id="run-a"} 8.0"#,
        ] {
            assert!(scrape.contains(line), "missing {line} in\n{scrape}");
        }
    }

    #[test]
    fn the_slowest_pass_in_the_window_is_kept_until_it_ages_out() {
        let hour = Duration::from_secs(3_600);
        let mut slowest = SlowestPass::new(hour);
        let start = Instant::now();
        let at = |seconds: u64| start + Duration::from_secs(seconds);
        let took = Duration::from_millis;
        assert_eq!(slowest.record(at(0), took(300)), took(300));
        assert_eq!(slowest.record(at(10), took(2_000)), took(2_000));
        // Faster passes after a slow one do not hide it.
        assert_eq!(slowest.record(at(20), took(250)), took(2_000));
        assert_eq!(slowest.record(at(3_610), took(400)), took(2_000));
        // An hour and more after it ended, it no longer counts.
        assert_eq!(slowest.record(at(3_611), took(400)), took(400));
    }
}
