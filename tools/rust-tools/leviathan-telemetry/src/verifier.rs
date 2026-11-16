use std::ops::AddAssign;

use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PassReport {
    /// Contributions replayed or compared and found within the band.
    pub cleared: u64,
    /// Contributions found outside the band: a fraud proof this verifier
    /// holds. Each is submitted, withheld, already on chain, failed, or
    /// against a committer this session already convicted.
    pub fraud_found: u64,
    /// Fraud proofs put on chain, as a committee vote or as a slash.
    pub submitted: u64,
    /// Fraud proofs the chain refused because it already holds them: the
    /// target convicted in this epoch, or this verifier's vote in the round.
    pub already_on_chain: u64,
    /// Attempts to put a fraud proof on chain that failed for any other
    /// reason. The proof is retried on the next pass.
    pub submission_failures: u64,
    /// Fraud proofs held back because the daemon runs in dry-run mode.
    pub withheld: u64,
    /// Signed payloads that could not be judged: they do not decode, or do not
    /// fit the model the reference was replayed on.
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
const PASS_COUNTERS: [(&str, &str, Count); 8] = [
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
        "leviathan_verifier_convictions_submitted",
        "Fraud proofs this verifier put on chain, as a committee vote or a slash",
        |r| r.submitted,
    ),
    (
        "leviathan_verifier_already_on_chain",
        "Fraud proofs the chain refused because it already holds a conviction of the target or this verifier's vote",
        |r| r.already_on_chain,
    ),
    (
        "leviathan_verifier_submission_failures",
        "Attempts to put a fraud proof on chain that failed for any other reason; retried every pass",
        |r| r.submission_failures,
    ),
    (
        "leviathan_verifier_convictions_withheld",
        "Fraud proofs held back because the daemon runs in dry-run mode",
        |r| r.withheld,
    ),
    (
        "leviathan_verifier_contributions_malformed",
        "Signed payloads that could not be judged: they do not decode or do not fit the replayed model",
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
}

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
        Self { pass, pass_errors }
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
}
