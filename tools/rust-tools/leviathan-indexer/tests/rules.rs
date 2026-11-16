//! Every series a kill-switch rule watches is one an exporter has.
//!
//! The rules used to name series nothing exported, so none of them could ever
//! fire and nothing said so (wienerlabs/leviathan-net#57). This reads the
//! rules Prometheus loads and checks each series they name against what the
//! indexer and the verifier daemon actually serve.

use std::collections::BTreeSet;

use leviathan_indexer::assess_security;
use leviathan_indexer::metrics::IndexerMetrics;
use leviathan_indexer::observe::RunObservation;
use leviathan_indexer::observe::SnapshotClock;
use leviathan_indexer::observe::TreasurerObservation;
use leviathan_indexer::verdicts::VerdictTally;
use leviathan_indexer::RunEconomics;
use leviathan_telemetry::exposition::render;
use leviathan_telemetry::exposition::sample_names;
use leviathan_telemetry::verifier::PassReport;
use leviathan_telemetry::verifier::VerifierMetrics;
use prometheus_client::registry::Registry;

const RULES: &str = include_str!("../../../../telemetry/leviathan-killswitch.rules.yml");
const SCRAPE_CONFIG: &str = include_str!("../../../../telemetry/prometheus.yml");

/// The PromQL of every rule: an `expr:` on one line, or a block scalar
/// (`expr: |`) running on while the lines are indented past the key.
fn expressions(rules: &str) -> Vec<String> {
    let mut expressions = Vec::new();
    let mut lines = rules.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((indent, value)) = line
            .find("expr:")
            .filter(|at| line[..*at].trim().is_empty())
            .map(|at| (at, line[at + "expr:".len()..].trim()))
        else {
            continue;
        };
        if value != "|" {
            expressions.push(value.to_string());
            continue;
        }
        let mut block = String::new();
        while let Some(next) = lines.peek() {
            let next_indent = next.len() - next.trim_start().len();
            if next.trim().is_empty() || next_indent <= indent {
                break;
            }
            block.push_str(next.trim());
            block.push('\n');
            lines.next();
        }
        expressions.push(block);
    }
    expressions
}

/// The Leviathan series an expression names.
fn series_named(expression: &str) -> BTreeSet<String> {
    expression
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
        .filter(|token| token.starts_with("leviathan_"))
        .map(str::to_string)
        .collect()
}

/// A run with every series populated: treasurer-managed, audited, with a
/// break-even penalty.
fn full_observation() -> RunObservation {
    let security = assess_security(
        0.1,
        &RunEconomics {
            reward: 500.0,
            bond: 500.0,
            slash_when_caught: 200.0,
        },
    );
    RunObservation {
        run_state: 3,
        training: true,
        epoch: 1,
        step: 10,
        registered_clients: 3,
        epoch_clients: [3, 0, 0, 0],
        min_clients: 2,
        audit_probability: 0.1,
        earned_points: 0,
        slashed_points: 0,
        convicted_clients: 0,
        clock: SnapshotClock {
            slot: 10,
            unix_timestamp: 1,
            tip_slot: 12,
        },
        treasurer: Some(TreasurerObservation {
            verifier_committee_size: 0,
            committee_drawn: true,
            min_verifier_committee: 2,
            epoch_earning_rate: 1_000,
            epoch_slashing_rate: 200,
            bond_minimum: 500,
            security,
            vault_balance: 1_500,
            bonded_amount: 1_500,
            unclaimed_earnings: 0,
            claimed_collateral: 0,
            bond_shortfall: 0,
            obligation_shortfall: 0,
            verdicts: VerdictTally::default(),
        }),
    }
}

/// What the indexer and the verifier daemon serve between them.
fn exported() -> BTreeSet<String> {
    let mut registry = Registry::default();
    let indexer = IndexerMetrics::register(&mut registry);
    let verifier = VerifierMetrics::register(&mut registry);
    indexer.observe("run", &full_observation());
    verifier.record("run", &PassReport::default());
    sample_names(&render(&registry).unwrap())
}

#[test]
fn every_rule_is_read() {
    let alerts = RULES.matches("- alert:").count();
    let expressions = expressions(RULES);
    assert_eq!(expressions.len(), alerts, "{expressions:#?}");
    assert!(alerts >= 12, "the kill switches went missing");
    assert!(expressions.iter().all(|expression| !expression.is_empty()));
}

#[test]
fn every_series_a_rule_watches_is_exported() {
    let exported = exported();
    let named: BTreeSet<String> = expressions(RULES)
        .iter()
        .flat_map(|expression| series_named(expression))
        .collect();
    assert!(
        named.contains("leviathan_run_economically_secure"),
        "the rules were not read: {named:?}"
    );
    let missing: Vec<&String> = named.difference(&exported).collect();
    assert!(
        missing.is_empty(),
        "the rules watch series no exporter has: {missing:?}\nexported: {exported:?}"
    );
}

#[test]
fn the_series_a_rule_names_are_found_in_block_and_inline_expressions() {
    let expressions = expressions(
        "      - alert: A\n        expr: leviathan_a > 0\n        for: 1m\n      - alert: B\n        expr: |\n          leviathan_b / leviathan_c\n          and up == 0\n        for: 2m\n",
    );
    assert_eq!(expressions.len(), 2);
    let named: Vec<String> = expressions.iter().flat_map(|e| series_named(e)).collect();
    assert_eq!(named, ["leviathan_a", "leviathan_b", "leviathan_c"]);
    // A series no exporter has is reported.
    assert!(!exported().contains("leviathan_b"));
}

/// The values of `key="..."` selectors in `text`.
fn quoted_values<'a>(text: &'a str, key: &str) -> BTreeSet<&'a str> {
    let opening = format!("{key}=\"");
    text.match_indices(&opening)
        .filter_map(|(at, _)| {
            let rest = &text[at + opening.len()..];
            rest.find('"').map(|end| &rest[..end])
        })
        .collect()
}

/// The jobs `prometheus.yml` scrapes.
fn scraped_jobs() -> BTreeSet<String> {
    SCRAPE_CONFIG
        .lines()
        .filter_map(|line| line.trim().strip_prefix("- job_name:"))
        .map(|job| {
            job.trim()
                .trim_matches(|c| c == '\'' || c == '"')
                .to_string()
        })
        .collect()
}

/// A rule on `up{job="..."}` for a job Prometheus does not scrape matches no
/// series, and so never fires.
#[test]
fn every_job_a_rule_watches_is_scraped() {
    let scraped = scraped_jobs();
    assert!(scraped.contains("leviathan-indexer"), "{scraped:?}");
    let watched: BTreeSet<String> = expressions(RULES)
        .iter()
        .flat_map(|expression| quoted_values(expression, "job"))
        .map(str::to_string)
        .collect();
    assert_eq!(
        watched,
        ["leviathan-indexer", "leviathan-verifier"]
            .map(str::to_string)
            .into(),
        "the rules watch the jobs {watched:?}"
    );
    let unscraped: Vec<&String> = watched.difference(&scraped).collect();
    assert!(
        unscraped.is_empty(),
        "the rules watch jobs nothing scrapes: {unscraped:?}"
    );
}
