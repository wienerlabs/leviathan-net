use prometheus_client::encoding::text::encode;
use prometheus_client::registry::Registry;

/// The media type of what [`render`] produces. Prometheus negotiates the
/// OpenMetrics text format by it, which is what keeps a counter's `_total`
/// suffix and the `# EOF` terminator meaningful to the scraper.
pub const CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// A registry as one scrape sees it, in the OpenMetrics text format.
pub fn render(registry: &Registry) -> Result<String, std::fmt::Error> {
    let mut body = String::new();
    encode(&mut body, registry)?;
    Ok(body)
}

/// The names of the samples in a rendered scrape, as a rule addresses them:
/// with a counter's `_total` suffix, without labels or values.
pub fn sample_names(scrape: &str) -> std::collections::BTreeSet<String> {
    scrape
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .filter_map(|line| line.split(['{', ' ']).next())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus_client::metrics::counter::Counter;
    use prometheus_client::metrics::gauge::Gauge;

    #[test]
    fn a_registry_renders_as_openmetrics_text() {
        let mut registry = Registry::default();
        let gauge: Gauge = Gauge::default();
        gauge.set(7);
        registry.register("leviathan_example", "An example gauge", gauge);
        let counter: Counter = Counter::default();
        counter.inc_by(3);
        registry.register("leviathan_example_events", "An example counter", counter);

        let scrape = render(&registry).unwrap();
        assert!(
            scrape.contains("# TYPE leviathan_example gauge"),
            "{scrape}"
        );
        assert!(scrape.contains("leviathan_example 7"), "{scrape}");
        assert!(
            scrape.contains("leviathan_example_events_total 3"),
            "{scrape}"
        );
        assert!(scrape.ends_with("# EOF\n"), "{scrape}");
        assert_eq!(
            sample_names(&scrape).into_iter().collect::<Vec<_>>(),
            vec!["leviathan_example", "leviathan_example_events_total"]
        );
    }
}
