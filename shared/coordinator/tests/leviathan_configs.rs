use psyche_coordinator::{CoordinatorConfig, model::Model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct RunConfig {
    config: CoordinatorConfig,
    model: Model,
}

fn string_leaves(value: &toml::Value, out: &mut Vec<String>) {
    match value {
        toml::Value::String(s) => out.push(s.clone()),
        toml::Value::Array(items) => items.iter().for_each(|v| string_leaves(v, out)),
        toml::Value::Table(table) => table.values().for_each(|v| string_leaves(v, out)),
        _ => {}
    }
}

#[test]
fn leviathan_run_configs_parse_and_pass_checks() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/leviathan");
    let mut checked = 0;
    for entry in std::fs::read_dir(dir).expect("config/leviathan exists") {
        let path = entry.expect("readable entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("readable config");
        let run: RunConfig = toml::from_str(&text)
            .unwrap_or_else(|e| panic!("{} does not parse: {e}", path.display()));
        if let Err(e) = run.config.check_error() {
            panic!("{} fails the coordinator check: {e:?}", path.display());
        }
        assert!(
            run.model.check(),
            "{} fails the model check",
            path.display()
        );

        let raw: toml::Value = toml::from_str(&text).expect("parses as toml");
        let stored = toml::Value::try_from(&run).expect("serializes back");
        let (mut written, mut kept) = (Vec::new(), Vec::new());
        string_leaves(&raw, &mut written);
        string_leaves(&stored, &mut kept);
        for s in written {
            assert!(
                kept.contains(&s),
                "{}: `{s}` is longer than its on-chain field and would be truncated",
                path.display()
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 3,
        "expected the Leviathan run configs, found {checked}"
    );
}
