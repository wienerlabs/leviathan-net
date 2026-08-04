//! The byte layout the web dashboard decodes.
//!
//! `CoordinatorAccount` is `#[repr(C)]` zero-copy, so the browser reads it by
//! offset rather than through Borsh. Those offsets are duplicated in
//! `leviathan-web/src/data/layout.ts`, and a decoder reading the wrong offset
//! does not fail loudly - it renders a plausible wrong number. This test is the
//! thing that fails instead.
//!
//! Regenerate `layout.json` with `UPDATE_WEB_LAYOUT=1 cargo test -p
//! psyche-solana-coordinator --test web_layout` and copy the values into the
//! web repo.

use std::collections::BTreeMap;
use std::mem::offset_of;
use std::mem::size_of;

use psyche_coordinator::Client as EpochClient;
use psyche_coordinator::Coordinator;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::CoordinatorEpochState;
use psyche_coordinator::CoordinatorProgress;
use psyche_coordinator::Round;
use psyche_solana_coordinator::Client as LedgerClient;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_coordinator::RunMetadata;
use psyche_solana_coordinator::clients_state::ClientsState;

fn layout() -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    let mut put = |key: &str, value: usize| {
        map.insert(key.to_string(), value);
    };

    put("discriminator", 8);

    put("size.CoordinatorAccount", size_of::<CoordinatorAccount>());
    put("size.CoordinatorInstanceState", size_of::<CoordinatorInstanceState>());
    put("size.Coordinator", size_of::<Coordinator>());
    put("size.CoordinatorEpochState", size_of::<CoordinatorEpochState>());
    put("size.CoordinatorConfig", size_of::<CoordinatorConfig>());
    put("size.CoordinatorProgress", size_of::<CoordinatorProgress>());
    put("size.ClientsState", size_of::<ClientsState>());
    put("size.RunMetadata", size_of::<RunMetadata>());
    put("size.Round", size_of::<Round>());
    put("size.EpochClient", size_of::<EpochClient>());
    put("size.LedgerClient", size_of::<LedgerClient>());

    put("CoordinatorAccount.version", offset_of!(CoordinatorAccount, version));
    put("CoordinatorAccount.state", offset_of!(CoordinatorAccount, state));
    put("CoordinatorAccount.nonce", offset_of!(CoordinatorAccount, nonce));

    put(
        "CoordinatorInstanceState.metadata",
        offset_of!(CoordinatorInstanceState, metadata),
    );
    put(
        "CoordinatorInstanceState.coordinator",
        offset_of!(CoordinatorInstanceState, coordinator),
    );
    put(
        "CoordinatorInstanceState.clients_state",
        offset_of!(CoordinatorInstanceState, clients_state),
    );
    put(
        "CoordinatorInstanceState.client_version",
        offset_of!(CoordinatorInstanceState, client_version),
    );

    put("Coordinator.run_id", offset_of!(Coordinator, run_id));
    put("Coordinator.run_state", offset_of!(Coordinator, run_state));
    put("Coordinator.model", offset_of!(Coordinator, model));
    put("Coordinator.config", offset_of!(Coordinator, config));
    put("Coordinator.progress", offset_of!(Coordinator, progress));
    put("Coordinator.epoch_state", offset_of!(Coordinator, epoch_state));
    put(
        "Coordinator.run_state_start_unix_timestamp",
        offset_of!(Coordinator, run_state_start_unix_timestamp),
    );

    put("CoordinatorConfig.warmup_time", offset_of!(CoordinatorConfig, warmup_time));
    put("CoordinatorConfig.cooldown_time", offset_of!(CoordinatorConfig, cooldown_time));
    put(
        "CoordinatorConfig.max_round_train_time",
        offset_of!(CoordinatorConfig, max_round_train_time),
    );
    put(
        "CoordinatorConfig.round_witness_time",
        offset_of!(CoordinatorConfig, round_witness_time),
    );
    put("CoordinatorConfig.epoch_time", offset_of!(CoordinatorConfig, epoch_time));
    put("CoordinatorConfig.total_steps", offset_of!(CoordinatorConfig, total_steps));
    put(
        "CoordinatorConfig.init_min_clients",
        offset_of!(CoordinatorConfig, init_min_clients),
    );
    put("CoordinatorConfig.min_clients", offset_of!(CoordinatorConfig, min_clients));
    put("CoordinatorConfig.witness_nodes", offset_of!(CoordinatorConfig, witness_nodes));
    put(
        "CoordinatorConfig.global_batch_size_start",
        offset_of!(CoordinatorConfig, global_batch_size_start),
    );
    put(
        "CoordinatorConfig.global_batch_size_end",
        offset_of!(CoordinatorConfig, global_batch_size_end),
    );
    put(
        "CoordinatorConfig.verification_percent",
        offset_of!(CoordinatorConfig, verification_percent),
    );

    put("CoordinatorEpochState.rounds", offset_of!(CoordinatorEpochState, rounds));
    put("CoordinatorEpochState.clients", offset_of!(CoordinatorEpochState, clients));
    put(
        "CoordinatorEpochState.exited_clients",
        offset_of!(CoordinatorEpochState, exited_clients),
    );
    put(
        "CoordinatorEpochState.rounds_head",
        offset_of!(CoordinatorEpochState, rounds_head),
    );
    put(
        "CoordinatorEpochState.start_step",
        offset_of!(CoordinatorEpochState, start_step),
    );
    put("CoordinatorEpochState.last_step", offset_of!(CoordinatorEpochState, last_step));
    put(
        "CoordinatorEpochState.start_timestamp",
        offset_of!(CoordinatorEpochState, start_timestamp),
    );

    put("Round.witnesses", offset_of!(Round, witnesses));
    put("Round.data_index", offset_of!(Round, data_index));
    put("Round.random_seed", offset_of!(Round, random_seed));
    put("Round.height", offset_of!(Round, height));
    put("Round.clients_len", offset_of!(Round, clients_len));
    put("Round.tie_breaker_tasks", offset_of!(Round, tie_breaker_tasks));

    put("EpochClient.id", offset_of!(EpochClient, id));
    put("EpochClient.state", offset_of!(EpochClient, state));
    put("EpochClient.exited_height", offset_of!(EpochClient, exited_height));

    // The earned and slashed ledger the leaderboard is built from. It is a
    // different `Client` from the epoch roster one, indexed differently, which
    // is exactly the confusion that produced wienerlabs/leviathan#15 finding 1.
    put("ClientsState.clients", offset_of!(ClientsState, clients));
    put("ClientsState.next_active", offset_of!(ClientsState, next_active));
    put("LedgerClient.id", offset_of!(LedgerClient, id));
    put("LedgerClient.earned", offset_of!(LedgerClient, earned));
    put("LedgerClient.slashed", offset_of!(LedgerClient, slashed));
    put("LedgerClient.active", offset_of!(LedgerClient, active));

    map
}

/// The layout the web decoder was written against.
///
/// A value moving here is not a cosmetic diff: it means every number the
/// dashboard prints for that field has been read out of the wrong bytes.
#[test]
fn layout_matches_the_web_decoder() {
    let actual = layout();
    let expected: BTreeMap<String, usize> =
        serde_json::from_str(include_str!("web_layout.json"))
            .expect("web_layout.json is not valid JSON");

    if std::env::var("UPDATE_WEB_LAYOUT").is_ok() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/web_layout.json");
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&actual).unwrap()),
        )
        .unwrap();
        println!("wrote {path}");
        return;
    }

    let mut drift: Vec<String> = Vec::new();
    for (key, value) in &actual {
        match expected.get(key) {
            Some(previous) if previous == value => {},
            Some(previous) => {
                drift.push(format!("  {key}: {previous} -> {value}"))
            },
            None => drift.push(format!("  {key}: new field at {value}")),
        }
    }
    for key in expected.keys() {
        if !actual.contains_key(key) {
            drift.push(format!("  {key}: gone"));
        }
    }

    assert!(
        drift.is_empty(),
        "coordinator layout moved, so leviathan-web/src/data/layout.ts now reads \
         the wrong bytes. Update both, then rerun with UPDATE_WEB_LAYOUT=1:\n{}",
        drift.join("\n")
    );
}

/// The decoder trusts `version` and the account length before it reads a single
/// field, so both have to be what the web repo checks for.
#[test]
fn account_is_the_size_the_dashboard_expects() {
    assert_eq!(CoordinatorAccount::VERSION, 1);
    assert_eq!(
        CoordinatorAccount::space_with_discriminator(),
        8 + size_of::<CoordinatorAccount>()
    );
}
