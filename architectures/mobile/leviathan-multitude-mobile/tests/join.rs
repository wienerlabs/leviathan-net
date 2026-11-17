use std::thread;
use std::time::{Duration, Instant};

use leviathan_multitude::adversary::Behavior;
use leviathan_multitude::aggregate::{aggregate, verify_aggregate, Expectation};
use leviathan_multitude::config::{AuditConfig, InnerConfig, ModelConfig, RunConfig};
use leviathan_multitude::device::{DeviceState, Network, Thermal};
use leviathan_multitude::Device;
use leviathan_multitude_mobile::{Activity, RelayClient, Report, Session};
use leviathan_multitude_relay::http::serve;
use leviathan_multitude_relay::{Options, Relay};

const TEXT: &str = include_str!("../../../../shared/client/src/state/prompt_texts/index.json");
const LAST_ROUND: u32 = 10;
const PLUGGED_IN: DeviceState = DeviceState {
    charging: true,
    battery_percent: 100,
    network: Network::Unmetered,
    thermal: Thermal::Nominal,
    user_active: false,
    free_memory_mb: u32::MAX,
};

fn config() -> RunConfig {
    RunConfig {
        model: ModelConfig {
            d_model: 32,
            n_heads: 2,
            n_layers: 1,
            context: 32,
            ..ModelConfig::default()
        },
        inner: InnerConfig {
            micro_batch: 4,
            min_steps: 2,
            max_steps: 6,
            ..InnerConfig::default()
        },
        audit: AuditConfig {
            verification_percent: 100,
            ..AuditConfig::default()
        },
        round_seconds: 2.0,
        ..RunConfig::default()
    }
}

fn start_relay() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let opts = Options {
        enforce_band: true,
        ..Options::default()
    };
    let relay = Relay::new(config(), TEXT.as_bytes().to_vec(), 1651, Device::Cpu, opts).unwrap();
    thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                serve(listener, relay).await.unwrap();
            })
    });
    url
}

fn run_phone(url: String, seed: u8, behavior: Behavior, state: std::path::PathBuf) -> Report {
    let deadline = Instant::now() + Duration::from_secs(90);
    let join = || {
        Session::join(&url, [seed; 32], Some(state.clone()))
            .unwrap()
            .with_behavior(behavior.clone())
    };
    let mut session = join();
    let mut restarted = false;
    loop {
        let report = session.tick(&PLUGGED_IN).unwrap();
        let round = report.round.as_ref().map_or(0, |r| r.round);
        if matches!(report.activity, Activity::Convicted(_)) || round >= LAST_ROUND {
            return report;
        }
        assert!(
            Instant::now() < deadline,
            "phone {seed} stalled: {report:?}"
        );
        if seed == 1
            && !restarted
            && matches!(report.activity, Activity::Trained { .. })
            && round >= 4
        {
            restarted = true;
            session = join();
        }
        if !matches!(report.activity, Activity::Trained { .. }) {
            thread::sleep(Duration::from_millis(50));
        }
    }
}

#[test]
fn phones_join_a_relay_and_the_cheater_is_convicted() {
    let url = start_relay();
    let relay = RelayClient::new(&url);
    let info = relay.run().unwrap();
    let start_loss = relay.round().unwrap().global_loss;
    let dir = std::env::temp_dir().join(format!("multitude-join-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let phones: Vec<_> = [
        (1u8, Behavior::Honest),
        (2u8, Behavior::Honest),
        (3u8, Behavior::SignFlip),
    ]
    .into_iter()
    .map(|(seed, behavior)| {
        let url = url.clone();
        let state = dir.join(format!("phone-{seed}.state"));
        thread::spawn(move || run_phone(url, seed, behavior, state))
    })
    .collect();
    let reports: Vec<Report> = phones.into_iter().map(|p| p.join().unwrap()).collect();

    for honest in &reports[..2] {
        assert!(
            !matches!(honest.activity, Activity::Convicted(_)),
            "{honest:?}"
        );
        assert!(honest.rounds_counted >= 6, "{honest:?}");
    }
    let Activity::Convicted(reason) = &reports[2].activity else {
        panic!(
            "the sign-flipping phone was not convicted: {:?}",
            reports[2]
        );
    };
    assert!(reason.contains("distance"), "{reason}");

    let end = relay.round().unwrap();
    assert!(
        end.global_loss < start_loss,
        "{} -> {}",
        start_loss,
        end.global_loss
    );

    let watched = end.round - 1;
    let result = relay.result(watched).unwrap().unwrap();
    let pool = relay.contributions(watched).unwrap().unwrap();
    let expect = Expectation::new(&info.cfg, info.run, watched, result.claim.base_checkpoint);
    let recount = aggregate(&expect, &pool);
    assert_eq!(recount.claim, result.claim);
    assert_eq!(
        verify_aggregate(&expect, &pool, &result.claim, &recount.tally),
        Ok(Vec::new())
    );
    std::fs::remove_dir_all(&dir).ok();
}
