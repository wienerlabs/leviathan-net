use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use leviathan_multitude::adversary::Behavior;
use leviathan_multitude::device::{DeviceState, Network, Thermal};
use leviathan_multitude_mobile::{generate_secret, Activity, Session};

const PLUGGED_IN: DeviceState = DeviceState {
    charging: true,
    battery_percent: 100,
    network: Network::Unmetered,
    thermal: Thermal::Nominal,
    user_active: false,
    free_memory_mb: u32::MAX,
};
const POLL: Duration = Duration::from_millis(500);
const RETRY: Duration = Duration::from_secs(3);

#[derive(Parser, Debug)]
#[command(
    name = "multitude-phone",
    about = "Join a Leviathan Multitude run from this machine, the way the phone app does"
)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080", help = "Relay URL")]
    relay: String,
    #[arg(
        long,
        help = "File with this phone's 32-byte key, created when missing"
    )]
    key: PathBuf,
    #[arg(
        long,
        help = "File that keeps the error buffer and the pending round across restarts"
    )]
    state: Option<PathBuf>,
    #[arg(long, help = "Stop after this many counted rounds")]
    rounds: Option<u32>,
    #[arg(
        long,
        default_value = "honest",
        value_parser = parse_behavior,
        help = "honest, or a cheat the relay should catch: sign-flip, random, lazy, withhold"
    )]
    behave: Behavior,
}

fn parse_behavior(value: &str) -> Result<Behavior, String> {
    match value {
        "honest" => Ok(Behavior::Honest),
        "sign-flip" => Ok(Behavior::SignFlip),
        "random" => Ok(Behavior::RandomVotes),
        "lazy" => Ok(Behavior::Lazy { real_steps: 1 }),
        "withhold" => Ok(Behavior::WithholdReveal),
        other => Err(format!("unknown behavior {other}")),
    }
}

fn load_or_create_key(path: &Path) -> Result<[u8; 32]> {
    if path.exists() {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let Ok(secret) = <[u8; 32]>::try_from(bytes.as_slice()) else {
            bail!("{} must hold exactly 32 bytes", path.display());
        };
        return Ok(secret);
    }
    let secret = generate_secret();
    std::fs::write(path, &secret).with_context(|| format!("writing {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(secret.try_into().expect("generate_secret returns 32 bytes"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let secret = load_or_create_key(&args.key)?;
    let mut session = Session::join(&args.relay, secret, args.state.clone())
        .with_context(|| format!("joining {}", args.relay))?
        .with_behavior(args.behave.clone());
    println!("Phone {} joined {}", session.address(), args.relay);

    let mut last = String::new();
    loop {
        let report = match session.tick(&PLUGGED_IN) {
            Ok(report) => report,
            Err(err) => {
                eprintln!("{err}");
                sleep(RETRY);
                continue;
            }
        };
        let line = match &report.activity {
            Activity::Trained {
                round,
                steps,
                loss,
                upload_bytes,
                seconds,
            } => format!(
                "r{round:03} trained {steps} steps in {seconds:.2}s, local loss {loss:.3}, sent {upload_bytes} bytes"
            ),
            Activity::Waiting(why) => why.clone(),
            Activity::Paused(why) => format!("paused: {why}"),
            Activity::Convicted(why) => format!("convicted: {why}"),
        };
        if line != last {
            let global = report
                .round
                .as_ref()
                .map(|r| {
                    format!(
                        "global loss {:.3}, {} phones",
                        r.global_loss, r.phones_last_round
                    )
                })
                .unwrap_or_default();
            println!(
                "{line} | {global} | counted {} | audits answered {}",
                report.rounds_counted, report.audits_answered
            );
            last = line;
        }
        if let Activity::Convicted(_) = report.activity {
            std::process::exit(2);
        }
        if args
            .rounds
            .is_some_and(|rounds| report.rounds_counted >= rounds)
        {
            return Ok(());
        }
        if !matches!(report.activity, Activity::Trained { .. }) {
            sleep(POLL);
        }
    }
}
