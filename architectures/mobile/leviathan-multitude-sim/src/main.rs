//! A Leviathan Multitude swarm in one process.
//!
//! Phones of four hardware classes train a byte-level transformer on the
//! Shakespeare prompts Leviathan's own client samples from, a relay tallies
//! their votes, a watcher recomputes the tally, the audit lottery makes phones
//! replay each other, and cheaters lose their bonds. Every step is the real
//! code path; only the network, the clock and the phones' sensors are
//! simulated.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;

use leviathan_multitude::adversary::{jitter, tamper_tally, Behavior};
use leviathan_multitude::aggregate::{aggregate, verify_aggregate, Expectation};
use leviathan_multitude::audit::{
    audit_contribution, calibrate_band, vote_distance, AuditContext, Verdict, DEFAULT_SAFETY_FACTOR,
};
use leviathan_multitude::compress::compress;
use leviathan_multitude::config::{AuditConfig, InnerConfig, ModelConfig, OuterConfig, RunConfig};
use leviathan_multitude::data::data_seed;
use leviathan_multitude::device::{
    decide, plan_inner_steps, Decision, DevicePolicy, DeviceState, Network, Thermal,
};
use leviathan_multitude::economics::assess;
use leviathan_multitude::lottery::{audit_probability, select_audits};
use leviathan_multitude::model::default_device;
use leviathan_multitude::protocol::short_address;
use leviathan_multitude::rng::{sha256_parts, SplitMix64};
use leviathan_multitude::train::{eval_loss, pseudo_gradient, train_local};
use leviathan_multitude::{Contribution, Corpus, Device, GlobalModel, PhoneNode, RoundJob};

const SHAKESPEARE: &str =
    include_str!("../../../../shared/client/src/state/prompt_texts/index.json");
const EVAL_WINDOWS: usize = 16;

#[derive(Parser, Debug)]
#[command(
    name = "multitude-sim",
    about = "Simulate a Leviathan Multitude phone swarm"
)]
struct Args {
    #[arg(long, default_value_t = 8)]
    phones: usize,
    #[arg(long, default_value_t = 30)]
    rounds: u32,
    #[arg(long, default_value_t = 1651)]
    seed: u64,
    /// Audit rate in percent. Leviathan's published operating point is 10;
    /// the demo default is higher so a short run catches its cheaters.
    #[arg(long, default_value_t = 25)]
    verification_percent: u8,
    /// A cheating phone, as `INDEX:KIND` with KIND one of sign-flip, random,
    /// lazy, withhold. Repeat for several.
    #[arg(long = "cheat", value_parser = parse_cheat)]
    cheats: Vec<(usize, Behavior)>,
    /// Round in which the relay publishes a tampered tally.
    #[arg(long)]
    evil_relay_round: Option<u32>,
    #[arg(long, default_value_t = 0.006)]
    outer_lr: f32,
    #[arg(long, default_value_t = 3e-3)]
    inner_lr: f64,
    #[arg(long, default_value_t = 12)]
    max_inner_steps: u16,
    /// Train on this text file instead of the bundled Shakespeare prompts.
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// Bond each phone posts, in collateral units.
    #[arg(long, default_value_t = 10.0)]
    bond: f64,
    /// Reward shared among a round's counted phones, as Leviathan shares
    /// `earning_rate_total_shared` among an epoch's finishers.
    #[arg(long, default_value_t = 1.0)]
    reward_per_round: f64,
    /// Share of a forfeited bond paid to whoever proved the fraud, in basis
    /// points, like the treasurer's `slash_bounty_bps`.
    #[arg(long, default_value_t = 5000)]
    slash_bounty_bps: u16,
    /// Write a JSON report here.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn parse_cheat(value: &str) -> Result<(usize, Behavior), String> {
    let (index, kind) = value
        .split_once(':')
        .ok_or_else(|| format!("expected INDEX:KIND, got {value}"))?;
    let index = index.parse::<usize>().map_err(|err| err.to_string())?;
    let behavior = match kind {
        "sign-flip" => Behavior::SignFlip,
        "random" => Behavior::RandomVotes,
        "lazy" => Behavior::Lazy { real_steps: 2 },
        "withhold" => Behavior::WithholdReveal,
        other => {
            return Err(format!(
                "unknown cheat {other}; use sign-flip, random, lazy or withhold"
            ))
        }
    };
    Ok((index, behavior))
}

/// An emulated hardware class. Throughput is the estimate for a 135M model on
/// that class and only drives how many inner steps a phone plans; the training
/// itself runs on this machine.
struct PhoneClass {
    name: &'static str,
    tokens_per_second: f32,
    jitter: f32,
    availability: f32,
    runs_hot: bool,
}

const CLASSES: [PhoneClass; 4] = [
    PhoneClass {
        name: "flagship-apple",
        tokens_per_second: 620.0,
        jitter: 0.0,
        availability: 0.95,
        runs_hot: false,
    },
    PhoneClass {
        name: "flagship-adreno",
        tokens_per_second: 480.0,
        jitter: 2e-4,
        availability: 0.92,
        runs_hot: false,
    },
    PhoneClass {
        name: "mid-range",
        tokens_per_second: 220.0,
        jitter: 4e-4,
        availability: 0.88,
        runs_hot: false,
    },
    PhoneClass {
        name: "budget",
        tokens_per_second: 110.0,
        jitter: 6e-4,
        availability: 0.85,
        runs_hot: true,
    },
];

struct SimPhone {
    node: PhoneNode,
    class: &'static PhoneClass,
    behavior: Behavior,
    streak: u32,
    rounds_counted: u32,
    earned: f64,
    bond: f64,
    bounty: f64,
    ejected: Option<(u32, String)>,
}

impl SimPhone {
    fn effective_behavior(&self) -> Behavior {
        match (&self.behavior, self.class.jitter) {
            (Behavior::Honest, scale) if scale > 0.0 => Behavior::Jitter { scale },
            (behavior, _) => behavior.clone(),
        }
    }

    fn sensors(&mut self, rng: &mut SplitMix64) -> DeviceState {
        let on_charger = rng.next_f32() < self.class.availability;
        let hot = self.class.runs_hot && self.streak >= 4;
        DeviceState {
            charging: on_charger,
            battery_percent: 85,
            network: Network::Unmetered,
            thermal: if hot {
                Thermal::Serious
            } else {
                Thermal::Nominal
            },
            user_active: false,
            free_memory_mb: 3072,
        }
    }
}

#[derive(Serialize)]
struct RoundReport {
    round: u32,
    trained: usize,
    paused: Vec<String>,
    counted: usize,
    min_steps: u16,
    max_steps: u16,
    mean_local_loss: f32,
    global_loss: f32,
    coverage: f32,
    upload_bytes_per_phone: usize,
    audits: usize,
    honest_max_distance: f32,
    convicted: Vec<String>,
    relay: String,
}

#[derive(Serialize)]
struct PhoneReport {
    address: String,
    class: &'static str,
    behavior: &'static str,
    rounds_counted: u32,
    earned: f64,
    bounty: f64,
    bond_left: f64,
    claimable: f64,
    ejected: Option<(u32, String)>,
}

#[derive(Serialize)]
struct Report {
    run_id: String,
    parameters: usize,
    band: f32,
    audit_probability: f64,
    start_loss: f32,
    end_loss: f32,
    rounds: Vec<RoundReport>,
    phones: Vec<PhoneReport>,
    honest_slashed: usize,
    relay_frauds_caught: Vec<u32>,
    expected_fraud_value_per_round: f64,
    economically_secure: bool,
}

fn load_corpus(path: Option<&PathBuf>) -> Result<Corpus> {
    if let Some(path) = path {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        return Ok(Corpus::from_text(&text));
    }
    #[derive(serde::Deserialize)]
    struct Prompt {
        text: String,
    }
    #[derive(serde::Deserialize)]
    struct Prompts {
        prompts: Vec<Prompt>,
    }
    let prompts: Prompts =
        serde_json::from_str(SHAKESPEARE).context("parsing the bundled prompts")?;
    let text = prompts
        .prompts
        .into_iter()
        .map(|p| p.text)
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(Corpus::from_text(&text))
}

/// Measure honest drift on the production path, class by class, and size the
/// band from the worst case with Leviathan's safety factor. Until a band is
/// measured for the hardware, the main network keeps its verifier in dry run;
/// this is that measurement.
fn calibrate(
    cfg: &RunConfig,
    global: &GlobalModel,
    corpus: &Corpus,
    device: &Device,
) -> Result<(f32, Vec<(String, f32)>)> {
    let probe = [0xC0u8; 32];
    let seed = data_seed(&global.seed, &probe);
    let local = train_local(
        cfg,
        &global.params,
        corpus,
        &seed,
        cfg.inner.max_steps,
        device,
    )?;
    let delta = pseudo_gradient(&global.params, &local.params);
    let zeros = vec![0.0; delta.len()];
    let (reference, _) = compress(&delta, &zeros, &cfg.compression)?;
    let mut per_class = Vec::new();
    let mut drifts = Vec::new();
    for class in &CLASSES {
        let mut worst = 0f32;
        for sample in 0..4u32 {
            let mut drifted = delta.clone();
            if class.jitter > 0.0 {
                jitter(&mut drifted, class.jitter, &probe, 10_000 + sample);
            }
            let (update, _) = compress(&drifted, &zeros, &cfg.compression)?;
            let distance = vote_distance(&update, &reference)?;
            drifts.push(distance);
            worst = worst.max(distance);
        }
        per_class.push((class.name.to_string(), worst));
    }
    let band = calibrate_band(&drifts, DEFAULT_SAFETY_FACTOR)?.max(0.01);
    Ok((band, per_class))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.phones < 2 {
        bail!("a swarm needs at least two phones");
    }
    for (index, _) in &args.cheats {
        if *index >= args.phones {
            bail!(
                "cheat index {index} is out of range for {} phones",
                args.phones
            );
        }
    }

    let mut cfg = RunConfig {
        run_id: "multitude-sim".into(),
        model: ModelConfig::default(),
        inner: InnerConfig {
            lr: args.inner_lr,
            max_steps: args.max_inner_steps,
            ..InnerConfig::default()
        },
        outer: OuterConfig {
            lr: args.outer_lr,
            total_rounds: args.rounds,
            ..OuterConfig::default()
        },
        audit: AuditConfig {
            verification_percent: args.verification_percent,
            ..AuditConfig::default()
        },
        ..RunConfig::default()
    };
    cfg.validate()?;

    let corpus = load_corpus(args.corpus.as_ref())?;
    corpus.check_context(cfg.model.context)?;
    let device = default_device();
    let run = cfg.fingerprint(&corpus.fingerprint());
    let mut global = GlobalModel::genesis(&cfg, args.seed)?;
    let num_params = cfg.model.num_params();

    let (band, per_class) = calibrate(&cfg, &global, &corpus, &device)?;
    cfg.audit.band = band;

    let mut phones: Vec<SimPhone> = (0..args.phones)
        .map(|i| {
            let secret = sha256_parts(&[
                b"multitude-sim/phone",
                &args.seed.to_le_bytes(),
                &(i as u64).to_le_bytes(),
            ]);
            let behavior = args
                .cheats
                .iter()
                .find(|(index, _)| *index == i)
                .map(|(_, behavior)| behavior.clone())
                .unwrap_or(Behavior::Honest);
            SimPhone {
                node: PhoneNode::new(secret, num_params),
                class: &CLASSES[i % CLASSES.len()],
                behavior,
                streak: 0,
                rounds_counted: 0,
                earned: 0.0,
                bond: args.bond,
                bounty: 0.0,
                ejected: None,
            }
        })
        .collect();

    let policy = DevicePolicy::default();
    let p = audit_probability(args.phones, cfg.audit.verification_percent);
    let start_loss = eval_loss(&cfg, &global.params, &corpus, EVAL_WINDOWS, &device)?;

    println!("Leviathan Multitude simulation");
    println!(
        "  model      {} parameters, byte-level, {} layers x {} wide, context {}",
        num_params, cfg.model.n_layers, cfg.model.d_model, cfg.model.context
    );
    println!(
        "  corpus     {} bytes ({})",
        corpus.len(),
        if args.corpus.is_some() {
            "custom"
        } else {
            "Leviathan's Shakespeare prompts"
        }
    );
    let selected = leviathan_multitude::compress::selected_count(
        num_params,
        cfg.compression.chunk,
        cfg.compression.topk,
    );
    let payload = selected + selected.div_ceil(8);
    println!(
        "  transport  top-{} of {}, 1-bit signs, error feedback: {} bytes of votes per round, {:.2} bits per parameter",
        cfg.compression.topk,
        cfg.compression.chunk,
        payload,
        payload as f64 * 8.0 / num_params as f64
    );
    println!(
        "  audits     verification_percent {} -> p = {:.3} per phone per round, about {:.1} rounds to catch a persistent cheater",
        cfg.audit.verification_percent,
        p,
        if p > 0.0 { 1.0 / p } else { f64::INFINITY }
    );
    println!("  band       calibrated on the replay path, worst drift x {DEFAULT_SAFETY_FACTOR}:");
    for (name, worst) in &per_class {
        println!("               {name:<16} worst honest drift {worst:.4}");
    }
    println!("               band = {band:.4}");
    println!("  phones");
    for phone in &phones {
        println!(
            "    {}  {:<16} {}",
            short_address(&phone.node.id()),
            phone.class.name,
            phone.behavior.label()
        );
    }
    println!(
        "  start loss {start_loss:.3} nats/byte (uniform guessing is {:.3})",
        (256f32).ln()
    );
    println!();

    let mut env = SplitMix64::new(args.seed ^ 0x5EED);
    let mut rounds = Vec::new();
    let mut relay_bond = 5.0 * args.bond;
    let mut relay_frauds = Vec::new();
    let mut honest_slashed = 0;
    let mut vault = 0.0f64;

    for _ in 0..args.rounds {
        let round = global.round;
        let base = global.checkpoint();
        let job = RoundJob {
            cfg: &cfg,
            run,
            round,
            round_seed: global.seed,
            global: &global.params,
            base_checkpoint: base,
            corpus: &corpus,
            device: &device,
        };

        let mut submitted: Vec<(usize, Contribution)> = Vec::new();
        let mut paused = Vec::new();
        let mut losses = Vec::new();
        let mut steps_range = (u16::MAX, 0u16);
        let mut upload = 0usize;
        for (index, phone) in phones.iter_mut().enumerate() {
            if phone.ejected.is_some() {
                continue;
            }
            let state = phone.sensors(&mut env);
            if let Decision::Pause(reason) = decide(&policy, &state) {
                paused.push(format!("{} {}", phone.class.name, reason));
                phone.streak = 0;
                continue;
            }
            let steps = plan_inner_steps(
                phone.class.tokens_per_second,
                cfg.tokens_per_step(),
                cfg.round_seconds,
                cfg.inner.min_steps,
                cfg.inner.max_steps,
            );
            if steps == 0 {
                paused.push(format!("{} too slow for the round", phone.class.name));
                continue;
            }
            let behavior = phone.effective_behavior();
            let (contribution, stats) = phone.node.contribute(&job, steps, &behavior)?;
            phone.streak += 1;
            steps_range = (steps_range.0.min(steps), steps_range.1.max(steps));
            if stats.mean_loss.is_finite() {
                losses.push(stats.mean_loss);
            }
            upload = upload.max(stats.upload_bytes);
            submitted.push((index, contribution));
        }

        let expect = Expectation::new(&cfg, run, round, base);
        let pool: Vec<Contribution> = submitted.iter().map(|(_, c)| c.clone()).collect();
        let aggregation = aggregate(&expect, &pool);

        // The relay publishes. In the scripted round it reverses 2% of the net
        // votes and hashes the tampered tally, so the claim is self-consistent
        // and only a recount can tell.
        let mut published = aggregation.tally.clone();
        let mut claim = aggregation.claim.clone();
        if args.evil_relay_round == Some(round) {
            tamper_tally(&mut published, 0.02, args.seed ^ round as u64);
            claim.tally_hash = published.hash();
        }

        // A watcher recounts from the contributions it saw.
        let watcher = submitted.first().map(|(index, _)| *index);
        let (tally, relay) = match verify_aggregate(&expect, &pool, &claim, &published) {
            Ok(omitted) if omitted.is_empty() => (published, "verified".to_string()),
            Ok(omitted) => (
                published,
                format!("verified, {} valid contributions omitted", omitted.len()),
            ),
            Err(fraud) => {
                relay_frauds.push(round);
                let forfeited = relay_bond;
                relay_bond = 0.0;
                let bounty = forfeited * args.slash_bounty_bps as f64 / 10_000.0;
                if let Some(watcher) = watcher {
                    phones[watcher].bounty += bounty;
                }
                vault += forfeited - bounty;
                (
                    aggregation.tally.clone(),
                    format!("FRAUD caught by recount ({fraud}); relay bond forfeited, honest tally used"),
                )
            }
        };

        let counted: Vec<usize> = claim
            .contributors
            .iter()
            .filter_map(|id| phones.iter().position(|phone| phone.node.id() == *id))
            .collect();
        for (index, contribution) in &submitted {
            if !counted.contains(index) {
                phones[*index].node.rollback(contribution.header.round);
            }
        }

        // The audit lottery, drawn from the round seed over the counted phones.
        let assignments =
            select_audits(&global.seed, counted.len(), cfg.audit.verification_percent);
        let ctx = AuditContext {
            cfg: &cfg,
            run,
            round,
            round_seed: global.seed,
            global: &global.params,
            base_checkpoint: base,
            corpus: &corpus,
            band,
            device: &device,
        };
        let mut convicted: Vec<(usize, usize, String)> = Vec::new();
        let mut honest_max = 0f32;
        for assignment in &assignments {
            let target = counted[assignment.target];
            let verifier = counted[assignment.verifier];
            if convicted.iter().any(|(t, _, _)| *t == target) {
                continue;
            }
            let contribution = &submitted
                .iter()
                .find(|(index, _)| *index == target)
                .expect("counted phones submitted")
                .1;
            let reveal = match phones[target].behavior {
                Behavior::WithholdReveal => None,
                _ => phones[target].node.reveal(round),
            };
            match audit_contribution(
                &ctx,
                contribution,
                reveal.as_ref(),
                assignment.target as u64,
            )? {
                Verdict::Honest { distance, .. } => honest_max = honest_max.max(distance),
                Verdict::Fraud { reason, .. } => {
                    convicted.push((target, verifier, reason.to_string()))
                }
            }
        }

        let lr = cfg.outer.lr_at(round);
        global.advance(&tally, lr);

        // Settle: the counted phones share the round's reward, convicted ones
        // forfeit their bond and their share.
        let finishers: Vec<usize> = counted
            .iter()
            .copied()
            .filter(|index| !convicted.iter().any(|(t, _, _)| t == index))
            .collect();
        let share = if finishers.is_empty() {
            0.0
        } else {
            args.reward_per_round / finishers.len() as f64
        };
        for index in &finishers {
            phones[*index].earned += share;
            phones[*index].rounds_counted += 1;
        }
        let mut convicted_labels = Vec::new();
        for (target, verifier, reason) in &convicted {
            let forfeited = phones[*target].bond;
            phones[*target].bond = 0.0;
            let bounty = forfeited * args.slash_bounty_bps as f64 / 10_000.0;
            phones[*verifier].bounty += bounty;
            vault += forfeited - bounty;
            if phones[*target].behavior.is_honest() {
                honest_slashed += 1;
            }
            phones[*target].ejected = Some((round, reason.clone()));
            convicted_labels.push(format!(
                "{} ({}, {})",
                short_address(&phones[*target].node.id()),
                phones[*target].behavior.label(),
                reason
            ));
        }

        let global_loss = eval_loss(&cfg, &global.params, &corpus, EVAL_WINDOWS, &device)?;
        let mean_local = if losses.is_empty() {
            f32::NAN
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        };
        let line = format!(
            "r{round:02} | {}/{} trained | steps {}-{} | local {:.3} | global {:.3} | {:.1} KB up | coverage {:>3.0}% | audits {}",
            submitted.len(),
            phones.iter().filter(|p| p.ejected.is_none() || p.ejected.as_ref().is_some_and(|(r, _)| *r == round)).count(),
            if steps_range.0 == u16::MAX { 0 } else { steps_range.0 },
            steps_range.1,
            mean_local,
            global_loss,
            upload as f32 / 1024.0,
            tally.coverage() * 100.0,
            assignments.len(),
        );
        println!("{line}");
        for label in &convicted_labels {
            println!("      convicted {label}");
        }
        if relay != "verified" {
            println!("      relay: {relay}");
        }
        for reason in &paused {
            println!("      paused: {reason}");
        }

        rounds.push(RoundReport {
            round,
            trained: submitted.len(),
            paused,
            counted: counted.len(),
            min_steps: if steps_range.0 == u16::MAX {
                0
            } else {
                steps_range.0
            },
            max_steps: steps_range.1,
            mean_local_loss: mean_local,
            global_loss,
            coverage: tally.coverage(),
            upload_bytes_per_phone: upload,
            audits: assignments.len(),
            honest_max_distance: honest_max,
            convicted: convicted_labels,
            relay,
        });
    }

    let end_loss = rounds.last().map(|r| r.global_loss).unwrap_or(start_loss);
    let honest_counted = phones
        .iter()
        .filter(|phone| phone.behavior.is_honest())
        .map(|phone| phone.rounds_counted as f64)
        .sum::<f64>();
    let honest_rounds_share = args.reward_per_round * args.rounds as f64 / honest_counted.max(1.0);
    let security = assess(p, honest_rounds_share, args.bond, args.bond);

    println!();
    println!("Result");
    println!(
        "  global loss {start_loss:.3} -> {end_loss:.3} nats/byte over {} rounds",
        args.rounds
    );
    let honest_max_overall = rounds
        .iter()
        .map(|r| r.honest_max_distance)
        .fold(0f32, f32::max);
    println!(
        "  honest phones slashed: {honest_slashed} (largest honest audit distance {honest_max_overall:.4} against band {band:.4})"
    );
    if relay_frauds.is_empty() {
        println!("  relay: every tally recounted and verified");
    } else {
        println!("  relay fraud caught by recount in round(s) {relay_frauds:?}");
    }
    println!(
        "  economics: reward share {:.3}/round, bond {:.1}, p {:.3} -> expected value of cheating {:+.3}/round ({})",
        honest_rounds_share,
        args.bond,
        p,
        security.expected_fraud_value_per_round,
        if security.economically_secure { "secure" } else { "NOT secure" }
    );
    println!("  run vault received {vault:.2} from forfeited bonds");
    println!();
    // Bond-gated claim, as in the treasurer's `participant_claim`: a phone
    // whose bond is gone cannot collect what it earned or the bounties it won,
    // so nothing is paid out without something left to slash.
    println!(
        "  {:<11} {:<16} {:<24} {:>7} {:>8} {:>7} {:>6} {:>9}  status",
        "phone", "class", "behavior", "counted", "earned", "bounty", "bond", "claimable"
    );
    let mut phone_reports = Vec::new();
    for phone in &phones {
        let status = match &phone.ejected {
            Some((round, _)) => format!("ejected in round {round}"),
            None => "active".into(),
        };
        let claimable = if phone.bond > 0.0 {
            phone.earned + phone.bounty
        } else {
            0.0
        };
        println!(
            "  {:<11} {:<16} {:<24} {:>7} {:>8.3} {:>7.2} {:>6.1} {:>9.3}  {}",
            short_address(&phone.node.id()),
            phone.class.name,
            phone.behavior.label(),
            phone.rounds_counted,
            phone.earned,
            phone.bounty,
            phone.bond,
            claimable,
            status
        );
        phone_reports.push(PhoneReport {
            address: phone.node.address(),
            class: phone.class.name,
            behavior: phone.behavior.label(),
            rounds_counted: phone.rounds_counted,
            earned: phone.earned,
            bounty: phone.bounty,
            bond_left: phone.bond,
            claimable,
            ejected: phone.ejected.clone(),
        });
    }

    if let Some(path) = &args.json {
        let report = Report {
            run_id: cfg.run_id.clone(),
            parameters: num_params,
            band,
            audit_probability: p,
            start_loss,
            end_loss,
            rounds,
            phones: phone_reports,
            honest_slashed,
            relay_frauds_caught: relay_frauds,
            expected_fraud_value_per_round: security.expected_fraud_value_per_round,
            economically_secure: security.economically_secure,
        };
        std::fs::write(path, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("writing {}", path.display()))?;
        println!();
        println!("  report written to {}", path.display());
    }
    Ok(())
}
