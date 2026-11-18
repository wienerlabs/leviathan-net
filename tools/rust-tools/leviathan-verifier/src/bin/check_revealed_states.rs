//! Checks that real verifiers fetched the state each payload they audit
//! commits to (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! A verifier keeps the state an audited payload was trained from beside the
//! payload's dump (`revealed_state_path`). For every state kept so, the
//! commitment record beside the dump has to name it: it hashes to the digest
//! the trainer signed, and it is as long as a state of the run's model. The
//! run's verifiers have to have revealed as many states as they say they did,
//! across rounds past the first of an epoch.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use leviathan_verifier::fidelity::read_client_dumps;
use leviathan_verifier::round_states::load_node_state;
use leviathan_verifier::{parse_device, parse_dtype, read_run_model};
use psyche_modeling::round_state_len;
use psyche_network::{commitment_record_path, revealed_state_path, CommitmentRecord};

#[derive(Parser, Debug)]
#[command(
    name = "check-revealed-states",
    about = "Check that real verifiers fetched the state each payload they audit commits to (wienerlabs/leviathan-net#89)"
)]
struct Args {
    /// The configuration the clients' run was started with.
    #[arg(long)]
    run_config: PathBuf,
    /// Each client's `--write-gradients-dir`.
    #[arg(long, required = true, num_args = 1..)]
    dumps: Vec<PathBuf>,
    /// How many states the clients say they revealed, which have to be kept.
    #[arg(long)]
    revealed: usize,
    /// Every step a payload was trained at has to have had a state revealed:
    /// for a run whose every training round seats a verifier.
    #[arg(long, default_value_t = false)]
    every_trained_step: bool,
    /// The device and dtype the clients trained in. The client loads its model
    /// in bf16.
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "bf16")]
    dtype: String,
}

/// A payload a verifier kept, and whether it kept the state it commits to.
struct Kept {
    epoch_start: u64,
    step: u32,
    revealed: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let llm = read_run_model(&args.run_config)?;
    let (model, distro) =
        load_node_state(&llm, parse_device(&args.device)?, parse_dtype(&args.dtype)?)?;
    let len = round_state_len(&*model, &distro)?;

    let mut kept = vec![];
    for dir in &args.dumps {
        for dump in read_client_dumps(dir)? {
            let record_path = commitment_record_path(&dump.path);
            if !record_path.exists() {
                continue;
            }
            let record = CommitmentRecord::from_bytes(
                &std::fs::read(&record_path)
                    .with_context(|| format!("reading {}", record_path.display()))?,
            )
            .map_err(|err| anyhow!("{}: {err}", record_path.display()))?;
            let state_path = revealed_state_path(&dump.path);
            let revealed = state_path.exists();
            if revealed {
                let digest = record.state_digest.ok_or_else(|| {
                    anyhow!(
                        "{} is kept beside a payload that commits to no state",
                        state_path.display()
                    )
                })?;
                let state = std::fs::read(&state_path)?;
                if state.len() as u64 != len {
                    bail!(
                        "{} is {} bytes, where a state of the run's model is {len}",
                        state_path.display(),
                        state.len()
                    );
                }
                if blake3::hash(&state).as_bytes() != &digest {
                    bail!(
                        "{} is not the state its payload commits to",
                        state_path.display()
                    );
                }
            }
            kept.push(Kept {
                epoch_start: record.epoch_start,
                step: record.step,
                revealed,
            });
        }
    }

    let revealed = kept.iter().filter(|payload| payload.revealed).count();
    if revealed != args.revealed {
        bail!(
            "{revealed} revealed states are kept, where the clients say they revealed {}",
            args.revealed
        );
    }
    let mut first_step_of_epoch: BTreeMap<u64, u32> = BTreeMap::new();
    for payload in &kept {
        let first = first_step_of_epoch
            .entry(payload.epoch_start)
            .or_insert(payload.step);
        *first = (*first).min(payload.step);
    }
    let revealed_steps: BTreeSet<u32> = kept
        .iter()
        .filter(|payload| payload.revealed)
        .map(|payload| payload.step)
        .collect();
    let past_first = kept.iter().filter(|payload| {
        payload.revealed && payload.step > first_step_of_epoch[&payload.epoch_start]
    });
    if revealed_steps.len() < 2 || past_first.count() == 0 {
        bail!(
            "states were revealed at steps {revealed_steps:?}; the check needs rounds past the first of an epoch"
        );
    }
    if args.every_trained_step {
        let trained_steps: BTreeSet<u32> = kept.iter().map(|payload| payload.step).collect();
        let unrevealed: Vec<u32> = trained_steps.difference(&revealed_steps).copied().collect();
        if !unrevealed.is_empty() {
            bail!(
                "no state was revealed at steps {unrevealed:?}, though payloads were trained there and every training round seats a verifier"
            );
        }
    }
    println!(
        "{revealed} states revealed to their verifiers, each the one its payload commits to ({len} bytes), at steps {revealed_steps:?}, of {} payloads kept",
        kept.len()
    );
    Ok(())
}
