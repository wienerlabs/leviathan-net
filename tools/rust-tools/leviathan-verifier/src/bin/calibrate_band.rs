//! Measures the honest drift of a one-step DisTrO delta across device / precision
//! classes and calibrates the tolerance band from it (issue #7).
//!
//! A verifier convicts a target when its honest replay differs from the
//! submission by more than the tolerance band. An honest node is not a cheater,
//! yet its delta still differs from a verifier's recomputation because they ran
//! on different hardware and, above all, different precision: the network trains
//! in bf16 while the reference replay is fp32. That gap is the band's floor -
//! set the band below the honest drift and honest nodes get convicted; set it
//! far above and a cheater's undetectable budget grows. This harness measures
//! the drift on whatever hardware it runs on and reports `calibrate_band` of it.
//!
//! It computes the same one-step delta twice - once in a reference config
//! (default fp32 on CPU) and once in a target config (default bf16 on the node's
//! accelerator) - for several distinct batches, each from the same fresh
//! checkpoint, then reports the drift distribution and the calibrated band. Run
//! it once per hardware class and collect the JSON.

use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::read_run_model;
use leviathan_verifier::replay_data_layout;
use leviathan_verifier::write_token_dataset;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::ReplayWork;
use leviathan_verifier::TrainerReplayEngine;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_coordinator::model::LLM;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::BatchDataCPU;
use psyche_verifier::calibrate_band;
use psyche_verifier::relative_l2_distance;
use psyche_verifier::DEFAULT_BAND;
use psyche_verifier::DEFAULT_SAFETY_FACTOR;

#[derive(Parser, Debug)]
#[command(
    name = "calibrate-band",
    about = "Measure honest replay drift across device/precision classes and calibrate the tolerance band (issue #7)"
)]
struct Args {
    /// The run configuration whose payloads are calibrated for
    /// (`config/leviathan/*.toml`). Its checkpoint, learning-rate schedule,
    /// optimizer and sequence length are what a node trains with and what the
    /// verifier replays with, 1-bit quantization included.
    #[arg(long)]
    run_config: PathBuf,
    /// A tokenized dataset directory. If omitted, a deterministic one is built.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// How many distinct one-step deltas to measure.
    #[arg(long, default_value_t = 16)]
    samples: usize,
    /// The reference (canonical) config the verifier is assumed to replay in.
    #[arg(long, default_value = "cpu")]
    reference_device: String,
    #[arg(long, default_value = "fp32")]
    reference_dtype: String,
    /// The target config a node actually trains in.
    #[arg(long, default_value = "auto")]
    target_device: String,
    #[arg(long, default_value = "bf16")]
    target_dtype: String,
    /// Vocabulary of the built-in dataset; keep it within the model's embedding table.
    #[arg(long, default_value_t = 30)]
    vocab_size: u16,
    #[arg(long, default_value_t = DEFAULT_SAFETY_FACTOR)]
    safety_factor: f32,
    /// A label for the target hardware class, echoed into the report (e.g. "rtx3090").
    #[arg(long, default_value = "unknown")]
    class: String,
    #[arg(long)]
    json_out: Option<PathBuf>,
}

fn parse_device(s: &str) -> Result<tch::Device> {
    match s.trim().to_lowercase().as_str() {
        "cpu" => Ok(tch::Device::Cpu),
        "mps" => Ok(tch::Device::Mps),
        "cuda" => Ok(tch::Device::Cuda(0)),
        "auto" => Ok(tch::Device::cuda_if_available()),
        other => other
            .strip_prefix("cuda:")
            .and_then(|n| n.parse::<usize>().ok())
            .map(tch::Device::Cuda)
            .ok_or_else(|| anyhow!("unknown device '{other}': expected cpu, mps, cuda, cuda:N, or auto")),
    }
}

fn parse_dtype(s: &str) -> Result<tch::Kind> {
    match s.trim().to_lowercase().as_str() {
        "fp32" | "f32" | "float" => Ok(tch::Kind::Float),
        "bf16" | "bfloat16" => Ok(tch::Kind::BFloat16),
        "fp16" | "f16" | "half" => Ok(tch::Kind::Half),
        other => Err(anyhow!("unknown dtype '{other}': expected fp32, bf16, or fp16")),
    }
}

fn device_label(d: tch::Device) -> String {
    match d {
        tch::Device::Cpu => "cpu".to_string(),
        tch::Device::Cuda(i) => format!("cuda:{i}"),
        other => format!("{other:?}").to_lowercase(),
    }
}

/// Compute one honest one-step delta for `batch` in the given device+dtype,
/// starting from a fresh checkpoint (a fresh trainer). Returns the dense
/// decompressed pseudo-gradient.
fn compute_delta(
    repo_files: &[PathBuf],
    device: tch::Device,
    kind: tch::Kind,
    step: u32,
    batch_id: BatchId,
    data: Vec<BatchDataCPU>,
    llm: &LLM,
) -> Result<Vec<f32>> {
    let config = ReplayTrainerConfig::for_run(llm, repo_files.to_vec(), device, kind);
    let trainer = build_replay_trainer(&config)
        .with_context(|| format!("building trainer on {} {kind:?}", device_label(device)))?;
    let assignment = ReplayAssignment {
        // A sample is not from a run, so it has no epoch to name.
        epoch_start: 0,
        step,
        // Each sample is a fresh trainer's first step, as in a run started
        // directly, whose first epoch has no cold-start warmup bounds.
        warmup_lr_between: None,
        batch_id,
        data,
    };
    let work = assignment.work();
    let engine =
        TrainerReplayEngine::new(trainer, vec![assignment], device, config.quantize_1bit());
    engine
        .replay_work(&work)
        .map_err(|e| anyhow!("replay on {}: {e}", device_label(device)))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let reference_device = parse_device(&args.reference_device)?;
    let reference_kind = parse_dtype(&args.reference_dtype)?;
    let target_device = parse_device(&args.target_device)?;
    let target_kind = parse_dtype(&args.target_dtype)?;

    if args.samples == 0 {
        return Err(anyhow!("--samples must be at least 1"));
    }
    eprintln!(
        "[calibrate] reference {} {reference_kind:?}  vs  target {} {target_kind:?}  ({} samples)",
        device_label(reference_device),
        device_label(target_device),
        args.samples,
    );

    let llm = read_run_model(&args.run_config)?;
    let repo_files = checkpoint_repo_files(&published_checkpoint(&llm)?)?;
    let sequence_length = llm.max_seq_len as usize;

    // Dataset: use the given one, or build a deterministic corpus with enough
    // sequences for `samples` distinct one-step batches.
    let data_dir = match &args.data_dir {
        Some(dir) => dir.clone(),
        None => {
            let tmp =
                std::env::temp_dir().join(format!("leviathan-calibrate-{}", std::process::id()));
            std::fs::create_dir_all(&tmp)?;
            write_token_dataset(
                &tmp.join("nano.ds"),
                args.vocab_size,
                sequence_length,
                args.samples + 8,
                1234,
            )?;
            tmp
        }
    };

    // A given dataset is read as the run lays its data out; the built-in one
    // is written as two-byte tokens in order.
    let (token_size, shuffle) = match &args.data_dir {
        Some(_) => replay_data_layout(&llm)?,
        None => (TokenSize::TwoBytes, Shuffle::DontShuffle),
    };
    // Load one distinct batch per sample up front (async), then compute deltas.
    let mut provider =
        LocalDataProvider::new_from_directory(&data_dir, token_size, sequence_length, shuffle)?;
    let mut batches = Vec::with_capacity(args.samples);
    for i in 0..args.samples {
        let batch_id = BatchId(ClosedInterval::new(i as u64, i as u64));
        let data: Vec<BatchDataCPU> = provider
            .get_samples(batch_id)
            .await?
            .into_iter()
            .map(|x| BatchDataCPU {
                input_ids: x.input_ids,
                labels: x.labels,
                position_ids: x.position_ids,
                sequence_lengths: x.sequence_lengths,
            })
            .collect();
        batches.push((batch_id, data));
    }

    let mut drifts = Vec::with_capacity(args.samples);
    for (i, (batch_id, data)) in batches.into_iter().enumerate() {
        let step = (i + 1) as u32;
        let reference = compute_delta(
            &repo_files,
            reference_device,
            reference_kind,
            step,
            batch_id,
            data.clone(),
            &llm,
        )?;
        let target = compute_delta(
            &repo_files,
            target_device,
            target_kind,
            step,
            batch_id,
            data,
            &llm,
        )?;
        let drift = relative_l2_distance(&target, &reference)
            .map_err(|e| anyhow!("distance on sample {i}: {e:?}"))?;
        eprintln!("[calibrate] sample {i:>3}: drift {drift:.6}");
        drifts.push(drift);
    }

    let band =
        calibrate_band(&drifts, args.safety_factor).map_err(|e| anyhow!("calibrating: {e:?}"))?;

    let n = drifts.len() as f32;
    let mean = drifts.iter().sum::<f32>() / n;
    let max = drifts.iter().copied().fold(0.0f32, f32::max);
    let min = drifts.iter().copied().fold(f32::INFINITY, f32::min);

    println!();
    println!("class               {}", args.class);
    println!("checkpoint          {}", llm.checkpoint);
    println!("run config          {}", args.run_config.display());
    println!(
        "reference           {} {}",
        device_label(reference_device),
        args.reference_dtype
    );
    println!(
        "target              {} {}",
        device_label(target_device),
        args.target_dtype
    );
    println!("samples             {}", drifts.len());
    println!("drift min/mean/max  {min:.6} / {mean:.6} / {max:.6}");
    println!("safety factor       {}", args.safety_factor);
    println!("calibrated band     {band:.6}");
    println!(
        "vs default band     {DEFAULT_BAND:.6} ({})",
        if band <= DEFAULT_BAND {
            "default is adequate for this class"
        } else {
            "default is TOO LOW for this class"
        }
    );

    if reference_device == target_device && reference_kind == target_kind {
        eprintln!(
            "[calibrate] WARNING: reference and target configs are identical, so the drift is ~0 \
             and there is nothing to calibrate. Vary --target-device or --target-dtype."
        );
    }

    if let Some(path) = &args.json_out {
        let samples_json = drifts
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            "{{\"class\":\"{}\",\"model\":\"{}\",\"reference_device\":\"{}\",\
             \"reference_dtype\":\"{}\",\"target_device\":\"{}\",\"target_dtype\":\"{}\",\
             \"samples\":[{}],\"drift_min\":{},\"drift_mean\":{},\"drift_max\":{},\
             \"safety_factor\":{},\"band\":{}}}\n",
            args.class,
            llm.checkpoint,
            device_label(reference_device),
            args.reference_dtype,
            device_label(target_device),
            args.target_dtype,
            samples_json,
            min,
            mean,
            max,
            args.safety_factor,
            band,
        );
        std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("[calibrate] wrote {}", path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devices_parse() {
        assert!(matches!(parse_device("cpu").unwrap(), tch::Device::Cpu));
        assert!(matches!(parse_device("CPU").unwrap(), tch::Device::Cpu));
        assert!(matches!(parse_device("cuda").unwrap(), tch::Device::Cuda(0)));
        assert!(matches!(parse_device("cuda:2").unwrap(), tch::Device::Cuda(2)));
        assert!(matches!(parse_device("mps").unwrap(), tch::Device::Mps));
        assert!(parse_device("auto").is_ok());
        assert!(parse_device("gpu").is_err());
        assert!(parse_device("cuda:x").is_err());
    }

    #[test]
    fn dtypes_parse() {
        assert!(matches!(parse_dtype("fp32").unwrap(), tch::Kind::Float));
        assert!(matches!(parse_dtype("float").unwrap(), tch::Kind::Float));
        assert!(matches!(parse_dtype("bf16").unwrap(), tch::Kind::BFloat16));
        assert!(matches!(parse_dtype("BFloat16").unwrap(), tch::Kind::BFloat16));
        assert!(matches!(parse_dtype("fp16").unwrap(), tch::Kind::Half));
        assert!(parse_dtype("int8").is_err());
    }

    #[test]
    fn device_labels_are_stable() {
        assert_eq!(device_label(tch::Device::Cpu), "cpu");
        assert_eq!(device_label(tch::Device::Cuda(1)), "cuda:1");
    }
}
