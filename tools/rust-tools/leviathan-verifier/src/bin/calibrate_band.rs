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
use leviathan_verifier::write_token_dataset;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::TrainerReplayEngine;
use psyche_core::BatchId;
use psyche_core::ClosedInterval;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_data_provider::download_model_repo_sync;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::BatchDataCPU;
use psyche_verifier::calibrate_band;
use psyche_verifier::relative_l2_distance;
use psyche_verifier::ReplayEngine;
use psyche_verifier::DEFAULT_BAND;
use psyche_verifier::DEFAULT_SAFETY_FACTOR;

#[derive(Parser, Debug)]
#[command(
    name = "calibrate-band",
    about = "Measure honest replay drift across device/precision classes and calibrate the tolerance band (issue #7)"
)]
struct Args {
    #[arg(long)]
    model: String,
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
    #[arg(long, default_value_t = 64)]
    sequence_length: usize,
    /// Vocabulary of the built-in dataset; keep it within the model's embedding table.
    #[arg(long, default_value_t = 30)]
    vocab_size: u16,
    #[arg(long, default_value_t = 4.0e-4)]
    lr: f64,
    #[arg(long, default_value_t = 0.999)]
    compression_decay: f32,
    #[arg(long, default_value_t = 8)]
    compression_topk: u16,
    #[arg(long, default_value_t = 64)]
    compression_chunk: u16,
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
    args: &Args,
) -> Result<Vec<f32>> {
    let trainer = build_replay_trainer(&ReplayTrainerConfig {
        repo_files: repo_files.to_vec(),
        sequence_length: args.sequence_length,
        micro_batch_size: 1,
        learning_rate: args.lr,
        compression_decay: args.compression_decay,
        compression_topk: args.compression_topk,
        compression_chunk: args.compression_chunk,
        clip_grad_norm: Some(1.0),
        quantize_1bit: false,
        device,
        kind,
    })
    .with_context(|| format!("building trainer on {} {kind:?}", device_label(device)))?;
    let engine = TrainerReplayEngine::new(
        trainer,
        vec![ReplayAssignment {
            target_index: 0,
            step,
            batch_id,
            data,
        }],
        device,
    );
    engine
        .replay(0)
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

    let repo_files = download_model_repo_sync(&args.model, None, None, None, true)
        .context("downloading the model repo")?;

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
                args.sequence_length,
                args.samples + 8,
                1234,
            )?;
            tmp
        }
    };

    // Load one distinct batch per sample up front (async), then compute deltas.
    let mut provider = LocalDataProvider::new_from_directory(
        &data_dir,
        TokenSize::TwoBytes,
        args.sequence_length,
        Shuffle::DontShuffle,
    )?;
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
            &args,
        )?;
        let target = compute_delta(
            &repo_files,
            target_device,
            target_kind,
            step,
            batch_id,
            data,
            &args,
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
    println!("model               {}", args.model);
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
            args.model,
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
