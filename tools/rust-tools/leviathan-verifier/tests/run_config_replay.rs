//! The replay has to reproduce what the run trains, and read what the run
//! sends (wienerlabs/leviathan-net#47).
//!
//! Everything here runs under a real run configuration, the genesis rehearsal
//! (`config/leviathan/genesis-rehearsal.toml`). That configuration trains with
//! a Cosine warmup and transmits 1-bit signs, as every Leviathan configuration
//! does.

use std::sync::Arc;

use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::cold_start_warmup_bounds;
use leviathan_verifier::decompress_results;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::read_run_model;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayTrainerConfig;
use leviathan_verifier::TrainerReplayEngine;
use psyche_coordinator::model::Model;
use psyche_coordinator::model::LLM;
use psyche_coordinator::Coordinator;
use psyche_coordinator::CoordinatorProgress;
use psyche_core::Barrier;
use psyche_core::BatchId;
use psyche_core::CancellableBarrier;
use psyche_core::ClosedInterval;
use psyche_core::OptimizerDefinition;
use psyche_modeling::auto_model_for_causal_lm_from_pretrained;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::CausalLM;
use psyche_modeling::DistroResult;
use psyche_modeling::LocalTrainer;
use psyche_modeling::ParallelModels;
use psyche_modeling::Trainer;
use psyche_network::SerializedDistroResult;
use psyche_verifier::audit_round;
use psyche_verifier::AuditOutcome;
use psyche_verifier::Contribution;
use psyche_verifier::DEFAULT_BAND;
use tch::Device;
use tch::Kind;
use tokio_util::sync::CancellationToken;

fn genesis() -> LLM {
    read_run_model(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../config/leviathan/genesis-rehearsal.toml"
    )))
    .unwrap()
}

fn repo(llm: &LLM) -> Vec<std::path::PathBuf> {
    checkpoint_repo_files(&published_checkpoint(llm).unwrap()).unwrap()
}

fn first_step() -> u32 {
    CoordinatorProgress::default().step
}

/// The coordinator of a run started directly with `llm`, in its first epoch:
/// where the client reads its warmup bounds and the verifier the step it can
/// replay.
fn first_epoch(llm: &LLM) -> Box<Coordinator> {
    let mut coordinator = Box::new(<Coordinator as bytemuck::Zeroable>::zeroed());
    coordinator.model = Model::LLM(*llm);
    coordinator.progress = CoordinatorProgress::default();
    coordinator.epoch_state.start_step = coordinator.progress.step;
    coordinator
}

fn batch(llm: &LLM) -> Vec<BatchDataCPU> {
    let input_ids: Vec<i32> = (0..llm.max_seq_len as i32).map(|i| i % 29).collect();
    vec![BatchDataCPU {
        input_ids,
        labels: None,
        position_ids: None,
        sequence_lengths: None,
    }]
}

/// A node's trainer, built the way the client builds it from the run's model
/// (`shared/client/src/state/init.rs`).
fn client_trainer(llm: &LLM) -> Trainer {
    let model: Box<dyn CausalLM> = auto_model_for_causal_lm_from_pretrained(
        repo(llm),
        Some(Kind::Float),
        None,
        Some(Device::Cpu),
        None,
        Some(llm.max_seq_len as usize),
    )
    .unwrap();
    model.prepare_for_training();
    LocalTrainer::new(
        ParallelModels {
            models: vec![model],
            barrier: Arc::new(CancellableBarrier::new(1)) as Arc<dyn Barrier>,
            data_parallel: None,
        },
        llm.lr_schedule,
        llm.optimizer,
        1,
        None,
        false,
    )
    .into()
}

/// What an honest node sends for the run's first step: trained, quantized if
/// the run says so, and read back off the wire.
fn honest_payload(llm: &LLM) -> (Vec<DistroResult>, Vec<DistroResult>) {
    let output = client_trainer(llm)
        .train(
            first_step(),
            Batch {
                id: BatchId(ClosedInterval::new(0, 0)),
                data: BatchData::CPU(batch(llm)),
            },
            first_epoch(llm).get_cold_start_warmup_bounds(),
            false,
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        )
        .unwrap();
    let raw = output.distro_results.unwrap();
    let OptimizerDefinition::Distro { quantize_1bit, .. } = llm.optimizer else {
        panic!("the run trains with DisTrO");
    };
    let sent = if quantize_1bit {
        Trainer::quantize_results(&raw)
    } else {
        raw.iter()
            .map(|r| DistroResult {
                stats: None,
                ..r.clone()
            })
            .collect()
    };
    let wire = sent
        .iter()
        .map(|r| {
            let serialized = SerializedDistroResult::try_from(r).unwrap();
            DistroResult::try_from(&serialized).unwrap()
        })
        .collect();
    (raw, wire)
}

/// A 1-bit payload carries the signs of the node's values, and the verifier
/// has to read them as the -1 and +1 that `Distro::apply` applies.
#[test]
fn a_one_bit_payload_decodes_to_the_signs_it_carries() {
    let llm = genesis();
    let (raw, wire) = honest_payload(&llm);
    let values = decompress_results(&raw, Device::Cpu).unwrap();
    let signs = decompress_results(&wire, Device::Cpu).unwrap();
    assert_eq!(values.len(), signs.len());
    let carried: Vec<(f32, f32)> = values
        .iter()
        .zip(&signs)
        .filter(|(value, _)| **value != 0.0)
        .map(|(value, sign)| (*value, *sign))
        .collect();
    assert!(!carried.is_empty());
    for (value, sign) in carried {
        assert_eq!(sign, value.signum(), "value {value} decoded as {sign}");
    }
}

/// The verdict the daemon reaches on `submitted`, replaying the run's first
/// step the way it does now: from the run's own configuration.
fn judged(llm: &LLM, submitted: Vec<f32>) -> psyche_verifier::BandVerdict {
    let config = ReplayTrainerConfig::for_run(llm, repo(llm), Device::Cpu, Kind::Float);
    let coordinator = first_epoch(llm);
    let step = coordinator
        .replayable_step()
        .expect("the run's first step is replayable");
    assert_eq!(step, first_step());
    let engine = TrainerReplayEngine::new(
        build_replay_trainer(&config).unwrap(),
        vec![ReplayAssignment {
            target_index: 0,
            step,
            warmup_lr_between: coordinator.get_cold_start_warmup_bounds(),
            batch_id: BatchId(ClosedInterval::new(0, 0)),
            data: batch(llm),
        }],
        Device::Cpu,
        config.quantize_1bit(),
    );
    let outcomes = audit_round(
        &engine,
        &[Contribution {
            target_index: 0,
            submitted,
        }],
        DEFAULT_BAND,
    );
    match outcomes.into_iter().next().unwrap() {
        AuditOutcome::Judged(report) => report.verdict,
        AuditOutcome::ReplayFailed { error, .. } => panic!("replay failed: {error}"),
        AuditOutcome::Malformed { error, .. } => panic!("malformed: {error}"),
    }
}

/// An honest node's first step clears the band. Replayed with the daemon's
/// old flags (a constant 4e-4, no quantization) the same contribution was
/// 50831 away and convicted.
#[test]
fn an_honest_first_step_under_the_runs_config_replays_within_band() {
    let llm = genesis();
    let (_, wire) = honest_payload(&llm);
    let verdict = judged(&llm, decompress_results(&wire, Device::Cpu).unwrap());
    assert!(
        !verdict.fraud,
        "an honest node was convicted, distance {}",
        verdict.distance
    );
}

/// A node that sends the opposite of what it computed is caught.
#[test]
fn a_forged_first_step_is_caught() {
    let llm = genesis();
    let (_, wire) = honest_payload(&llm);
    let forged: Vec<f32> = decompress_results(&wire, Device::Cpu)
        .unwrap()
        .iter()
        .map(|value| -value)
        .collect();
    let verdict = judged(&llm, forged);
    assert!(
        verdict.fraud,
        "a sign-flipped payload cleared, distance {}",
        verdict.distance
    );
}

/// The offline tools have no coordinator to ask for the warmup bounds of an
/// epoch flagged as a cold start, and work them out the way the coordinator
/// does.
#[test]
fn the_offline_warmup_bounds_are_the_coordinators() {
    for warmup_steps in [0, 3] {
        let mut llm = genesis();
        llm.cold_start_warmup_steps = warmup_steps;
        let mut coordinator = Box::new(<Coordinator as bytemuck::Zeroable>::zeroed());
        coordinator.model = Model::LLM(llm);
        coordinator.epoch_state.cold_start_epoch = true.into();
        coordinator.epoch_state.start_step = first_step();
        assert_eq!(
            cold_start_warmup_bounds(&llm, first_step()),
            coordinator.get_cold_start_warmup_bounds(),
            "{warmup_steps} warmup steps"
        );
    }
}

/// A replay trainer is reused across targets, and weight decay would change
/// its weights under each later replay, so a run that decays is refused.
#[test]
fn a_run_with_weight_decay_is_refused() {
    let mut llm = genesis();
    let OptimizerDefinition::Distro { weight_decay, .. } = &mut llm.optimizer else {
        panic!("the run trains with DisTrO");
    };
    *weight_decay = Some(0.1);
    let config = ReplayTrainerConfig::for_run(&llm, repo(&llm), Device::Cpu, Kind::Float);
    let Err(refused) = build_replay_trainer(&config) else {
        panic!("a decaying run was accepted");
    };
    assert!(
        refused.to_string().contains("decays its weights"),
        "{refused:#}"
    );
}
