use std::sync::Arc;

use leviathan_verifier::PayloadBand;
use leviathan_verifier::audit_with_replay;
use leviathan_verifier::decompress_results;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::replay_epoch;
use leviathan_verifier::DumpKey;
use leviathan_verifier::write_token_dataset;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::ReplayWork;
use leviathan_verifier::TrainerReplayEngine;
use psyche_core::Barrier;
use psyche_core::BatchId;
use psyche_core::CancellableBarrier;
use psyche_core::ClosedInterval;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_data_provider::download_model_repo_sync;
use psyche_data_provider::LocalDataProvider;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::auto_model_for_causal_lm_from_pretrained;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::BatchDataCPU;
use psyche_modeling::CausalLM;
use psyche_modeling::CompressDCT;
use psyche_modeling::DistroResult;
use psyche_modeling::LocalTrainer;
use psyche_network::distro_results_to_bytes;
use psyche_network::SerializedDistroResult;
use psyche_modeling::ParallelModels;
use psyche_modeling::Trainer;
use psyche_verifier::audit_contribution;
use psyche_verifier::AuditReport;
use psyche_verifier::ReplayError;
use psyche_verifier::DEFAULT_BAND;
use tch::Device;
use tch::Kind;
use tokio_util::sync::CancellationToken;

const MODEL: &str = "pefontana/Nano-Llama";
const SEQUENCE_LENGTH: usize = 64;
const STEP: u32 = 1;
/// When the epoch the suite's dumps were trained in started.
const EPOCH_START: u64 = 1_700_000_000;

fn distro_optimizer() -> OptimizerDefinition {
    OptimizerDefinition::Distro {
        clip_grad_norm: Some(1.0),
        compression_decay: 0.999,
        compression_topk: 8,
        compression_chunk: 64,
        quantize_1bit: false,
        robust_aggregation: false,
        weight_decay: None,
    }
}

/// Whether this harness's optimizer sends 1-bit signs, as the engine has to
/// know to compare a replay with what a node sends.
fn quantize_1bit() -> bool {
    matches!(
        distro_optimizer(),
        OptimizerDefinition::Distro {
            quantize_1bit: true,
            ..
        }
    )
}

fn write_dump(dir: &std::path::Path, key: &str, dense: &[f32]) {
    let rows = 6;
    let x = tch::Tensor::from_slice(dense)
        .reshape([rows, (dense.len() / rows as usize) as i64])
        .to_kind(Kind::Float);
    let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(&x, 8);
    let result = DistroResult {
        sparse_idx,
        sparse_val,
        xshape,
        totalk,
        stats: None,
    };
    let serialized: SerializedDistroResult = (&result).try_into().unwrap();
    let bytes = distro_results_to_bytes(&[serialized]).unwrap();
    std::fs::write(dir.join(format!("result-node0-{key}.vec-postcard")), bytes).unwrap();
}

fn sample_batch() -> Vec<BatchDataCPU> {
    let input_ids: Vec<i32> = (0..SEQUENCE_LENGTH as i32).map(|i| i % 29).collect();
    vec![BatchDataCPU {
        input_ids,
        labels: None,
        position_ids: None,
        sequence_lengths: None,
    }]
}

fn build_trainer() -> Trainer {
    let repo_files = download_model_repo_sync(&MODEL.to_string(), None, None, None, true)
        .expect("Nano-Llama must be available in the local hub cache");
    let model: Box<dyn CausalLM> = auto_model_for_causal_lm_from_pretrained(
        repo_files,
        Some(Kind::Float),
        None,
        Some(Device::Cpu),
        None,
        Some(SEQUENCE_LENGTH),
    )
    .expect("the nano model must load");
    model.prepare_for_training();
    LocalTrainer::new(
        ParallelModels {
            models: vec![model],
            barrier: Arc::new(CancellableBarrier::new(1)) as Arc<dyn Barrier>,
            data_parallel: None,
        },
        LearningRateSchedule::Constant(ConstantLR::new(4.0e-4, 0, 0.0)),
        distro_optimizer(),
        1,
        None,
        false,
    )
    .into()
}

fn train_once(trainer: Trainer, data: Vec<BatchDataCPU>) -> (Trainer, Vec<f32>) {
    let output = trainer
        .train(
            STEP,
            Batch {
                id: BatchId(ClosedInterval::new(0, 0)),
                data: BatchData::CPU(data),
            },
            None,
            true,
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        )
        .expect("training a single batch must succeed");
    let results = output
        .distro_results
        .clone()
        .expect("a DisTrO run must produce compressed results");
    let dense = decompress_results(&results, Device::Cpu).expect("results must decompress");
    (output.trainer, dense)
}

/// The work the suite's dumps are for: batch `batch` of the run's first step.
fn work(batch: u64) -> DumpKey {
    DumpKey {
        epoch_start: EPOCH_START,
        step: STEP,
        batch_id: BatchId(ClosedInterval::new(batch, batch)),
    }
}

/// A batch of other samples than `sample_batch`'s, so its honest result is
/// another.
fn other_batch() -> Vec<BatchDataCPU> {
    let input_ids: Vec<i32> = (0..SEQUENCE_LENGTH as i32).map(|i| (i * 7 + 3) % 29).collect();
    vec![BatchDataCPU {
        input_ids,
        labels: None,
        position_ids: None,
        sequence_lengths: None,
    }]
}

fn assignment(work: DumpKey, data: Vec<BatchDataCPU>) -> ReplayAssignment {
    ReplayAssignment {
        epoch_start: work.epoch_start,
        step: work.step,
        warmup_lr_between: None,
        batch_id: work.batch_id,
        data,
    }
}

fn engine_with_assignment(trainer: Trainer) -> TrainerReplayEngine {
    TrainerReplayEngine::new(
        trainer,
        vec![assignment(work(0), sample_batch())],
        Device::Cpu,
        quantize_1bit(),
    )
}

/// `submitted` judged against `engine`'s replay of `work`.
fn judge(engine: &TrainerReplayEngine, work: DumpKey, submitted: &[f32]) -> AuditReport {
    let recomputed = engine
        .replay_work(&work)
        .unwrap_or_else(|error| panic!("replay failed: {error}"));
    audit_contribution(0, submitted, &recomputed, DEFAULT_BAND)
        .unwrap_or_else(|error| panic!("malformed: {error}"))
}

#[test]
fn recomputed_reference_clears_an_honest_contribution() {
    let (_, submitted) = train_once(build_trainer(), sample_batch());
    assert!(
        submitted.iter().any(|v| v.abs() > 0.0),
        "the nano model must produce a delta with actual gradient signal"
    );

    let engine = engine_with_assignment(build_trainer());
    let report = judge(&engine, work(0), &submitted);
    assert!(
        !report.verdict.fraud,
        "an honestly trained delta must clear a recomputed reference, distance was {}",
        report.verdict.distance
    );
}

#[test]
fn recomputed_reference_catches_a_forged_contribution() {
    let (_, honest) = train_once(build_trainer(), sample_batch());
    let forged: Vec<f32> = honest.iter().map(|v| -5.0 * v).collect();

    let engine = engine_with_assignment(build_trainer());
    let report = judge(&engine, work(0), &forged);
    assert!(
        report.verdict.fraud,
        "a sign-flipped delta must be caught, distance was {}",
        report.verdict.distance
    );
    assert!(report.proof.is_some(), "a conviction must carry a fraud proof");
}

/// An engine holding several pieces of work replays the one it is asked for,
/// by the work's own name, in whatever order it was given them. An honest
/// result clears the replay of its own batch and is far from another's, which
/// is why a payload must never be paired with a replay by anything but the
/// work it names: paired by roster position, an honest node whose seat moved
/// would be judged against someone else's batch and convicted.
#[test]
fn a_replay_is_of_the_work_it_names() {
    let (_, honest) = train_once(build_trainer(), sample_batch());

    let engine = TrainerReplayEngine::new(
        build_trainer(),
        vec![
            assignment(work(1), other_batch()),
            assignment(work(0), sample_batch()),
        ],
        Device::Cpu,
        quantize_1bit(),
    );
    let own = judge(&engine, work(0), &honest);
    assert!(
        !own.verdict.fraud,
        "an honest result was convicted against the replay of its own batch, distance {}",
        own.verdict.distance
    );
    let other = judge(&engine, work(1), &honest);
    assert!(
        other.verdict.fraud,
        "the replay of another batch is not a reference for this one, distance {}",
        other.verdict.distance
    );
}

#[test]
fn a_dump_directory_is_audited_against_a_recomputed_reference() {
    let dir = std::env::temp_dir().join(format!("levreplay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let (_, honest) = train_once(build_trainer(), sample_batch());
    write_dump(&dir, "epoch1700000000-step1-batchB[0, 0]", &honest);

    let engine = engine_with_assignment(build_trainer());
    let summary = audit_with_replay(
        &dir,
        &engine,
        PayloadBand::Distance(DEFAULT_BAND),
        Device::Cpu,
    )
    .expect("the audit must run");
    assert_eq!(summary.audited(), 1, "the dump must be judged, not skipped");
    assert_eq!(
        summary.fraud(),
        0,
        "an honest dump must clear a reference this verifier recomputed itself"
    );

    let forged: Vec<f32> = honest.iter().map(|v| -5.0 * v).collect();
    let fraud_dir = dir.join("fraud");
    std::fs::create_dir_all(&fraud_dir).unwrap();
    write_dump(&fraud_dir, "epoch1700000000-step1-batchB[0, 0]", &forged);

    let engine = engine_with_assignment(build_trainer());
    let summary = audit_with_replay(
        &fraud_dir,
        &engine,
        PayloadBand::Distance(DEFAULT_BAND),
        Device::Cpu,
    )
    .expect("the audit must run");
    assert_eq!(summary.fraud(), 1, "a forged dump must be convicted");
    assert_eq!(summary.proofs().len(), 1, "a conviction must carry a proof");
}

#[tokio::test]
async fn a_verifier_replays_the_exact_batch_its_target_pulled_from_the_dataset() {
    let dir = std::env::temp_dir().join(format!("levdataset-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let data_dir = dir.join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    write_token_dataset(&data_dir.join("nano.ds"), 30, SEQUENCE_LENGTH, 8, 42).unwrap();

    let batch_id = BatchId(ClosedInterval::new(0, 0));
    let mut provider = LocalDataProvider::new_from_directory(
        &data_dir,
        TokenSize::TwoBytes,
        SEQUENCE_LENGTH,
        Shuffle::DontShuffle,
    )
    .expect("a nano-sized dataset must load");
    let data: Vec<BatchDataCPU> = provider
        .get_samples(batch_id)
        .await
        .expect("the batch must be servable")
        .into_iter()
        .map(|x| BatchDataCPU {
            input_ids: x.input_ids,
            labels: x.labels,
            position_ids: x.position_ids,
            sequence_lengths: x.sequence_lengths,
        })
        .collect();
    assert!(!data.is_empty(), "the batch must carry real tokens");

    let (_, honest) = train_once(build_trainer(), data.clone());
    write_dump(&dir, "epoch1700000000-step1-batchB[0, 0]", &honest);

    let engine = TrainerReplayEngine::new(
        build_trainer(),
        vec![ReplayAssignment {
            epoch_start: EPOCH_START,
            step: STEP,
            warmup_lr_between: None,
            batch_id,
            data,
        }],
        Device::Cpu,
        quantize_1bit(),
    );
    let summary = audit_with_replay(
        &dir,
        &engine,
        PayloadBand::Distance(DEFAULT_BAND),
        Device::Cpu,
    )
    .expect("the audit must run");
    assert_eq!(summary.audited(), 1);
    assert_eq!(
        summary.fraud(),
        0,
        "a contribution trained on the dataset must clear a reference replayed from that same dataset"
    );
}

#[test]
fn assignment_keys_carry_the_step_and_batch_the_target_trained() {
    let key =
        parse_assignment_key("epoch1700000000-step7-batchB[12, 19]").expect("a real dump name");
    assert_eq!(key.epoch_start, 1_700_000_000);
    assert_eq!(key.step, 7);
    assert_eq!(key.batch_id, BatchId(ClosedInterval::new(12, 19)));
    // A batch of one sample is named the way `BatchId` displays it, without
    // brackets, and is read back all the same.
    let single = BatchId(ClosedInterval::new(15, 15));
    assert_eq!(
        parse_assignment_key(&format!("epoch1700000090-step3-batch{single}")),
        Some(DumpKey {
            epoch_start: 1_700_000_090,
            step: 3,
            batch_id: single,
        })
    );
    // A name without its epoch is a dump from before the epoch named the
    // round, which cannot be placed (wienerlabs/leviathan-net#67).
    assert!(parse_assignment_key("step7-batchB[12, 19]").is_none());
    assert!(parse_assignment_key("not-a-dump-name").is_none());
}

/// Dumps of the first step from one epoch are replayed without naming it.
/// From two, as after a run restarts from its first step, the epoch has to be
/// named: they were trained from other weights (wienerlabs/leviathan-net#67).
#[test]
fn the_epoch_replayed_is_the_only_one_or_the_one_named() {
    let key = |epoch_start: u64, step: u32| DumpKey {
        epoch_start,
        step,
        batch_id: BatchId(ClosedInterval::new(0, 3)),
    };
    let one = [key(100, 1), key(100, 1), key(100, 2), key(190, 2)];
    assert_eq!(replay_epoch(&one, 1, None).unwrap(), Some(100));
    assert_eq!(replay_epoch(&one, 1, Some(190)).unwrap(), None);
    assert_eq!(replay_epoch(&one, 3, None).unwrap(), None);

    let restarted = [key(100, 1), key(190, 1)];
    assert!(replay_epoch(&restarted, 1, None).is_err());
    assert_eq!(replay_epoch(&restarted, 1, Some(190)).unwrap(), Some(190));
}

/// Work the engine was not given is refused rather than replayed from some
/// other assignment: another batch of the step, or the same batch trained in
/// another epoch.
#[test]
fn unassigned_work_is_refused_rather_than_convicted() {
    let engine = engine_with_assignment(build_trainer());
    let elsewhere = DumpKey {
        epoch_start: EPOCH_START + 90,
        ..work(0)
    };
    for unassigned in [work(7), elsewhere] {
        assert_eq!(
            engine.replay_work(&unassigned),
            Err(ReplayError::WorkNotAssigned(unassigned.to_string())),
            "{unassigned} was not assigned and must not be judged",
        );
    }
}

/// The work's name is the one its dump carries, so a work key read from a
/// dump finds the assignment made for it.
#[test]
fn a_work_key_reads_back_from_its_name() {
    for key in [
        work(0),
        work(15),
        DumpKey {
            batch_id: BatchId(ClosedInterval::new(12, 19)),
            ..work(0)
        },
    ] {
        assert_eq!(parse_assignment_key(&key.to_string()), Some(key));
    }
}
