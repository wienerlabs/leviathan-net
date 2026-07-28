use std::sync::Arc;

use leviathan_verifier::audit_with_replay;
use leviathan_verifier::decompress_results;
use leviathan_verifier::parse_assignment_key;
use leviathan_verifier::write_token_dataset;
use leviathan_verifier::ReplayAssignment;
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
use psyche_verifier::audit_round;
use psyche_verifier::AuditOutcome;
use psyche_verifier::Contribution;
use psyche_verifier::DEFAULT_BAND;
use tch::Device;
use tch::Kind;
use tokio_util::sync::CancellationToken;

const MODEL: &str = "pefontana/Nano-Llama";
const SEQUENCE_LENGTH: usize = 64;
const STEP: u32 = 1;

fn distro_optimizer() -> OptimizerDefinition {
    OptimizerDefinition::Distro {
        clip_grad_norm: Some(1.0),
        compression_decay: 0.999,
        compression_topk: 8,
        compression_chunk: 64,
        quantize_1bit: false,
        weight_decay: None,
    }
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

fn engine_with_assignment(trainer: Trainer) -> TrainerReplayEngine {
    TrainerReplayEngine::new(
        trainer,
        vec![ReplayAssignment {
            target_index: 0,
            step: STEP,
            batch_id: BatchId(ClosedInterval::new(0, 0)),
            data: sample_batch(),
        }],
        Device::Cpu,
    )
}

#[test]
fn recomputed_reference_clears_an_honest_contribution() {
    let (_, submitted) = train_once(build_trainer(), sample_batch());
    assert!(
        submitted.iter().any(|v| v.abs() > 0.0),
        "the nano model must produce a delta with actual gradient signal"
    );

    let engine = engine_with_assignment(build_trainer());
    let outcomes = audit_round(
        &engine,
        &[Contribution {
            target_index: 0,
            submitted,
        }],
        DEFAULT_BAND,
    );

    match &outcomes[0] {
        AuditOutcome::Judged(report) => assert!(
            !report.verdict.fraud,
            "an honestly trained delta must clear a recomputed reference, distance was {}",
            report.verdict.distance
        ),
        AuditOutcome::ReplayFailed { error, .. } => panic!("replay failed: {error}"),
        AuditOutcome::Malformed { error, .. } => panic!("malformed: {error}"),
    }
}

#[test]
fn recomputed_reference_catches_a_forged_contribution() {
    let (_, honest) = train_once(build_trainer(), sample_batch());
    let forged: Vec<f32> = honest.iter().map(|v| -5.0 * v).collect();

    let engine = engine_with_assignment(build_trainer());
    let outcomes = audit_round(
        &engine,
        &[Contribution {
            target_index: 0,
            submitted: forged,
        }],
        DEFAULT_BAND,
    );

    match &outcomes[0] {
        AuditOutcome::Judged(report) => {
            assert!(
                report.verdict.fraud,
                "a sign-flipped delta must be caught, distance was {}",
                report.verdict.distance
            );
            assert!(report.proof.is_some(), "a conviction must carry a fraud proof");
        }
        AuditOutcome::ReplayFailed { error, .. } => panic!("replay failed: {error}"),
        AuditOutcome::Malformed { error, .. } => panic!("malformed: {error}"),
    }
}

#[test]
fn a_dump_directory_is_audited_against_a_recomputed_reference() {
    let dir = std::env::temp_dir().join(format!("levreplay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let (_, honest) = train_once(build_trainer(), sample_batch());
    write_dump(&dir, "step1-batchB[0, 0]", &honest);

    let engine = engine_with_assignment(build_trainer());
    let summary =
        audit_with_replay(&dir, &engine, DEFAULT_BAND, Device::Cpu).expect("the audit must run");
    assert_eq!(summary.audited(), 1, "the dump must be judged, not skipped");
    assert_eq!(
        summary.fraud(),
        0,
        "an honest dump must clear a reference this verifier recomputed itself"
    );

    let forged: Vec<f32> = honest.iter().map(|v| -5.0 * v).collect();
    let fraud_dir = dir.join("fraud");
    std::fs::create_dir_all(&fraud_dir).unwrap();
    write_dump(&fraud_dir, "step1-batchB[0, 0]", &forged);

    let engine = engine_with_assignment(build_trainer());
    let summary = audit_with_replay(&fraud_dir, &engine, DEFAULT_BAND, Device::Cpu)
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
    write_dump(&dir, "step1-batchB[0, 0]", &honest);

    let engine = TrainerReplayEngine::new(
        build_trainer(),
        vec![ReplayAssignment {
            target_index: 0,
            step: STEP,
            batch_id,
            data,
        }],
        Device::Cpu,
    );
    let summary =
        audit_with_replay(&dir, &engine, DEFAULT_BAND, Device::Cpu).expect("the audit must run");
    assert_eq!(summary.audited(), 1);
    assert_eq!(
        summary.fraud(),
        0,
        "a contribution trained on the dataset must clear a reference replayed from that same dataset"
    );
}

#[test]
fn assignment_keys_carry_the_step_and_batch_the_target_trained() {
    let (step, batch) = parse_assignment_key("step7-batchB[12, 19]").expect("a real dump name");
    assert_eq!(step, 7);
    assert_eq!(batch, BatchId(ClosedInterval::new(12, 19)));
    assert!(parse_assignment_key("not-a-dump-name").is_none());
}

#[test]
fn an_unassigned_target_is_refused_rather_than_convicted() {
    let trainer = build_trainer();
    let engine = engine_with_assignment(trainer);
    let outcomes = audit_round(
        &engine,
        &[Contribution {
            target_index: 7,
            submitted: vec![0.0; 8],
        }],
        DEFAULT_BAND,
    );

    assert!(
        matches!(&outcomes[0], AuditOutcome::ReplayFailed { .. }),
        "a target this verifier was not assigned must not be judged",
    );
}
