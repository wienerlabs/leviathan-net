//! The error-feedback measurement replays what it simulates exactly on the
//! same hardware, resets and resyncs where it says it does, and records the
//! rounds a run trains (wienerlabs/leviathan-net#48). That it trains a run the
//! way real clients do is checked against real clients by
//! `scripts/error-feedback-fidelity.sh`.
//!
//! Under the genesis rehearsal (`config/leviathan/genesis-rehearsal.toml`):
//! its checkpoint, 1-bit DisTrO and global batch of 8. Its Cosine schedule
//! warms up from zero over 250 steps, so its first steps barely move the
//! model; the simulations train the same run at its base rate from the
//! start, where every step moves it.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use leviathan_verifier::build_replay_trainer;
use leviathan_verifier::checkpoint_repo_files;
use leviathan_verifier::error_feedback::check_cohort;
use leviathan_verifier::error_feedback::error_feedback_bytes;
use leviathan_verifier::error_feedback::read_rounds;
use leviathan_verifier::error_feedback::reveal_bytes;
use leviathan_verifier::error_feedback::schedule;
use leviathan_verifier::error_feedback::simulate;
use leviathan_verifier::error_feedback::Round;
use leviathan_verifier::error_feedback::ShadowStart;
use leviathan_verifier::error_feedback::Simulation;
use leviathan_verifier::error_feedback::StepRecord;
use leviathan_verifier::published_checkpoint;
use leviathan_verifier::read_run_global_batch;
use leviathan_verifier::read_run_model;
use leviathan_verifier::read_run_verification_percent;
use leviathan_verifier::write_token_dataset;
use leviathan_verifier::ReplayTrainerConfig;
use psyche_coordinator::model::LLM;
use psyche_coordinator::CoordinatorProgress;
use psyche_core::ConstantLR;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_core::Shuffle;
use psyche_core::TokenSize;
use psyche_data_provider::LocalDataProvider;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use tch::Device;
use tch::Kind;
use tch::Tensor;
use tokio_util::sync::CancellationToken;

fn run_config() -> PathBuf {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../config/leviathan/genesis-rehearsal.toml"
    ))
    .to_path_buf()
}

fn genesis() -> LLM {
    read_run_model(&run_config()).unwrap()
}

/// The genesis rehearsal at its Cosine schedule's base rate from the first
/// step, where every step moves the model.
fn genesis_at_base_rate() -> LLM {
    let mut llm = genesis();
    let LearningRateSchedule::Cosine(cosine) = llm.lr_schedule else {
        panic!("the genesis rehearsal trains on a Cosine schedule");
    };
    // Where the warmup ends, the schedule is at its base rate.
    let base_lr = llm.lr_schedule.get_lr(cosine.get_warmup_steps());
    llm.lr_schedule = LearningRateSchedule::Constant(ConstantLR::new(base_lr, 0, 0.0));
    llm
}

fn trainer(llm: &LLM, kind: Kind) -> ReplayTrainerConfig {
    let files = checkpoint_repo_files(&published_checkpoint(llm).unwrap()).unwrap();
    ReplayTrainerConfig::for_run(llm, files, Device::Cpu, kind)
}

fn first_step() -> u32 {
    CoordinatorProgress::default().step
}

/// A directory of one test's own, removed when the test is done with it.
struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "leviathan-error-feedback-{}-{:?}-{name}",
            std::process::id(),
            std::thread::current().id()
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("could not remove {}: {error}", self.0.display());
        }
    }
}

/// Copies every file in `from` into `to`, and returns the copies.
fn copy_files(from: &Path, to: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(from)
        .unwrap()
        .map(|entry| {
            let file = entry.unwrap().path();
            let copy = to.join(file.file_name().unwrap());
            std::fs::copy(&file, &copy).unwrap();
            copy
        })
        .collect()
}

/// The rounds of a run that trains `training_rounds` rounds, `epoch_rounds` in
/// each epoch, split between two nodes, from a corpus the model's vocabulary
/// covers.
async fn rounds(llm: &LLM, training_rounds: u32, epoch_rounds: u32) -> Vec<Round> {
    let global_batch = read_run_global_batch(&run_config()).unwrap();
    let slots = schedule(first_step(), training_rounds, epoch_rounds);
    let dir = TestDir::new("corpus");
    let sequence_length = llm.max_seq_len as usize;
    write_token_dataset(
        &dir.0.join("corpus.ds"),
        30,
        sequence_length,
        global_batch * slots.len() + 1,
        48,
    )
    .unwrap();
    let mut provider = LocalDataProvider::new_from_directory(
        &dir.0,
        TokenSize::TwoBytes,
        sequence_length,
        Shuffle::DontShuffle,
    )
    .unwrap();
    read_rounds(&mut provider, &slots, first_step(), global_batch, 2)
        .await
        .unwrap()
}

fn run(simulation: &Simulation, rounds: Vec<Round>) -> Vec<StepRecord> {
    simulate(simulation, rounds, &mut |_| {}).unwrap()
}

fn simulation(nodes: ReplayTrainerConfig, shadow: Option<ReplayTrainerConfig>) -> Simulation {
    Simulation {
        nodes,
        node_count: 2,
        shadow,
        reset_every: 0,
        shadow_start: ShadowStart::Own,
    }
}

#[test]
fn the_run_trains_a_fixed_global_batch_of_eight() {
    assert_eq!(read_run_global_batch(&run_config()).unwrap(), 8);
}

#[test]
fn the_measured_runs_seat_no_verifier_among_their_nodes() {
    let config = |name: &str| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../config")
            .join(name)
    };
    for (name, percent) in [
        ("leviathan/genesis-rehearsal.toml", 10),
        ("leviathan/fineweb-rehearsal.toml", 10),
        // The fidelity check's server state: a centralized run, with its id.
        ("error-feedback-fidelity/state.toml", 0),
    ] {
        let path = config(name);
        assert_eq!(
            read_run_verification_percent(&path).unwrap(),
            percent,
            "{name}"
        );
        let llm = read_run_model(&path).unwrap();
        check_cohort(2, percent, llm.cold_start_warmup_steps).unwrap();
    }
    // At ten nodes the rehearsals seat a verifier every round.
    assert!(check_cohort(10, 10, 0).is_err());
}

#[tokio::test]
async fn only_training_rounds_are_recorded_and_epochs_close_on_their_own() {
    let llm = genesis_at_base_rate();
    // Three training rounds an epoch: steps 1-3 train, 4-5 close the epoch,
    // 6-7 train in the next.
    let data = rounds(&llm, 5, 3).await;
    assert_eq!(data.len(), 7);
    let mut seen = vec![];
    let records = simulate(
        &simulation(trainer(&llm, Kind::Float), None),
        data,
        &mut |observed| {
            assert_eq!(observed.shares.len(), 2);
            assert_eq!(observed.sent.len(), 2);
            seen.push(observed.record.step);
        },
    )
    .unwrap();
    let steps: Vec<u32> = records.iter().map(|r| r.step).collect();
    assert_eq!(steps, vec![1, 2, 3, 6, 7]);
    assert_eq!(seen, steps);
    assert!(records.iter().all(|r| r.loss.is_finite() && r.loss > 0.0));
}

#[tokio::test]
async fn a_verifier_on_the_nodes_own_hardware_replays_every_round_exactly() {
    // At the base rate every step moves the model, so a shadow that applied
    // anything other than what the nodes applied would land elsewhere.
    let llm = genesis_at_base_rate();
    let data = rounds(&llm, 6, 3).await;
    for reset_every in [0, 2] {
        let mut simulation =
            simulation(trainer(&llm, Kind::Float), Some(trainer(&llm, Kind::Float)));
        simulation.reset_every = reset_every;
        let records = run(&simulation, data.clone());
        assert_eq!(records.len(), 6);
        for record in &records {
            let drift = record.shadow.expect("a shadow ran");
            assert_eq!(drift.distance, 0.0, "reset_every {reset_every}: {record:?}");
            assert_eq!(drift.churn, 0.0, "reset_every {reset_every}: {record:?}");
            assert_eq!(drift.flips, 0.0, "reset_every {reset_every}: {record:?}");
        }
        let since: Vec<u32> = records.iter().map(|r| r.since_reset).collect();
        match reset_every {
            0 => assert_eq!(since, vec![1, 2, 3, 4, 5, 6]),
            _ => assert_eq!(since, vec![1, 2, 1, 2, 1, 2]),
        }
    }
}

#[tokio::test]
async fn a_reset_changes_the_run_from_its_round_on_and_not_before() {
    let llm = genesis_at_base_rate();
    let data = rounds(&llm, 4, 0).await;
    let kept = run(&simulation(trainer(&llm, Kind::Float), None), data.clone());
    let mut resetting = simulation(trainer(&llm, Kind::Float), None);
    resetting.reset_every = 2;
    let reset = run(&resetting, data);
    assert_eq!(
        reset.iter().map(|r| r.since_reset).collect::<Vec<_>>(),
        vec![1, 2, 1, 2]
    );
    // Until the reset of the third round the runs are the same run.
    assert_eq!(kept[0].loss, reset[0].loss);
    assert_eq!(kept[1].loss, reset[1].loss);
    // From it on, the nodes train without the buffer they had built.
    assert_ne!(kept[2].loss, reset[2].loss);
    assert_ne!(kept[3].loss, reset[3].loss);
}

#[tokio::test]
async fn a_verifier_in_another_dtype_drifts_from_what_the_node_sent() {
    let llm = genesis_at_base_rate();
    let data = rounds(&llm, 8, 0).await;
    let records = run(
        &simulation(
            trainer(&llm, Kind::Float),
            Some(trainer(&llm, Kind::BFloat16)),
        ),
        data,
    );
    let drifts: Vec<_> = records.iter().filter_map(|r| r.shadow).collect();
    assert!(
        drifts.iter().any(|drift| drift.distance > 0.0),
        "a bf16 replay of an fp32 node matched it exactly: {records:?}"
    );
    // Which of the two the drift shows up as depends on the platform: a
    // coordinate the replay picked instead of node 0's, or a sign it got
    // wrong. Over 1-bit payloads of equal top-k counts the distance is made of
    // exactly those: distance² = 2·churn + 4·(1 − churn)·flips, since flips are
    // a share of the coordinates both sent (docs/ERROR_FEEDBACK.md).
    for drift in &drifts {
        let (d, c, f) = (
            drift.distance as f64,
            drift.churn as f64,
            drift.flips as f64,
        );
        assert!(
            (d * d - (2.0 * c + 4.0 * (1.0 - c) * f)).abs() < 1e-5,
            "{drift:?} does not split into churn and flips"
        );
    }
}

#[tokio::test]
async fn a_verifier_resynced_at_a_reset_takes_node_zeros_weights_and_replays_it_exactly() {
    let llm = genesis_at_base_rate();
    let data = rounds(&llm, 6, 3).await;
    let nodes = trainer(&llm, Kind::Float);

    // Other weights than the nodes start from: node 0's at the third round,
    // which hold the first round's update, as the resync itself writes them
    // at that round's reset.
    let written = TestDir::new("written");
    let mut first = simulation(nodes.clone(), Some(nodes.clone()));
    first.reset_every = 2;
    first.shadow_start = ShadowStart::Resync(written.0.join("checkpoint"));
    run(&first, data[..3].to_vec());
    let elsewhere = TestDir::new("elsewhere");
    let elsewhere_files = copy_files(&written.0.join("checkpoint"), &elsewhere.0);

    // A shadow that starts from them replays node 0 from the wrong model
    // until the first reset after the first round. There it takes node 0's
    // weights with a zeroed buffer, as a fresh trainer from a canonical
    // checkpoint would. From then on it replays node 0 exactly, across the
    // rounds that close the epoch and the next one's start, so node 0's reset
    // left it where such a fresh trainer starts, and the checkpoint carried
    // its weights bit for bit.
    let resync = TestDir::new("resync");
    let mut resynced = simulation(
        nodes.clone(),
        Some(ReplayTrainerConfig {
            repo_files: elsewhere_files,
            ..nodes
        }),
    );
    resynced.reset_every = 2;
    resynced.shadow_start = ShadowStart::Resync(resync.0.join("checkpoint"));
    let records = run(&resynced, data);
    assert_eq!(
        records.iter().map(|r| r.since_reset).collect::<Vec<_>>(),
        vec![1, 2, 1, 2, 1, 2]
    );
    let drifts: Vec<_> = records
        .iter()
        .map(|r| r.shadow.expect("a shadow ran"))
        .collect();
    assert!(
        drifts[..2].iter().any(|drift| drift.distance > 0.0),
        "a shadow on other weights replayed node 0 exactly: {records:?}"
    );
    for (record, drift) in records.iter().zip(&drifts).skip(2) {
        assert_eq!(drift.distance, 0.0, "{record:?}");
        assert_eq!(drift.churn, 0.0, "{record:?}");
        assert_eq!(drift.flips, 0.0, "{record:?}");
    }
    // The last reset wrote node 0's weights beside the model's configuration.
    assert!(resync.0.join("checkpoint/model.safetensors").exists());
    assert!(resync.0.join("checkpoint/config.json").exists());
}

#[tokio::test]
async fn a_verifier_given_what_node_zero_reveals_replays_each_round_exactly() {
    let llm = genesis_at_base_rate();
    let data = rounds(&llm, 6, 3).await;
    let nodes = trainer(&llm, Kind::Float);

    // A shadow on other weights than the nodes', with no buffer of its own:
    // node 0's at the third round, as the resync writes them.
    let written = TestDir::new("written");
    let mut first = simulation(nodes.clone(), Some(nodes.clone()));
    first.reset_every = 2;
    first.shadow_start = ShadowStart::Resync(written.0.join("checkpoint"));
    run(&first, data[..3].to_vec());
    let elsewhere = TestDir::new("elsewhere");
    let elsewhere_files = copy_files(&written.0.join("checkpoint"), &elsewhere.0);
    let shadow = ReplayTrainerConfig {
        repo_files: elsewhere_files,
        ..nodes.clone()
    };

    // On its own it drifts from node 0. Given node 0's weights, buffer and
    // previous result at the start of every round, across an epoch's close,
    // it sends what node 0 sent: the state a node reveals is all a round
    // depends on.
    let own = run(
        &simulation(nodes.clone(), Some(shadow.clone())),
        data.clone(),
    );
    assert!(
        own.iter()
            .any(|r| r.shadow.is_some_and(|d| d.distance > 0.0)),
        "a shadow on other weights replayed node 0 exactly: {own:?}"
    );
    let mut revealed = simulation(nodes, Some(shadow));
    revealed.shadow_start = ShadowStart::Reveal;
    let records = run(&revealed, data);
    assert_eq!(records.len(), 6);
    for record in &records {
        let drift = record.shadow.expect("a shadow ran");
        assert_eq!(drift.distance, 0.0, "{record:?}");
        assert_eq!(drift.churn, 0.0, "{record:?}");
        assert_eq!(drift.flips, 0.0, "{record:?}");
    }
}

#[tokio::test]
async fn a_trainer_takes_another_ones_state_across_dtypes_and_refuses_one_that_does_not_fit() {
    let llm = genesis_at_base_rate();
    let data = rounds(&llm, 1, 0).await;
    let share = data[0].shares.as_ref().expect("the round trains")[0].clone();

    // A bf16 node after a round: its buffer holds what it did not send.
    let node = build_replay_trainer(&trainer(&llm, Kind::BFloat16)).unwrap();
    let mut node = node
        .train(
            1,
            Batch {
                id: share.batch,
                data: BatchData::CPU(share.data),
            },
            None,
            false,
            vec![],
            Some(vec![]),
            CancellationToken::new(),
        )
        .unwrap()
        .trainer;
    let weights = node.extract().unwrap();
    let buffer: Vec<Vec<Tensor>> = node
        .error_feedback()
        .unwrap()
        .into_iter()
        .map(|buffer| buffer.expect("the node trains with DisTrO"))
        .collect();
    assert!(buffer[0]
        .iter()
        .any(|tensor| tensor.abs().sum(None).double_value(&[]) > 0.0));

    // An fp32 verifier takes it exactly, each value cast up.
    let mut verifier = build_replay_trainer(&trainer(&llm, Kind::Float)).unwrap();
    verifier.load(&weights, Some(&buffer)).unwrap();
    let taken_weights = verifier.extract().unwrap();
    for (name, tensor) in &weights {
        assert!(
            taken_weights[name].equal(&tensor.to_kind(Kind::Float)),
            "{name}"
        );
    }
    let taken = verifier.error_feedback().unwrap();
    let taken = taken[0].as_ref().expect("the verifier trains with DisTrO");
    for (taken, given) in taken.iter().zip(&buffer[0]) {
        assert_eq!(taken.kind(), Kind::Float);
        assert!(taken.equal(&given.to_kind(Kind::Float)));
    }

    // A state that does not fit is refused, and leaves the verifier as it was.
    let fewer: HashMap<String, Tensor> = weights
        .iter()
        .skip(1)
        .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
        .collect();
    assert!(verifier.load(&fewer, Some(&buffer)).is_err());
    let zeros = || -> Vec<Tensor> { buffer[0].iter().map(Tensor::zeros_like).collect() };
    let doubled = vec![zeros(), zeros()];
    assert!(verifier.load(&weights, Some(&doubled)).is_err());
    let mut reshaped = zeros();
    let matrix = reshaped
        .iter()
        .rposition(|tensor| tensor.dim() >= 2)
        .expect("the model has a matrix");
    reshaped[matrix] = reshaped[matrix].flatten(0, -1);
    let reshaped = vec![reshaped];
    let renamed: HashMap<String, Tensor> = weights
        .iter()
        .map(|(name, tensor)| (format!("{name}.renamed"), tensor.zeros_like()))
        .collect();
    assert!(verifier.load(&weights, Some(&reshaped)).is_err());
    assert!(verifier.load(&renamed, Some(&[zeros()])).is_err());
    let after_weights = verifier.extract().unwrap();
    for (name, tensor) in &taken_weights {
        assert!(after_weights[name].equal(tensor), "{name}");
    }
    let after = verifier.error_feedback().unwrap();
    let after = after[0].as_ref().expect("the verifier trains with DisTrO");
    for (after, taken) in after.iter().zip(taken) {
        assert!(after.equal(taken));
    }
}

#[test]
fn the_buffer_holds_a_value_for_every_parameter_in_the_training_dtype() {
    let llm = genesis();
    let fp32 = error_feedback_bytes(&trainer(&llm, Kind::Float)).unwrap();
    let bf16 = error_feedback_bytes(&trainer(&llm, Kind::BFloat16)).unwrap();
    assert!(fp32 > 0);
    assert_eq!(fp32 % 4, 0);
    assert_eq!(fp32, 2 * bf16);

    // A bf16 node that keeps its buffer in fp32 holds the fp32 size, and a
    // reveal of it sends bf16 weights beside it (wienerlabs/leviathan-net#90).
    let kept = ReplayTrainerConfig {
        error_feedback_kind: Some(Kind::Float),
        ..trainer(&llm, Kind::BFloat16)
    };
    let (weights, buffer) = reveal_bytes(&kept).unwrap();
    assert_eq!(buffer, fp32);
    assert_eq!(weights, bf16);
}

/// A bf16 trainer told to keep its buffer in fp32 starts it from zeros in
/// fp32, beside bf16 weights. One whose optimizer keeps no error feedback
/// refuses (wienerlabs/leviathan-net#90).
#[test]
fn a_trainer_keeps_its_buffer_in_the_dtype_it_is_given() {
    let llm = genesis();
    let kept = ReplayTrainerConfig {
        error_feedback_kind: Some(Kind::Float),
        ..trainer(&llm, Kind::BFloat16)
    };
    let mut node = build_replay_trainer(&kept).unwrap();
    let buffers = node.error_feedback().unwrap();
    let buffer = buffers[0].as_ref().expect("the run trains with DisTrO");
    assert!(!buffer.is_empty());
    for tensor in buffer {
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(tensor.abs().sum(None).double_value(&[]), 0.0);
    }
    let weights = node.extract().unwrap();
    assert!(!weights.is_empty());
    assert!(weights
        .values()
        .all(|tensor| tensor.kind() == Kind::BFloat16));

    let adamw = ReplayTrainerConfig {
        optimizer: OptimizerDefinition::AdamW {
            betas: [0.9, 0.95],
            weight_decay: 0.0,
            eps: 1e-8,
            clip_grad_norm: None,
        },
        ..kept
    };
    let error = match build_replay_trainer(&adamw) {
        Ok(_) => panic!("a trainer without DisTrO keeps no error feedback to move"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("keeps no error feedback"), "{error}");
}
