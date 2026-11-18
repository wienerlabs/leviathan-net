//! A round trained from a state its trainer's last round did not leave is
//! convicted by continuity (wienerlabs/leviathan-net#89,
//! `docs/ROUND_AUDIT.md`).

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use leviathan_telemetry::verifier::PassReport;
use leviathan_verifier::write_results_dump;
use leviathan_verifier::PayloadBand;
use leviathan_verifier::ReplayAssignment;
use leviathan_verifier::TrainerReplayEngine;
use psyche_coordinator::model::LLM;
use psyche_coordinator::Coordinator;
use psyche_core::BatchId;
use psyche_modeling::read_round_state;
use psyche_modeling::round_state_file;
use psyche_modeling::Batch;
use psyche_modeling::BatchData;
use psyche_modeling::DistroResult;
use psyche_modeling::TrainOutput;
use psyche_modeling::Trainer;
use psyche_network::applied_results_path;
use psyche_network::previous_round_path;
use psyche_network::revealed_state_path;
use psyche_network::round_training_path;
use psyche_network::AppliedResults;
use psyche_network::PreviousRound;
use psyche_solana_tooling::daemon::audit_pass;
use psyche_solana_tooling::daemon::AuditConfig;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_tick_and_reveal;
use psyche_verifier::SignBand;
use solana_sdk::signer::Signer;
use tch::Tensor;
use tokio_util::sync::CancellationToken;

use super::chain_state::coordinator_state;
use super::memnet_verifier_daemon_replay::distinct_batch_data;
use super::memnet_verifier_daemon_replay::genesis_model;
use super::memnet_verifier_daemon_replay::scratch;
use super::memnet_verifier_daemon_replay::signed_commitment_header;
use super::memnet_verifier_daemon_replay::start_replay_run;
use super::memnet_verifier_daemon_replay::trainer_for;
use super::memnet_verifier_daemon_replay::witness_round;
use super::memnet_verifier_daemon_replay::ReplayRun;
use super::memnet_verifier_daemon_replay::COOLDOWN_TIME;
use super::memnet_verifier_daemon_replay::SLASHING_RATE;
use super::signed_payload::identity;
use super::signed_payload::keep_signed_payload;
use super::signed_payload::payload_header;

const RUN_ID_CONTINUITY: &str = "Leviathan replay continuity";

/// A node's first two rounds, trained as a client trains them, with the
/// states they commit to.
struct Rounds {
    first: Vec<DistroResult>,
    second: Vec<DistroResult>,
    first_state: [u8; 32],
    first_state_file: PathBuf,
    second_state: [u8; 32],
}

/// Trains `trainer` on `batch_id` at `step`, taking `sent_before` out of its
/// buffer first.
fn train(
    llm: &LLM,
    coordinator: &Coordinator,
    trainer: Trainer,
    (step, batch_id): (u32, BatchId),
    sent_before: Vec<Vec<DistroResult>>,
) -> TrainOutput {
    trainer
        .train(
            step,
            Batch {
                id: batch_id,
                data: BatchData::CPU(distinct_batch_data(llm, batch_id)),
            },
            coordinator.get_cold_start_warmup_bounds(),
            false,
            vec![],
            Some(sent_before),
            CancellationToken::new(),
        )
        .expect("the run's model trains a batch")
}

/// A node's first two rounds, keeping their states in `keep_in`. Forging,
/// the node trains its second round from the state its first left with the
/// buffer halved, and as a replay from that state would: what it takes out
/// of its buffer first is read off it, so its second round passes the band
/// when replayed from the state it commits to.
fn rounds(
    llm: &LLM,
    coordinator: &Coordinator,
    first: (u32, BatchId),
    second: (u32, BatchId),
    keep_in: PathBuf,
    forge: bool,
) -> Rounds {
    let (trainer, config) = trainer_for(llm);
    #[allow(irrefutable_let_patterns)]
    let Trainer::Local(local) = trainer
    else {
        panic!("a replay trainer is a local one");
    };
    let trainer: Trainer =
        local.keeping_round_states_in(keep_in.clone()).into();
    let send = |results: Vec<DistroResult>| match config.quantize_1bit() {
        true => Trainer::quantize_results(&results),
        false => results,
    };
    let first_round = train(llm, coordinator, trainer, first, vec![]);
    let first_state = first_round.state_digest.unwrap();
    let sent_first = first_round.distro_results.clone().unwrap();
    let honest_second = train(
        llm,
        coordinator,
        first_round.trainer,
        second,
        vec![sent_first.clone()],
    );
    let second_round = match forge {
        false => honest_second,
        true => {
            let left = round_state_file(
                &keep_in,
                &honest_second.state_digest.unwrap(),
            );
            let state = read_round_state(&fs::read(left).unwrap()).unwrap();
            let weights: HashMap<String, Tensor> =
                state.variables.into_iter().collect();
            let buffers: Vec<Tensor> = state
                .buffers
                .into_iter()
                .map(|(_, tensor)| tensor * 0.5)
                .collect();
            let (mut forger, _) = trainer_for(llm);
            forger.load(&weights, Some(&[buffers])).unwrap();
            let read_off =
                forger.previous_results(&send(sent_first.clone())).unwrap();
            train(llm, coordinator, forger, second, vec![read_off])
        },
    };
    Rounds {
        first: send(sent_first),
        second: send(second_round.distro_results.unwrap()),
        first_state,
        first_state_file: round_state_file(&keep_in, &first_state),
        second_state: second_round.state_digest.unwrap(),
    }
}

/// A round trained from a state its trainer's last round did not leave is
/// convicted (wienerlabs/leviathan-net#89). Two nodes train their batches of
/// the run's first two rounds, keeping their states. One trains its second
/// round from a state with its buffer halved, as consistently as a replay
/// from that state would, and commits to it: replayed from the state it
/// commits to, its second round would pass. The verifier audited both
/// nodes' first rounds, and kept beside each the state its trainer revealed,
/// what it did the round before and what it trained in the round; and its
/// own node kept that the second round applied nothing, as an epoch's first
/// two rounds do not. The daemon, checking continuity, replays each first
/// round from its revealed state: the honest node's leaves the state its
/// second round committed to, and the forger's does not, so its second
/// payload is convicted and the forger slashed.
#[tokio::test]
pub async fn a_round_trained_from_a_state_its_last_round_did_not_leave_is_convicted(
) {
    let cheater = 0usize;
    let honest = 1usize;
    let ReplayRun {
        mut endpoint,
        payer,
        main_authority,
        ticker,
        run,
        coordinator_instance,
        coordinator_account,
        clients,
        p2ps,
    } = start_replay_run(77, RUN_ID_CONTINUITY, &[cheater]).await;

    let first_round =
        coordinator_state(&mut endpoint, &coordinator_account).await;
    let run_id = String::from(&first_round.coordinator.run_id);
    witness_round(
        &mut endpoint,
        &payer,
        &ticker,
        &clients,
        &coordinator_instance,
        &coordinator_account,
    )
    .await;
    let second_round =
        coordinator_state(&mut endpoint, &coordinator_account).await;
    // Through the run's last round, so the cooldown after the audit charges
    // the epoch.
    for _ in 0..3 {
        witness_round(
            &mut endpoint,
            &payer,
            &ticker,
            &clients,
            &coordinator_instance,
            &coordinator_account,
        )
        .await;
    }
    let roster = coordinator_state(&mut endpoint, &coordinator_account).await;
    let epoch_start = second_round.coordinator.epoch_state.start_timestamp;
    assert_eq!(
        roster.coordinator.epoch_state.start_timestamp, epoch_start,
        "the audit is in the epoch the payloads were trained in"
    );

    let llm = genesis_model();
    let submitted_dir = scratch("continuity-submitted");
    let staging = scratch("continuity-staging");
    let mut assignments = vec![];
    let mut second_step = None;
    for node in [cheater, honest] {
        let id = identity(&clients[node], &p2ps[node]);
        let first = payload_header(&first_round.coordinator, &id);
        let second = payload_header(&second_round.coordinator, &id);
        assert_eq!(second.step, first.step + 1);
        second_step = Some(second.step);
        let trained = rounds(
            &llm,
            &first_round.coordinator,
            (first.step, first.batch_id),
            (second.step, second.batch_id),
            scratch(&format!("continuity-states-{node}")),
            node == cheater,
        );
        let mut kept = vec![];
        for (header, payload, state) in [
            (first, trained.first, trained.first_state),
            (second, trained.second, trained.second_state),
        ] {
            let staged =
                staging.join(format!("{node}-{}.vec-postcard", header.step));
            write_results_dump(&staged, &payload).unwrap();
            let header = signed_commitment_header(header, state);
            kept.push(keep_signed_payload(
                &submitted_dir,
                &staged,
                &clients[node],
                &p2ps[node],
                &run_id,
                header,
            ));
            assignments.push(ReplayAssignment {
                epoch_start: header.epoch_start,
                step: header.step,
                warmup_lr_between: roster
                    .coordinator
                    .get_cold_start_warmup_bounds(),
                batch_id: header.batch_id,
                data: distinct_batch_data(&llm, header.batch_id),
            });
        }
        // What the verifier kept beside the first round's payload: the state
        // its trainer revealed, that the round opens the epoch, and that the
        // trainer trained this batch alone in it.
        fs::copy(&trained.first_state_file, revealed_state_path(&kept[0]))
            .unwrap();
        fs::write(previous_round_path(&kept[0]), PreviousRound::Idle.encode())
            .unwrap();
        fs::write(
            round_training_path(&kept[0]),
            PreviousRound::Trained(vec![first.batch_id]).encode(),
        )
        .unwrap();
    }
    // What the verifier's own node applied at the start of the second round:
    // nothing, as in every epoch's first two.
    fs::write(
        applied_results_path(&submitted_dir, epoch_start, second_step.unwrap()),
        AppliedResults::Skipped.encode(),
    )
    .unwrap();

    let (mut trainer, replay_config) = trainer_for(&llm);
    let checkpoint = trainer
        .extract()
        .unwrap()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.copy()))
        .collect();
    let engine = TrainerReplayEngine::new(
        trainer,
        assignments,
        tch::Device::Cpu,
        replay_config.quantize_1bit(),
    )
    .with_checkpoint(checkpoint);
    // The nodes trained in the replay's own class, the CPU in fp32, so a
    // replay is exact.
    let band = PayloadBand::Signs(SignBand {
        churn: 0.0,
        flips: 0.0,
    });
    let config = |continuity| AuditConfig {
        run_id: RUN_ID_CONTINUITY.to_string(),
        submitted_dir: submitted_dir.clone(),
        reference_dir: None,
        band,
        audit_assigned: false,
        dry_run: false,
        verdict_mode: false,
        continuity,
    };

    // Without continuity, both first rounds clear from their revealed
    // states, and neither second round has a state beside it to replay
    // from: the forger is not caught.
    let mut judged = HashSet::new();
    let report = audit_pass(
        &mut endpoint,
        &main_authority,
        &coordinator_account,
        &run,
        &config(false),
        Some(&engine),
        &mut judged,
    )
    .await
    .unwrap();
    assert_eq!(
        report,
        PassReport {
            cleared: 2,
            ..PassReport::default()
        }
    );

    // With it, the forger's second payload is convicted, and the honest
    // node's is not.
    let report = audit_pass(
        &mut endpoint,
        &main_authority,
        &coordinator_account,
        &run,
        &config(true),
        Some(&engine),
        &mut judged,
    )
    .await
    .unwrap();
    assert_eq!(
        report,
        PassReport {
            fraud_found: 1,
            submitted: 1,
            ..PassReport::default()
        }
    );
    assert!(judged.contains(&clients[cheater].pubkey().to_string()));
    assert!(
        !judged.contains(&clients[honest].pubkey().to_string()),
        "the honest node was convicted"
    );

    endpoint
        .forward_clock_unix_timestamp(COOLDOWN_TIME)
        .await
        .unwrap();
    process_coordinator_tick_and_reveal(
        &mut endpoint,
        &payer,
        &ticker,
        &coordinator_instance,
        &coordinator_account,
    )
    .await
    .unwrap();
    let state = coordinator_state(&mut endpoint, &coordinator_account).await;
    let slashed = |node: usize| {
        state
            .clients_state
            .clients
            .iter()
            .find(|c| *c.id.signer() == clients[node].pubkey().to_bytes())
            .unwrap()
            .slashed
    };
    assert_eq!(slashed(cheater), SLASHING_RATE);
    assert_eq!(slashed(honest), 0);
}
