use crate::{
    fetch_data::{BatchIdSet, DataFetcher, TrainingDataForStep},
    state::types::{DeserializeError, PayloadState},
};

use futures::{StreamExt, future::try_join_all, stream::FuturesUnordered};
use psyche_coordinator::{
    BLOOM_FALSE_RATE, Commitment, CommitteeSelection, Coordinator, CoordinatorError, HealthChecks,
    assign_data_for_state, get_batch_ids_for_node, get_batch_ids_for_round, model,
};
use psyche_core::{BatchId, Bloom, IntegrationTestLogMarker, NodeIdentity, OptimizerDefinition};
use psyche_event_sourcing::event;
use psyche_modeling::{
    ApplyDistroResultError, Batch, BatchData, DistroResult, TrainOutput, Trainer,
    TrainerThreadCommunicationError,
};
use psyche_network::{
    CommitmentRecord, EquivocationRecord, Hash, SerializeDistroResultError, SerializedDistroResult,
    TransmittableDistroResult, commitment_record_path, distro_results_to_bytes,
    equivocation_record_path, payload_dump_name,
};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio::{io::AsyncWriteExt, sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, error, info, trace, trace_span, warn};

use super::{
    evals::{MaybeRunningEvals, ModelTaskRunner},
    round_state::RoundState,
    types::DistroBroadcastAndPayload,
};

#[derive(Debug)]
pub struct FinishedTrainers {
    pub evals_or_trainers: MaybeRunningEvals,
    pub round_losses: Vec<f32>,
    pub optim_stats: HashMap<String, f64>,
    pub round_duration: Duration,
}

#[derive(Error, Debug)]
pub enum TrainError {
    #[error("No trainers available when entering training step.")]
    NoTrainers,

    #[error("No training round in-progress")]
    NoActiveRound,

    #[error("No committee info for this round ")]
    NoCommitteeInfo,

    #[error("We're not in this round")]
    NotInThisRound,

    #[error("Apply thread crashed")]
    ApplyCrashed,

    #[error("Failed to apply distro results: {0}")]
    Apply(#[from] ApplyError),

    #[error("Training thread crashed")]
    TrainCrashed,

    #[error("Transmit thread crashed")]
    TransmitCrashed,

    #[error("Failed to train on batch: {0}")]
    TrainOnBatch(#[from] TrainerThreadCommunicationError),

    #[error("Failed to serialize distro result: {0}")]
    SerializeDistroResult(SerializeDistroResultError),

    #[error("Failed to hash distro result for its commitment: {0}")]
    HashDistroResult(postcard::Error),

    #[error("Failed to send distro result, channel must be closed")]
    SendDistroResult,

    #[error("Failed to send health checks, channel must be closed")]
    SendHealthChecks,

    #[error("Health check thread crashed")]
    HealthCheckCrashed,

    #[error("Coordinator error: {0}")]
    CoordinatorError(CoordinatorError),
}

pub struct TrainingStepMetadata {
    pub identity: NodeIdentity,
    pub data_fetcher: DataFetcher,
    pub tx_health_check: mpsc::UnboundedSender<HealthChecks>,
    pub tx_distro_result: mpsc::UnboundedSender<DistroBroadcastAndPayload>,

    pub write_gradients_dir: Option<PathBuf>,

    pub model_task_runner: ModelTaskRunner,
}

#[derive(Debug)]
pub struct TrainingStep {
    sending_health_checks: Option<JoinHandle<Result<(), TrainError>>>,
    cancel_training: CancellationToken,

    applying_and_training: JoinHandle<Result<FinishedTrainers, TrainError>>,
    finished: Arc<AtomicBool>,
}

impl TrainingStep {
    pub async fn finish(self) -> Result<FinishedTrainers, TrainError> {
        self.cancel_training.cancel();
        if let Some(hc) = self.sending_health_checks {
            hc.await.map_err(|_| TrainError::HealthCheckCrashed)??;
        }

        let finished = self.finished.clone();

        let trainers: FinishedTrainers = self
            .applying_and_training
            .await
            .map_err(|_| TrainError::TrainCrashed)??;

        if !finished.load(Ordering::SeqCst) {
            warn!("Training didn't finish when the Training round ended, we are likely to desync.");
        }

        Ok(trainers)
    }

    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
}

impl TrainingStepMetadata {
    pub fn start(
        &mut self,
        client_index: u64,
        state: &Coordinator,
        trainers: Vec<Trainer>,
        previous_round: &mut RoundState,
        current_round: &mut RoundState,
    ) -> Result<TrainingStep, TrainError> {
        if trainers.is_empty() {
            return Err(TrainError::NoTrainers);
        }

        let applying = self.apply_results(trainers, state, previous_round, current_round)?;

        let sending_health_checks =
            start_sending_health_checks(current_round, state, self.tx_health_check.clone())?;

        debug!("Transitioning to train step {}", state.progress.step);

        let cancel_training = CancellationToken::new();
        let round_start = Instant::now();

        let round = state.current_round().ok_or(TrainError::NoActiveRound)?;

        *previous_round = std::mem::take(current_round);

        let committee_selection = CommitteeSelection::new(
            round.tie_breaker_tasks as usize,
            state.config.witness_nodes as usize,
            state.config.verification_percent,
            state.epoch_state.clients.len(),
            round.random_seed,
        )
        .map_err(TrainError::CoordinatorError)?;

        let have_training = !state.epoch_state.last_step_set();
        let (data_assignments, num_all_batch_ids, batch_ids_not_yet_trained_on) = if have_training {
            let data_assignments = assign_data_for_state(state, &committee_selection);
            let all_batch_ids = get_batch_ids_for_round(
                state.current_round().unwrap(),
                state,
                committee_selection.get_num_trainer_nodes(),
            );
            let num_all_batch_ids = all_batch_ids.len();
            let batch_ids_not_yet_trained_on: BatchIdSet = all_batch_ids.into_iter().collect();
            (
                data_assignments,
                num_all_batch_ids,
                Arc::new(Mutex::new(Some(batch_ids_not_yet_trained_on))),
            )
        } else {
            (BTreeMap::new(), 0, Arc::new(Mutex::new(None)))
        };

        let committee_proof = committee_selection.get_committee(client_index);
        let witness_proof = committee_selection.get_witness(client_index);

        let blooms = {
            let participant_bloom =
                Bloom::random(state.epoch_state.clients.len(), BLOOM_FALSE_RATE);
            let broadcast_bloom = Bloom::random(num_all_batch_ids, BLOOM_FALSE_RATE);
            trace!(
                "Participant bloom size: {} bits, {} keys",
                participant_bloom.bits.0.len(),
                participant_bloom.keys.len()
            );
            trace!(
                "Broadcast bloom size: {} bits, {} keys",
                broadcast_bloom.bits.0.len(),
                broadcast_bloom.keys.len()
            );
            Arc::new(Mutex::new(Some((participant_bloom, broadcast_bloom))))
        };

        *current_round = RoundState {
            height: round.height,
            step: state.progress.step,
            epoch_start: state.epoch_state.start_timestamp,
            sent_witness: false,
            sent_finished: false,
            downloads: Default::default(),
            results: Default::default(),
            broadcasts: Default::default(),
            clients_finished: Default::default(),
            data_assignments: data_assignments.clone(),
            blooms,
            committee_info: Some((committee_proof, witness_proof, committee_selection)),
            batch_ids_not_yet_trained_on,
            self_distro_results: vec![],
        };

        let warmup_lr_between = state.get_cold_start_warmup_bounds();
        let zero_optim = warmup_lr_between.is_some_and(|_| round.height == 0);
        let epoch = state.progress.epoch;

        event!(train::WitnessElected {
            step: state.progress.step as u64,
            round: round.height as u64,
            epoch: epoch as u64,
            index: client_index,
            committee_position: committee_proof.position,
            is_witness: witness_proof.witness.into(),
        });

        let assigned_batches = get_batch_ids_for_node(&data_assignments, &self.identity);
        info!(
            integration_test_log_marker = %IntegrationTestLogMarker::WitnessElected,
            step = state.progress.step,
            round = round.height,
            epoch = epoch,
            index = client_index,
            committee_position = committee_proof.position,
            committee = %committee_proof.committee,
            witness_position = witness_proof.position,
            witness = %witness_proof.witness,
            warmup_lr_between = ?warmup_lr_between,
            assigned_batches = ?assigned_batches,
            "Got training assignment for step {} (round {}/epoch {}): index={} committee position={} committee={} witness position={} witness={} warmup_lr_between={:?}",
            state.progress.step, round.height, epoch, client_index, committee_proof.position, committee_proof.committee, witness_proof.position, witness_proof.witness, warmup_lr_between
        );

        for batch_id in &assigned_batches {
            event!(train::BatchAssigned {
                batch_id: *batch_id
            });
        }
        let model_task_runner = self.model_task_runner.clone();
        let finished = Arc::new(AtomicBool::new(false));

        let prev_self_distro_results = previous_round.self_distro_results.clone();
        let applying_and_training: JoinHandle<Result<FinishedTrainers, TrainError>> =
            if !have_training {
                let finished = finished.clone();

                // the last two rounds have no training (just applying the final results)
                tokio::task::spawn(async move {
                    let round_duration = Instant::now() - round_start;
                    debug!("Training for round finished, duration {:?}", round_duration);
                    finished.store(true, Ordering::SeqCst);
                    Ok(FinishedTrainers {
                        evals_or_trainers: MaybeRunningEvals::Running(
                            model_task_runner
                                .start(applying.await.map_err(|_| TrainError::ApplyCrashed)??),
                        ),
                        round_losses: vec![],
                        optim_stats: HashMap::new(),
                        round_duration,
                    })
                })
            } else {
                let identity = self.identity;
                // The run a payload's commitment is bound to, and the epoch it
                // names the round by (wienerlabs/leviathan-net#67).
                let run_id = String::from(&state.run_id);
                let epoch_start = state.epoch_state.start_timestamp;
                let cancel_training = cancel_training.clone();
                let write_gradients_dir = self.write_gradients_dir.clone();
                let tx_distro_result = self.tx_distro_result.clone();
                let quantize = match &state.model {
                    model::Model::LLM(llm) => match llm.optimizer {
                        OptimizerDefinition::Distro { quantize_1bit, .. } => quantize_1bit,
                        _ => false,
                    },
                };
                let finished = finished.clone();

                let TrainingDataForStep {
                    step,
                    mut next_sample,
                } = self
                    .data_fetcher
                    .fetch_data(state, &data_assignments, &self.identity);

                tokio::task::spawn(async move {
                    let mut round_losses: Vec<f32> = Vec::new();
                    let mut optim_stats: HashMap<String, f64> = HashMap::new();

                    let mut available_trainers =
                        applying.await.map_err(|_| TrainError::ApplyCrashed)??;

                    while let Some(data) = next_sample.recv().await {
                        let mut in_progress = FuturesUnordered::new();

                        // reset the DP barriers
                        if let Some(trainer) = available_trainers.first() {
                            #[allow(irrefutable_let_patterns)]
                            if let Trainer::Local(trainer) = trainer {
                                if available_trainers.len() != trainer.data_parallel_world_size() {
                                    error!("Available trainers does not equal DP world size");
                                    return Err(TrainError::TrainCrashed);
                                }
                                trainer.data_parallel_barrier();
                            }
                        } else {
                            error!("No available trainers");
                            return Err(TrainError::TrainCrashed);
                        }

                        let batches = match &data.data {
                            BatchData::CPU(items) => {
                                let total_size = items.len();
                                let num_trainers = available_trainers.len();
                                let chunk_size = total_size / num_trainers;
                                let mut batches = items
                                    .chunks(chunk_size)
                                    .map(|x| x.to_owned())
                                    .collect::<Vec<_>>();
                                if batches.len() == num_trainers + 1 {
                                    let last = batches.pop().unwrap();
                                    for (i, sample) in last.into_iter().enumerate() {
                                        batches[i].push(sample);
                                    }
                                }
                                if batches.len() != num_trainers {
                                    error!("Batches does not match DP world size");
                                }
                                batches
                            }
                            BatchData::GPU(_) => {
                                error!("Got data on GPU before distribution to trainers");
                                return Err(TrainError::TrainCrashed);
                            }
                        };

                        for (trainer, batch_data) in available_trainers.drain(..).zip(batches) {
                            let batch_id = data.id;
                            let batch_data = batch_data.to_vec();
                            let cancel_training = cancel_training.clone();
                            let prev_self_distro_results = prev_self_distro_results.clone();
                            in_progress.push(tokio::task::spawn_blocking(move || {
                                event!(train::TrainingStarted { batch_id });
                                trainer.train(
                                    step,
                                    Batch {
                                        id: batch_id,
                                        data: BatchData::CPU(batch_data),
                                    },
                                    warmup_lr_between,
                                    zero_optim,
                                    Vec::new(),
                                    Some(prev_self_distro_results),
                                    cancel_training,
                                )
                            }));
                        }

                        // the distro results are identical across all ranks, so we just send the first one we get
                        let mut sent_results = false;

                        while let Some(completed_trainer) = in_progress.next().await {
                            let TrainOutput {
                                batch_id,
                                trainer,
                                loss,
                                step,
                                distro_results,
                                cancelled,
                                nonce,
                            } = completed_trainer.map_err(|_| TrainError::TrainCrashed)??;

                            event!(train::TrainingFinished {
                                batch_id,
                                step: step.into(),
                                loss: Some(loss.into())
                            });

                            debug!(step=step, loss=loss, batch_id=%batch_id, "Got training output, DisTrO results generated");

                            available_trainers.push(trainer);

                            if !sent_results {
                                let distro_results = distro_results.unwrap_or_default();

                                for result in &distro_results {
                                    if let Some(stats) = &result.stats {
                                        for (name, value) in stats {
                                            // a rolling average for this step :)
                                            optim_stats
                                                .entry(name.clone())
                                                .and_modify(|e| *e = (*e + value) / 2.0)
                                                .or_insert(*value);
                                        }
                                    }
                                }
                                let write_gradients_dir = write_gradients_dir.clone();
                                let tx_distro_result = tx_distro_result.clone();
                                let run_id = run_id.clone();
                                let res: Result<(), TrainError> = tokio::task::spawn_blocking(move || {
                                    if cancelled {
                                        trace!("However, we were cancelled, so we're throwing away this result.");
                                        // we're throwing away this result.
                                        return Ok(());
                                    }

                                    let to_transmit = if quantize { Trainer::quantize_results(&distro_results) } else { distro_results.clone()};
                                    let to_transmit = inject_fake_delta(to_transmit);

                                    let transmittable_distro_result = TransmittableDistroResult {
                                        epoch_start,
                                        step,
                                        batch_id,
                                        distro_results: to_transmit
                                            .into_iter()
                                            .map(|x| SerializedDistroResult::try_from(&x))
                                            .collect::<std::result::Result<Vec<_>, _>>()
                                            .map_err(TrainError::SerializeDistroResult)?,
                                        trainer_nonce: nonce,
                                    };

                                    if let Some(dir) = write_gradients_dir {
                                        let transmittable_distro_result = transmittable_distro_result.clone();
                                        let dir = dir.clone();
                                        tokio::spawn(async move {
                                            if let Err(err) =
                                                write_gradients_to_disk(dir, identity, transmittable_distro_result, None).await
                                            {
                                                error!("Failed to write gradients to disk: {err:#}");
                                            }
                                        });
                                    }

                                    let commitment_data_hash = transmittable_distro_result
                                        .comptue_hash(&run_id)
                                        .map_err(TrainError::HashDistroResult)?;

                                    trace!("trying to queue tx distro result...");
                                    tx_distro_result
                                        .send(DistroBroadcastAndPayload {
                                            step,
                                            batch_id,
                                            commitment_data_hash,
                                            proof: committee_proof,
                                            distro_result: transmittable_distro_result,
                                            original_distro_result: distro_results,
                                        })
                                        .map_err(|_| TrainError::SendDistroResult)?;
                                    trace!("successfully queued tx distro result");
                                    Ok(())
                                }).await.map_err(|_| TrainError::TransmitCrashed)?;
                                res?;

                                round_losses.push(loss);
                                sent_results = true;
                            }
                        }
                    }

                    let evals = if cancel_training.is_cancelled() {
                        // we got timed out, don't bother starting evals
                        MaybeRunningEvals::NotRunning(available_trainers)
                    } else {
                        // we finished before getting cancelled, have some time to start evals.
                        MaybeRunningEvals::Running(model_task_runner.start(available_trainers))
                    };
                    let round_duration = Instant::now() - round_start;
                    debug!("Training for round finished, duration {:?}", round_duration);
                    finished.store(true, Ordering::SeqCst);
                    Ok(FinishedTrainers {
                        evals_or_trainers: evals,
                        round_losses,
                        optim_stats,
                        round_duration,
                    })
                })
            };

        Ok(TrainingStep {
            applying_and_training,
            cancel_training,
            sending_health_checks,
            finished,
        })
    }

    fn apply_results(
        &mut self,
        trainers: Vec<Trainer>,
        state: &Coordinator,
        previous_round: &mut RoundState,
        current_round: &mut RoundState,
    ) -> Result<JoinHandle<Result<Vec<Trainer>, ApplyError>>, ApplyError> {
        if current_round.height == 0 {
            // the first TWO training step of each epoch has no apply phase.
            // but, because we call this once with the default initalized RoundState (round 0)
            // and a second time (when transitioning from round 0 -> round 1), this check will skip
            // the two phases
            trace!("Skipping early apply");
            return Ok(tokio::task::spawn(async move { Ok(trainers) }));
        }

        let apply_start = Instant::now();
        let step = state.progress.step;
        let witness_quorum = state.witness_quorum(
            state
                .previous_round()
                .ok_or(ApplyError::NoActiveRound)?
                .witnesses
                .len() as u16,
        );
        let (cold_start_warmup_steps, checkpoint_is_p2p) = match &state.model {
            model::Model::LLM(llm) => (
                llm.cold_start_warmup_steps,
                matches!(
                    llm.checkpoint,
                    model::Checkpoint::P2P(_) | model::Checkpoint::P2PGcs(_)
                ),
            ),
        };
        let warmup_lr_between = state.get_cold_start_warmup_bounds();

        // coordinator has already advanced to the next round (unless we're in cooldown) but we haven't started ours yet.
        // so our current_round corresponds to the coordinator's previous_round
        // `previous_round` -> state.previous_previous_round()
        // `current_round` -> state.previous_round()
        let payloads = std::mem::take(&mut previous_round.downloads);
        let commitments = std::mem::take(&mut previous_round.results);

        // here, when dealing with the coordinator,
        let witnesses = state
            .previous_round()
            .ok_or(ApplyError::NoActiveRound)?
            .witnesses;
        let batch_ids = get_batch_ids_for_round(
            state
                .previous_previous_round()
                .ok_or(ApplyError::NoActiveRound)?,
            state,
            previous_round
                .committee_info
                .as_ref()
                .ok_or(ApplyError::NoActiveRound)?
                .2
                .get_num_trainer_nodes(),
        );

        let data_assignments = previous_round.data_assignments.clone();

        Ok(tokio::task::spawn(async move {
                let payloads = payloads.clone();
                let mut distro_results: Vec<Vec<DistroResult>> = Vec::new();

                trace!("Have commitments for batches {:?}", commitments.keys().collect::<Vec<_>>());
                trace!("Have payloads for hashes {:?}", payloads.lock().unwrap().keys().collect::<Vec<_>>());

                for batch_id in batch_ids {
                    let batch_commitments = match commitments.get(&batch_id) {
                        Some(x) => x,
                        None => {
                            let expected_trainer = data_assignments.get(&batch_id);
                            event!(train::UntrainedBatchWarning {
                                batch_id,
                                expected_trainer: expected_trainer.map(|t| format!("{:?}", t)),
                            });
                            warn!(
                                integration_test_log_marker = %IntegrationTestLogMarker::UntrainedBatches,
                                batch_id = %batch_id,
                                expected_trainer = ?expected_trainer,
                                "No commitments for batch {batch_id}, assigned to node {expected_trainer:?}",
                            );
                            continue;
                        }
                    };
                    trace!("Commitments for batch {batch_id}: {batch_commitments:?}");
                    let consensus = match Coordinator::select_consensus_commitment_by_witnesses(
                        &batch_commitments
                            .iter()
                            .map(|x| x.1.0)
                            .collect::<Vec<_>>(),
                        &witnesses,
                        witness_quorum,
                    ) {
                        Some(x) => x,
                        None => {
                            warn!("No consensus commitment for batch {}", batch_id);
                            continue;
                        }
                    };
                    trace!("Consensus commitment for batch {batch_id}: {consensus:?}");

                    let (commitment, result) = &batch_commitments[consensus].1;
                    let payload_remove_result = payloads.lock().unwrap().remove(&result.ticket.hash());
                    let maybe_results: Result<(Vec<DistroResult>, u32), DeserializeError> = match payload_remove_result {
                        Some(PayloadState::Deserializing(x)) => match x.is_finished() {
                            true => x.await.unwrap(),
                            false => {
                                return Err(ApplyError::DidNotFinishDeserializingCommitment(
                                    Box::new(*commitment),
                                    batch_id,
                                ));
                            }
                        },
                        Some(PayloadState::Downloading((_, _, ticket))) => {
                            return Err(ApplyError::DidNotBeginDownloadingCommitment(
                                Box::new(*commitment),
                                batch_id,
                                ticket.hash()
                            ));
                        }
                        None => {
                            return Err(ApplyError::UnknownCommitment(
                                Box::new(*commitment),
                                batch_id,
                            ))
                        }
                    };

                    match maybe_results {
                        Ok((results, trainer_nonce)) => {
                            if trainer_nonce < cold_start_warmup_steps && checkpoint_is_p2p {
                                // Only filter results from trainers that are still warming up their optimizer,
                                // and only when the checkpoint is P2P (meaning other clients exist from a previous epoch).
                                // When checkpoint is Hub (first epoch or mass restart), all trainers are equally new so no filtering is needed.
                                // note, we are relying on honest communication of this value here -- will need to harden with verification.
                                info!("Skipping apply of batch {batch_id}, trainer warming up ({trainer_nonce}/{cold_start_warmup_steps})");
                            } else {
                                distro_results.push(results);
                            }
                        }
                        Err(err) => warn!("DESYNC: Got the following error when deserializing results for commitment 0x{}: {}", hex::encode(commitment.data_hash), err),
                    }
                }

                event!(train::ApplyDistroResultsStart);
                let futures: Vec<JoinHandle<std::result::Result<Trainer, ApplyDistroResultError>>> =
                    trainers
                        .into_iter()
                        .map(|trainer| {
                            let distro_results = Some(distro_results.clone());

                            tokio::task::spawn_blocking(move || {
                                trainer.optimize(step, warmup_lr_between, distro_results)
                            })
                        })
                        .collect::<Vec<_>>();
                let apply_result: Result<Vec<Trainer>, ApplyError> = async {
                    let results = try_join_all(futures)
                        .await
                        .map_err(|_| ApplyDistroResultError::ThreadCrashed)?;
                    let trainers: Vec<_> = results.into_iter().collect::<Result<_, _>>()?;
                    Ok(trainers)
                }.await;
                let trainers: Vec<_> = match apply_result {
                    Ok(trainers) => {
                        event!(train::ApplyDistroResultsComplete(Ok(())));
                        trainers
                    }
                    Err(e) => {
                        event!(train::ApplyDistroResultsComplete(Err(e.to_string())));
                        return Err(e);
                    }
                };
                trace!(
                    "Apply time: {:.1}s, {} trainers ready",
                    (Instant::now() - apply_start).as_secs_f32(),
                    trainers.len()
                );
                Ok(trainers)
            }.instrument(trace_span!("Applying distro results"))))
    }
}

fn start_sending_health_checks(
    round_state: &mut RoundState,
    state: &Coordinator,
    tx_health_check: mpsc::UnboundedSender<HealthChecks>,
) -> Result<Option<JoinHandle<Result<(), TrainError>>>, TrainError> {
    // we won't have any information to health check with until at least one round of training has finished
    if round_state.height == 0 {
        return Ok(None);
    }
    let (_, witness_proof, committee_selection) = round_state
        .committee_info
        .as_ref()
        .ok_or(TrainError::NoCommitteeInfo)?;
    // The committee these checks are judged by was drawn over the previous
    // round's roster, seat by seat. If anyone has left since, the seats no
    // longer hold the clients it seated, and the coordinator refuses every
    // check this round (`CoordinatorError::RosterMoved`).
    let roster_moved = state
        .previous_round()
        .is_some_and(|round| round.clients_len as usize != state.epoch_state.clients.len());
    if roster_moved {
        debug!("The roster moved since the previous round was drawn, so no health check is sent");
        return Ok(None);
    }
    Ok(
        if state.epoch_state.first_round.is_false() && witness_proof.witness.is_true() {
            let clients = state.epoch_state.clients;
            let committee_selection = committee_selection.clone();
            let state = *state;
            Some(tokio::task::spawn(async move {
                let mut checks = HealthChecks::new();
                for (index, client) in clients.iter().enumerate() {
                    let proof = committee_selection.get_committee(index as u64);
                    if !state.healthy(&client.id, &proof).unwrap_or(false) {
                        event!(client::HealthCheckFailed {
                            index: index as u64,
                            round: state.epoch_state.rounds_head as u64,
                        });
                        warn!(
                            integration_test_log_marker = %IntegrationTestLogMarker::HealthCheck,
                            index = index,
                            client_id = %&client.id,
                            current_step = state.epoch_state.rounds_head,
                            "Found unhealthy trainer at index: {}", index,
                        );
                        checks.push((client.id, proof));
                    }
                }

                if !checks.is_empty() {
                    info!("Sending health check for following indices: {:?}", checks);
                    tx_health_check
                        .send(checks)
                        .map_err(|_| TrainError::SendHealthChecks)
                } else {
                    Ok(())
                }
            }))
        } else {
            None
        },
    )
}

#[derive(Error, Debug)]
pub enum ApplyError {
    #[error("no active round")]
    NoActiveRound,

    #[error("failed to apply distro result: {0}")]
    BadResult(#[from] ApplyDistroResultError),

    #[error("DESYNC: Did not finish deserializing payload for consensus commitment 0x{commitment} for batch {1}", commitment=hex::encode(.0.data_hash))]
    DidNotFinishDeserializingCommitment(Box<Commitment>, BatchId),

    #[error("DESYNC: Did not begin downloading payload for consensus commitment 0x{commitment} for batch {1} with blob hash {2}", commitment=hex::encode(.0.data_hash))]
    DidNotBeginDownloadingCommitment(Box<Commitment>, BatchId, Hash),

    #[error("DESYNC: Unknown consensus commitment 0x{commitment} for batch {1}", commitment=hex::encode(.0.data_hash))]
    UnknownCommitment(Box<Commitment>, BatchId),
}

#[derive(Debug, Error)]
pub(crate) enum WriteGradientsError {
    #[error("Failed to create write_gradients_dir: {0}")]
    CreateDir(tokio::io::Error),

    #[error("Failed to serialize distro result data {fname} to bytes: {err}")]
    Serialize { fname: String, err: postcard::Error },

    #[error("Failed to write distro result data {fname}: {err}")]
    Write {
        fname: String,
        err: tokio::io::Error,
    },
}

fn inject_fake_delta(results: Vec<DistroResult>) -> Vec<DistroResult> {
    let mode = match std::env::var("LEVIATHAN_FAKE_DELTA") {
        Ok(mode) if !mode.is_empty() => mode,
        _ => return results,
    };
    warn!("LEVIATHAN_FAKE_DELTA={mode}: broadcasting a fraudulent gradient (test only)");
    results
        .into_iter()
        .map(|mut result| {
            result.sparse_val = match mode.as_str() {
                "lazy" => result.sparse_val.zeros_like(),
                "gaussian" => result.sparse_val.randn_like(),
                _ => result.sparse_val * -5.0,
            };
            result
        })
        .collect()
}

/// Writes `distro_result` to disk, as a payload dump the verifier daemon reads.
///
/// A peer's payload comes with `commitment`, the record of what the peer
/// signed for it, which is what a conviction proves on chain
/// (wienerlabs/leviathan-net#43). The pair has to hold together for a reader
/// at any moment:
///
/// - **One payload per committer, step and batch, the first kept.** A peer that
///   signs two payloads for the same batch would otherwise have them written
///   over each other. That can leave one's record beside the other's dump, and
///   nothing could be convicted on it. The record is created only if none
///   exists, so the first payload owns the name.
/// - **The record before the dump, and the dump all at once.** The record is
///   on disk before the dump is written under a temporary name and renamed
///   into place, so a reader that finds the dump also finds its record,
///   complete, beside it. A dump that cannot be written takes its record
///   with it.
pub(crate) async fn write_gradients_to_disk(
    write_gradients_dir: PathBuf,
    identity: NodeIdentity,
    distro_result: TransmittableDistroResult,
    commitment: Option<CommitmentRecord>,
) -> Result<(), WriteGradientsError> {
    debug!("Trying to write distro result to disk...");
    tokio::fs::create_dir_all(&write_gradients_dir)
        .await
        .map_err(WriteGradientsError::CreateDir)?;

    // The full signer, base58 and untruncated. The daemon parses this segment
    // back out and matches it against the roster, so a truncated `Display` made
    // the audit pipeline join on eight characters - and any change to how
    // identities are printed silently broke the match, with the daemon skipping
    // every contribution while reporting a clean pass
    // (wienerlabs/leviathan#15, finding 20).
    let fname = payload_dump_name(
        &bs58::encode(identity.signer()).into_string(),
        distro_result.epoch_start,
        distro_result.step,
        distro_result.batch_id,
    );
    let fpath = write_gradients_dir.join(&fname);
    let serialized = distro_results_to_bytes(&distro_result.distro_results).map_err(|err| {
        WriteGradientsError::Serialize {
            fname: fname.clone(),
            err,
        }
    })?;
    let Some(commitment) = commitment else {
        write_whole(&write_gradients_dir, &fname, &serialized).await?;
        debug!("Wrote distro result {fname}.");
        return Ok(());
    };
    let record_path = commitment_record_path(&fpath);
    match claim_record(&record_path, &commitment, &fpath).await? {
        Claim::Owned => {}
        Claim::Duplicate => {
            debug!("Already keeping this payload for {fname}.");
            return Ok(());
        }
        Claim::Unreadable => {
            warn!(
                "Already keeping a payload for {fname} under a record that does not read; keeping it"
            );
            return Ok(());
        }
        Claim::Contradicted => {
            warn!(
                "Already keeping a payload for {fname}; the committer signed another for the same batch, keeping the first and the proof that it equivocated"
            );
            return keep_equivocation(&fpath, &distro_result, commitment).await;
        }
    }
    if let Err(err) = write_whole(&write_gradients_dir, &fname, &serialized).await {
        // A record whose dump never lands would keep this payload out for good,
        // so it goes too, and the payload can be kept when it comes again.
        if let Err(cleanup) = tokio::fs::remove_file(&record_path).await {
            warn!(
                "Could not remove {} after its dump failed: {cleanup}",
                record_path.display()
            );
        }
        return Err(err);
    }
    debug!("Wrote distro result {fname}.");
    Ok(())
}

/// What taking the record for a payload's name found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Claim {
    /// The name is this payload's: write its dump.
    Owned,
    /// This payload is already kept under the name.
    Duplicate,
    /// Another payload is kept under the name, with a commitment that
    /// contradicts this one: the committer signed two for one batch.
    Contradicted,
    /// A record holds the name but does not read, so it proves nothing.
    Unreadable,
}

/// Takes the commitment record for the dump at `dump`, which says this payload
/// is the one kept under that name, or says who holds it.
///
/// A record left without its dump, by a crash or a failed write, is still
/// this payload's if the same commitment arrives again, and the dump is then
/// finished rather than the payload refused.
async fn claim_record(
    record_path: &Path,
    commitment: &CommitmentRecord,
    dump: &Path,
) -> Result<Claim, WriteGradientsError> {
    let write_error = |err| WriteGradientsError::Write {
        fname: record_path.display().to_string(),
        err,
    };
    // On disk, whole, before the dump can appear beside it.
    match publish_new(record_path, &commitment.to_bytes()).await {
        Ok(Published::New) => Ok(Claim::Owned),
        Ok(Published::Taken) => {
            let held = tokio::fs::read(record_path).await.map_err(write_error)?;
            let Ok(held) = CommitmentRecord::from_bytes(&held) else {
                return Ok(Claim::Unreadable);
            };
            if held != *commitment {
                return Ok(Claim::Contradicted);
            }
            let dump_kept = tokio::fs::try_exists(dump).await.map_err(write_error)?;
            Ok(if dump_kept {
                Claim::Duplicate
            } else {
                Claim::Owned
            })
        }
        Err(err) => Err(write_error(err)),
    }
}

/// Keeps what the committer signed for `payload`, the second payload under
/// the name of the dump at `dump`, as an equivocation record beside it: the
/// proof, with the kept pair, that the committer signed two payloads for one
/// batch (wienerlabs/leviathan-net#46). The first contradiction is kept; a
/// later one adds nothing a conviction needs.
async fn keep_equivocation(
    dump: &Path,
    payload: &TransmittableDistroResult,
    commitment: CommitmentRecord,
) -> Result<(), WriteGradientsError> {
    let path = equivocation_record_path(dump);
    let write_error = |err| WriteGradientsError::Write {
        fname: path.display().to_string(),
        err,
    };
    let record = EquivocationRecord::new(payload, commitment)
        .map_err(|err| write_error(std::io::Error::new(std::io::ErrorKind::InvalidData, err)))?;
    publish_new(&path, &record.to_bytes())
        .await
        .map(|_| ())
        .map_err(write_error)
}

/// Whether `publish_new` put its bytes in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Published {
    /// The path was free and now holds them.
    New,
    /// Something was there already and is left as it was.
    Taken,
}

/// Puts `bytes` at `path` only if nothing is there yet, and only whole. They
/// go to a temporary name of their own, reach the disk, and are linked to
/// `path`, which fails if it exists. A reader of `path` finds nothing or all
/// of it: a second payload arriving while the first record is written reads
/// the whole record, not a short one, and a crash leaves no partial file to
/// hold the name for good. Either used to lose a proof of equivocation
/// (wienerlabs/leviathan-net#46).
async fn publish_new(path: &Path, bytes: &[u8]) -> std::io::Result<Published> {
    publish_new_linking(path, bytes, |from, to| async move {
        tokio::fs::hard_link(from, to).await
    })
    .await
}

/// `publish_new`, linking the finished file into place with `link`.
///
/// A filesystem that cannot hard-link (some FUSE, SMB and FAT mounts) gets the
/// bytes written in place under `create_new` instead. They still never land
/// over another file, though a reader may find them part-written while the
/// write lasts. Failing there instead would keep no peer payload at all.
async fn publish_new_linking<F, Linked>(
    path: &Path,
    bytes: &[u8],
    link: F,
) -> std::io::Result<Published>
where
    F: FnOnce(PathBuf, PathBuf) -> Linked,
    Linked: std::future::Future<Output = std::io::Result<()>>,
{
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| std::io::Error::other(format!("{} has no file name", path.display())))?;
    let partial = path.with_file_name(format!("{name}.{:016x}.partial", rand::random::<u64>()));
    let linked = async {
        let mut file = tokio::fs::File::create(&partial).await?;
        file.write_all(bytes).await?;
        file.sync_all().await?;
        link(partial.clone(), path.to_path_buf()).await
    }
    .await;
    if let Err(cleanup) = tokio::fs::remove_file(&partial).await {
        if cleanup.kind() != std::io::ErrorKind::NotFound {
            warn!("Could not remove {}: {cleanup}", partial.display());
        }
    }
    match linked {
        Ok(()) => Ok(Published::New),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(Published::Taken),
        Err(err) if cannot_link(&err) => {
            warn!(
                "{} cannot be hard-linked into place ({err}); writing it in place",
                path.display()
            );
            create_in_place(path, bytes).await
        }
        Err(err) => Err(err),
    }
}

/// Whether `err` is a filesystem refusing hard links, rather than a failure to
/// write at all.
fn cannot_link(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::Unsupported
            | std::io::ErrorKind::PermissionDenied
            | std::io::ErrorKind::CrossesDevices
    )
}

/// Writes `bytes` at `path` only if nothing is there yet, in place.
async fn create_in_place(path: &Path, bytes: &[u8]) -> std::io::Result<Published> {
    match tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await
    {
        Ok(mut file) => {
            file.write_all(bytes).await?;
            file.sync_all().await?;
            Ok(Published::New)
        }
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(Published::Taken),
        Err(err) => Err(err),
    }
}

/// Writes `bytes` to `dir/fname` so that a reader finds either nothing there
/// or all of it. They go to a temporary name of their own, reach the disk,
/// and are renamed into place.
async fn write_whole(dir: &Path, fname: &str, bytes: &[u8]) -> Result<(), WriteGradientsError> {
    let partial = dir.join(format!("{fname}.{:016x}.partial", rand::random::<u64>()));
    let written = async {
        let mut file = tokio::fs::File::create(&partial).await?;
        file.write_all(bytes).await?;
        file.sync_all().await?;
        tokio::fs::rename(&partial, dir.join(fname)).await
    }
    .await;
    if let Err(err) = written {
        if let Err(cleanup) = tokio::fs::remove_file(&partial).await {
            if cleanup.kind() != std::io::ErrorKind::NotFound {
                warn!("Could not remove {}: {cleanup}", partial.display());
            }
        }
        return Err(WriteGradientsError::Write {
            fname: fname.to_string(),
            err,
        });
    }
    Ok(())
}

/// A verifier convicts a peer on the payload it keeps and the record of what
/// the peer signed for it, so the pair has to hold together whenever the
/// verifier reads it (wienerlabs/leviathan-net#43).
#[cfg(test)]
mod write_gradients_tests {
    use super::*;
    use psyche_core::ClosedInterval;
    use psyche_network::SecretKey;
    use tch::{Kind, Tensor};

    const RUN: &str = "leviathan kept payloads";

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("levkept-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A payload of one compressed result whose values are `values`.
    fn payload(values: &[f32]) -> TransmittableDistroResult {
        let result = DistroResult {
            sparse_idx: Tensor::from_slice(&[0i64, 1, 2]).to_kind(Kind::Int16),
            sparse_val: Tensor::from_slice(values).to_kind(Kind::Float),
            xshape: vec![1, 3],
            totalk: 3,
            stats: None,
        };
        TransmittableDistroResult {
            epoch_start: 1_700_000_000,
            step: 5,
            trainer_nonce: 2,
            batch_id: BatchId(ClosedInterval::new(8, 9)),
            distro_results: vec![SerializedDistroResult::try_from(&result).unwrap()],
        }
    }

    fn signed(key: &SecretKey, payload: &TransmittableDistroResult) -> CommitmentRecord {
        let data_hash = payload.comptue_hash(RUN).unwrap();
        CommitmentRecord::new(payload, data_hash, key.sign(&data_hash).to_bytes())
    }

    fn peer(key: &SecretKey) -> NodeIdentity {
        NodeIdentity::new(*key.public().as_bytes(), *key.public().as_bytes())
    }

    fn dump_of(dir: &Path, peer: &NodeIdentity, payload: &TransmittableDistroResult) -> PathBuf {
        dir.join(format!(
            "result-{}-epoch{}-step{}-batch{}.vec-postcard",
            bs58::encode(peer.signer()).into_string(),
            payload.epoch_start,
            payload.step,
            payload.batch_id
        ))
    }

    fn kept(dump: &Path) -> (CommitmentRecord, Vec<SerializedDistroResult>) {
        let record =
            CommitmentRecord::from_bytes(&std::fs::read(commitment_record_path(dump)).unwrap())
                .unwrap();
        let results = psyche_network::distro_results_from_reader(std::io::Cursor::new(
            std::fs::read(dump).unwrap(),
        ))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        (record, results)
    }

    #[tokio::test]
    async fn a_peer_payload_is_kept_beside_the_record_it_was_signed_with() {
        let dir = scratch("pair");
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(&[1.0, -2.0, 3.0]);
        write_gradients_to_disk(
            dir.clone(),
            peer(&key),
            payload.clone(),
            Some(signed(&key, &payload)),
        )
        .await
        .unwrap();

        let dump = dump_of(&dir, &peer(&key), &payload);
        let (record, results) = kept(&dump);
        assert_eq!(
            record.verify(&results, RUN, key.public().as_bytes()),
            Ok(())
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".partial"))
            .collect();
        assert!(leftovers.is_empty(), "left behind {leftovers:?}");
    }

    /// A run restarted from an earlier step trains the step again in a new
    /// epoch. The payload the peer signs then is another round's, and is kept
    /// beside the first under its own name, with its own record, rather than
    /// refused for a name the first one holds (wienerlabs/leviathan-net#67).
    #[tokio::test]
    async fn the_same_step_trained_again_in_a_later_epoch_is_kept_too() {
        let dir = scratch("restart");
        let key = SecretKey::generate(&mut rand::rng());
        let first = payload(&[1.0, -2.0, 3.0]);
        let again = TransmittableDistroResult {
            epoch_start: first.epoch_start + 90,
            ..payload(&[4.0, 5.0, -6.0])
        };
        for payload in [&first, &again] {
            write_gradients_to_disk(
                dir.clone(),
                peer(&key),
                payload.clone(),
                Some(signed(&key, payload)),
            )
            .await
            .unwrap();
        }
        for payload in [&first, &again] {
            let (record, results) = kept(&dump_of(&dir, &peer(&key), payload));
            assert_eq!(record.epoch_start, payload.epoch_start);
            assert_eq!(record.data_hash, payload.comptue_hash(RUN).unwrap());
            assert_eq!(
                record.verify(&results, RUN, key.public().as_bytes()),
                Ok(())
            );
        }
    }

    /// A peer that signs two payloads for one batch gets the first kept, with
    /// its own record, rather than one written over the other.
    #[tokio::test]
    async fn the_first_payload_a_peer_signed_for_a_batch_is_the_one_kept() {
        let dir = scratch("first");
        let key = SecretKey::generate(&mut rand::rng());
        let first = payload(&[1.0, -2.0, 3.0]);
        let second = payload(&[4.0, 5.0, -6.0]);
        assert_ne!(
            first.comptue_hash(RUN).unwrap(),
            second.comptue_hash(RUN).unwrap()
        );

        for payload in [&first, &second] {
            write_gradients_to_disk(
                dir.clone(),
                peer(&key),
                payload.clone(),
                Some(signed(&key, payload)),
            )
            .await
            .unwrap();
        }

        let (record, results) = kept(&dump_of(&dir, &peer(&key), &first));
        assert_eq!(record.data_hash, first.comptue_hash(RUN).unwrap());
        assert_eq!(
            record.verify(&results, RUN, key.public().as_bytes()),
            Ok(())
        );
    }

    /// The second payload a peer signed for the batch is not kept, but what it
    /// signed is: beside the kept pair, an equivocation record that proves the
    /// peer contradicted itself (wienerlabs/leviathan-net#46). A third keeps
    /// the first equivocation.
    #[tokio::test]
    async fn a_second_payload_for_a_batch_leaves_the_proof_of_equivocation() {
        let dir = scratch("equivocation");
        let key = SecretKey::generate(&mut rand::rng());
        let first = payload(&[1.0, -2.0, 3.0]);
        let second = payload(&[4.0, 5.0, -6.0]);
        let third = payload(&[7.0, 8.0, 9.0]);
        for payload in [&first, &second, &third] {
            write_gradients_to_disk(
                dir.clone(),
                peer(&key),
                payload.clone(),
                Some(signed(&key, payload)),
            )
            .await
            .unwrap();
        }

        let dump = dump_of(&dir, &peer(&key), &first);
        let (kept_record, kept_results) = kept(&dump);
        let equivocation = EquivocationRecord::from_bytes(
            &std::fs::read(equivocation_record_path(&dump)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            equivocation.record.data_hash,
            second.comptue_hash(RUN).unwrap()
        );
        assert_eq!(
            equivocation.proves_against(
                &kept_record,
                &psyche_network::results_digest(&kept_results).unwrap(),
                RUN,
                key.public().as_bytes(),
            ),
            Ok(())
        );
    }

    /// Payloads for one batch arriving together, handled at once on several
    /// threads: one is kept with its own whole record, and a contradicting
    /// one is the proof beside it. A handler used to read a record another
    /// had created but not yet written, find it short, and drop the proof
    /// (wienerlabs/leviathan-net#46).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn payloads_arriving_together_still_leave_the_proof() {
        for round in 0..64 {
            let dir = scratch(&format!("together-{round}"));
            let key = SecretKey::generate(&mut rand::rng());
            let one = payload(&[1.0, -2.0, 3.0]);
            let other = payload(&[4.0, 5.0, -6.0]);
            let arrivals: Vec<_> = (0..8)
                .map(|n| {
                    let payload = if n % 2 == 0 {
                        one.clone()
                    } else {
                        other.clone()
                    };
                    let commitment = signed(&key, &payload);
                    tokio::spawn(write_gradients_to_disk(
                        dir.clone(),
                        peer(&key),
                        payload,
                        Some(commitment),
                    ))
                })
                .collect();
            for arrival in arrivals {
                arrival.await.unwrap().unwrap();
            }

            let dump = dump_of(&dir, &peer(&key), &one);
            let (kept_record, kept_results) = kept(&dump);
            assert_eq!(
                kept_record.verify(&kept_results, RUN, key.public().as_bytes()),
                Ok(()),
                "round {round}: the kept pair does not hold"
            );
            let equivocation = EquivocationRecord::from_bytes(
                &std::fs::read(equivocation_record_path(&dump))
                    .unwrap_or_else(|err| panic!("round {round}: the proof was lost: {err}")),
            )
            .unwrap();
            assert_eq!(
                equivocation.proves_against(
                    &kept_record,
                    &psyche_network::results_digest(&kept_results).unwrap(),
                    RUN,
                    key.public().as_bytes(),
                ),
                Ok(()),
                "round {round}: the proof does not hold"
            );
            let partials: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .filter(|name| name.ends_with(".partial"))
                .collect();
            assert!(
                partials.is_empty(),
                "round {round}: left behind {partials:?}"
            );
        }
    }

    /// A record is put in place whole, and never over another.
    #[tokio::test]
    async fn a_record_is_published_whole_and_only_once() {
        let dir = scratch("publish");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("result-x.commitment");
        assert_eq!(publish_new(&path, b"first").await.unwrap(), Published::New);
        assert_eq!(
            publish_new(&path, b"second").await.unwrap(),
            Published::Taken
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec!["result-x.commitment".to_string()]);
    }

    /// Where the filesystem refuses hard links, a record is still published,
    /// in place, and still never over another.
    #[tokio::test]
    async fn a_record_is_published_where_links_are_refused() {
        let dir = scratch("nolinks");
        std::fs::create_dir_all(&dir).unwrap();
        for refusal in [
            std::io::ErrorKind::Unsupported,
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::CrossesDevices,
        ] {
            let path = dir.join(format!("result-{refusal:?}.commitment"));
            let refuse =
                |_: PathBuf, _: PathBuf| async move { Err::<(), _>(std::io::Error::from(refusal)) };
            assert_eq!(
                publish_new_linking(&path, b"first", refuse).await.unwrap(),
                Published::New
            );
            assert_eq!(
                publish_new_linking(&path, b"second", refuse).await.unwrap(),
                Published::Taken
            );
            assert_eq!(std::fs::read(&path).unwrap(), b"first");
        }
        let partials: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".partial"))
            .collect();
        assert!(partials.is_empty(), "left behind {partials:?}");
    }

    /// A record that does not read holds its name and proves nothing: the
    /// payload is not kept beside a record that is not its own, no proof is
    /// written against it, and the write does not fail.
    #[tokio::test]
    async fn a_record_that_does_not_read_keeps_the_name() {
        let dir = scratch("unreadable");
        std::fs::create_dir_all(&dir).unwrap();
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(&[1.0, -2.0, 3.0]);
        let dump = dump_of(&dir, &peer(&key), &payload);
        std::fs::write(commitment_record_path(&dump), b"not a record").unwrap();
        write_gradients_to_disk(
            dir.clone(),
            peer(&key),
            payload.clone(),
            Some(signed(&key, &payload)),
        )
        .await
        .unwrap();
        assert!(!dump.exists());
        assert!(!equivocation_record_path(&dump).exists());
        assert_eq!(
            std::fs::read(commitment_record_path(&dump)).unwrap(),
            b"not a record"
        );
    }

    /// The same payload arriving again is not a contradiction, and leaves no
    /// equivocation record.
    #[tokio::test]
    async fn the_same_payload_twice_leaves_no_equivocation() {
        let dir = scratch("repeat");
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(&[1.0, -2.0, 3.0]);
        for _ in 0..2 {
            write_gradients_to_disk(
                dir.clone(),
                peer(&key),
                payload.clone(),
                Some(signed(&key, &payload)),
            )
            .await
            .unwrap();
        }
        let dump = dump_of(&dir, &peer(&key), &payload);
        assert!(!equivocation_record_path(&dump).exists());
    }

    /// A record left without its dump is finished by the same commitment when
    /// it comes again, rather than keeping the payload out.
    #[tokio::test]
    async fn a_record_left_without_its_dump_is_finished_by_the_same_payload() {
        let dir = scratch("unfinished");
        std::fs::create_dir_all(&dir).unwrap();
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(&[1.0, -2.0, 3.0]);
        let record = signed(&key, &payload);
        let dump = dump_of(&dir, &peer(&key), &payload);
        std::fs::write(commitment_record_path(&dump), record.to_bytes()).unwrap();

        write_gradients_to_disk(dir.clone(), peer(&key), payload.clone(), Some(record))
            .await
            .unwrap();
        let (kept_record, results) = kept(&dump);
        assert_eq!(
            kept_record.verify(&results, RUN, key.public().as_bytes()),
            Ok(())
        );
    }

    /// A dump that cannot be written takes its record with it, so the payload
    /// is not kept out for good and no partial file is left. The rename fails
    /// here because a directory sits at the dump's name.
    #[tokio::test]
    async fn a_dump_that_cannot_be_written_leaves_no_record_behind() {
        let dir = scratch("failed");
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(&[1.0, -2.0, 3.0]);
        let dump = dump_of(&dir, &peer(&key), &payload);
        std::fs::create_dir_all(dump.join("occupied")).unwrap();

        let result = write_gradients_to_disk(
            dir.clone(),
            peer(&key),
            payload.clone(),
            Some(signed(&key, &payload)),
        )
        .await;
        assert!(result.is_err(), "a dump written over a directory succeeded");
        assert!(
            !commitment_record_path(&dump).exists(),
            "the record outlived its dump"
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".partial"))
            .collect();
        assert!(leftovers.is_empty(), "left behind {leftovers:?}");
    }

    /// A node's own result is kept with no record: nobody convicts itself.
    #[tokio::test]
    async fn an_own_result_is_kept_without_a_record() {
        let dir = scratch("own");
        let key = SecretKey::generate(&mut rand::rng());
        let payload = payload(&[1.0, -2.0, 3.0]);
        write_gradients_to_disk(dir.clone(), peer(&key), payload.clone(), None)
            .await
            .unwrap();
        let dump = dump_of(&dir, &peer(&key), &payload);
        assert!(dump.exists());
        assert!(!commitment_record_path(&dump).exists());
    }
}
