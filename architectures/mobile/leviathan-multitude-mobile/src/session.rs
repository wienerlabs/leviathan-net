use std::path::{Path, PathBuf};

use leviathan_multitude::adversary::Behavior;
use leviathan_multitude::device::{
    decide, plan_inner_steps, Decision, DevicePolicy, DeviceState, Network, ThroughputMeter,
};
use leviathan_multitude::model::default_device;
use leviathan_multitude::protocol::weights_hash;
use leviathan_multitude::wire::{RoundInfo, RunInfo};
use leviathan_multitude::{Contribution, Corpus, Device, NodeState, PhoneNode, RoundJob};
use serde::{Deserialize, Serialize};

use crate::relay::{RelayClient, RelayError};

const THROUGHPUT_SMOOTHING: f32 = 0.3;
const CONVICTED: u16 = 403;

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error(transparent)]
    Relay(#[from] RelayError),
    #[error("training failed: {0}")]
    Training(#[from] leviathan_multitude::Error),
    #[error("could not save the phone's state: {0}")]
    Storage(String),
    #[error("the relay's corpus does not match its fingerprint")]
    Corpus,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Activity {
    Trained {
        round: u32,
        steps: u16,
        loss: f32,
        upload_bytes: usize,
        seconds: f32,
    },
    Waiting(String),
    Paused(String),
    Convicted(String),
}

#[derive(Clone, Debug)]
pub struct Report {
    pub activity: Activity,
    pub round: Option<RoundInfo>,
    pub rounds_counted: u32,
    pub audits_answered: u32,
}

#[derive(Serialize, Deserialize)]
struct Saved {
    run: [u8; 32],
    node: NodeState,
    pending: Option<Contribution>,
    rounds_counted: u32,
    audits_answered: u32,
}

pub struct Session {
    relay_url: String,
    relay: RelayClient,
    secret: [u8; 32],
    info: RunInfo,
    corpus: Corpus,
    node: PhoneNode,
    device: Device,
    policy: DevicePolicy,
    meter: ThroughputMeter,
    behavior: Behavior,
    pending: Option<Contribution>,
    convicted: Option<String>,
    last_round: Option<RoundInfo>,
    rounds_counted: u32,
    audits_answered: u32,
    state_path: Option<PathBuf>,
}

fn load(path: &Path) -> Option<Saved> {
    postcard::from_bytes(&std::fs::read(path).ok()?).ok()
}

fn storage(err: impl std::fmt::Display) -> SessionError {
    SessionError::Storage(err.to_string())
}

impl Session {
    pub fn join(
        relay_url: &str,
        secret: [u8; 32],
        state_path: Option<PathBuf>,
    ) -> Result<Self, SessionError> {
        let relay = RelayClient::new(relay_url);
        let info = relay.run()?;
        let corpus = Corpus::from_bytes(relay.corpus()?);
        if corpus.fingerprint() != info.corpus {
            return Err(SessionError::Corpus);
        }
        let num_params = info.cfg.model.num_params();
        let saved = state_path
            .as_deref()
            .and_then(load)
            .filter(|saved| saved.run == info.run && saved.node.error.len() == num_params);
        let (node, pending, rounds_counted, audits_answered) = match saved {
            Some(saved) => (
                PhoneNode::restore(secret, saved.node),
                saved.pending,
                saved.rounds_counted,
                saved.audits_answered,
            ),
            None => (PhoneNode::new(secret, num_params), None, 0, 0),
        };
        Ok(Self {
            relay_url: relay_url.to_string(),
            relay,
            secret,
            info,
            corpus,
            node,
            device: default_device(),
            policy: DevicePolicy::default(),
            meter: ThroughputMeter::new(THROUGHPUT_SMOOTHING),
            behavior: Behavior::Honest,
            pending,
            convicted: None,
            last_round: None,
            rounds_counted,
            audits_answered,
            state_path,
        })
    }

    pub fn with_behavior(mut self, behavior: Behavior) -> Self {
        self.behavior = behavior;
        self
    }

    pub fn address(&self) -> String {
        self.node.address()
    }

    pub fn tick(&mut self, device: &DeviceState) -> Result<Report, SessionError> {
        if let Some(reason) = self.convicted.clone() {
            return Ok(self.report(Activity::Convicted(reason)));
        }
        if device.network == Network::Offline {
            return Ok(self.report(Activity::Paused("offline".into())));
        }
        let round = self.relay.round()?;
        if round.run != self.info.run {
            let fresh = Self::join(&self.relay_url, self.secret, self.state_path.clone())?
                .with_behavior(self.behavior.clone());
            *self = fresh;
            return Ok(self.report(Activity::Waiting("the relay started a new run".into())));
        }
        self.last_round = Some(round.clone());
        self.settle(&round)?;
        if let Decision::Pause(reason) = decide(&self.policy, device) {
            return Ok(self.report(Activity::Paused(reason.to_string())));
        }
        if let Some(pending) = &self.pending {
            let waiting = format!("waiting for round {} to close", pending.header.round);
            return Ok(self.report(Activity::Waiting(waiting)));
        }
        let steps = self.plan(&round);
        if steps == 0 {
            let waiting = format!("too little time left in round {}", round.round);
            return Ok(self.report(Activity::Waiting(waiting)));
        }
        let (params_round, params) = self.relay.params()?;
        if params_round != round.round || weights_hash(&params) != round.base_checkpoint {
            return Ok(self.report(Activity::Waiting(
                "the round moved on during the download".into(),
            )));
        }

        let job = RoundJob {
            cfg: &self.info.cfg,
            run: self.info.run,
            round: round.round,
            round_seed: round.seed,
            global: &params,
            base_checkpoint: round.base_checkpoint,
            corpus: &self.corpus,
            device: &self.device,
        };
        let (contribution, stats) = self.node.contribute(&job, steps, &self.behavior)?;
        self.meter.record(stats.tokens, stats.seconds);
        let activity = match self.relay.contribute(&contribution) {
            Ok(()) => {
                self.pending = Some(contribution);
                Activity::Trained {
                    round: round.round,
                    steps,
                    loss: stats.mean_loss,
                    upload_bytes: stats.upload_bytes,
                    seconds: stats.seconds,
                }
            }
            Err(RelayError::Refused { status, message }) => {
                self.node.rollback(round.round);
                if status == CONVICTED {
                    let reason = message.strip_prefix("convicted: ").unwrap_or(&message);
                    self.convicted = Some(reason.to_string());
                    Activity::Convicted(reason.to_string())
                } else {
                    Activity::Waiting(format!(
                        "the relay did not take round {}: {message}",
                        round.round
                    ))
                }
            }
            Err(err) => {
                self.pending = Some(contribution);
                self.save()?;
                return Err(err.into());
            }
        };
        self.save()?;
        Ok(self.report(activity))
    }

    fn plan(&self, round: &RoundInfo) -> u16 {
        let inner = &self.info.cfg.inner;
        match self.meter.tokens_per_second() {
            Some(rate) => plan_inner_steps(
                rate,
                self.info.cfg.tokens_per_step(),
                round.closes_in_ms as f32 / 1000.0,
                inner.min_steps,
                inner.max_steps,
            ),
            None => inner.min_steps,
        }
    }

    fn settle(&mut self, round: &RoundInfo) -> Result<(), SessionError> {
        let Some(pending) = self.pending.clone() else {
            return Ok(());
        };
        let sent = pending.header.round;
        if sent == round.round {
            return Ok(());
        }
        let result = if sent < round.round {
            self.relay.result(sent)?
        } else {
            None
        };
        if let Some(result) = &result {
            if result.audits(&self.node.id()) && self.behavior != Behavior::WithholdReveal {
                if let Some(reveal) = self.node.reveal(sent) {
                    match self.relay.reveal(&reveal) {
                        Ok(_) => self.audits_answered += 1,
                        Err(RelayError::Refused { .. }) => {}
                        Err(err) => return Err(err.into()),
                    }
                }
            }
        }
        if result.is_some_and(|result| result.counts(&pending)) {
            self.rounds_counted += 1;
        } else {
            self.node.rollback(sent);
        }
        self.pending = None;
        self.save()
    }

    fn report(&self, activity: Activity) -> Report {
        Report {
            activity,
            round: self.last_round.clone(),
            rounds_counted: self.rounds_counted,
            audits_answered: self.audits_answered,
        }
    }

    fn save(&self) -> Result<(), SessionError> {
        let Some(path) = &self.state_path else {
            return Ok(());
        };
        let saved = Saved {
            run: self.info.run,
            node: self.node.state(),
            pending: self.pending.clone(),
            rounds_counted: self.rounds_counted,
            audits_answered: self.audits_answered,
        };
        let bytes = postcard::to_stdvec(&saved).map_err(storage)?;
        let partial = path.with_extension("partial");
        std::fs::write(&partial, bytes)
            .and_then(|()| std::fs::rename(&partial, path))
            .map_err(storage)
    }
}
