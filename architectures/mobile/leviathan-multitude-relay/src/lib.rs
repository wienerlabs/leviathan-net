use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use leviathan_multitude::aggregate::{aggregate, Expectation};
use leviathan_multitude::audit::{audit_contribution, AuditContext, FraudReason, Verdict};
use leviathan_multitude::lottery::select_audits;
use leviathan_multitude::protocol::{address, short_address, weights_hash};
use leviathan_multitude::rng::sha256_parts;
use leviathan_multitude::train::eval_loss;
use leviathan_multitude::wire::{RoundInfo, RoundResult, RunInfo};
use leviathan_multitude::{Contribution, Corpus, Device, ErrorReveal, GlobalModel, RunConfig};
use serde::Serialize;

pub mod http;

const EVAL_WINDOWS: usize = 16;
const DISTINCT_PER_ROUND: usize = 2;

#[derive(Clone, Debug)]
pub struct Options {
    pub reveal_rounds: u32,
    pub max_phones: usize,
    pub enforce_band: bool,
    pub keep_rounds: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            reveal_rounds: 30,
            max_phones: 64,
            enforce_band: false,
            keep_rounds: 64,
        }
    }
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum Reject {
    #[error("{0}")]
    Invalid(String),
    #[error("convicted: {0}")]
    Convicted(String),
    #[error("the run already has its {0} phones")]
    Full(usize),
    #[error("no audit of this phone is open for round {0}")]
    NoAudit(u32),
}

#[derive(Default)]
struct Standing {
    counted: u32,
    convicted: Option<String>,
}

struct Published {
    result: RoundResult,
    contributions: Vec<Contribution>,
}

struct OpenAudit {
    round: u32,
    contribution: Contribution,
    seed: [u8; 32],
    base: [u8; 32],
    params: Arc<Vec<f32>>,
    due: u32,
}

#[derive(Debug, Serialize)]
pub struct Status {
    pub run_id: String,
    pub round: u32,
    pub global_loss: f32,
    pub phones_last_round: u32,
    pub open_audits: usize,
    pub enforce_band: bool,
    pub phones: Vec<PhoneStatus>,
}

#[derive(Debug, Serialize)]
pub struct PhoneStatus {
    pub address: String,
    pub rounds_counted: u32,
    pub convicted: Option<String>,
}

pub struct Relay {
    info: RunInfo,
    corpus: Corpus,
    corpus_bytes: Vec<u8>,
    global: GlobalModel,
    checkpoint: [u8; 32],
    global_loss: f32,
    phones_last_round: u32,
    device: Device,
    opts: Options,
    deadline: Instant,
    pool: BTreeMap<[u8; 32], Vec<Contribution>>,
    standing: BTreeMap<[u8; 32], Standing>,
    published: BTreeMap<u32, Published>,
    audits: Vec<OpenAudit>,
}

fn round_length(cfg: &RunConfig) -> Duration {
    Duration::from_secs_f32(cfg.round_seconds)
}

impl Relay {
    pub fn new(
        cfg: RunConfig,
        corpus_bytes: Vec<u8>,
        genesis_seed: u64,
        device: Device,
        mut opts: Options,
    ) -> leviathan_multitude::Result<Self> {
        if !(cfg.round_seconds.is_finite() && cfg.round_seconds >= 1.0) {
            return Err(leviathan_multitude::Error::Invalid(
                "round_seconds must be at least 1".into(),
            ));
        }
        opts.keep_rounds = opts.keep_rounds.max(opts.reveal_rounds as usize + 1);
        let corpus = Corpus::from_bytes(corpus_bytes.clone());
        corpus.check_context(cfg.model.context)?;
        let global = GlobalModel::genesis(&cfg, genesis_seed)?;
        let checkpoint = global.checkpoint();
        let info = RunInfo {
            run: sha256_parts(&[
                b"multitude/relay-run",
                &cfg.fingerprint(&corpus.fingerprint()),
                &checkpoint,
            ]),
            corpus: corpus.fingerprint(),
            cfg,
        };
        let global_loss = eval_loss(&info.cfg, &global.params, &corpus, EVAL_WINDOWS, &device)?;
        let deadline = Instant::now() + round_length(&info.cfg);
        Ok(Self {
            info,
            corpus,
            corpus_bytes,
            global,
            checkpoint,
            global_loss,
            phones_last_round: 0,
            device,
            opts,
            deadline,
            pool: BTreeMap::new(),
            standing: BTreeMap::new(),
            published: BTreeMap::new(),
            audits: Vec::new(),
        })
    }

    pub fn info(&self) -> &RunInfo {
        &self.info
    }

    pub fn corpus(&self) -> &[u8] {
        &self.corpus_bytes
    }

    pub fn params(&self) -> (u32, &[f32]) {
        (self.global.round, &self.global.params)
    }

    pub fn round(&self, now: Instant) -> RoundInfo {
        RoundInfo {
            run: self.info.run,
            round: self.global.round,
            seed: self.global.seed,
            base_checkpoint: self.checkpoint,
            closes_in_ms: self.deadline.saturating_duration_since(now).as_millis() as u64,
            global_loss: self.global_loss,
            phones_last_round: self.phones_last_round,
        }
    }

    pub fn result(&self, round: u32) -> Option<&RoundResult> {
        self.published.get(&round).map(|p| &p.result)
    }

    pub fn contributions(&self, round: u32) -> Option<&[Contribution]> {
        self.published
            .get(&round)
            .map(|p| p.contributions.as_slice())
    }

    fn expectation(&self) -> Expectation {
        Expectation::new(
            &self.info.cfg,
            self.info.run,
            self.global.round,
            self.checkpoint,
        )
    }

    pub fn submit(&mut self, contribution: Contribution) -> Result<(), Reject> {
        contribution
            .verify_signature()
            .map_err(|err| Reject::Invalid(err.to_string()))?;
        let phone = contribution.header.phone;
        if let Some(reason) = self.standing.get(&phone).and_then(|s| s.convicted.clone()) {
            return Err(Reject::Convicted(reason));
        }
        self.expectation()
            .check(&contribution)
            .map_err(|reason| Reject::Invalid(format!("contribution {reason}")))?;
        if !self.standing.contains_key(&phone) && self.standing.len() >= self.opts.max_phones {
            return Err(Reject::Full(self.opts.max_phones));
        }
        self.standing.entry(phone).or_default();
        let sent = self.pool.entry(phone).or_default();
        let digest = contribution.digest();
        if sent.iter().any(|c| c.digest() == digest) {
            return Ok(());
        }
        if sent.len() >= DISTINCT_PER_ROUND {
            return Err(Reject::Invalid(
                "already sent two contributions this round".into(),
            ));
        }
        sent.push(contribution);
        Ok(())
    }

    pub fn close_if_due(&mut self, now: Instant) -> leviathan_multitude::Result<Vec<String>> {
        if now < self.deadline {
            return Ok(Vec::new());
        }
        self.deadline = now + round_length(&self.info.cfg);
        if self.pool.is_empty() {
            return Ok(Vec::new());
        }

        let round = self.global.round;
        let expect = self.expectation();
        let pool: Vec<Contribution> = std::mem::take(&mut self.pool)
            .into_values()
            .flatten()
            .collect();
        let aggregation = aggregate(&expect, &pool);
        let mut log = Vec::new();
        for phone in &aggregation.equivocators {
            log.push(self.convict(phone, format!("signed two contributions for round {round}")));
        }

        let params = Arc::new(self.global.params.clone());
        let seed = self.global.seed;
        let base = self.checkpoint;
        self.global
            .advance(&aggregation.tally, self.info.cfg.outer.lr_at(round));
        self.checkpoint = self.global.checkpoint();
        self.global_loss = eval_loss(
            &self.info.cfg,
            &self.global.params,
            &self.corpus,
            EVAL_WINDOWS,
            &self.device,
        )?;

        let counted = aggregation.claim.contributors.clone();
        self.phones_last_round = counted.len() as u32;
        for phone in &counted {
            self.standing.entry(*phone).or_default().counted += 1;
        }

        let mut targets: Vec<usize> = select_audits(
            &self.global.seed,
            counted.len(),
            self.info.cfg.audit.verification_percent,
        )
        .into_iter()
        .map(|assignment| assignment.target)
        .collect();
        targets.sort_unstable();
        targets.dedup();
        let due = round + self.opts.reveal_rounds;
        for &index in &targets {
            let digest = aggregation.digests[index];
            if let Some(contribution) = pool.iter().find(|c| c.digest() == digest) {
                self.audits.push(OpenAudit {
                    round,
                    contribution: contribution.clone(),
                    seed,
                    base,
                    params: params.clone(),
                    due,
                });
            }
        }
        log.push(format!(
            "r{round:03} | {} counted | global loss {:.3} | audits {}",
            counted.len(),
            self.global_loss,
            targets.len()
        ));
        self.published.insert(
            round,
            Published {
                result: RoundResult {
                    claim: aggregation.claim,
                    digests: aggregation.digests,
                    audit_targets: targets.iter().map(|&i| counted[i]).collect(),
                },
                contributions: pool,
            },
        );
        while self.published.len() > self.opts.keep_rounds {
            self.published.pop_first();
        }

        let next = self.global.round;
        let (expired, open): (Vec<OpenAudit>, Vec<OpenAudit>) = std::mem::take(&mut self.audits)
            .into_iter()
            .partition(|audit| audit.due < next);
        self.audits = open;
        for audit in expired {
            log.push(self.convict(
                &audit.contribution.header.phone,
                format!("did not answer the audit of round {}", audit.round),
            ));
        }
        Ok(log)
    }

    pub fn reveal(&mut self, reveal: ErrorReveal) -> Result<String, Reject> {
        let Some(index) = self.audits.iter().position(|audit| {
            audit.round == reveal.round && audit.contribution.header.phone == reveal.phone
        }) else {
            return Err(Reject::NoAudit(reveal.round));
        };
        let audit = &self.audits[index];
        if reveal.buffer.len() != audit.params.len()
            || weights_hash(&reveal.buffer) != audit.contribution.header.error_commitment
        {
            return Err(Reject::Invalid(
                "the revealed buffer does not match its commitment".into(),
            ));
        }
        let ctx = AuditContext {
            cfg: &self.info.cfg,
            run: self.info.run,
            round: audit.round,
            round_seed: audit.seed,
            global: &audit.params,
            base_checkpoint: audit.base,
            corpus: &self.corpus,
            band: self.info.cfg.audit.band,
            device: &self.device,
        };
        let verdict = audit_contribution(&ctx, &audit.contribution, Some(&reveal), index as u64)
            .map_err(|err| Reject::Invalid(err.to_string()))?;
        self.audits.remove(index);

        let target = short_address(&reveal.phone);
        let round = reveal.round;
        Ok(match verdict {
            Verdict::Honest { distance, band } => {
                format!("r{round:03} audit {target}: honest, distance {distance:.4} against band {band:.4}")
            }
            Verdict::Fraud {
                reason: FraudReason::BeyondBand { distance, band },
                ..
            } if !self.opts.enforce_band => format!(
                "r{round:03} audit {target}: distance {distance:.4} against band {band:.4}, dry run, not convicted"
            ),
            Verdict::Fraud { reason, .. } => {
                self.convict(&reveal.phone, format!("round {round}: {reason}"))
            }
        })
    }

    fn convict(&mut self, phone: &[u8; 32], reason: String) -> String {
        self.audits
            .retain(|audit| audit.contribution.header.phone != *phone);
        let standing = self.standing.entry(*phone).or_default();
        if standing.convicted.is_none() {
            standing.convicted = Some(reason.clone());
        }
        format!("convicted {}: {reason}", short_address(phone))
    }

    pub fn status(&self) -> Status {
        Status {
            run_id: self.info.cfg.run_id.clone(),
            round: self.global.round,
            global_loss: self.global_loss,
            phones_last_round: self.phones_last_round,
            open_audits: self.audits.len(),
            enforce_band: self.opts.enforce_band,
            phones: self
                .standing
                .iter()
                .map(|(phone, standing)| PhoneStatus {
                    address: address(phone),
                    rounds_counted: standing.counted,
                    convicted: standing.convicted.clone(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use leviathan_multitude::adversary::Behavior;
    use leviathan_multitude::config::{AuditConfig, InnerConfig, ModelConfig};
    use leviathan_multitude::{PhoneNode, RoundJob};

    use super::*;

    const TEXT: &str = include_str!("../../../../shared/client/src/state/prompt_texts/index.json");

    fn relay() -> Relay {
        let cfg = RunConfig {
            model: ModelConfig {
                d_model: 32,
                n_heads: 2,
                n_layers: 1,
                context: 32,
                ..ModelConfig::default()
            },
            inner: InnerConfig {
                micro_batch: 4,
                min_steps: 2,
                max_steps: 4,
                ..InnerConfig::default()
            },
            audit: AuditConfig {
                verification_percent: 100,
                ..AuditConfig::default()
            },
            ..RunConfig::default()
        };
        let opts = Options {
            reveal_rounds: 1,
            ..Options::default()
        };
        Relay::new(cfg, TEXT.as_bytes().to_vec(), 7, Device::Cpu, opts).unwrap()
    }

    fn signed(relay: &Relay, phone: &mut PhoneNode) -> Contribution {
        let round = relay.round(Instant::now());
        let params = relay.params().1.to_vec();
        let corpus = Corpus::from_bytes(relay.corpus().to_vec());
        let job = RoundJob {
            cfg: &relay.info.cfg,
            run: relay.info.run,
            round: round.round,
            round_seed: round.seed,
            global: &params,
            base_checkpoint: round.base_checkpoint,
            corpus: &corpus,
            device: &Device::Cpu,
        };
        phone.contribute(&job, 2, &Behavior::Honest).unwrap().0
    }

    fn contribute(relay: &mut Relay, phone: &mut PhoneNode) {
        let contribution = signed(relay, phone);
        relay.submit(contribution).unwrap();
    }

    fn close(relay: &mut Relay) -> Vec<String> {
        let due = relay.deadline;
        relay.close_if_due(due).unwrap()
    }

    fn convicted(relay: &Relay, phone: &PhoneNode) -> Option<String> {
        relay
            .status()
            .phones
            .into_iter()
            .find(|p| p.address == phone.address())
            .and_then(|p| p.convicted)
    }

    #[test]
    fn a_round_nobody_trained_stays_open() {
        let mut relay = relay();
        assert!(close(&mut relay).is_empty());
        assert_eq!(relay.round(Instant::now()).round, 0);
        assert!(relay.round(Instant::now()).closes_in_ms > 0);
    }

    #[test]
    fn a_forged_reveal_is_refused_and_silence_is_convicted() {
        let mut relay = relay();
        let num_params = relay.params().1.len();
        let mut answers = PhoneNode::new([1u8; 32], num_params);
        let mut silent = PhoneNode::new([2u8; 32], num_params);
        contribute(&mut relay, &mut answers);
        contribute(&mut relay, &mut silent);
        close(&mut relay);
        let result = relay.result(0).unwrap();
        assert!(result.audits(&answers.id()) && result.audits(&silent.id()));

        let mut forged = answers.reveal(0).unwrap();
        forged.buffer[0] += 1.0;
        assert!(matches!(relay.reveal(forged), Err(Reject::Invalid(_))));
        assert_eq!(convicted(&relay, &answers), None);

        let honest = relay.reveal(answers.reveal(0).unwrap()).unwrap();
        assert!(honest.contains("honest"), "{honest}");
        assert_eq!(
            relay.reveal(answers.reveal(0).unwrap()),
            Err(Reject::NoAudit(0))
        );

        contribute(&mut relay, &mut answers);
        let log = close(&mut relay);
        assert!(
            log.iter().any(|line| line.contains("did not answer")),
            "{log:?}"
        );
        assert!(convicted(&relay, &silent).unwrap().contains("round 0"));
        assert_eq!(convicted(&relay, &answers), None);
        let late = signed(&relay, &mut silent);
        assert_eq!(
            relay.submit(late),
            Err(Reject::Convicted(
                "did not answer the audit of round 0".into()
            ))
        );
    }
}
