//! End-to-end: phones train a real transformer, a relay tallies, phones audit
//! each other, and the global model learns.

use candle_core::Device;
use leviathan_multitude::adversary::Behavior;
use leviathan_multitude::aggregate::{aggregate, verify_aggregate, Expectation};
use leviathan_multitude::audit::{audit_contribution, AuditContext, FraudReason, Verdict};
use leviathan_multitude::config::{InnerConfig, ModelConfig, RunConfig};
use leviathan_multitude::data::{data_seed, Corpus};
use leviathan_multitude::train::{eval_loss, train_local};
use leviathan_multitude::{Contribution, GlobalModel, PhoneNode, RoundJob};

const TEXT: &str = include_str!("../../../../shared/client/src/state/prompt_texts/index.json");

fn small() -> RunConfig {
    RunConfig {
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
            max_steps: 6,
            ..InnerConfig::default()
        },
        ..RunConfig::default()
    }
}

struct Fixture {
    cfg: RunConfig,
    corpus: Corpus,
    run: [u8; 32],
    global: GlobalModel,
    device: Device,
}

impl Fixture {
    fn new() -> Self {
        let cfg = small();
        let corpus = Corpus::from_text(TEXT);
        let run = cfg.fingerprint(&corpus.fingerprint());
        let global = GlobalModel::genesis(&cfg, 1651).unwrap();
        Self {
            cfg,
            corpus,
            run,
            global,
            device: Device::Cpu,
        }
    }

    fn job(&self) -> RoundJob<'_> {
        RoundJob {
            cfg: &self.cfg,
            run: self.run,
            round: self.global.round,
            round_seed: self.global.seed,
            global: &self.global.params,
            base_checkpoint: self.global.checkpoint(),
            corpus: &self.corpus,
            device: &self.device,
        }
    }

    fn audit_ctx(&self, band: f32) -> AuditContext<'_> {
        AuditContext {
            cfg: &self.cfg,
            run: self.run,
            round: self.global.round,
            round_seed: self.global.seed,
            global: &self.global.params,
            base_checkpoint: self.global.checkpoint(),
            corpus: &self.corpus,
            band,
            device: &self.device,
        }
    }

    fn phone(&self, n: u8) -> PhoneNode {
        PhoneNode::new([n; 32], self.cfg.model.num_params())
    }

    fn contribute(&self, phone: &mut PhoneNode, behavior: Behavior) -> Contribution {
        phone
            .contribute(&self.job(), self.cfg.inner.max_steps, &behavior)
            .unwrap()
            .0
    }
}

#[test]
fn local_training_reduces_loss() {
    let f = Fixture::new();
    let seed = data_seed(&f.global.seed, &[1u8; 32]);
    let before = eval_loss(&f.cfg, &f.global.params, &f.corpus, 8, &f.device).unwrap();
    let outcome = train_local(&f.cfg, &f.global.params, &f.corpus, &seed, 40, &f.device).unwrap();
    let after = eval_loss(&f.cfg, &outcome.params, &f.corpus, 8, &f.device).unwrap();
    assert!(after < before - 0.5, "loss went from {before} to {after}");
}

#[test]
fn a_local_round_is_a_pure_function_of_its_inputs() {
    let f = Fixture::new();
    let seed = data_seed(&f.global.seed, &[1u8; 32]);
    let a = train_local(&f.cfg, &f.global.params, &f.corpus, &seed, 5, &f.device).unwrap();
    let b = train_local(&f.cfg, &f.global.params, &f.corpus, &seed, 5, &f.device).unwrap();
    assert_eq!(
        a.params, b.params,
        "replay must be bit-identical on the same device"
    );
}

#[test]
fn an_honest_phone_passes_its_audit() {
    let f = Fixture::new();
    let mut phone = f.phone(1);
    let contribution = f.contribute(&mut phone, Behavior::Honest);
    let reveal = phone.reveal(f.global.round);
    let verdict =
        audit_contribution(&f.audit_ctx(0.05), &contribution, reveal.as_ref(), 0).unwrap();
    assert_eq!(
        verdict,
        Verdict::Honest {
            distance: 0.0,
            band: 0.05
        }
    );
}

#[test]
fn honest_audits_still_pass_in_later_rounds_with_a_live_error_buffer() {
    let mut f = Fixture::new();
    let mut phones: Vec<PhoneNode> = (1..=3).map(|n| f.phone(n)).collect();
    for _ in 0..3 {
        let contributions: Vec<Contribution> = phones
            .iter_mut()
            .map(|phone| f.contribute(phone, Behavior::Honest))
            .collect();
        for (index, (phone, contribution)) in phones.iter().zip(&contributions).enumerate() {
            let reveal = phone.reveal(f.global.round);
            let verdict = audit_contribution(
                &f.audit_ctx(0.05),
                contribution,
                reveal.as_ref(),
                index as u64,
            )
            .unwrap();
            assert!(
                !verdict.is_fraud(),
                "round {} phone {index}: {verdict:?}",
                f.global.round
            );
        }
        let expect = Expectation::new(&f.cfg, f.run, f.global.round, f.global.checkpoint());
        let aggregation = aggregate(&expect, &contributions);
        let lr = f.cfg.outer.lr_at(f.global.round);
        f.global.advance(&aggregation.tally, lr);
    }
}

#[test]
fn cheaters_are_convicted() {
    let f = Fixture::new();
    let ctx = f.audit_ctx(0.05);

    let mut flipper = f.phone(2);
    let flipped = f.contribute(&mut flipper, Behavior::SignFlip);
    match audit_contribution(&ctx, &flipped, flipper.reveal(0).as_ref(), 1).unwrap() {
        Verdict::Fraud {
            reason: FraudReason::BeyondBand { distance, .. },
            proof: Some(proof),
        } => {
            assert!(
                (distance - 2.0).abs() < 1e-6,
                "a full sign flip sits at distance 2, got {distance}"
            );
            assert_ne!(proof.committed_hash, proof.replayed_hash);
        }
        other => panic!("sign flip not convicted: {other:?}"),
    }

    let mut rider = f.phone(3);
    let random = f.contribute(&mut rider, Behavior::RandomVotes);
    assert!(
        audit_contribution(&ctx, &random, rider.reveal(0).as_ref(), 2)
            .unwrap()
            .is_fraud()
    );

    let mut lazy = f.phone(4);
    let short = f.contribute(&mut lazy, Behavior::Lazy { real_steps: 1 });
    assert!(audit_contribution(&ctx, &short, lazy.reveal(0).as_ref(), 3)
        .unwrap()
        .is_fraud());
}

#[test]
fn a_phone_that_cannot_back_its_commitment_is_convicted() {
    let f = Fixture::new();
    let ctx = f.audit_ctx(0.05);
    let mut phone = f.phone(5);
    let contribution = f.contribute(&mut phone, Behavior::Honest);

    let silent = audit_contribution(&ctx, &contribution, None, 0).unwrap();
    assert!(matches!(
        silent,
        Verdict::Fraud {
            reason: FraudReason::NoReveal,
            ..
        }
    ));

    let mut forged = phone.reveal(0).unwrap();
    forged.buffer[0] += 1.0;
    let verdict = audit_contribution(&ctx, &contribution, Some(&forged), 0).unwrap();
    assert!(matches!(
        verdict,
        Verdict::Fraud {
            reason: FraudReason::ErrorCommitmentMismatch,
            ..
        }
    ));
}

#[test]
fn the_swarm_learns_and_every_replica_agrees() {
    let mut f = Fixture::new();
    let mut phones: Vec<PhoneNode> = (1..=4).map(|n| f.phone(n)).collect();
    let start = eval_loss(&f.cfg, &f.global.params, &f.corpus, 8, &f.device).unwrap();

    for _ in 0..12 {
        let contributions: Vec<Contribution> = phones
            .iter_mut()
            .map(|phone| f.contribute(phone, Behavior::Honest))
            .collect();
        let expect = Expectation::new(&f.cfg, f.run, f.global.round, f.global.checkpoint());
        let aggregation = aggregate(&expect, &contributions);
        assert_eq!(aggregation.claim.contributors.len(), 4);
        verify_aggregate(
            &expect,
            &contributions,
            &aggregation.claim,
            &aggregation.tally,
        )
        .unwrap();

        // Every phone applies the tally to its own copy; all copies must hash
        // to the same checkpoint.
        let lr = f.cfg.outer.lr_at(f.global.round);
        let mut replica = f.global.params.clone();
        aggregation.tally.apply(&mut replica, lr);
        f.global.advance(&aggregation.tally, lr);
        assert_eq!(
            leviathan_multitude::protocol::weights_hash(&replica),
            f.global.checkpoint()
        );
    }

    let end = eval_loss(&f.cfg, &f.global.params, &f.corpus, 8, &f.device).unwrap();
    assert!(end < start - 0.5, "global loss went from {start} to {end}");
}
