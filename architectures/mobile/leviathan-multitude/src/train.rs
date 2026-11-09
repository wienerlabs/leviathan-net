//! The DiLoCo inner loop and the global model's score.

use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use crate::config::RunConfig;
use crate::data::Corpus;
use crate::model::Replica;
use crate::{Error, Result};

/// What a phone's local round produced.
#[derive(Clone, Debug)]
pub struct LocalOutcome {
    pub params: Vec<f32>,
    pub first_loss: f32,
    pub mean_loss: f32,
    pub tokens: u64,
}

/// Train `steps` AdamW steps starting from `start`.
///
/// This is a pure function of its arguments, which is what makes a round
/// auditable: the optimizer starts from zero state every round and the windows
/// come from `data_seed`, so a verifier handed the same inputs computes the
/// same weights. Leviathan's replay audits rest on the same property, that a
/// local round is a function of `(checkpoint, seed, data)`.
pub fn train_local(
    cfg: &RunConfig,
    start: &[f32],
    corpus: &Corpus,
    data_seed: &[u8; 32],
    steps: u16,
    device: &Device,
) -> Result<LocalOutcome> {
    corpus.check_context(cfg.model.context)?;
    let replica = Replica::new(&cfg.model, device)?;
    replica.load(start)?;
    let mut optimizer = AdamW::new(
        replica.vars(),
        ParamsAdamW {
            lr: cfg.inner.lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: cfg.inner.weight_decay,
        },
    )?;
    let (micro_batch, context) = (cfg.inner.micro_batch, cfg.model.context);
    let mut first_loss = f32::NAN;
    let mut total = 0f64;
    for step in 0..steps {
        let (inputs, targets) = corpus.batch(data_seed, step as u32, micro_batch, context);
        let inputs = Tensor::from_vec(inputs, (micro_batch, context), device)?;
        let targets = Tensor::from_vec(targets, (micro_batch, context), device)?;
        let loss = replica.model.loss(&inputs, &targets)?;
        let value = loss.to_scalar::<f32>()?;
        if !value.is_finite() {
            return Err(Error::NonFiniteLoss { step });
        }
        if step == 0 {
            first_loss = value;
        }
        total += value as f64;
        optimizer.backward_step(&loss)?;
    }
    let params = replica.flatten()?;
    if let Some(index) = params.iter().position(|p| !p.is_finite()) {
        return Err(Error::Invalid(format!(
            "weight {index} is not finite after local training"
        )));
    }
    Ok(LocalOutcome {
        params,
        first_loss,
        mean_loss: if steps == 0 {
            f32::NAN
        } else {
            (total / steps as f64) as f32
        },
        tokens: steps as u64 * cfg.tokens_per_step(),
    })
}

/// The DiLoCo outer gradient: where the round started minus where it ended.
pub fn pseudo_gradient(start: &[f32], end: &[f32]) -> Vec<f32> {
    start.iter().zip(end).map(|(s, e)| s - e).collect()
}

/// Mean next-byte loss of `params` on the corpus's fixed evaluation windows.
pub fn eval_loss(
    cfg: &RunConfig,
    params: &[f32],
    corpus: &Corpus,
    windows: usize,
    device: &Device,
) -> Result<f32> {
    corpus.check_context(cfg.model.context)?;
    let replica = Replica::new(&cfg.model, device)?;
    replica.load(params)?;
    let context = cfg.model.context;
    let (inputs, targets) = corpus.eval_windows(context, windows);
    let inputs = Tensor::from_vec(inputs, (windows, context), device)?;
    let targets = Tensor::from_vec(targets, (windows, context), device)?;
    Ok(replica.model.loss(&inputs, &targets)?.to_scalar::<f32>()?)
}
