//! A small decoder-only transformer, sized for a phone.
//!
//! Byte-level vocabulary, learned positions, pre-norm blocks with RMSNorm,
//! causal multi-head attention and a GELU MLP. Every operation is a candle
//! primitive with a backward pass, so the same code trains on the CPU of an
//! Android phone, on Apple's GPU through the `metal` feature, and on the
//! verifier that replays it.

use candle_core::{DType, Device, Module, Tensor, Var, D};
use candle_nn::{Embedding, Init, Linear, VarBuilder, VarMap};

use crate::config::ModelConfig;
use crate::rng::SplitMix64;
use crate::{Error, Result};

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", Init::Const(1.0))?;
        Ok(Self { weight, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean_square = x.sqr()?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(mean_square + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)
    }
}

struct Block {
    norm1: RmsNorm,
    qkv: Linear,
    proj: Linear,
    norm2: RmsNorm,
    up: Linear,
    down: Linear,
    n_heads: usize,
}

impl Block {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let d = cfg.d_model;
        let hidden = cfg.mlp_ratio * d;
        Ok(Self {
            norm1: RmsNorm::new(d, vb.pp("norm1"))?,
            qkv: candle_nn::linear_no_bias(d, 3 * d, vb.pp("qkv"))?,
            proj: candle_nn::linear_no_bias(d, d, vb.pp("proj"))?,
            norm2: RmsNorm::new(d, vb.pp("norm2"))?,
            up: candle_nn::linear_no_bias(d, hidden, vb.pp("up"))?,
            down: candle_nn::linear_no_bias(hidden, d, vb.pp("down"))?,
            n_heads: cfg.n_heads,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let head_dim = c / self.n_heads;
        let h = self.norm1.forward(x)?;
        let qkv = self.qkv.forward(&h)?;
        let heads = |i: usize| -> candle_core::Result<Tensor> {
            qkv.narrow(2, i * c, c)?
                .reshape((b, t, self.n_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (heads(0)?, heads(1)?, heads(2)?);
        let scores = (q.matmul(&k.t()?.contiguous()?)? * (1.0 / (head_dim as f64).sqrt()))?;
        let mask = mask.narrow(0, 0, t)?.narrow(1, 0, t)?;
        let weights = candle_nn::ops::softmax(&scores.broadcast_add(&mask)?, D::Minus1)?;
        let attended = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;
        let x = (x + self.proj.forward(&attended)?)?;
        let h = self.norm2.forward(&x)?;
        let h = self.down.forward(&self.up.forward(&h)?.gelu()?)?;
        x + h
    }
}

/// The model itself.
pub struct TinyLm {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    norm_f: RmsNorm,
    head: Linear,
    mask: Tensor,
}

impl TinyLm {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let ctx = cfg.context;
        let mask: Vec<f32> = (0..ctx)
            .flat_map(|i| (0..ctx).map(move |j| if j > i { -1e9 } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (ctx, ctx), vb.device())?;
        let blocks = (0..cfg.n_layers)
            .map(|i| Block::new(cfg, vb.pp(format!("blocks.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            tok_emb: candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?,
            pos_emb: candle_nn::embedding(ctx, cfg.d_model, vb.pp("pos_emb"))?,
            blocks,
            norm_f: RmsNorm::new(cfg.d_model, vb.pp("norm_f"))?,
            head: candle_nn::linear_no_bias(cfg.d_model, cfg.vocab_size, vb.pp("head"))?,
            mask,
        })
    }

    /// Logits of shape `(batch, time, vocab)` for token ids of shape
    /// `(batch, time)`.
    pub fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        let (_, t) = ids.dims2()?;
        let positions = Tensor::arange(0u32, t as u32, ids.device())?;
        let mut x = self
            .tok_emb
            .forward(ids)?
            .broadcast_add(&self.pos_emb.forward(&positions)?)?;
        for block in &self.blocks {
            x = block.forward(&x, &self.mask)?;
        }
        self.head.forward(&self.norm_f.forward(&x)?)
    }

    /// Mean next-byte cross-entropy, in nats.
    pub fn loss(&self, ids: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(ids)?;
        let (b, t, v) = logits.dims3()?;
        candle_nn::loss::cross_entropy(&logits.reshape((b * t, v))?, &targets.reshape(b * t)?)
    }
}

/// A model instance plus its parameters in a fixed order.
///
/// `VarMap` keeps variables in a hash map, so the order is pinned here by
/// sorting on the name. Flattened weights, pseudo-gradients, compressed
/// updates and tallies all index parameters in this order.
pub struct Replica {
    pub model: TinyLm,
    vars: Vec<(String, Var)>,
    _varmap: VarMap,
}

impl Replica {
    pub fn new(cfg: &ModelConfig, device: &Device) -> Result<Self> {
        cfg.validate()?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = TinyLm::new(cfg, vb)?;
        let mut vars: Vec<(String, Var)> = varmap
            .data()
            .lock()
            .map_err(|_| Error::Invalid("variable map lock poisoned".into()))?
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Self {
            model,
            vars,
            _varmap: varmap,
        })
    }

    pub fn num_params(&self) -> usize {
        self.vars.iter().map(|(_, var)| var.elem_count()).sum()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.vars.iter().map(|(name, _)| name.as_str())
    }

    pub fn vars(&self) -> Vec<Var> {
        self.vars.iter().map(|(_, var)| var.clone()).collect()
    }

    pub fn flatten(&self) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.num_params());
        for (_, var) in &self.vars {
            out.extend(var.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(out)
    }

    pub fn load(&self, flat: &[f32]) -> Result<()> {
        if flat.len() != self.num_params() {
            return Err(Error::Invalid(format!(
                "weights have {} values, the model has {}",
                flat.len(),
                self.num_params()
            )));
        }
        let mut offset = 0;
        for (_, var) in &self.vars {
            let n = var.elem_count();
            let tensor = Tensor::from_slice(&flat[offset..offset + n], var.dims(), var.device())?;
            var.set(&tensor)?;
            offset += n;
        }
        Ok(())
    }
}

/// Genesis weights: normal(0, 0.02), residual projections scaled down by
/// `sqrt(2 * n_layers)`, norms at one.
///
/// The transcendental functions behind the normal draw are not bit-identical
/// across platforms, so phones never recompute these: the run publishes the
/// genesis weights with their hash.
pub fn genesis_params(cfg: &ModelConfig, seed: u64) -> Result<Vec<f32>> {
    let replica = Replica::new(cfg, &Device::Cpu)?;
    let mut rng = SplitMix64::new(seed);
    let residual_std = 0.02 / (2.0 * cfg.n_layers as f32).sqrt();
    let mut out = Vec::with_capacity(replica.num_params());
    for (name, var) in &replica.vars {
        let n = var.elem_count();
        if name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name == "norm_f.weight"
        {
            out.extend(std::iter::repeat_n(1.0f32, n));
        } else {
            let std = if name.ends_with("proj.weight") || name.ends_with("down.weight") {
                residual_std
            } else {
                0.02
            };
            out.extend((0..n).map(|_| rng.normal() * std));
        }
    }
    Ok(out)
}

/// The CPU everywhere, Apple's GPU when built with `--features metal`.
pub fn default_device() -> Device {
    #[cfg(feature = "metal")]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count_matches_the_config() {
        let cfg = ModelConfig::default();
        let replica = Replica::new(&cfg, &Device::Cpu).unwrap();
        assert_eq!(replica.num_params(), cfg.num_params());
    }

    #[test]
    fn flatten_and_load_round_trip() {
        let cfg = ModelConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            context: 8,
            ..ModelConfig::default()
        };
        let weights = genesis_params(&cfg, 3).unwrap();
        let replica = Replica::new(&cfg, &Device::Cpu).unwrap();
        replica.load(&weights).unwrap();
        assert_eq!(replica.flatten().unwrap(), weights);
        assert!(replica.load(&weights[1..]).is_err());
    }

    #[test]
    fn genesis_is_seeded() {
        let cfg = ModelConfig {
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            context: 8,
            ..ModelConfig::default()
        };
        assert_eq!(
            genesis_params(&cfg, 9).unwrap(),
            genesis_params(&cfg, 9).unwrap()
        );
        assert_ne!(
            genesis_params(&cfg, 9).unwrap(),
            genesis_params(&cfg, 10).unwrap()
        );
    }
}
