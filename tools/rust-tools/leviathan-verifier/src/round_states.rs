//! The state a node of a run trains each round from, as a verifier works out
//! what to expect of it (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).

use anyhow::{anyhow, Result};
use psyche_coordinator::model::LLM;
use psyche_core::OptimizerDefinition;
use psyche_modeling::{auto_model_for_causal_lm_from_pretrained, CausalLM, Distro};
use tch::{Device, Kind};

use crate::{checkpoint_repo_files, published_checkpoint};

/// The run's published checkpoint, loaded on `device` in `kind`, as a node
/// that trains there loads it, with the run's DisTrO and a zeroed buffer: the
/// state a node trains its first round from, and the shapes and dtypes of
/// every state after it.
pub fn load_node_state(
    llm: &LLM,
    device: Device,
    kind: Kind,
) -> Result<(Box<dyn CausalLM>, Distro)> {
    let OptimizerDefinition::Distro {
        compression_decay,
        compression_chunk,
        compression_topk,
        quantize_1bit,
        robust_aggregation,
        weight_decay,
        ..
    } = llm.optimizer
    else {
        return Err(anyhow!("the run does not train with DisTrO"));
    };
    let files = checkpoint_repo_files(&published_checkpoint(llm)?)?;
    let model = auto_model_for_causal_lm_from_pretrained(
        files,
        Some(kind),
        None,
        Some(device),
        None,
        Some(llm.max_seq_len as usize),
    )?;
    model.prepare_for_training();
    let distro = Distro::new(
        &*model,
        compression_decay as f64,
        compression_chunk as i64,
        compression_topk as i64,
        weight_decay.unwrap_or(0.0) as f64,
        robust_aggregation,
        quantize_1bit,
    );
    Ok((model, distro))
}
