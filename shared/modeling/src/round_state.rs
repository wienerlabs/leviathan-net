//! The state a node trains a round from, and its digest
//! (wienerlabs/leviathan-net#89, `docs/ROUND_AUDIT.md`).
//!
//! At round `h` a node trains from its weights, after applying round `h - 2`'s
//! results, and its DisTrO error-feedback buffer, before round `h`'s step adds
//! its gradient. A node commits to that state in the payload it signs, and an
//! audited node reveals it, so a verifier can replay the round. The state is
//! written in one canonical encoding, and its digest is the BLAKE3 hash of
//! that encoding: iroh names a blob by the same hash, so the commitment names
//! a blob any peer can fetch and check by its address alone.

use std::io::Write;

use tch::{Device, Kind, Tensor};
use thiserror::Error;

use crate::{CausalLM, Distro};

#[cfg(not(target_endian = "little"))]
compile_error!("a round's state is encoded little-endian, as tensor bytes are copied");

/// What the encoding of a round's state starts with, so no other bytes a node
/// writes can stand in for it.
pub const ROUND_STATE_DOMAIN: &[u8] = b"leviathan/round-state/v1";

#[derive(Debug, Error)]
pub enum RoundStateError {
    #[error("a {0:?} tensor has no encoding in a round's state")]
    Kind(Kind),
    #[error("writing a round's state failed: {0}")]
    Io(#[from] std::io::Error),
}

/// The tag a tensor's dtype is written under. Only the dtypes a model trains
/// or keeps its buffer in have one.
fn kind_tag(kind: Kind) -> Result<u8, RoundStateError> {
    match kind {
        Kind::Float => Ok(1),
        Kind::BFloat16 => Ok(2),
        Kind::Half => Ok(3),
        Kind::Double => Ok(4),
        other => Err(RoundStateError::Kind(other)),
    }
}

/// Writes `tensor` under `name`: the name's length and bytes, the dtype's
/// tag, the number of dimensions and each, then the length of the data and the
/// data itself, little-endian, from the CPU.
fn write_tensor(out: &mut impl Write, name: &str, tensor: &Tensor) -> Result<(), RoundStateError> {
    let tensor = tensor.to_device(Device::Cpu).contiguous();
    let tag = kind_tag(tensor.kind())?;
    out.write_all(&(name.len() as u32).to_le_bytes())?;
    out.write_all(name.as_bytes())?;
    out.write_all(&[tag])?;
    let shape = tensor.size();
    out.write_all(&(shape.len() as u32).to_le_bytes())?;
    for dim in &shape {
        out.write_all(&dim.to_le_bytes())?;
    }
    let numel = tensor.numel();
    let mut data = vec![0u8; numel * tensor.kind().elt_size_in_bytes()];
    tensor.copy_data_u8(&mut data, numel);
    out.write_all(&(data.len() as u64).to_le_bytes())?;
    out.write_all(&data)?;
    Ok(())
}

/// Writes the state `model` and its optimizer `distro` train a round from: the
/// domain, then the model's variables in the model's order, then the
/// error-feedback buffer in the same order, each count first. Each tensor is
/// gathered whole, so every rank of a sharded model has to call this together.
pub fn write_round_state(
    out: &mut impl Write,
    model: &dyn CausalLM,
    distro: &Distro,
) -> Result<(), RoundStateError> {
    let _no_grad = tch::no_grad_guard();
    out.write_all(ROUND_STATE_DOMAIN)?;
    let variables: Vec<_> = model.variables().collect();
    out.write_all(&(variables.len() as u32).to_le_bytes())?;
    for variable in &variables {
        write_tensor(out, variable.name(), &variable.gather_full_tensor())?;
    }
    let buffers: Vec<_> = distro.error_feedback_variables().collect();
    out.write_all(&(buffers.len() as u32).to_le_bytes())?;
    for buffer in buffers {
        write_tensor(out, buffer.name(), &buffer.gather_full_tensor())?;
    }
    Ok(())
}

/// The digest of the state `model` and `distro` train a round from: the
/// BLAKE3 hash of `write_round_state`'s encoding.
pub fn round_state_digest(
    model: &dyn CausalLM,
    distro: &Distro,
) -> Result<[u8; 32], RoundStateError> {
    let mut hasher = blake3::Hasher::new();
    write_round_state(&mut hasher, model, distro)?;
    Ok(*hasher.finalize().as_bytes())
}
