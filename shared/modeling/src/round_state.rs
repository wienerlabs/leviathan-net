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
//!
//! A node keeps each state it commits to in a file named by the digest
//! (`keep_round_state`), written in the same pass that hashes it.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// How many bytes `write_tensor` writes for a tensor of `shape` and `kind`
/// under `name`.
fn tensor_len(name: &str, shape: &[i64], kind: Kind) -> Result<u64, RoundStateError> {
    kind_tag(kind)?;
    let numel: i64 = shape.iter().product();
    let header = 4 + name.len() + 1 + 4 + 8 * shape.len() + 8;
    Ok(header as u64 + numel as u64 * kind.elt_size_in_bytes() as u64)
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

/// The length of `write_round_state`'s encoding of the state `model` and
/// `distro` train from, from the shapes and dtypes alone. A verifier expects
/// a revealed state of this length before it fetches one, so a node cannot
/// make it download more than the state of the run's model.
pub fn round_state_len(model: &dyn CausalLM, distro: &Distro) -> Result<u64, RoundStateError> {
    let mut len = ROUND_STATE_DOMAIN.len() as u64 + 4 + 4;
    for variable in model.variables() {
        let kind = variable.logical_tensor().kind();
        len += tensor_len(variable.name(), &variable.full_tensor_shape(), kind)?;
    }
    for buffer in distro.error_feedback_variables() {
        let kind = buffer.logical_tensor().kind();
        len += tensor_len(buffer.name(), &buffer.full_tensor_shape(), kind)?;
    }
    Ok(len)
}

/// The file in `dir` that holds the state named by `digest`: the digest in
/// hex, so a state is found by the digest a payload commits to.
pub fn round_state_file(dir: &Path, digest: &[u8; 32]) -> PathBuf {
    dir.join(format!(
        "{}.{ROUND_STATE_EXTENSION}",
        blake3::Hash::from_bytes(*digest).to_hex()
    ))
}

/// The extension of a kept state's file.
pub const ROUND_STATE_EXTENSION: &str = "state";

/// The extension of a state's file while it is written, before it is whole.
pub const PARTIAL_ROUND_STATE_EXTENSION: &str = "partial";

/// A state a node committed to, and where it keeps it.
#[derive(Debug)]
pub struct KeptRoundState {
    pub digest: [u8; 32],
    /// The file that holds the state, or why it could not be written.
    pub file: std::io::Result<PathBuf>,
}

/// Where the encoding goes as it is written: always into the hash, and into
/// the file until the file fails.
enum Sink {
    File(BufWriter<File>),
    Failed(std::io::Error),
}

/// Hashes everything written to it, and copies it to a file. A failing file
/// never fails a write: each rank of a sharded model gathers every tensor
/// together, so the encoding has to run to its end, and the digest is the
/// state's whether or not the file holds it.
struct HashingFile {
    hasher: blake3::Hasher,
    sink: Sink,
}

impl Write for HashingFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.hasher.update(buf);
        if let Sink::File(file) = &mut self.sink {
            if let Err(err) = file.write_all(buf) {
                self.sink = Sink::Failed(err);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Names the files a state is written under before it is whole, so two
/// trainers of one process never write the same one.
static PARTIAL_FILES: AtomicU64 = AtomicU64::new(0);

/// Writes the state `model` and `distro` train a round from into `dir`, in a
/// file named by its digest (`round_state_file`), hashing it in the same pass,
/// and returns the digest and the file. The encoding is written under a
/// temporary name and renamed once whole, so a file under a digest's name
/// always holds that state. A file that cannot be written leaves the digest
/// as it is, and says why. Every rank of a sharded model has to call this or
/// `write_round_state` together.
pub fn keep_round_state(
    dir: &Path,
    model: &dyn CausalLM,
    distro: &Distro,
) -> Result<KeptRoundState, RoundStateError> {
    let partial = dir.join(format!(
        ".{}-{}.{PARTIAL_ROUND_STATE_EXTENSION}",
        std::process::id(),
        PARTIAL_FILES.fetch_add(1, Ordering::Relaxed)
    ));
    let sink = match File::create(&partial) {
        Ok(file) => Sink::File(BufWriter::with_capacity(WRITE_BUFFER_BYTES, file)),
        Err(err) => Sink::Failed(err),
    };
    let mut out = HashingFile {
        hasher: blake3::Hasher::new(),
        sink,
    };
    if let Err(err) = write_round_state(&mut out, model, distro) {
        discard(&partial);
        return Err(err);
    }
    let digest = *out.hasher.finalize().as_bytes();
    let file = match out.sink {
        Sink::File(file) => file
            .into_inner()
            .map_err(|err| err.into_error())
            .and_then(|_| {
                let path = round_state_file(dir, &digest);
                std::fs::rename(&partial, &path).map(|()| path)
            }),
        Sink::Failed(err) => Err(err),
    };
    if file.is_err() {
        discard(&partial);
    }
    Ok(KeptRoundState { digest, file })
}

/// The buffer a state's file is written through.
const WRITE_BUFFER_BYTES: usize = 1 << 20;

/// Removes a state's file that was never made whole. One already gone is
/// fine; one that cannot be removed is logged, and the store sweeps it when
/// it next opens.
fn discard(partial: &Path) {
    match std::fs::remove_file(partial) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => tracing::warn!(
            path = %partial.display(),
            "Could not remove a round state's partial file: {err}"
        ),
    }
}
