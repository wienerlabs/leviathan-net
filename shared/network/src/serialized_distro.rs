use psyche_core::{payload_commitment_hash, payload_commitment_run, BatchId};
use psyche_modeling::DistroResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fmt,
    io::{BufReader, Read},
    num::TryFromIntError,
};
use tch::Device;
use thiserror::Error;

use crate::serializable_tensor::SerializableTensor;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SerializedDistroResult {
    pub sparse_idx: SerializableTensor,
    pub sparse_val: SerializableTensor,
    pub xshape: Vec<u16>,
    pub totalk: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransmittableDistroResult {
    /// When the epoch the payload was trained in started: the coordinator's
    /// `epoch_state.start_timestamp`. With the step, it names the round, which
    /// the step alone does not once a run restarts from an earlier step
    /// (wienerlabs/leviathan-net#67).
    pub epoch_start: u64,
    pub step: u32,
    pub trainer_nonce: u32,
    pub batch_id: BatchId,
    pub distro_results: Vec<SerializedDistroResult>,
    /// The digest of the state the trainer trained these results from: its
    /// weights and error-feedback buffer as the round began. The commitment
    /// binds it, so an audit can replay the round from the state the trainer
    /// reveals (wienerlabs/leviathan-net#89). None from a trainer that
    /// committed to no state.
    pub state_digest: Option<[u8; 32]>,
}

impl TransmittableDistroResult {
    /// The hash a sender signs and a receiver checks a download against.
    ///
    /// It is `payload_commitment_hash` over the run `run_id`, this payload's
    /// epoch, step, nonce and batch, and the digest of its results. With the
    /// run in it, a signature made in one run convicts in no other, and with
    /// the epoch, one made in a round convicts in no other round
    /// (wienerlabs/leviathan-net#67). The header is kept apart from the results
    /// so the chain can rebuild the hash from a few dozen bytes. It can then
    /// check that a conviction names a payload the target signed for exactly
    /// that step and batch (wienerlabs/leviathan-net#43).
    ///
    /// The struct is destructured rather than read field by field: a new field
    /// is then a compile error here, not a silent hole in the commitment.
    pub fn comptue_hash(&self, run_id: &str) -> Result<[u8; 32], postcard::Error> {
        let TransmittableDistroResult {
            epoch_start,
            step,
            trainer_nonce,
            batch_id,
            distro_results,
            state_digest,
        } = self;
        Ok(payload_commitment_hash(
            &payload_commitment_run(run_id),
            *epoch_start,
            *step,
            *trainer_nonce,
            batch_id.0.start,
            batch_id.0.end,
            &results_digest(distro_results)?,
            state_digest.as_ref(),
        ))
    }

    /// The digest of this payload's results, as its commitment covers them.
    pub fn results_digest(&self) -> Result<[u8; 32], postcard::Error> {
        results_digest(&self.distro_results)
    }
}

/// The digest a payload commitment takes of the results: the sha256 of their
/// wire encoding, which is the same bytes `distro_results_to_bytes` writes and
/// a payload dump holds.
///
/// The encoding carries every field. That covers the bytes, and also the
/// fields that say how the bytes are read: the shape, the dtype, the encoding
/// tag, `xshape` and `totalk`, which are what `CompressDCT::decompress`
/// rebuilds with. A hash that left those out let a sender commit to one
/// payload and hand over another that decompressed differently and still
/// verified (wienerlabs/leviathan#15, finding 26). A field added to a result
/// later is covered without anyone having to remember it.
///
/// Each result's encoding is self-delimiting, so one run of bytes divided into
/// results one way cannot read back as the same run divided another way. And
/// because the digest is of the dump bytes themselves, anyone holding a dump
/// can check it against a conviction recorded on chain with nothing but
/// sha256.
pub fn results_digest(
    distro_results: &[SerializedDistroResult],
) -> Result<[u8; 32], postcard::Error> {
    let mut hasher = Sha256::new();
    for result in distro_results {
        hasher = postcard::to_io(result, hasher)?;
    }
    Ok(hasher.finalize().into())
}

#[derive(Debug, Error)]
pub enum SerializeDistroResultError {
    #[error("Torch error: {0}")]
    Tch(#[from] tch::TchError),
    #[error("Shape had invalid u16: {0}")]
    ShapeInt(#[from] TryFromIntError),
}

impl TryFrom<&DistroResult> for SerializedDistroResult {
    type Error = SerializeDistroResultError;
    fn try_from(value: &DistroResult) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            sparse_idx: (&value.sparse_idx).try_into()?,
            sparse_val: (&value.sparse_val).try_into()?,
            xshape: value
                .xshape
                .iter()
                .map(|&x| u16::try_from(x))
                .collect::<Result<Vec<u16>, _>>()?,
            totalk: value.totalk as u32,
        })
    }
}

/// The most dimensions a parameter tensor is allowed to claim. Real ones have
/// two or three; this leaves room and still bounds the rank.
pub const MAX_XSHAPE_RANK: usize = 8;

/// The most elements a parameter tensor is allowed to claim. `xshape` is a
/// `Vec<u16>` off the wire and `CompressDCT::decompress` opens by allocating
/// `Tensor::zeros(xshape)`, so without a ceiling three entries of 65535 ask the
/// receiver for about a petabyte (wienerlabs/leviathan#15, finding 28).
///
/// A better bound is the shape the receiver's own model says this parameter
/// has, which it knows and never consults. Until that is threaded through, this
/// is the ceiling: generous next to any real parameter, finite next to what a
/// peer can ask for.
pub const MAX_XSHAPE_ELEMENTS: i64 = 1 << 32;

/// Checks a peer-supplied parameter shape before anything allocates from it.
pub fn validate_xshape(xshape: &[u16]) -> Result<Vec<i64>, tch::TchError> {
    if xshape.is_empty() {
        return Err(tch::TchError::Shape("empty parameter shape".to_string()));
    }
    if xshape.len() > MAX_XSHAPE_RANK {
        return Err(tch::TchError::Shape(format!(
            "parameter shape has rank {}, the most allowed is {MAX_XSHAPE_RANK}",
            xshape.len()
        )));
    }
    let mut elements: i64 = 1;
    for dim in xshape {
        if *dim == 0 {
            return Err(tch::TchError::Shape(format!(
                "zero dimension in parameter shape {xshape:?}"
            )));
        }
        elements = elements
            .checked_mul(*dim as i64)
            .filter(|n| *n <= MAX_XSHAPE_ELEMENTS)
            .ok_or_else(|| {
                tch::TchError::Shape(format!(
                    "parameter shape {xshape:?} claims more than {MAX_XSHAPE_ELEMENTS} elements"
                ))
            })?;
    }
    Ok(xshape.iter().map(|x| *x as i64).collect())
}

impl TryFrom<&SerializedDistroResult> for DistroResult {
    type Error = tch::TchError;

    fn try_from(value: &SerializedDistroResult) -> std::result::Result<Self, Self::Error> {
        let mut distro_result = Self {
            sparse_idx: (&value.sparse_idx).try_into()?,
            sparse_val: (&value.sparse_val).try_into()?,
            xshape: validate_xshape(&value.xshape)?,
            totalk: value.totalk as i64,
            stats: None,
        };
        // only pin if we have a device to pin to
        let potential_cuda_device = Device::cuda_if_available();
        if potential_cuda_device.is_cuda() {
            distro_result.sparse_idx = distro_result.sparse_idx.pin_memory();
            distro_result.sparse_val = distro_result.sparse_val.pin_memory();
        }
        Ok(distro_result)
    }
}

pub fn distro_results_to_bytes(
    results: &[SerializedDistroResult],
) -> Result<Vec<u8>, postcard::Error> {
    let mut buf = Vec::new();
    for result in results {
        buf.extend(postcard::to_stdvec(result)?);
    }
    Ok(buf)
}

pub fn distro_results_from_reader<R: Read>(reader: R) -> DistroResultIterator<R> {
    DistroResultIterator::new(reader)
}

pub enum DistroResultsReaderError {
    Postcard(postcard::Error),
    Io(std::io::Error),
}

impl Error for DistroResultsReaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            DistroResultsReaderError::Postcard(err) => Some(err),
            DistroResultsReaderError::Io(err) => Some(err),
        }
    }
}

impl fmt::Display for DistroResultsReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistroResultsReaderError::Postcard(err) => write!(f, "Postcard error: {err}"),
            DistroResultsReaderError::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl fmt::Debug for DistroResultsReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistroResultsReaderError::Postcard(err) => write!(f, "Postcard({err:?})"),
            DistroResultsReaderError::Io(err) => write!(f, "Io({err:?})"),
        }
    }
}

/// Reads payload results one after another from a stream.
///
/// A result is parsed from what has been read so far. Short of it, the
/// iterator reads at least as much again as it holds, so a result that
/// declares a field longer than the stream is parsed a logarithmic number of
/// times, not once per kilobyte. Consumed bytes are skipped by offset, not
/// copied away after every result. A stranger's payload made either quadratic
/// (wienerlabs/leviathan-net#59).
///
/// It ends after the first error. It used to yield the same error again on
/// every call, so a caller that logged errors and read on never finished.
pub struct DistroResultIterator<R: Read> {
    reader: BufReader<R>,
    buffer: Vec<u8>,
    start: usize,
    finished: bool,
}

impl<R: Read> DistroResultIterator<R> {
    pub fn new(reader: R) -> Self {
        DistroResultIterator {
            reader: BufReader::new(reader),
            buffer: Vec::new(),
            start: 0,
            finished: false,
        }
    }

    /// Reads at least as many bytes again as are pending, or a kilobyte.
    /// `Ok(0)` is the end of the stream.
    fn read_more(&mut self) -> std::io::Result<usize> {
        self.buffer.drain(..self.start);
        self.start = 0;
        let want = self.buffer.len().max(1024) as u64;
        (&mut self.reader).take(want).read_to_end(&mut self.buffer)
    }
}

impl<R: Read> Iterator for DistroResultIterator<R> {
    type Item = Result<SerializedDistroResult, DistroResultsReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let item = self.read_result();
        self.finished = !matches!(item, Some(Ok(_)));
        item
    }
}

impl<R: Read> DistroResultIterator<R> {
    fn read_result(&mut self) -> Option<Result<SerializedDistroResult, DistroResultsReaderError>> {
        loop {
            match postcard::take_from_bytes::<SerializedDistroResult>(&self.buffer[self.start..]) {
                Ok((result, remaining)) => {
                    self.start = self.buffer.len() - remaining.len();
                    return Some(Ok(result));
                }
                Err(postcard::Error::DeserializeUnexpectedEnd) => match self.read_more() {
                    Ok(0) if self.buffer.is_empty() => return None,
                    Ok(0) => {
                        return Some(Err(DistroResultsReaderError::Postcard(
                            postcard::Error::DeserializeUnexpectedEnd,
                        )));
                    }
                    Ok(_) => {}
                    Err(e) => return Some(Err(DistroResultsReaderError::Io(e))),
                },
                Err(e) => return Some(Err(DistroResultsReaderError::Postcard(e))),
            }
        }
    }
}

/// A stranger's payload can declare a field far longer than the stream holds.
/// Reading it has to cost time in proportion to the stream, not to its square
/// (wienerlabs/leviathan-net#59).
#[cfg(test)]
mod reader_cost_tests {
    use super::*;
    use crate::serializable_tensor::SerializableTensor;
    use std::time::{Duration, Instant};
    use tch::{Device, Kind, Tensor};

    fn result(xshape: Vec<u16>) -> SerializedDistroResult {
        let tensor = Tensor::from_slice(&[1.0f32, 2.0])
            .to_kind(Kind::Float)
            .to(Device::Cpu);
        let serialized = SerializableTensor::try_from(&tensor).unwrap();
        SerializedDistroResult {
            sparse_idx: serialized.clone(),
            sparse_val: serialized,
            xshape,
            totalk: 2,
        }
    }

    /// A result whose shape lists millions of dimensions, cut short by one
    /// byte, is refused after reading the stream a handful of times. Read a
    /// kilobyte at a time and parsed from the start after each, it took the
    /// verifier minutes.
    #[test]
    fn a_result_cut_short_is_refused_in_time_linear_in_its_length() {
        let mut bytes = postcard::to_allocvec(&result(vec![1; 8_000_000])).unwrap();
        bytes.pop();
        let started = Instant::now();
        let parsed: Vec<_> = distro_results_from_reader(std::io::Cursor::new(bytes)).collect();
        assert!(matches!(
            parsed.as_slice(),
            [Err(DistroResultsReaderError::Postcard(
                postcard::Error::DeserializeUnexpectedEnd
            ))]
        ));
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "took {:?}",
            started.elapsed()
        );
    }

    /// After an error the stream ends, rather than yielding it forever.
    #[test]
    fn the_stream_ends_after_its_first_error() {
        let mut bytes = distro_results_to_bytes(&[result(vec![1, 2])]).unwrap();
        bytes.extend_from_slice(&[0xff; 3]);
        let mut results = distro_results_from_reader(std::io::Cursor::new(bytes));
        assert!(matches!(results.next(), Some(Ok(_))));
        assert!(matches!(results.next(), Some(Err(_))));
        assert!(results.next().is_none());
        assert!(results.next().is_none());
    }

    /// A reader handing over a byte at a time, and interrupting itself
    /// between bytes, as a pipe may, still yields every result.
    #[test]
    fn a_reader_that_trickles_still_yields_every_result() {
        struct Trickle {
            bytes: std::io::Cursor<Vec<u8>>,
            interrupt: bool,
        }
        impl Read for Trickle {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.interrupt = !self.interrupt;
                if self.interrupt {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                let one = buf.len().min(1);
                self.bytes.read(&mut buf[..one])
            }
        }
        let results: Vec<_> = (0..5).map(|n| result(vec![1, n + 1])).collect();
        let trickle = Trickle {
            bytes: std::io::Cursor::new(distro_results_to_bytes(&results).unwrap()),
            interrupt: false,
        };
        let parsed = distro_results_from_reader(trickle)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(parsed, results);
    }

    /// Many results in one stream are read one after another, each once.
    #[test]
    fn many_results_are_each_read_once() {
        let results: Vec<_> = (0..2_000)
            .map(|n| result(vec![1, (n % 7 + 1) as u16]))
            .collect();
        let bytes = distro_results_to_bytes(&results).unwrap();
        let parsed = distro_results_from_reader(std::io::Cursor::new(bytes))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(parsed, results);
    }
}

#[cfg(test)]
mod tests {
    use psyche_modeling::CompressDCT;
    use tch::{Device, Kind, Tensor};

    use crate::serializable_tensor::SerializableTensor;

    #[test]
    fn test_roundtrip_distro_result_1bit() {
        let truth = Tensor::from_slice2(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ])
        .to_kind(Kind::Float)
        .to(Device::Cpu);

        let (sparse_idx, raw_sparse_val, xshape, totalk) = CompressDCT::compress(&truth, i64::MAX);
        // turn raw sparse vals into bools
        let bool_sparse_val = raw_sparse_val.greater(0);

        // and compress to 1bit
        let ser_sparse_val = SerializableTensor::try_from(&bool_sparse_val).unwrap();

        // decompress back into bool tensor
        let sparse_val = Tensor::try_from(&ser_sparse_val).unwrap();

        assert_eq!(sparse_val.kind(), Kind::Bool);

        // when it's quantized to bools, we need to transform it back into -1/+1.
        let sparse_val = sparse_val.to_kind(Kind::Int8) * 2 - 1;

        // finally decompress back to ground truth
        let decompressed_signed = CompressDCT::decompress(
            &sparse_idx,
            &sparse_val,
            &xshape,
            totalk,
            truth.kind(),
            Device::Cpu,
        );
        let signed_truth = truth.sign();

        assert!(decompressed_signed.equal(&signed_truth));
    }
}

/// The commitment hash has to cover everything that decides how the bytes are
/// read, not only the bytes. wienerlabs/leviathan#15, finding 26.
#[cfg(test)]
mod commitment_binding_tests {
    use super::*;
    use psyche_core::{BatchId, ClosedInterval};
    use tch::{Device, Kind, Tensor};

    fn tensor(values: &[f32], dims: &[i64]) -> SerializableTensor {
        let t = Tensor::from_slice(values)
            .to_kind(Kind::Float)
            .to(Device::Cpu)
            .reshape(dims);
        SerializableTensor::try_from(&t).unwrap()
    }

    fn payload(results: Vec<SerializedDistroResult>) -> TransmittableDistroResult {
        TransmittableDistroResult {
            epoch_start: 1_700_000_000,
            step: 7,
            trainer_nonce: 1,
            batch_id: BatchId(ClosedInterval::new(0, 3)),
            distro_results: results,
            state_digest: Some(psyche_core::sha256(b"the state trained from")),
        }
    }

    const VALUES: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    const RUN: &str = "leviathan commitment binding";

    fn one_result(dims: &[i64], xshape: Vec<u16>, totalk: u32) -> TransmittableDistroResult {
        payload(vec![SerializedDistroResult {
            sparse_idx: tensor(&VALUES, dims),
            sparse_val: tensor(&VALUES, dims),
            xshape,
            totalk,
        }])
    }

    /// The same bytes in a different shape are a different commitment, so a
    /// signature over the hash now says which shape the sender meant.
    #[test]
    fn shape_changes_the_commitment() {
        assert_ne!(
            one_result(&[4], vec![2, 2], 4).comptue_hash(RUN).unwrap(),
            one_result(&[2, 2], vec![2, 2], 4)
                .comptue_hash(RUN)
                .unwrap(),
        );
    }

    /// The same payload in another round, here an epoch that started later
    /// because the run restarted from an earlier step, is another commitment
    /// (wienerlabs/leviathan-net#67).
    #[test]
    fn the_epoch_changes_the_commitment() {
        let first = one_result(&[4], vec![2, 2], 4);
        let again = TransmittableDistroResult {
            epoch_start: first.epoch_start + 1,
            ..first.clone()
        };
        assert_ne!(
            first.comptue_hash(RUN).unwrap(),
            again.comptue_hash(RUN).unwrap()
        );
    }

    /// So do the two fields `CompressDCT::decompress` actually rebuilds with.
    #[test]
    fn xshape_and_totalk_change_the_commitment() {
        let base = one_result(&[4], vec![2, 2], 4).comptue_hash(RUN).unwrap();
        assert_ne!(
            base,
            one_result(&[4], vec![4, 1], 4).comptue_hash(RUN).unwrap()
        );
        assert_ne!(
            base,
            one_result(&[4], vec![2, 2], 9999)
                .comptue_hash(RUN)
                .unwrap()
        );
    }

    /// And how a run of bytes is divided into results, which the length
    /// prefixes now pin down.
    #[test]
    fn the_split_between_results_changes_the_commitment() {
        let one = one_result(&[4], vec![2, 2], 4);
        let two = payload(vec![
            SerializedDistroResult {
                sparse_idx: tensor(&VALUES[..2], &[2]),
                sparse_val: tensor(&VALUES[2..], &[2]),
                xshape: vec![2, 2],
                totalk: 4,
            },
            SerializedDistroResult {
                sparse_idx: tensor(&VALUES[..2], &[2]),
                sparse_val: tensor(&VALUES[2..], &[2]),
                xshape: vec![2, 2],
                totalk: 4,
            },
        ]);
        assert_ne!(
            one.comptue_hash(RUN).unwrap(),
            two.comptue_hash(RUN).unwrap()
        );
    }

    /// The encoding tag is covered too: the same buffer read as packed bits and
    /// as raw values is two different tensors and must be two commitments.
    #[test]
    fn the_encoding_changes_the_commitment() {
        let bools = Tensor::from_slice(&[1i64, 0, 1, 0])
            .to_kind(Kind::Bool)
            .to(Device::Cpu);
        let packed = SerializableTensor::try_from(&bools).unwrap();
        let raw = tensor(&VALUES, &[4]);
        let a = payload(vec![SerializedDistroResult {
            sparse_idx: packed,
            sparse_val: raw.clone(),
            xshape: vec![2, 2],
            totalk: 4,
        }]);
        let b = payload(vec![SerializedDistroResult {
            sparse_idx: raw.clone(),
            sparse_val: raw,
            xshape: vec![2, 2],
            totalk: 4,
        }]);
        assert_ne!(a.comptue_hash(RUN).unwrap(), b.comptue_hash(RUN).unwrap());
    }

    /// The bytes are still covered, and the step and batch bounds still are.
    #[test]
    fn the_fields_that_were_already_covered_still_are() {
        let a = one_result(&[4], vec![2, 2], 4);
        let mut b = one_result(&[4], vec![2, 2], 4);
        b.distro_results[0].sparse_idx = tensor(&[1.0, 2.0, 3.0, 5.0], &[4]);
        assert_ne!(a.comptue_hash(RUN).unwrap(), b.comptue_hash(RUN).unwrap());

        let mut c = one_result(&[4], vec![2, 2], 4);
        c.step = 8;
        assert_ne!(a.comptue_hash(RUN).unwrap(), c.comptue_hash(RUN).unwrap());

        let mut d = one_result(&[4], vec![2, 2], 4);
        d.batch_id = BatchId(ClosedInterval::new(0, 4));
        assert_ne!(a.comptue_hash(RUN).unwrap(), d.comptue_hash(RUN).unwrap());
    }

    /// What a sender signs is the payload commitment over this payload's header
    /// and results digest, the same hash the chain rebuilds from those fields.
    #[test]
    fn the_commitment_is_the_payload_commitment_over_header_and_digest() {
        let payload = one_result(&[4], vec![2, 2], 4);
        let digest = results_digest(&payload.distro_results).unwrap();
        assert_eq!(
            payload.comptue_hash(RUN).unwrap(),
            payload_commitment_hash(
                &payload_commitment_run(RUN),
                payload.epoch_start,
                payload.step,
                payload.trainer_nonce,
                payload.batch_id.0.start,
                payload.batch_id.0.end,
                &digest,
                payload.state_digest.as_ref(),
            )
        );
        assert_eq!(payload.results_digest().unwrap(), digest);
    }

    /// The state the payload was trained from is part of what a sender signs
    /// (wienerlabs/leviathan-net#89).
    #[test]
    fn the_state_trained_from_changes_the_commitment() {
        let payload = one_result(&[4], vec![2, 2], 4);
        let base = payload.comptue_hash(RUN).unwrap();
        for other in [Some(psyche_core::sha256(b"another state")), None] {
            let claimed = TransmittableDistroResult {
                state_digest: other,
                ..payload.clone()
            };
            assert_ne!(claimed.comptue_hash(RUN).unwrap(), base, "{other:?}");
        }
    }

    /// The same payload committed in another run is another commitment.
    #[test]
    fn the_run_changes_the_commitment() {
        let payload = one_result(&[4], vec![2, 2], 4);
        assert_ne!(
            payload.comptue_hash(RUN).unwrap(),
            payload.comptue_hash("another run").unwrap()
        );
    }

    /// The results digest is the sha256 of the dump bytes, so a dump on disk
    /// can be checked against a conviction with sha256 alone.
    #[test]
    fn the_results_digest_is_the_sha256_of_the_dump_bytes() {
        let payload = payload(vec![
            one_result(&[4], vec![2, 2], 4).distro_results.remove(0),
            one_result(&[2, 2], vec![4, 1], 4).distro_results.remove(0),
        ]);
        let dump = distro_results_to_bytes(&payload.distro_results).unwrap();
        assert_eq!(
            payload.results_digest().unwrap(),
            <[u8; 32]>::from(Sha256::digest(&dump))
        );
    }

    /// The nonce is in the header the signature covers, not in the digest.
    #[test]
    fn the_nonce_changes_the_commitment() {
        let a = one_result(&[4], vec![2, 2], 4);
        let mut b = one_result(&[4], vec![2, 2], 4);
        b.trainer_nonce += 1;
        assert_ne!(a.comptue_hash(RUN).unwrap(), b.comptue_hash(RUN).unwrap());
        assert_eq!(a.results_digest().unwrap(), b.results_digest().unwrap());
    }

    /// Identical payloads still agree, or a sender could not commit to its own.
    #[test]
    fn the_same_payload_still_hashes_the_same() {
        assert_eq!(
            one_result(&[4], vec![2, 2], 4).comptue_hash(RUN).unwrap(),
            one_result(&[4], vec![2, 2], 4).comptue_hash(RUN).unwrap(),
        );
    }
}

/// `xshape` is a peer-supplied allocation size, so it is bounded before
/// `CompressDCT::decompress` allocates from it. wienerlabs/leviathan#15,
/// finding 28.
#[cfg(test)]
mod xshape_bound_tests {
    use super::*;

    #[test]
    fn a_shape_that_asks_for_a_petabyte_is_refused() {
        let err = validate_xshape(&[65535, 65535, 65535]).expect_err("2.8e14 elements");
        assert!(format!("{err:?}").contains("more than"));
    }

    #[test]
    fn rank_is_bounded() {
        assert!(validate_xshape(&[2; MAX_XSHAPE_RANK]).is_ok());
        assert!(validate_xshape(&[2; MAX_XSHAPE_RANK + 1]).is_err());
    }

    #[test]
    fn degenerate_shapes_are_refused() {
        assert!(validate_xshape(&[]).is_err(), "no shape at all");
        assert!(validate_xshape(&[4, 0, 4]).is_err(), "a zero dimension");
    }

    #[test]
    fn the_shapes_a_real_parameter_has_still_pass() {
        // A 4096-wide linear layer, an embedding row, a bias, a conv kernel.
        for shape in [
            vec![4096u16, 4096],
            vec![32000, 512],
            vec![4096],
            vec![64, 3, 7, 7],
        ] {
            assert!(
                validate_xshape(&shape).is_ok(),
                "{shape:?} is an ordinary parameter shape"
            );
        }
    }
}
