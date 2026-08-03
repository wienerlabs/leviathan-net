use psyche_core::BatchId;
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
    pub step: u32,
    pub trainer_nonce: u32,
    pub batch_id: BatchId,
    pub distro_results: Vec<SerializedDistroResult>,
}

impl TransmittableDistroResult {
    /// The hash a sender signs and a receiver checks a download against.
    ///
    /// It has to cover everything that decides what gradient comes out of this
    /// payload, not only the tensor bytes: the shape, the dtype and the
    /// encoding say how the bytes are read, and `xshape` and `totalk` are what
    /// `CompressDCT::decompress` rebuilds with. Leaving them out let a sender
    /// commit to one payload and hand over another that decompressed
    /// differently and still verified (wienerlabs/leviathan#15, finding 26).
    ///
    /// Lengths go in ahead of anything variable, so one run of bytes divided
    /// into results one way cannot hash the same as the same run divided
    /// another way.
    ///
    /// Both structs are destructured rather than read field by field: a new
    /// field is then a compile error here, not a silent hole in the commitment.
    pub fn comptue_hash(&self) -> [u8; 32] {
        let TransmittableDistroResult {
            step,
            trainer_nonce,
            batch_id,
            distro_results,
        } = self;

        let mut hasher = Sha256::new();
        hasher.update(step.to_be_bytes());
        hasher.update(trainer_nonce.to_be_bytes());
        hasher.update(batch_id.0.start.to_be_bytes());
        hasher.update(batch_id.0.end.to_be_bytes());
        hasher.update((distro_results.len() as u64).to_be_bytes());

        for result in distro_results {
            let SerializedDistroResult {
                sparse_idx,
                sparse_val,
                xshape,
                totalk,
            } = result;
            sparse_idx.hash_into(&mut hasher);
            sparse_val.hash_into(&mut hasher);
            hasher.update((xshape.len() as u64).to_be_bytes());
            for dim in xshape {
                hasher.update(dim.to_be_bytes());
            }
            hasher.update(totalk.to_be_bytes());
        }
        hasher.finalize().into()
    }
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

pub struct DistroResultIterator<R: Read> {
    reader: BufReader<R>,
    buffer: Vec<u8>,
}

impl<R: Read> DistroResultIterator<R> {
    pub fn new(reader: R) -> Self {
        DistroResultIterator {
            reader: BufReader::new(reader),
            buffer: Vec::new(),
        }
    }
}

impl<R: Read> Iterator for DistroResultIterator<R> {
    type Item = Result<SerializedDistroResult, DistroResultsReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match postcard::take_from_bytes::<SerializedDistroResult>(&self.buffer) {
                Ok((result, remaining)) => {
                    self.buffer = remaining.to_vec();
                    return Some(Ok(result));
                }
                Err(postcard::Error::DeserializeUnexpectedEnd) => {
                    // Not enough data, need to read more
                    let mut chunk = [0u8; 1024]; // Adjust chunk size as needed
                    match self.reader.read(&mut chunk) {
                        Ok(0) if self.buffer.is_empty() => return None, // EOF and no partial data
                        Ok(0) => {
                            return Some(Err(DistroResultsReaderError::Postcard(
                                postcard::Error::DeserializeUnexpectedEnd,
                            )));
                        }
                        Ok(n) => self.buffer.extend_from_slice(&chunk[..n]),
                        Err(e) => return Some(Err(DistroResultsReaderError::Io(e))),
                    }
                }
                Err(e) => return Some(Err(DistroResultsReaderError::Postcard(e))),
            }
        }
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
            step: 7,
            trainer_nonce: 1,
            batch_id: BatchId(ClosedInterval::new(0, 3)),
            distro_results: results,
        }
    }

    const VALUES: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

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
            one_result(&[4], vec![2, 2], 4).comptue_hash(),
            one_result(&[2, 2], vec![2, 2], 4).comptue_hash(),
        );
    }

    /// So do the two fields `CompressDCT::decompress` actually rebuilds with.
    #[test]
    fn xshape_and_totalk_change_the_commitment() {
        let base = one_result(&[4], vec![2, 2], 4).comptue_hash();
        assert_ne!(base, one_result(&[4], vec![4, 1], 4).comptue_hash());
        assert_ne!(base, one_result(&[4], vec![2, 2], 9999).comptue_hash());
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
        assert_ne!(one.comptue_hash(), two.comptue_hash());
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
        assert_ne!(a.comptue_hash(), b.comptue_hash());
    }

    /// The bytes are still covered, and the step and batch bounds still are.
    #[test]
    fn the_fields_that_were_already_covered_still_are() {
        let a = one_result(&[4], vec![2, 2], 4);
        let mut b = one_result(&[4], vec![2, 2], 4);
        b.distro_results[0].sparse_idx = tensor(&[1.0, 2.0, 3.0, 5.0], &[4]);
        assert_ne!(a.comptue_hash(), b.comptue_hash());

        let mut c = one_result(&[4], vec![2, 2], 4);
        c.step = 8;
        assert_ne!(a.comptue_hash(), c.comptue_hash());

        let mut d = one_result(&[4], vec![2, 2], 4);
        d.batch_id = BatchId(ClosedInterval::new(0, 4));
        assert_ne!(a.comptue_hash(), d.comptue_hash());
    }

    /// Identical payloads still agree, or a sender could not commit to its own.
    #[test]
    fn the_same_payload_still_hashes_the_same() {
        assert_eq!(
            one_result(&[4], vec![2, 2], 4).comptue_hash(),
            one_result(&[4], vec![2, 2], 4).comptue_hash(),
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
