use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::convert::TryFrom;
use tch::{Device, Kind, TchError, Tensor};

use crate::serializable_kind::SerializableKind;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum SerializableTensorData {
    Full(#[serde(with = "serde_bytes")] Vec<u8>),
    OneBit(#[serde(with = "serde_bytes")] Vec<u8>),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SerializableTensor {
    dims: Vec<i64>,
    kind: SerializableKind,
    requires_grad: bool,
    data: SerializableTensorData,
}

impl SerializableTensor {
    pub fn raw_tensor_data(&self) -> &[u8] {
        match &self.data {
            SerializableTensorData::Full(items) => items,
            SerializableTensorData::OneBit(items) => items,
        }
    }

    /// Feeds every field into a commitment hash.
    ///
    /// The bytes alone do not say how they are to be read, and the fields that
    /// do say - the shape, the dtype, which of the two encodings this is - decide
    /// what gradient comes out the other end. A commitment that leaves them out
    /// does not bind what the receiver computes (wienerlabs/leviathan#15,
    /// finding 26).
    ///
    /// The destructuring is the point: adding a field to this struct without
    /// deciding whether it belongs in the commitment will not compile.
    pub fn hash_into(&self, hasher: &mut Sha256) {
        let SerializableTensor {
            dims,
            kind,
            requires_grad,
            data,
        } = self;

        hasher.update((dims.len() as u64).to_be_bytes());
        for dim in dims {
            hasher.update(dim.to_be_bytes());
        }
        hasher.update([crate::serializable_kind::kind_to_u8(&kind.clone().into_inner())]);
        hasher.update([u8::from(*requires_grad)]);

        // The encoding tag matters as much as the bytes: the same buffer read as
        // packed bits and as raw values is two different tensors.
        let (tag, bytes) = match data {
            SerializableTensorData::Full(bytes) => (0u8, bytes),
            SerializableTensorData::OneBit(bytes) => (1u8, bytes),
        };
        hasher.update([tag]);
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
}

impl TryFrom<&Tensor> for SerializableTensor {
    type Error = TchError;

    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        // tensor must be on cpu & contiguous to read as &[u8]
        let tensor = match (tensor.device(), tensor.is_contiguous()) {
            (Device::Cpu, true) => tensor.shallow_clone(),
            (Device::Cpu, false) => tensor.contiguous(),
            (_, true) => tensor.to_device(Device::Cpu),
            (_, false) => tensor.to_device(Device::Cpu).contiguous(),
        };

        debug_assert!(tensor.is_contiguous());
        debug_assert_eq!(tensor.device(), Device::Cpu);

        let dims = tensor.size();
        let kind = tensor.kind().into();
        let requires_grad = tensor.requires_grad();

        fn tensor_to_bytes(tensor: &Tensor) -> Vec<u8> {
            let num_elements = tensor.numel();
            let elt_size = tensor.kind().elt_size_in_bytes();
            let mut data = vec![0u8; num_elements * elt_size];
            tensor.copy_data_u8(&mut data, num_elements);
            data
        }

        let data = if tensor.kind() == Kind::Bool {
            // this pad and reshape operation is equivalent to taking a tensor of
            // [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1]
            // and transforming it into [0b01101110, 0b01101111]
            let n_bits = tensor.numel() as i64;
            let n_bytes = (n_bits + 7) / 8;

            // first we pad lengths to multiple of 8, since final array should be &[u8]
            let pad_size = (8 - (n_bits % 8)) % 8;
            let padded = if pad_size > 0 {
                Tensor::f_pad(&tensor.flatten(0, -1), [0, pad_size], "constant", Some(0.0))?
            } else {
                tensor.flatten(0, -1)
            };

            // then we reshape to (..., N/8, 8)
            let reshaped = padded.reshape([n_bytes, 8]);

            // make a tensor of bit weights (LSB first)
            // which we will multiply with each value consecutively
            // to create packable bits
            let bit_weights = Tensor::from_slice(&[1u8, 2, 4, 8, 16, 32, 64, 128])
                .to_device(tensor.device())
                .to_kind(Kind::Uint8);

            // multiply and sum to pack bits
            let packed = (reshaped.to_kind(Kind::Uint8) * bit_weights).sum_dim_intlist(
                -1,
                false,
                Kind::Uint8,
            );

            SerializableTensorData::OneBit(tensor_to_bytes(&packed))
        } else {
            SerializableTensorData::Full(tensor_to_bytes(&tensor))
        };

        Ok(SerializableTensor {
            dims,
            kind,
            requires_grad,
            data,
        })
    }
}

/// How many elements `dims` describes, or an error if it describes something a
/// tensor cannot be.
///
/// Every one of these numbers arrives from a peer, so this is checked
/// arithmetic: a negative dimension is not a shape, and a product that leaves
/// `i64` is not a size. Neither can be handed on to libtorch, which takes the
/// element count on trust and reads that far.
fn element_count(dims: &[i64]) -> Result<i64, TchError> {
    let mut count: i64 = 1;
    for dim in dims {
        if *dim < 0 {
            return Err(TchError::Shape(format!(
                "negative dimension {dim} in shape {dims:?}"
            )));
        }
        count = count.checked_mul(*dim).ok_or_else(|| {
            TchError::Shape(format!("shape {dims:?} overflows the element count"))
        })?;
    }
    Ok(count)
}

impl TryFrom<&SerializableTensor> for Tensor {
    type Error = TchError;

    fn try_from(value: &SerializableTensor) -> Result<Self, Self::Error> {
        let elements = element_count(&value.dims)?;
        let tensor = match &value.data {
            SerializableTensorData::Full(data) => {
                // `f_from_data_size` passes the pointer and drops the length, and
                // libtorch then copies `elements * elt_size` bytes from it. The
                // buffer has to be measured here or not at all.
                let kind: Kind = (&value.kind).into();
                let needed = elements
                    .checked_mul(kind.elt_size_in_bytes() as i64)
                    .ok_or_else(|| {
                        TchError::Shape(format!(
                            "shape {:?} of {kind:?} overflows a byte count",
                            value.dims
                        ))
                    })?;
                if data.len() as i64 != needed {
                    return Err(TchError::Shape(format!(
                        "shape {:?} of {kind:?} needs {needed} bytes, got {}",
                        value.dims,
                        data.len()
                    )));
                }
                Tensor::f_from_data_size(data, &value.dims, kind)?
            }
            SerializableTensorData::OneBit(bytes) => {
                // One bit per element, so the buffer has to hold at least that
                // many bits. The rounding up is the sender's padding.
                let needed_bytes = elements
                    .checked_add(7)
                    .ok_or_else(|| TchError::Shape("bit count overflows".to_string()))?
                    / 8;
                if (bytes.len() as i64) < needed_bytes {
                    return Err(TchError::Shape(format!(
                        "shape {:?} needs {needed_bytes} packed bytes, got {}",
                        value.dims,
                        bytes.len()
                    )));
                }

                // packed bytes are just a flat 1d slice of bits
                let packed = Tensor::f_from_slice(bytes)?.f_to_kind(Kind::Uint8)?;

                // make a tensor of bit weights (LSB first) to unpack
                let bit_weights =
                    Tensor::f_from_slice(&[1u8, 2, 4, 8, 16, 32, 64, 128])?.f_to_kind(Kind::Uint8)?;

                // reshape packed to [..., 1] for broadcasting
                let reshaped_packed = packed.f_reshape([-1, 1])?;

                // unpack bits
                let bits = reshaped_packed
                    .f_bitwise_and_tensor(&bit_weights)?
                    .f_to_kind(Kind::Bool)?;

                // flatten, select needed bits, and reshape
                let flat_bits = bits.f_flatten(0, -1)?;
                let needed_bits = flat_bits.f_slice(0, 0, elements, 1)?;

                needed_bits.f_reshape(&value.dims)?
            }
        };

        Ok(if value.requires_grad {
            tensor.f_set_requires_grad(true)?
        } else {
            tensor
        })
    }
}

#[cfg(test)]
mod tests {
    use psyche_modeling::set_torch_rng_seed;
    use tch::{Device, Kind, Tensor};

    use crate::serializable_tensor::SerializableTensor;

    #[test]
    fn test_roundtrip_tensor1d() {
        let truth = Tensor::from_slice(&[0.6533, 0.2706, -0.2706, -0.6533])
            .to_kind(Kind::Float)
            .to(Device::Cpu);

        let serializable = SerializableTensor::try_from(&truth).unwrap();

        let result = Tensor::try_from(&serializable).unwrap();

        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_roundtrip_tensor2d() {
        let truth = Tensor::from_slice2(&[
            [0.6533, 0.2706, -0.2706, -0.6533],
            [230.4230, -25774.5, 0.0, 25.0],
        ])
        .to_kind(Kind::Float)
        .to(Device::Cpu);

        let serializable = SerializableTensor::try_from(&truth).unwrap();

        let result = Tensor::try_from(&serializable).unwrap();

        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_roundtrip_tensor_manyd() {
        set_torch_rng_seed();

        // some random # of dimensions
        let dims = [2, 16, 2, 25, 2, 215, 6];

        // rand between -500 and +500
        let truth = (Tensor::rand(dims, (Kind::Float, Device::Cpu)) - 0.5) * 1000;

        // roundtrip
        let serializable = SerializableTensor::try_from(&truth).unwrap();
        let result = Tensor::try_from(&serializable).unwrap();

        // roundtripped bools === original bools
        assert!(result.equal(&truth));
    }

    #[test]
    fn test_roundtrip_bool_tensor1d() {
        let truth = Tensor::from_slice(&[1, 0, 0, 1, 0, 1, 1, 1])
            .to_kind(Kind::Bool)
            .to(Device::Cpu);

        let serializable = SerializableTensor::try_from(&truth).unwrap();

        let result = Tensor::try_from(&serializable).unwrap();

        assert!(result.equal(&truth));
    }

    #[test]
    fn test_roundtrip_bool_tensor2d() {
        let truth = Tensor::from_slice2(&[[1, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1]])
            .to_kind(Kind::Bool)
            .to(Device::Cpu);

        let serializable = SerializableTensor::try_from(&truth).unwrap();
        let result = Tensor::try_from(&serializable).unwrap();

        assert!(result.equal(&truth));
    }

    #[test]
    fn test_roundtrip_bool_tensor_manyd() {
        set_torch_rng_seed();

        // some random # of dimensions
        let dims = [2, 16, 2, 25, 2, 215, 6];

        // rand between -0.5 and +0.5
        let rand_tensor = Tensor::rand(dims, (Kind::Float, Device::Cpu)) - 0.5;

        // make a baseline that's true and false
        let truth = rand_tensor.signbit();
        // roundtrip
        let serializable = SerializableTensor::try_from(&truth).unwrap();
        let result = Tensor::try_from(&serializable).unwrap();

        // roundtripped bools === original bools
        assert!(result.equal(&truth));
    }

    #[test]
    fn test_roundtrip_bool_tensor_non_divisible_by_8() {
        // Test with 5 elements (not divisible by 8)
        let truth = Tensor::from_slice(&[1, 0, 1, 0, 1])
            .to_kind(Kind::Bool)
            .to(Device::Cpu);

        let serializable = SerializableTensor::try_from(&truth).unwrap();
        let result = Tensor::try_from(&serializable).unwrap();

        assert!(result.equal(&truth));
    }

    #[test]
    fn test_roundtrip_bool_tensor_single_element() {
        let truth = Tensor::from_slice(&[1]).to_kind(Kind::Bool).to(Device::Cpu);

        let serializable = SerializableTensor::try_from(&truth).unwrap();
        let result = Tensor::try_from(&serializable).unwrap();

        assert!(result.equal(&truth));
    }

    #[test]
    fn test_roundtrip_bool_tensor_unusual_shape() {
        let truth = Tensor::from_slice(&[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
            .to_kind(Kind::Bool)
            .to(Device::Cpu)
            .reshape([1, 3, 5]);

        let serializable = SerializableTensor::try_from(&truth).unwrap();
        let result = Tensor::try_from(&serializable).unwrap();

        assert!(result.equal(&truth));
    }
}

#[cfg(test)]
mod hostile_input_tests {
    use super::*;
    use crate::serializable_kind::SerializableKind;

    /// Builds a tensor the way the wire does: every field independently chosen,
    /// because every field arrives from a peer.
    fn from_wire(dims: Vec<i64>, kind: Kind, data: SerializableTensorData) -> SerializableTensor {
        SerializableTensor {
            dims,
            kind: SerializableKind::from(kind),
            requires_grad: false,
            data,
        }
    }

    /// `f_from_data_size` hands libtorch a pointer and drops the slice length;
    /// `at_tensor_of_data` then copies `numel * element_size` bytes from it. A
    /// declared shape larger than the buffer is a read past the end of that
    /// buffer, so the length has to be compared here, before the pointer leaves
    /// Rust. wienerlabs/leviathan#15, finding 25.
    #[test]
    fn a_shape_larger_than_the_bytes_is_refused() {
        let sixteen_bytes_claiming_a_thousand_floats = from_wire(
            vec![1000],
            Kind::Float,
            SerializableTensorData::Full(vec![0u8; 16]),
        );
        let err = Tensor::try_from(&sixteen_bytes_claiming_a_thousand_floats)
            .expect_err("16 bytes are not 1000 floats");
        assert!(format!("{err:?}").contains("needs 4000 bytes"));
    }

    /// The overshoot that matters is the small one - large enough to read
    /// adjacent heap, small enough not to fault - so it is refused too.
    #[test]
    fn a_shape_that_overshoots_by_a_little_is_refused_as_well() {
        let four_floats_claiming_sixty_four = from_wire(
            vec![64],
            Kind::Float,
            SerializableTensorData::Full(vec![0u8; 16]),
        );
        assert!(Tensor::try_from(&four_floats_claiming_sixty_four).is_err());
    }

    /// Trailing bytes are refused as well as missing ones: the shape and the
    /// buffer have to agree exactly, so there is no slack for a sender to hide
    /// anything in.
    #[test]
    fn extra_bytes_are_refused_too() {
        let too_many = from_wire(
            vec![2],
            Kind::Float,
            SerializableTensorData::Full(vec![0u8; 64]),
        );
        assert!(Tensor::try_from(&too_many).is_err());
    }

    /// A shape whose product leaves `i64` is rejected before anything is
    /// multiplied out, rather than wrapping into a small number that passes.
    #[test]
    fn an_overflowing_shape_is_refused() {
        let overflowing = from_wire(
            vec![i64::MAX, 4],
            Kind::Bool,
            SerializableTensorData::OneBit(vec![0x00u8; 8]),
        );
        let err = Tensor::try_from(&overflowing).expect_err("the product overflows");
        assert!(format!("{err:?}").contains("overflows"));
    }

    /// A negative dimension is not a shape.
    #[test]
    fn a_negative_dimension_is_refused() {
        let negative = from_wire(
            vec![-1, 4],
            Kind::Float,
            SerializableTensorData::Full(vec![0u8; 16]),
        );
        let err = Tensor::try_from(&negative).expect_err("-1 is not a dimension");
        assert!(format!("{err:?}").contains("negative dimension"));
    }

    /// The one-bit branch now fails the way the other one does - an `Err` the
    /// caller can act on - instead of panicking out of the thread. finding 27.
    #[test]
    fn the_one_bit_branch_returns_an_error_rather_than_panicking() {
        let one_byte_claiming_a_thousand_bits = from_wire(
            vec![1000],
            Kind::Bool,
            SerializableTensorData::OneBit(vec![0xFFu8]),
        );
        let outcome =
            std::panic::catch_unwind(|| Tensor::try_from(&one_byte_claiming_a_thousand_bits));
        let result = outcome.expect("no panic escapes the conversion any more");
        assert!(result.is_err(), "it reports the mismatch instead");
    }

    /// Honest payloads still round-trip, including the bit-packed ones whose
    /// buffer is padded up to a whole byte.
    #[test]
    fn honest_payloads_still_round_trip() {
        for dims in [vec![4i64], vec![2, 2], vec![1, 3, 5]] {
            let n: i64 = dims.iter().product();
            let floats = Tensor::from_slice(&vec![1.5f32; n as usize])
                .to_kind(Kind::Float)
                .reshape(&dims);
            let wire = SerializableTensor::try_from(&floats).unwrap();
            let back = Tensor::try_from(&wire).unwrap();
            assert!(back.equal(&floats), "float round-trip for {dims:?}");

            let bools = Tensor::from_slice(&vec![1i64; n as usize])
                .to_kind(Kind::Bool)
                .reshape(&dims);
            let wire = SerializableTensor::try_from(&bools).unwrap();
            let back = Tensor::try_from(&wire).unwrap();
            assert!(back.equal(&bools), "bool round-trip for {dims:?}");
        }
    }
}
