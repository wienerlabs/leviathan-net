use serde::{Deserialize, Serialize};
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

impl TryFrom<&SerializableTensor> for Tensor {
    type Error = TchError;

    fn try_from(value: &SerializableTensor) -> Result<Self, Self::Error> {
        let tensor = match &value.data {
            SerializableTensorData::Full(data) => {
                Tensor::f_from_data_size(data, &value.dims, (&value.kind).into())?
            }
            SerializableTensorData::OneBit(bytes) => {
                // packed bytes are just a flat 1d slice of bits
                let packed = Tensor::from_slice(bytes).to_kind(Kind::Uint8);

                // make a tensor of bit weights (LSB first) to unpack
                let bit_weights =
                    Tensor::from_slice(&[1u8, 2, 4, 8, 16, 32, 64, 128]).to_kind(Kind::Uint8);

                // reshape packed to [..., 1] for broadcasting
                let reshaped_packed = packed.reshape([-1, 1]);

                // unpack bits
                let bits = reshaped_packed
                    .bitwise_and_tensor(&bit_weights)
                    .to_kind(Kind::Bool);

                // flatten, select needed bits, and reshape
                let flat_bits = bits.flatten(0, -1);
                let total_elements: i64 = value.dims.iter().product();
                let needed_bits = flat_bits.slice(0, 0, total_elements, 1);

                needed_bits.reshape(&value.dims)
            }
        };

        Ok(if value.requires_grad {
            tensor.set_requires_grad(true)
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

    /// The `Full` path does not check that the declared shape fits the bytes
    /// that arrived. `f_from_data_size` passes only a raw pointer to libtorch -
    /// the slice length is dropped - and `at_tensor_of_data` then memcpys
    /// `numel * element_size` bytes from it. Both `dims` and `kind` come off the
    /// wire, so a peer chooses how far past the buffer that read goes.
    #[test]
    fn full_path_accepts_a_shape_larger_than_the_bytes_that_arrived() {
        let sixteen_bytes_claiming_a_thousand_floats = from_wire(
            vec![1000],
            Kind::Float,
            SerializableTensorData::Full(vec![0u8; 16]),
        );
        let tensor = Tensor::try_from(&sixteen_bytes_claiming_a_thousand_floats)
            .expect("BUG: 16 bytes are accepted as 1000 floats");
        assert_eq!(
            tensor.numel(),
            1000,
            "BUG: libtorch was told to copy 4000 bytes out of a 16-byte buffer"
        );
    }

    /// The same call, arranged so the memory past the buffer is known rather
    /// than arbitrary: the pointer handed over belongs to a longer allocation
    /// whose tail is a fixed pattern. Every element past the fourth is read from
    /// bytes that were never part of the slice, and comes back as that pattern.
    #[test]
    fn the_read_runs_past_the_slice_it_was_given() {
        // 0xAA repeated is -3.0316488e-13 when four of them are read as an f32.
        // The first sixteen bytes - the part we actually hand over - are zero,
        // so the two regions are told apart by their contents.
        let mut backing = vec![0xAAu8; 4096];
        backing[..16].fill(0);
        let handed_over = &backing[..16]; // four floats, and that is all we pass
        let tensor = Tensor::f_from_data_size(handed_over, &[64], Kind::Float)
            .expect("the length of the slice is never checked");
        let values: Vec<f32> = Vec::<f32>::try_from(&tensor).unwrap();
        assert_eq!(values.len(), 64);
        assert_eq!(values[0], 0.0, "the four floats we did pass are zero");
        assert_eq!(values[3], 0.0);
        let pattern = f32::from_le_bytes([0xAA; 4]);
        assert_eq!(
            values[4], pattern,
            "BUG: element 4 was read from memory past the end of the slice"
        );
        assert!(
            values[4..].iter().all(|v| *v == pattern),
            "BUG: 240 bytes beyond the buffer are copied into the tensor"
        );
    }

    /// The `OneBit` path does not: it reaches for panicking tch calls, so the
    /// same class of mismatch aborts the thread instead of returning
    /// `Err(TchError)` for the caller to handle. Recorded in the third pass of
    /// the internal review (wienerlabs/leviathan#15).
    #[test]
    fn one_bit_path_panics_where_the_full_path_would_have_errored() {
        let one_byte_claiming_a_thousand_bits =
            from_wire(vec![1000], Kind::Bool, SerializableTensorData::OneBit(vec![0xFFu8]));
        let outcome = std::panic::catch_unwind(|| {
            Tensor::try_from(&one_byte_claiming_a_thousand_bits)
        });
        assert!(
            outcome.is_err(),
            "BUG: the OneBit branch panics rather than returning Err, so its \
             errors are not the caller's to handle"
        );
    }

    /// Same asymmetry reached through an overflowing shape rather than a short
    /// buffer: `dims.iter().product()` is computed on numbers a peer chose.
    #[test]
    fn one_bit_path_panics_on_a_shape_whose_product_overflows() {
        let overflowing = from_wire(
            vec![i64::MAX, 4],
            Kind::Bool,
            SerializableTensorData::OneBit(vec![0x00u8; 8]),
        );
        let outcome = std::panic::catch_unwind(|| Tensor::try_from(&overflowing));
        assert!(
            outcome.is_err(),
            "BUG: an overflowing dimension product is not rejected, it aborts"
        );
    }
}
