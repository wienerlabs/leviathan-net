use crate::{CausalLM, StableVariableIterator, Variable};

use std::{cmp::Ordering, collections::HashMap, f64::consts::PI};
use tch::{COptimizer, Device, Kind, Tensor};
use tracing::warn;

pub struct TransformDCT {
    shape_dict: HashMap<i64, i64>,
    f_dict: HashMap<i64, Tensor>,
    b_dict: HashMap<i64, Tensor>,
}

impl TransformDCT {
    pub fn new(variables: StableVariableIterator, target_chunk: i64) -> Self {
        let _no_grad = tch::no_grad_guard();
        let mut shape_dict = HashMap::new();
        let mut f_dict = HashMap::new();
        let mut b_dict = HashMap::new();

        // Get all variants of model tensor sizes
        // Generate all possible valid DCT sizes for model tensors
        for variable in variables {
            let size = variable.full_tensor_shape();
            let variable = variable.local_tensor();
            for s in size {
                // Get the closest smallest divisor to the targeted DCT size
                let sc = match shape_dict.get(&s) {
                    Some(sc) => *sc,
                    None => {
                        let sc = Self::get_smaller_split(s, target_chunk);
                        shape_dict.insert(s, sc);
                        sc
                    }
                };

                // Pregenerate DCT basis matrices
                if let std::collections::hash_map::Entry::Vacant(e) = f_dict.entry(sc) {
                    let i = Tensor::eye(sc, (Kind::Float, variable.device()));
                    e.insert(
                        Self::dct(&i, true)
                            .to_kind(variable.kind())
                            .to(variable.device()),
                    );
                    b_dict.insert(
                        sc,
                        Self::idct(&i, true)
                            .to_kind(variable.kind())
                            .to(variable.device()),
                    );
                }
            }
        }
        Self {
            shape_dict,
            f_dict,
            b_dict,
        }
    }

    fn get_prime_divisors(mut n: i64) -> Vec<i64> {
        if n == 0 {
            return Vec::new();
        }
        let mut divisors = Vec::new();
        while n % 2 == 0 {
            divisors.push(2);
            n /= 2;
        }
        while n % 3 == 0 {
            divisors.push(3);
            n /= 3;
        }
        let mut i = 5;
        while i * i <= n {
            for k in [i, i + 2].iter() {
                while n % k == 0 {
                    divisors.push(*k);
                    n /= k;
                }
            }
            i += 6;
        }
        if n > 1 {
            divisors.push(n);
        }
        divisors
    }

    fn get_divisors(n: i64) -> Vec<i64> {
        let mut divisors = Vec::new();
        match n.cmp(&1) {
            Ordering::Equal => {
                divisors.push(1);
            }
            Ordering::Greater => {
                let prime_factors = Self::get_prime_divisors(n);
                divisors = vec![1];
                let mut last_prime = 0;
                let mut factor = 0;
                let mut slice_len = 0;
                // Find all the products that are divisors of n
                for prime in prime_factors {
                    if last_prime != prime {
                        slice_len = divisors.len();
                        factor = prime;
                    } else {
                        factor *= prime;
                    }
                    for i in 0..slice_len {
                        divisors.push(divisors[i] * factor);
                    }
                    last_prime = prime;
                }
                divisors.sort_unstable();
            }
            Ordering::Less => {}
        }
        divisors
    }

    fn get_smaller_split(n: i64, close_to: i64) -> i64 {
        let all_divisors = Self::get_divisors(n);
        for (ix, &val) in all_divisors.iter().enumerate() {
            if val == close_to {
                return val;
            }
            if val > close_to {
                if ix == 0 {
                    return val;
                }
                return all_divisors[ix - 1];
            }
        }
        n
    }

    fn dct_fft_impl(v: &Tensor) -> Tensor {
        v.fft_fft(None, 1, "backward").view_as_real()
    }

    #[allow(unused)]
    fn dct(x: &Tensor, ortho: bool) -> Tensor {
        let x_shape = x.size();
        let n = { *x_shape.last().unwrap() };
        let x = x.contiguous().view([-1, n]);

        let v = Tensor::cat(
            &[x.slice(1, 0, None, 2), x.slice(1, 1, None, 2).flip([1])],
            1,
        );

        let vc = Self::dct_fft_impl(&v);

        let k = -Tensor::arange(n, (Kind::Float, x.device()))
            .unsqueeze(0)
            .g_mul_scalar(PI / (2.0 * n as f64));
        let w_r = k.cos();
        let w_i = k.sin();

        let mut v = vc.select(2, 0) * &w_r - vc.select(2, 1) * &w_i;

        if ortho {
            v.select(1, 0).g_div_scalar_((n as f64).sqrt() * 2.0);
            v.slice(1, 1, None, 1)
                .g_div_scalar_((n as f64 / 2.0).sqrt() * 2.0);
        }

        v.g_mul_scalar_(2.0).view(x_shape.as_slice())
    }

    fn idct_irfft_impl(v: &Tensor) -> Tensor {
        let complex_v = v.view_as_complex();
        let n = v.size()[1];
        complex_v.fft_irfft(Some(n), 1, "backward")
    }

    #[allow(unused)]
    fn idct(x: &Tensor, ortho: bool) -> Tensor {
        let x_shape = x.size();
        let n = { *x_shape.last().unwrap() };

        let mut x_v = x.contiguous().view([-1, n]).f_div_scalar(2.0).unwrap();

        if ortho {
            x_v.slice(1, 0, 1, 1)
                .f_mul_scalar_((n as f64).sqrt() * 2.0)
                .unwrap();
            x_v.slice(1, 1, n, 1)
                .f_mul_scalar_((n as f64 / 2.0).sqrt() * 2.0)
                .unwrap();
        }

        let k = Tensor::arange(n, (Kind::Float, x.device()))
            .f_mul_scalar(PI / (2.0 * n as f64))
            .unwrap()
            .unsqueeze(0);

        let w_r = k.cos();
        let w_i = k.sin();

        let v_t_r = &x_v;
        let v_t_i = Tensor::cat(
            &[
                x_v.slice(1, 0, 1, 1).f_mul_scalar(0.0).unwrap(),
                x_v.flip([1]).slice(1, 0, n - 1, 1).f_neg().unwrap(),
            ],
            1,
        );

        let v_r = v_t_r.f_mul(&w_r).unwrap() - v_t_i.f_mul(&w_i).unwrap();
        let v_i = v_t_r.f_mul(&w_i).unwrap() + v_t_i.f_mul(&w_r).unwrap();

        let v = Tensor::cat(&[v_r.unsqueeze(2), v_i.unsqueeze(2)], 2);

        let v = Self::idct_irfft_impl(&v);

        let mut x = Tensor::zeros(v.size(), (Kind::Float, v.device()));

        x.slice(1, 0, n, 2)
            .f_add_(&v.slice(1, 0, n - (n / 2), 1))
            .unwrap();
        x.slice(1, 1, n, 2)
            .f_add_(&v.flip([1]).slice(1, 0, n / 2, 1))
            .unwrap();

        x.view(x_shape.as_slice())
    }

    fn einsum_2d(x: &Tensor, b: &Tensor, d: Option<&Tensor>) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        match d {
            None => Tensor::einsum("...ij, jb -> ...ib", &[x, b], None::<i64>),
            Some(d_tensor) => {
                // Note: b-c axis output is transposed to chunk DCT in 2D
                Tensor::einsum("...ijkl, jb, ld -> ...ikbd", &[x, b, d_tensor], None::<i64>)
            }
        }
    }

    fn einsum_2d_t(x: &Tensor, b: &Tensor, d: Option<&Tensor>) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        match d {
            None => Tensor::einsum("...ij, jb -> ...ib", &[x, b], None::<i64>),
            Some(d_tensor) => {
                // Note: b-c axis output is transposed to chunk DCT in 2D
                Tensor::einsum("...ijkl, kb, ld -> ...ibjd", &[x, b, d_tensor], None::<i64>)
            }
        }
    }

    /// The encoded shape and `totalk` that honest results carry for
    /// `variable`. `Distro::generate` encodes the full, gathered tensor, so
    /// this is keyed on the full shape, which is the same on every rank of a
    /// tensor-parallel group, never on a rank's local shard.
    pub fn expected_encoding(&self, variable: &dyn Variable) -> Option<(Vec<i64>, i64)> {
        let encoded = self.encoded_shape(&variable.full_tensor_shape())?;
        let totalk = *flattened(&encoded).last()?;
        Some((encoded, totalk))
    }

    /// The shape `encode` gives a tensor of `shape`, or `None` if this
    /// transform has no chunk size for one of its dimensions, which is the
    /// case for a shape no variable of the model has.
    pub fn encoded_shape(&self, shape: &[i64]) -> Option<Vec<i64>> {
        let ndim = shape.len();
        match ndim {
            0 => None,
            1 => {
                let n1 = *self.shape_dict.get(&shape[0])?;
                Some(vec![shape[0] / n1, n1])
            }
            _ => {
                let n1 = *self.shape_dict.get(&shape[ndim - 2])?;
                let n2 = *self.shape_dict.get(&shape[ndim - 1])?;
                let mut encoded = shape[..ndim - 2].to_vec();
                encoded.extend([shape[ndim - 2] / n1, shape[ndim - 1] / n2, n1, n2]);
                Some(encoded)
            }
        }
    }

    pub fn encode(&mut self, x: &Tensor) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let shape = x.size();
        let ndim = shape.len();

        if ndim > 1 {
            // 2D+ weights - get chunk sizes for last two dimensions
            let n1 = *self.shape_dict.get(&shape[ndim - 2]).unwrap();
            let n2 = *self.shape_dict.get(&shape[ndim - 1]).unwrap();
            let n1w = self.f_dict.get(&n1).unwrap().to_device(x.device());
            let n2w = self.f_dict.get(&n2).unwrap().to_device(x.device());
            self.f_dict.insert(n1, n1w.copy());
            self.f_dict.insert(n2, n2w.copy());

            // Equivalent to rearrange(x, "... (y h) (x w) -> ... y h x w", h=n1, w=n2)
            let mut new_shape: Vec<i64> = shape[..ndim - 2].to_vec();
            new_shape.push(shape[ndim - 2] / n1); // y
            new_shape.push(n1); // h
            new_shape.push(shape[ndim - 1] / n2); // x
            new_shape.push(n2); // w

            let x = x.view(new_shape.as_slice());
            Self::einsum_2d(&x, &n1w, Some(&n2w))
        } else {
            // 1D weights
            let n1 = *self.shape_dict.get(&shape[0]).unwrap();
            let n1w = self.f_dict.get(&n1).unwrap().to_device(x.device());
            self.f_dict.insert(n1, n1w.copy());

            // Equivalent to rearrange(x, "(x w) -> x w", w=n1)
            let x = x.view([-1, n1]);
            Self::einsum_2d(&x, &n1w, None)
        }
    }

    pub fn decode(&mut self, x: &Tensor) -> Tensor {
        let _no_grad = tch::no_grad_guard();
        let x_shape = x.size();
        let ndim = x_shape.len();

        if ndim > 2 {
            // 2D+ weights - n1 and n2 are at positions -2 and -1 of the encoded tensor
            let n1 = x_shape[ndim - 2];
            let n2 = x_shape[ndim - 1];
            let device = x.device();

            let n1w = self.b_dict.get(&n1).unwrap().to_device(device);
            let n2w = self.b_dict.get(&n2).unwrap().to_device(device);

            self.b_dict.insert(n1, n1w.copy());
            self.b_dict.insert(n2, n2w.copy());

            let x = Self::einsum_2d_t(x, &n1w, Some(&n2w));
            let x_shape = x.size();
            let x_ndim = x_shape.len();

            // Equivalent to rearrange(x, "... y h x w -> ... (y h) (x w)")
            let mut new_shape: Vec<i64> = x_shape[..x_ndim - 4].to_vec();
            new_shape.push(x_shape[x_ndim - 4] * x_shape[x_ndim - 3]); // y * h
            new_shape.push(x_shape[x_ndim - 2] * x_shape[x_ndim - 1]); // x * w

            x.reshape(new_shape.as_slice())
        } else {
            // 1D weights
            let n1 = x_shape[1];
            let device = x.device();

            let n1w = self.b_dict.get(&n1).unwrap().to_device(device);
            self.b_dict.insert(n1, n1w.copy());

            let x = Self::einsum_2d_t(x, &n1w, None);
            let x_shape = x.size();

            // Equivalent to rearrange(x, "x w -> (x w)")
            let (x_, w) = (x_shape[0], x_shape[1]);
            x.reshape([x_ * w])
        }
    }
}

pub struct CompressDCT;

impl CompressDCT {
    fn clamp_topk(x: &Tensor, topk: i64) -> i64 {
        let last_dim = x.size()[x.dim() - 1];

        if topk > last_dim {
            last_dim
        } else if topk < 1 {
            1
        } else {
            topk
        }
    }

    pub fn compress(x: &Tensor, topk: i64) -> (Tensor, Tensor, Vec<i64>, i64) {
        let _no_grad = tch::no_grad_guard();
        let xshape = x.size();
        let ndim = xshape.len();

        let x = if ndim > 2 {
            // Equivalent to rearrange(x, "... y x h w -> ... y x (h w)")
            let mut new_shape: Vec<i64> = xshape[..ndim - 2].to_vec();
            new_shape.push(xshape[ndim - 2] * xshape[ndim - 1]);
            x.view(new_shape.as_slice())
        } else {
            x.shallow_clone()
        };

        let totalk = *x.size().last().unwrap();
        let topk = Self::clamp_topk(&x, topk);

        let idx = x.abs().topk(topk, -1, true, false).1;
        let val = x.gather(-1, &idx, false);

        let idx = compress_idx(totalk, &idx);

        (idx, val, xshape, totalk)
    }

    pub fn decompress(
        idx: &Tensor,
        val: &Tensor,
        xshape: &[i64],
        totalk: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let totalk = totalk.abs();
        let idx = decompress_idx(totalk, idx);
        let val = val.to_kind(kind);
        let ndim = xshape.len();

        let mut x: Tensor = Tensor::zeros(xshape, (kind, device));

        if ndim > 2 {
            // Equivalent to rearrange(x, "... y x h w -> ... y x (h w)")
            let mut new_shape: Vec<i64> = xshape[..ndim - 2].to_vec();
            new_shape.push(xshape[ndim - 2] * xshape[ndim - 1]);
            x = x.view(new_shape.as_slice());
        }

        let _ = x.internal_scatter_reduce_(-1, &idx, &val, "mean", false);

        x.reshape(xshape)
    }

    pub fn batch_decompress(
        idx: &[Tensor],
        val: &[Tensor],
        xshape: &[i64],
        totalk: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let idx_concat = Tensor::cat(idx, -1).to_device(device);
        let val_concat = Tensor::cat(val, -1).to_device(device);
        // Call the decompress method
        Self::decompress(&idx_concat, &val_concat, xshape, totalk, kind, device)
    }
}

fn compress_idx(max_value: i64, idx: &Tensor) -> Tensor {
    if max_value <= 256 {
        idx.to_kind(Kind::Uint8)
    } else if max_value <= 65536 {
        idx.to_kind(Kind::UInt16).view_dtype(Kind::Uint8)
    } else if max_value <= 4294967296 {
        idx.to_kind(Kind::UInt32).view_dtype(Kind::Uint8)
    } else {
        idx.shallow_clone()
    }
}

fn decompress_idx(max_value: i64, idx: &Tensor) -> Tensor {
    if max_value <= 256 {
        idx.view_dtype(Kind::Uint8)
    } else if max_value <= 65536 {
        idx.view_dtype(Kind::UInt16)
    } else if max_value <= 4294967296 {
        idx.view_dtype(Kind::UInt32)
    } else {
        idx.shallow_clone()
    }
    .to_kind(Kind::Int64)
}

/// The shape `CompressDCT::compress` scatters into for an encoded tensor of
/// shape `encoded`: the last two dimensions of a 2D+ encoding flattened.
fn flattened(encoded: &[i64]) -> Vec<i64> {
    let ndim = encoded.len();
    if ndim > 2 {
        let mut flat = encoded[..ndim - 2].to_vec();
        flat.push(encoded[ndim - 2] * encoded[ndim - 1]);
        flat
    } else {
        encoded.to_vec()
    }
}

/// The index width `compress_idx` sends for `totalk`, and the dtype it reads
/// back as.
fn index_encoding(totalk: i64) -> (Kind, i64) {
    if totalk <= 256 {
        (Kind::Uint8, 1)
    } else if totalk <= 65536 {
        (Kind::UInt16, 2)
    } else if totalk <= 4294967296 {
        (Kind::UInt32, 4)
    } else {
        (Kind::Int64, 8)
    }
}

/// Why a peer's results do not fit the local model, so that applying them
/// would index past them or fail inside libtorch (wienerlabs/leviathan-net#41).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DistroResultMismatch {
    #[error("{got} results for a model with {expected} variables")]
    Count { expected: usize, got: usize },
    #[error("variable {index} ({name}) has a shape the transform cannot encode")]
    Unencodable { index: usize, name: String },
    #[error("variable {index} ({name}): xshape {got:?}, the model encodes it as {expected:?}")]
    Shape {
        index: usize,
        name: String,
        expected: Vec<i64>,
        got: Vec<i64>,
    },
    #[error("variable {index} ({name}): totalk {got}, the model encodes it with {expected}")]
    Totalk {
        index: usize,
        name: String,
        expected: i64,
        got: i64,
    },
    #[error("variable {index} ({name}): indices {reason}")]
    Indices {
        index: usize,
        name: String,
        reason: String,
    },
    #[error("variable {index} ({name}): values {reason}")]
    Values {
        index: usize,
        name: String,
        reason: String,
    },
}

struct State {
    delta: Box<dyn Variable>,
}

#[derive(Debug)]
pub struct DistroResult {
    pub sparse_idx: Tensor,
    pub sparse_val: Tensor,
    pub xshape: Vec<i64>,
    pub totalk: i64,
    pub stats: Option<HashMap<String, f64>>,
}

impl Clone for DistroResult {
    fn clone(&self) -> Self {
        Self {
            sparse_idx: self.sparse_idx.shallow_clone(),
            sparse_val: self.sparse_val.shallow_clone(),
            xshape: self.xshape.clone(),
            totalk: self.totalk,
            stats: self.stats.clone(),
        }
    }
}

/// Why an error-feedback buffer does not fit an optimizer's.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ErrorFeedbackMismatch {
    #[error("{given} buffers for {expected} variables")]
    Count { given: usize, expected: usize },
    #[error("variable {index} has shape {expected:?}, its buffer {given:?}")]
    Shape {
        index: usize,
        expected: Vec<i64>,
        given: Vec<i64>,
    },
}

pub struct Distro {
    sgd: COptimizer,
    compression_decay: f64,
    compression_topk: i64,
    weight_decay: f64,
    /// The run's aggregation rule, `OptimizerDefinition::Distro`'s
    /// `robust_aggregation`. Nothing about the process this runs in can
    /// change it, so every node of a run applies the same update
    /// (wienerlabs/leviathan-net#60).
    robust_aggregation: bool,
    state: Vec<State>,
    transform: TransformDCT,
}

impl Distro {
    pub fn new(
        vs: &dyn CausalLM,
        compression_decay: f64,
        compression_chunk: i64,
        compression_topk: i64,
        weight_decay: f64,
        robust_aggregation: bool,
    ) -> Self {
        let _no_grad = tch::no_grad_guard();
        let mut sgd = COptimizer::sgd(0.1, 0.0, 0.0, 0.0, false).unwrap();

        let mut state = Vec::new();
        for variable in vs.variables() {
            state.push(State {
                delta: variable.zeros_like(format!("{}.delta", variable.name())),
            });

            let logical_tensor = variable.logical_tensor();
            sgd.add_parameters(&logical_tensor, 0).unwrap();
            variable.zero_grad();
        }

        let transform = TransformDCT::new(vs.variables(), compression_chunk);

        Self {
            sgd,
            compression_decay,
            compression_topk,
            weight_decay,
            robust_aggregation,
            state,
            transform,
        }
    }

    pub fn generate(
        &mut self,
        variables: &dyn CausalLM,
        prev_self_results: &[Vec<DistroResult>],
        prev_lr: f64,
        lr: f64,
        stats: bool,
    ) -> Vec<DistroResult> {
        let _no_grad = tch::no_grad_guard();

        let mut ret = Vec::new();
        for (index, var) in variables.variables().enumerate() {
            let mut variable = var.logical_tensor();

            let grad_energy: Option<f64> = match stats {
                true => Some(
                    variable
                        .grad()
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                _ => None,
            };

            let delta_var = &mut self.state.get_mut(index).unwrap().delta;
            let mut delta = delta_var.logical_tensor();

            let _t = variable.g_add_(&delta.sign().multiply_scalar(prev_lr));

            if !prev_self_results.is_empty() {
                let device = variable.device();
                let indicies = prev_self_results
                    .iter()
                    .map(|x| x[index].sparse_idx.to_device(device))
                    .collect::<Vec<_>>();

                let val_kind: Kind = variable.kind();
                let values = prev_self_results
                    .iter()
                    .map(|x| {
                        let sparse_val = x[index].sparse_val.to_device(device);
                        if sparse_val.kind() == Kind::Bool {
                            Self::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                        } else {
                            sparse_val
                        }
                    })
                    .collect::<Vec<_>>();

                // Decode grad from all nodes
                let decompressed = CompressDCT::batch_decompress(
                    &indicies,
                    &values,
                    &prev_self_results[0][index].xshape,
                    prev_self_results[0][index].totalk,
                    val_kind,
                    device,
                );
                let transmit_grad = self.transform.decode(&decompressed);

                // Remove transmitted from delta
                let _t = delta.g_sub_(&var.shard_other_tensor_like_me(transmit_grad));
            }

            // weight decay
            if self.weight_decay != 0.0 {
                let _t = variable.g_mul_scalar_(1.0 - lr * self.weight_decay);
            }

            // decay delta
            if self.compression_decay != 1.0 {
                let _t = delta.g_mul_scalar_(self.compression_decay);
            }

            // add delta to new gradient
            let _t = delta.g_add_(&variable.grad().multiply_scalar(lr));

            // Compress delta
            let full_delta = delta_var.gather_full_tensor();
            let (sparse_idx, sparse_val, xshape, totalk) =
                CompressDCT::compress(&self.transform.encode(&full_delta), self.compression_topk);

            let delta_energy: Option<f64> = match stats {
                true => Some(
                    full_delta
                        .norm_scalaropt_dtype(1, Kind::Float)
                        .try_into()
                        .unwrap(),
                ),
                false => None,
            };

            ret.push(DistroResult {
                sparse_idx,
                sparse_val,
                xshape,
                totalk,
                stats: match stats {
                    true => {
                        let name = var.name();
                        Some(HashMap::from([
                            (format!("{name}.delta_energy"), delta_energy.unwrap()),
                            (format!("{name}.grad_energy"), grad_energy.unwrap()),
                        ]))
                    }
                    false => None,
                },
            });
        }
        ret
    }

    /// Checks one peer's results against the local model, read from the
    /// model itself: one result per variable, each with the encoded shape and
    /// `totalk` that variable's own encoding has, indices that decode under
    /// that `totalk` and stay inside it, in the shape the encoding scatters
    /// into, and as many finite values or 1-bit signs as indices.
    ///
    /// A commitment hash proves a payload is the one its trainer signed, not
    /// that it fits the model, and `apply` indexes and scatters with what is
    /// in it (wienerlabs/leviathan-net#41).
    pub fn check_results(
        &self,
        vars: &dyn CausalLM,
        results: &[DistroResult],
    ) -> Result<(), DistroResultMismatch> {
        let expected_count = vars.variables().count();
        if results.len() != expected_count {
            return Err(DistroResultMismatch::Count {
                expected: expected_count,
                got: results.len(),
            });
        }
        for (index, (var, result)) in vars.variables().zip(results).enumerate() {
            let name = var.name().to_string();
            let Some((expected, totalk)) = self.transform.expected_encoding(var.as_ref()) else {
                return Err(DistroResultMismatch::Unencodable { index, name });
            };
            if result.xshape != expected {
                return Err(DistroResultMismatch::Shape {
                    index,
                    name,
                    expected,
                    got: result.xshape.clone(),
                });
            }
            if result.totalk != totalk {
                return Err(DistroResultMismatch::Totalk {
                    index,
                    name,
                    expected: totalk,
                    got: result.totalk,
                });
            }
            let flat = flattened(&expected);
            Self::check_sparse(&flat, totalk, result).map_err(|(on_indices, reason)| {
                match on_indices {
                    true => DistroResultMismatch::Indices {
                        index,
                        name: name.clone(),
                        reason,
                    },
                    false => DistroResultMismatch::Values {
                        index,
                        name: name.clone(),
                        reason,
                    },
                }
            })?;
        }
        Ok(())
    }

    /// Checks each of a stranger's results as a DisTrO encoding of its own
    /// shape, with no model at hand: its `totalk` is the one its encoded shape
    /// gives, and its indices and values are what that encoding scatters, as
    /// `check_results` checks them against the model's shapes.
    ///
    /// `CompressDCT::decompress` allocates the shape a result claims and
    /// scatters into it with libtorch's infallible operations. A result that
    /// does not hold together makes it index past that tensor and panic, so
    /// nothing that decompresses a stranger's results may do so before this
    /// passes. It does not say the results fit any model: one result short of
    /// the model, or another variable's shape, holds together, and only
    /// `check_results`, or a comparison with an honest result, finds it out
    /// (wienerlabs/leviathan-net#59).
    pub fn check_encodings(results: &[DistroResult]) -> Result<(), DistroResultMismatch> {
        for (index, result) in results.iter().enumerate() {
            let name = format!("result {index}");
            let flat = flattened(&result.xshape);
            let Some(&totalk) = flat.last() else {
                return Err(DistroResultMismatch::Unencodable { index, name });
            };
            if result.totalk != totalk {
                return Err(DistroResultMismatch::Totalk {
                    index,
                    name,
                    expected: totalk,
                    got: result.totalk,
                });
            }
            Self::check_sparse(&flat, totalk, result).map_err(|(on_indices, reason)| {
                match on_indices {
                    true => DistroResultMismatch::Indices {
                        index,
                        name: name.clone(),
                        reason,
                    },
                    false => DistroResultMismatch::Values {
                        index,
                        name: name.clone(),
                        reason,
                    },
                }
            })?;
        }
        Ok(())
    }

    /// The sparse half of `check_results`: `Err((true, _))` for the indices,
    /// `Err((false, _))` for the values. Every tensor operation here is the
    /// fallible one, because the input is a stranger's.
    fn check_sparse(
        flat: &[i64],
        totalk: i64,
        result: &DistroResult,
    ) -> Result<(), (bool, String)> {
        let numel: i64 = flat.iter().product();
        let (wire_kind, width) = index_encoding(totalk);
        let sparse_idx = &result.sparse_idx;
        let expected_kind = match wire_kind {
            Kind::Int64 => Kind::Int64,
            _ => Kind::Uint8,
        };
        if sparse_idx.kind() != expected_kind {
            return Err((
                true,
                format!(
                    "are {:?}, totalk {totalk} sends {:?}",
                    sparse_idx.kind(),
                    expected_kind
                ),
            ));
        }
        let wire_shape = sparse_idx.size();
        let Some(&wire_last) = wire_shape.last() else {
            return Err((true, "have no dimensions".to_string()));
        };
        if expected_kind == Kind::Uint8 && wire_last % width != 0 {
            return Err((
                true,
                format!("are {wire_last} bytes, not a whole number of {width}-byte indices"),
            ));
        }
        let decoded = match wire_kind {
            Kind::Uint8 | Kind::Int64 => sparse_idx.shallow_clone(),
            kind => sparse_idx
                .f_contiguous()
                .and_then(|idx| idx.f_view_dtype(kind))
                .map_err(|error| (true, format!("do not decode: {error}")))?,
        };
        let decoded = decoded
            .f_to_kind(Kind::Int64)
            .map_err(|error| (true, format!("do not decode: {error}")))?;
        let shape = decoded.size();
        let k = *shape.last().expect("checked to have a dimension above");
        if shape.len() != flat.len() || shape[..shape.len() - 1] != flat[..flat.len() - 1] {
            return Err((
                true,
                format!("have shape {shape:?}, the encoding scatters into {flat:?}"),
            ));
        }
        if k == 0 || k > totalk {
            return Err((
                true,
                format!("number {k} per row, the encoding has {totalk}"),
            ));
        }
        let lowest = decoded
            .f_min()
            .and_then(|t| t.f_int64_value(&[]))
            .map_err(|error| (true, format!("cannot be read: {error}")))?;
        let highest = decoded
            .f_max()
            .and_then(|t| t.f_int64_value(&[]))
            .map_err(|error| (true, format!("cannot be read: {error}")))?;
        if lowest < 0 || highest >= totalk {
            return Err((
                true,
                format!("reach {lowest}..={highest}, outside 0..{totalk}"),
            ));
        }
        // Each row names each of its coordinates once, as a top-k does. A
        // coordinate named twice is read as the mean of its values where a
        // payload is decompressed on its own, as a verifier does, but counts
        // twice where peers' indices are decompressed together, as `apply`
        // does, so one payload would say one thing to the verifier and another
        // to every node (wienerlabs/leviathan-net#49).
        if k > 1 {
            let (sorted, _) = decoded.reshape([-1, k]).sort(-1, false);
            let repeated = sorted
                .narrow(-1, 1, k - 1)
                .eq_tensor(&sorted.narrow(-1, 0, k - 1))
                .any()
                .to_kind(Kind::Int64)
                .f_int64_value(&[])
                .map_err(|error| (true, format!("cannot be read: {error}")))?;
            if repeated != 0 {
                return Err((
                    true,
                    "name a coordinate more than once in a row".to_string(),
                ));
            }
        }

        let values = &result.sparse_val;
        if values.size() != shape {
            return Err((
                false,
                format!("have shape {:?}, the indices {shape:?}", values.size()),
            ));
        }
        match values.kind() {
            Kind::Bool => Ok(()),
            Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16 => {
                // Read in fp32 whatever the local model's dtype, so every node
                // reaches the same verdict. A value is only usable if it stays
                // finite through what `apply` does with it: the DCT decode
                // and, under robust aggregation, sums of squares over the
                // whole variable. Bounding each value by half the square root
                // of fp32's range over the variable's size keeps both finite.
                // Honest values are many orders of magnitude below it.
                let bound = (f32::MAX / numel as f32).sqrt() / 2.0;
                let largest = values
                    .f_to_kind(Kind::Float)
                    .and_then(|t| t.f_abs())
                    .and_then(|t| t.f_max())
                    .and_then(|t| t.f_double_value(&[]))
                    .map_err(|error| (false, format!("cannot be read: {error}")))?;
                if largest.is_finite() && largest <= bound as f64 {
                    Ok(())
                } else {
                    Err((
                        false,
                        format!("reach {largest:e}, beyond {bound:e} for a variable of {numel} elements"),
                    ))
                }
            }
            kind => Err((
                false,
                format!("are {kind:?}, neither floating point nor 1-bit signs"),
            )),
        }
    }

    pub fn apply(&mut self, vars: &dyn CausalLM, results: &[Vec<DistroResult>], lr: f64) {
        let _no_grad = tch::no_grad_guard();

        // Every honest node holds the same model, so every one skips the same
        // peers and the replicas stay identical. A peer that does not fit is
        // never indexed, and decoding uses the shapes checked against the
        // local model, not whatever the first peer sent
        // (wienerlabs/leviathan-net#41).
        let results: Vec<&Vec<DistroResult>> = results
            .iter()
            .enumerate()
            .filter_map(
                |(peer, peer_results)| match self.check_results(vars, peer_results) {
                    Ok(()) => Some(peer_results),
                    Err(mismatch) => {
                        warn!(peer, %mismatch, "skipping DisTrO results that do not fit the model");
                        None
                    }
                },
            )
            .collect();
        if results.is_empty() {
            return;
        }

        let robust = self.robust_aggregation;

        for (index, var) in vars.variables().enumerate() {
            let variable = var.logical_tensor();
            let device = variable.device();
            let indicies = results
                .iter()
                .map(|x| x[index].sparse_idx.to_device(device))
                .collect::<Vec<_>>();

            let val_kind: Kind = variable.kind();
            let values = results
                .iter()
                .map(|x| {
                    let sparse_val = x[index].sparse_val.to_device(device);
                    if sparse_val.kind() == Kind::Bool {
                        Self::unpack_tensor_sign_from_boolean(sparse_val, val_kind)
                    } else {
                        sparse_val
                    }
                })
                .collect::<Vec<_>>();

            // Every accepted peer was checked to carry exactly the local
            // encoding of this variable, so any of them names it.
            let xshape = &results[0][index].xshape;
            let totalk = results[0][index].totalk;

            // Decode grad from all nodes
            let decompressed = if robust {
                let per: Vec<Tensor> = indicies
                    .iter()
                    .zip(values.iter())
                    .map(|(idx, val)| {
                        CompressDCT::decompress(idx, val, xshape, totalk, val_kind, device)
                    })
                    .collect();
                let shape = per[0].size();
                let flat = Tensor::stack(&per, 0)
                    .reshape([per.len() as i64, -1])
                    .to_kind(Kind::Float);
                let center = leviathan_robust_aggregate(&flat, 3.0, 3);
                center.reshape(shape.as_slice()).to_kind(val_kind)
            } else {
                CompressDCT::batch_decompress(&indicies, &values, xshape, totalk, val_kind, device)
            };

            // Set the gradients!!!
            var.set_grad(self.transform.decode(&decompressed));

            // Sign-SGD
            let _t = variable.grad().sign_();
        }
        // SGD step
        self.sgd.set_learning_rate(lr).unwrap();
        let _ = self.sgd.step();
        for var in vars.variables() {
            var.zero_grad();
        }
    }

    pub fn error_correction(&mut self, vars: &dyn CausalLM, prev_lr: f64) {
        let _no_grad = tch::no_grad_guard();
        for (index, var) in vars.variables().enumerate() {
            let mut variable = var.logical_tensor();

            let state = self.state.get_mut(index).unwrap();

            // Apply lookahead, the signed delta, multiplied by lr
            let _t = variable.g_sub_(&state.delta.logical_tensor().sign().multiply_scalar(prev_lr));
        }
    }

    pub fn zero_optim(&mut self) {
        for state in &mut self.state {
            let _ = state.delta.logical_tensor().zero_();
        }
    }

    /// A copy of the error-feedback buffer on the CPU: what this optimizer
    /// has computed and not yet sent, one tensor per model variable in the
    /// model's order. With the node's weights, it is the state a node would
    /// reveal for an audit of its next round (wienerlabs/leviathan-net#48).
    pub fn error_feedback(&self) -> Vec<Tensor> {
        let _no_grad = tch::no_grad_guard();
        self.state
            .iter()
            .map(|state| state.delta.local_tensor().to_device(Device::Cpu).copy())
            .collect()
    }

    /// Replaces the error-feedback buffer with `buffer`, as
    /// [`Self::error_feedback`] returns it, cast to this buffer's dtype and
    /// device. A buffer that does not fit the model changes nothing.
    pub fn load_error_feedback(&mut self, buffer: &[Tensor]) -> Result<(), ErrorFeedbackMismatch> {
        if buffer.len() != self.state.len() {
            return Err(ErrorFeedbackMismatch::Count {
                given: buffer.len(),
                expected: self.state.len(),
            });
        }
        for (index, (state, given)) in self.state.iter().zip(buffer).enumerate() {
            let expected = state.delta.local_tensor().size();
            if given.size() != expected {
                return Err(ErrorFeedbackMismatch::Shape {
                    index,
                    expected,
                    given: given.size(),
                });
            }
        }
        let _no_grad = tch::no_grad_guard();
        for (state, given) in self.state.iter_mut().zip(buffer) {
            let mut delta = state.delta.local_tensor();
            delta.copy_(&given.to_device(delta.device()).to_kind(delta.kind()));
        }
        Ok(())
    }

    pub fn quantize_nozeros_tensor_to_boolean_sign(tensor: &Tensor) -> Tensor {
        let original_size = tensor.size();
        let tensor = tensor.signbit();
        debug_assert_eq!(tensor.kind(), Kind::Bool);
        debug_assert_eq!(tensor.size(), original_size);
        tensor
    }

    /// The -1 and +1 a 1-bit payload's packed signs stand for, as `apply`
    /// reads them: a set bit is a negative value.
    pub fn unpack_tensor_sign_from_boolean(tensor: Tensor, unpack_kind: Kind) -> Tensor {
        tensor.to_kind(unpack_kind) * -2 + 1
    }
}

unsafe impl Send for Distro {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Variable, set_torch_rng_seed};
    use itertools::iproduct;

    impl Variable for Tensor {
        fn name(&self) -> &str {
            unimplemented!()
        }

        fn local_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn logical_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn gather_full_tensor(&self) -> Tensor {
            self.shallow_clone()
        }

        fn shard_other_tensor_like_me(&self, tensor: Tensor) -> Tensor {
            tensor
        }

        fn full_tensor_shape(&self) -> Vec<i64> {
            self.size()
        }

        fn is_sharded(&self) -> bool {
            false
        }

        fn zeros_like(&self, _name: String) -> Box<dyn Variable> {
            Box::new(self.zeros_like())
        }

        fn set_grad(&self, tensor: Tensor) {
            self.grad().copy_(&tensor);
        }

        fn zero_grad(&self) {
            let grad = self.grad();
            if grad.defined() {
                let _ = self.grad().zero_();
            }
        }
    }

    fn vars(vars: Vec<Tensor>) -> StableVariableIterator {
        Box::new(vars.into_iter().map(|x| Box::new(x) as Box<dyn Variable>))
    }

    #[test]
    fn test_get_prime_divisors() {
        assert_eq!(TransformDCT::get_prime_divisors(1), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_prime_divisors(2), vec![2]);
        assert_eq!(TransformDCT::get_prime_divisors(12), vec![2, 2, 3]);
        assert_eq!(TransformDCT::get_prime_divisors(15), vec![3, 5]);
        assert_eq!(TransformDCT::get_prime_divisors(100), vec![2, 2, 5, 5]);
        assert_eq!(TransformDCT::get_prime_divisors(2310), vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_get_divisors() {
        assert_eq!(TransformDCT::get_divisors(1), vec![1]);
        assert_eq!(TransformDCT::get_divisors(2), vec![1, 2]);
        assert_eq!(TransformDCT::get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(TransformDCT::get_divisors(15), vec![1, 3, 5, 15]);
        assert_eq!(
            TransformDCT::get_divisors(100),
            vec![1, 2, 4, 5, 10, 20, 25, 50, 100]
        );
    }

    #[test]
    fn test_get_smaller_split() {
        assert_eq!(TransformDCT::get_smaller_split(12, 3), 3);
        assert_eq!(TransformDCT::get_smaller_split(12, 4), 4);
        assert_eq!(TransformDCT::get_smaller_split(12, 5), 4);
        assert_eq!(TransformDCT::get_smaller_split(100, 7), 5);
        assert_eq!(TransformDCT::get_smaller_split(100, 26), 25);
        assert_eq!(TransformDCT::get_smaller_split(100, 101), 100);
        assert_eq!(TransformDCT::get_smaller_split(1, 1), 1);
    }

    #[test]
    fn test_edge_cases() {
        assert_eq!(TransformDCT::get_prime_divisors(0), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_divisors(0), Vec::<i64>::new());
        assert_eq!(TransformDCT::get_smaller_split(0, 1), 0);
    }

    #[test]
    fn test_large_numbers() {
        assert_eq!(
            TransformDCT::get_prime_divisors(1000000007),
            vec![1000000007]
        ); // Large prime
        assert_eq!(TransformDCT::get_divisors(1000000007), vec![1, 1000000007]);
        assert_eq!(TransformDCT::get_smaller_split(1000000007, 500000000), 1);
    }

    #[test]
    fn test_dct() {
        let eye = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [0.5000, 0.6533, 0.5000, 0.2706],
            [0.5000, 0.2706, -0.5000, -0.6533],
            [0.5000, -0.2706, -0.5000, 0.6533],
            [0.5000, -0.6533, 0.5000, -0.2706],
        ]);
        let result = TransformDCT::dct(&eye, true);
        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    fn _2d_float<T: AsRef<[f64]>>(x: &[T]) -> Tensor {
        Tensor::from_slice2(x).to_kind(Kind::Float).to(Device::Cpu)
    }

    fn _2d_int<T: AsRef<[i64]>>(x: &[T]) -> Tensor {
        Tensor::from_slice2(x).to_kind(Kind::Int64).to(Device::Cpu)
    }

    fn _1d_float(x: &[f64]) -> Tensor {
        Tensor::from_slice(x).to_kind(Kind::Float).to(Device::Cpu)
    }

    fn _1d_int(x: &[i64]) -> Tensor {
        Tensor::from_slice(x).to_kind(Kind::Int64).to(Device::Cpu)
    }

    #[test]
    fn test_idct() {
        let eye = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ]);
        let result = TransformDCT::idct(&eye, true);
        assert!(result.allclose(&truth, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_compress_2d() {
        let r = _2d_float(&[
            [0.1911, 0.4076, 0.1649, 0.8059],
            [0.2803, 0.9381, 0.9071, 0.2573],
            [0.4070, 0.5765, 0.7226, 0.9486],
            [0.0737, 0.7378, 0.1898, 0.2990],
        ]);
        let truth = (
            _2d_int(&[[3, 1], [1, 2], [3, 2], [1, 3]]),
            _2d_float(&[
                [0.8059, 0.4076],
                [0.9381, 0.9071],
                [0.9486, 0.7226],
                [0.7378, 0.2990],
            ]),
            vec![4i64, 4i64],
            4i64,
        );
        let ret = CompressDCT::compress(&r, 2);
        assert_eq!(truth.0, ret.0);
        assert!(truth.1.allclose(&ret.1, 1e-4, 1e-8, false));
        assert_eq!(truth.2, ret.2);
        assert_eq!(4, ret.3);
    }

    #[test]
    fn test_compress_1d() {
        let r = _1d_float(&[
            0.5223, 0.9625, 0.5487, 0.2152, 0.2161, 0.0363, 0.4944, 0.0974,
        ]);
        let truth = (
            _1d_int(&[1, 2]),
            _1d_float(&[0.9625, 0.5487]),
            vec![8i64],
            8i64,
        );
        let ret = CompressDCT::compress(&r, 2);
        assert_eq!(truth.0, ret.0);
        assert!(truth.1.allclose(&ret.1, 1e-4, 1e-8, false));
        assert_eq!(truth.2, ret.2);
        assert_eq!(8, ret.3);
    }

    #[test]
    fn test_decompress_1d() {
        let p = _1d_float(&[0.0]);
        let idx = _1d_int(&[1, 2]);
        let val = _1d_float(&[0.9625, 0.5487]);
        let xshape = vec![8i64];
        let truth = _1d_float(&[
            0.0000, 0.9625, 0.5487, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        ]);
        let ret = CompressDCT::decompress(&idx, &val, &xshape, i64::MAX, p.kind(), p.device());
        assert!(truth.allclose(&ret, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_decompress_2d() {
        let p = _1d_float(&[0.0]);
        let idx = _2d_int(&[[0, 2], [1, 2], [2, 3], [3, 1]]);
        let val = _2d_float(&[
            [0.8988, 0.5175],
            [0.9882, 0.8945],
            [0.8285, 0.8163],
            [0.9093, 0.7600],
        ]);
        let xshape = vec![4i64, 4i64];
        let truth = _2d_float(&[
            [0.8988, 0.0000, 0.5175, 0.0000],
            [0.0000, 0.9882, 0.8945, 0.0000],
            [0.0000, 0.0000, 0.8285, 0.8163],
            [0.0000, 0.7600, 0.0000, 0.9093],
        ]);
        let ret = CompressDCT::decompress(&idx, &val, &xshape, i64::MAX, p.kind(), p.device());
        assert!(truth.allclose(&ret, 1e-4, 1e-8, false));
    }

    #[test]
    fn test_encode_1d() {
        let a = Tensor::arange(8, (Kind::Float, Device::Cpu));
        let truth = _1d_float(&[
            9.8995e+00,
            -6.4423e+00,
            -4.7684e-07,
            -6.7345e-01,
            2.3842e-07,
            -2.0090e-01,
            -1.1921e-07,
            -5.0702e-02,
        ]);
        let ret = TransformDCT::new(vars(vec![a.copy()]), 64)
            .encode(&a)
            .squeeze();
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_encode_2d() {
        let b = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let truth = _2d_float(&[
            [1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00, 0.0000e+00, -5.9605e-08],
            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
            [0.0000e+00, -5.9605e-08, 0.0000e+00, 1.0000e+00],
        ]);
        let ret = TransformDCT::new(vars(vec![b.copy()]), 64)
            .encode(&b)
            .squeeze();
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_decode_1d() {
        let a = Tensor::arange(8, (Kind::Float, Device::Cpu));
        let a_ = _2d_float(&[[
            9.8995e+00,
            -6.4423e+00,
            -4.7684e-07,
            -6.7345e-01,
            2.3842e-07,
            -2.0090e-01,
            -1.1921e-07,
            -5.0702e-02,
        ]]);
        let truth = _1d_float(&[
            -2.2352e-07,
            1.0000e+00,
            2.0000e+00,
            3.0000e+00,
            4.0000e+00,
            5.0000e+00,
            6.0000e+00,
            7.0000e+00,
        ]);
        let ret = TransformDCT::new(vars(vec![a]), 64).decode(&a_);
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_decode_2d() {
        let b = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let b_ = _2d_float(&[
            [1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00, 0.0000e+00, -5.9605e-08],
            [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
            [0.0000e+00, -5.9605e-08, 0.0000e+00, 1.0000e+00],
        ])
        .unsqueeze(0)
        .unsqueeze(0);
        let truth = _2d_float(&[
            [1.0000e+00, 1.4901e-08, 4.4703e-08, 4.4703e-08],
            [2.9802e-08, 1.0000e+00, -2.9802e-08, 4.4703e-08],
            [4.4703e-08, -2.9802e-08, 1.0000e+00, 2.9802e-08],
            [4.4703e-08, 4.4703e-08, 1.4901e-08, 1.0000e+00],
        ]);
        let ret = TransformDCT::new(vars(vec![b]), 64).decode(&b_);
        assert!(truth.allclose(&ret, 1e-4, 1e-4, false));
    }

    #[test]
    fn test_signed_vals_reconstructs_original_sign() {
        let truth = Tensor::from_slice2(&[
            [0.5000, 0.5000, 0.5000, 0.5000],
            [0.6533, 0.2706, -0.2706, -0.6533],
            [0.5000, -0.5000, -0.5000, 0.5000],
            [0.2706, -0.6533, 0.6533, -0.2706],
        ])
        .to_kind(Kind::Float)
        .to(Device::Cpu);

        let signed_truth = truth.sign();

        let (sparse_idx, sparse_val, xshape, totalk) = CompressDCT::compress(&truth, i64::MAX);
        let signed_sparse_val = sparse_val.sign();

        let decompressed_signed = CompressDCT::decompress(
            &sparse_idx,
            &signed_sparse_val,
            &xshape,
            totalk,
            truth.kind(),
            Device::Cpu,
        );
        assert!(decompressed_signed.equal(&signed_truth));
    }

    #[test]
    fn test_artifical_distro_results_roundtrip() {
        use tch::{Kind, Tensor};

        /// Generates a dummy estimate_val tensor of shape (r0, r1, k), where r is the remainder shape after DCT chunking
        /// r1 can be set to 0 to simulate a 1D DCT
        fn generate_random_estimate_val(r0: i64, r1: i64, k: i64, dtype: Kind) -> Tensor {
            // Warning: only works if dtype bits size is divisible by 8, should always be true for current torch tensors
            // but who knows what would happen one day... fp4?

            let randbytes = match dtype {
                Kind::BFloat16 => 2,
                Kind::Float => 4,
                Kind::Double => 8,
                _ => panic!("Unsupported dtype"),
            };

            // 1D DCT estimates
            let randsize = if r1 == 0 {
                vec![r0, k * randbytes]
            }
            // 2D DCT estimates
            else {
                vec![r0, r1, k * randbytes]
            };

            Tensor::randint(256, &randsize, (Kind::Uint8, tch::Device::Cpu)).view_dtype(dtype)
        }

        /// Generates a dummy indices tensor when given estimate_val. indices are between 0 and s0*s1 (exclusive),
        /// where s0 and s1 is the DCT chunk shape
        /// s1 can be set to 0 to simulate a 1D DCT
        fn generate_random_estimate_idx(val: &Tensor, s0: i64, s1: i64) -> (Tensor, i64) {
            // Note: Some indices will collide, just like real estimates
            // Warning: At the current moment of writing this test, we assume indices must always be int64
            // for correct torch indexing

            // 1D DCT estimates
            let s1 = if s1 == 0 { 1 } else { s1 };

            let max_value = s0 * s1;
            (
                Tensor::randint(max_value, val.size(), (Kind::Int64, tch::Device::Cpu)),
                max_value,
            )
        }

        set_torch_rng_seed();

        let range_r0 = 1..10;
        let range_r1 = 0..10;
        let range_s0 = [1, 7, 512];
        let range_s1 = [1, 4, 64];
        let range_k = [1, 2, 3, 4, 5, 7, 9, 16, 32, 64, 96, 128];
        let range_dtype = [Kind::BFloat16, Kind::Float];

        for (r0, r1, s0, s1, k, d) in
            iproduct!(range_r0, range_r1, range_s0, range_s1, range_k, range_dtype)
        {
            let val = generate_random_estimate_val(r0, r1, k, d);
            let (idx, max_idx_val) = generate_random_estimate_idx(&val, s0, s1);

            let roundtripped_val = Distro::unpack_tensor_sign_from_boolean(
                Distro::quantize_nozeros_tensor_to_boolean_sign(&val),
                val.kind(),
            );

            // we need to make a reference to compare the compression to.
            // this compression should hold Infinity and +0 and some NaNs as 1
            // and -Infinity and -0 and some NaNs as -1
            let val_signed: Tensor = (-2.0 * val.signbit().to_kind(Kind::Float)) + 1.0;
            assert!(val_signed.equal(&roundtripped_val));

            let roundtripped_idx = decompress_idx(max_idx_val, &compress_idx(max_idx_val, &idx));
            assert!(idx.equal(&roundtripped_idx));
        }
    }
    /// `generate` encodes the full, gathered tensor, so on a tensor-parallel
    /// rank a sharded variable's honest results carry the full encoding, not
    /// its local shard's (wienerlabs/leviathan-net#41).
    #[test]
    fn a_sharded_variable_is_expected_in_its_full_encoding() {
        let sharded = || -> Box<dyn Variable> {
            Box::new((
                "sharded".to_string(),
                Tensor::zeros([64, 96], (Kind::Float, Device::Cpu)),
                Some(tch::nn::Shard {
                    dim: 0,
                    rank: 0,
                    world_size: 2,
                }),
                None::<std::sync::Arc<crate::Communicator>>,
            ))
        };
        let variable = sharded();
        assert_eq!(variable.full_tensor_shape(), vec![128, 96]);

        let mut transform = TransformDCT::new(Box::new(vec![sharded()].into_iter()), 64);
        let (expected, totalk) = transform.expected_encoding(variable.as_ref()).unwrap();
        let full = Tensor::zeros([128, 96], (Kind::Float, Device::Cpu));
        let encoded = transform.encode(&full);
        assert_eq!(encoded.size(), expected);
        let (_, _, xshape, compressed_totalk) = CompressDCT::compress(&encoded, 8);
        assert_eq!(xshape, expected);
        assert_eq!(compressed_totalk, totalk);
        assert_ne!(
            transform.encoded_shape(&variable.logical_tensor().size()),
            Some(expected),
            "the local shard's shape would expect a different encoding"
        );
    }

    #[test]
    fn test_1bit_matches_non_quant() {
        set_torch_rng_seed();
        let input = Tensor::rand(
            [51, 35, 5, 13, 6],
            (Kind::BFloat16, Device::cuda_if_available()),
        ) - 0.5;
        // ensure no zeros in our ground truth!
        let input = (&input) + (input.sign() + 0.1);

        let quant = Distro::quantize_nozeros_tensor_to_boolean_sign(&input);
        let unquant = Distro::unpack_tensor_sign_from_boolean(quant, input.kind());

        assert!(input.sign().equal(&unquant));
    }
}

// #[cfg(test)]
// #[cfg(feature = "parallelism")]
// mod tp_tests {
//     use super::*;
//     use crate::tensor_parallelism::CommunicatorId;
//     use crate::{
//         set_suggested_env_vars, set_torch_rng_seed, unsharded_cpu_variables, ColumnParallelLinear,
//     };
//     use std::sync::{Arc, Barrier, Mutex};
//     use tch::{nn, Device, Kind, Tensor, CNCCL};

//     const TEST_LR: f64 = 0.01;
//     const COMPRESSION_DECAY: f64 = 0.99;
//     const COMPRESSION_CHUNK: i64 = 64;
//     const COMPRESSION_TOPK: i64 = 16;
//     const WEIGHT_DECAY: f64 = 0.0;
//     const NUM_STEPS: u32 = 10;

//     fn run_parallel_test<F>(world_size: usize, test_fn: F)
//     where
//         F: Fn(Arc<CommunicatorId>, usize, Arc<Barrier>, Device) -> anyhow::Result<()>
//             + Send
//             + Sync
//             + 'static,
//     {
//         if !tch::utils::has_cuda() || tch::Cuda::device_count() < world_size as i64 {
//             println!(
//                 "Skipping parallel test: requires CUDA and {} GPUs.",
//                 world_size
//             );
//             return;
//         }

//         let barrier = Arc::new(Barrier::new(world_size));
//         let comm_id = Arc::new(CommunicatorId::new());
//         let test_fn = Arc::new(test_fn);

//         let threads: Vec<_> = (0..world_size)
//             .map(|rank| {
//                 let barrier = barrier.clone();
//                 let comm_id = comm_id.clone();
//                 let test_fn = test_fn.clone();
//                 let device = Device::Cuda(rank);

//                 std::thread::spawn(move || {
//                     test_fn(comm_id, rank, barrier, device).unwrap();
//                 })
//             })
//             .collect();

//         for thread in threads {
//             thread.join().expect("Thread panicked");
//         }
//     }

//     // Helper to run a simple training loop step with Distro
//     fn run_distro_step(
//         step_num: u32,
//         model: &dyn nn::Module,
//         input: &Tensor,
//         target: &Tensor,
//         optimizer: &mut Distro,
//         lr: f64,
//         all_rank_results: Arc<Mutex<HashMap<u32, Vec<Vec<DistroResult>>>>>,
//         _rank: usize,
//         _world_size: usize,
//         _comm: &Option<Arc<Communicator>>,
//         barrier: &Arc<Barrier>,
//     ) -> anyhow::Result<Vec<DistroResult>> {
//         optimizer.zero_grad();
//         barrier.wait();

//         let output = model.forward(input);
//         let loss = output.mse_loss(target, tch::Reduction::Mean);
//         barrier.wait();

//         loss.backward();
//         barrier.wait();

//         let current_step_results = optimizer.generate(&vec![], 0.0, lr, false);
//         barrier.wait();

//         {
//             let mut results_map = all_rank_results.lock().unwrap();
//             let step_results = results_map.entry(step_num).or_default();
//             step_results.push(current_step_results.clone());
//         }
//         barrier.wait();

//         let results_to_apply = {
//             let results_map = all_rank_results.lock().unwrap();
//             results_map
//                 .get(&step_num)
//                 .expect(&format!("missing results for current step {step_num}"))
//                 .clone()
//         };
//         barrier.wait();

//         optimizer.apply(&results_to_apply, lr);
//         barrier.wait();

//         Ok(current_step_results)
//     }

//     #[test]
//     fn test_distro_tp_consistency() -> anyhow::Result<()> {
//         const WORLD_SIZE: usize = 8;
//         const BATCH_SIZE: i64 = 4;
//         const SEQ_LEN: i64 = 32;
//         const IN_FEATURES: i64 = 128;
//         const OUT_FEATURES: i64 = 256;

//         set_suggested_env_vars();
//         set_torch_rng_seed();

//         let device = Device::cuda_if_available();
//         if !device.is_cuda() {
//             println!("Skipping TP test as CUDA is not available.");
//             return Ok(());
//         }

//         let input = Arc::new(Mutex::new(Tensor::randn(
//             &[BATCH_SIZE, SEQ_LEN, IN_FEATURES],
//             (Kind::Float, device),
//         )));
//         let target = Arc::new(Mutex::new(Tensor::randn(
//             &[BATCH_SIZE, SEQ_LEN, OUT_FEATURES],
//             (Kind::Float, device),
//         )));

//         // single gpu
//         let (final_weights_non_tp, linear_layer_weights) = {
//             let vs_non_tp = nn::VarStore::new(device);
//             let model_non_tp = nn::linear(
//                 vs_non_tp.root() / "layer",
//                 IN_FEATURES,
//                 OUT_FEATURES,
//                 nn::LinearConfig {
//                     bias: false,
//                     ..Default::default()
//                 },
//             );
//             let original_weights = model_non_tp.ws.copy();

//             let mut optimizer_non_tp = Distro::new(
//                 &vs_non_tp,
//                 COMPRESSION_DECAY,
//                 COMPRESSION_CHUNK,
//                 COMPRESSION_TOPK,
//                 WEIGHT_DECAY,
//                 None,
//             );

//             let dummy_barrier = Arc::new(Barrier::new(1));
//             let dummy_all_results = Arc::new(Mutex::new(HashMap::new()));

//             for step in 0..NUM_STEPS {
//                 let _ = run_distro_step(
//                     step,
//                     &model_non_tp,
//                     &input.lock().unwrap(),
//                     &target.lock().unwrap(),
//                     &mut optimizer_non_tp,
//                     TEST_LR,
//                     dummy_all_results.clone(),
//                     0,
//                     1,
//                     &None,
//                     &dummy_barrier,
//                 )?;
//             }

//             let mut final_weights = HashMap::new();
//             for (name, tensor) in vs_non_tp.variables() {
//                 final_weights.insert(name, tensor.detach().to_device(Device::Cpu));
//             }
//             (final_weights, original_weights)
//         };

//         let final_weights_tp_rank0 = Arc::new(Mutex::new(HashMap::new()));
//         let all_rank_results_tp: Arc<Mutex<HashMap<u32, Vec<Vec<DistroResult>>>>> =
//             Arc::new(Mutex::new(HashMap::new()));

//         {
//             let final_weights_tp_rank0 = final_weights_tp_rank0.clone();
//             let all_rank_results_tp = all_rank_results_tp.clone();
//             let ref_linear_weights = Arc::new(Mutex::new(linear_layer_weights));

//             run_parallel_test(
//                 WORLD_SIZE,
//                 move |comm_id, rank, barrier, device| -> anyhow::Result<()> {
//                     let vs_tp = nn::VarStore::new(device);
//                     let comm = Arc::new(CNCCL::new(
//                         comm_id.clone(),
//                         rank as i64,
//                         WORLD_SIZE as i64,
//                         device,
//                     )?);

//                     let mut model_tp = ColumnParallelLinear::new(
//                         vs_tp.root() / "layer",
//                         IN_FEATURES,
//                         OUT_FEATURES,
//                         false,
//                         true,
//                         Some(comm.clone()),
//                     );

//                     let (input, target) = {
//                         let _no_grad = tch::no_grad_guard();
//                         model_tp.linear.ws.copy_(&tensor_shard(
//                             &ref_linear_weights.lock().unwrap(),
//                             &Shard {
//                                 dim: 0,
//                                 rank,
//                                 world_size: WORLD_SIZE,
//                             },
//                         ));

//                         barrier.wait();

//                         comm.group_start().unwrap();
//                         if rank == 0 {
//                             let input = input.lock().unwrap();
//                             for i in 0..WORLD_SIZE {
//                                 comm.send(&[input.as_ref()], i as i64).unwrap();
//                             }
//                         }
//                         let input = Tensor::zeros(
//                             &[BATCH_SIZE, SEQ_LEN, IN_FEATURES],
//                             (Kind::Float, device),
//                         );
//                         comm.recv(&[input.shallow_clone()], 0).unwrap();
//                         comm.group_end().unwrap();

//                         barrier.wait();

//                         comm.group_start().unwrap();
//                         if rank == 0 {
//                             let target = target.lock().unwrap();
//                             for i in 0..WORLD_SIZE {
//                                 comm.send(&[target.as_ref()], i as i64).unwrap();
//                             }
//                         }
//                         let target = Tensor::zeros(
//                             &[BATCH_SIZE, SEQ_LEN, OUT_FEATURES],
//                             (Kind::Float, device),
//                         );
//                         comm.recv(&[target.shallow_clone()], 0).unwrap();
//                         comm.group_end().unwrap();

//                         barrier.wait();

//                         (input, target)
//                     };

//                     let mut optimizer_tp = Distro::new(
//                         &vs_tp,
//                         COMPRESSION_DECAY,
//                         COMPRESSION_CHUNK,
//                         COMPRESSION_TOPK,
//                         WEIGHT_DECAY,
//                         Some(comm.clone()),
//                     );

//                     for step in 0..NUM_STEPS {
//                         let current_rank_results = run_distro_step(
//                             step,
//                             &model_tp,
//                             &input,
//                             &target,
//                             &mut optimizer_tp,
//                             TEST_LR,
//                             all_rank_results_tp.clone(),
//                             rank,
//                             WORLD_SIZE,
//                             &Some(comm.clone()),
//                             &barrier,
//                         )?;
//                         let _ = current_rank_results;
//                         barrier.wait();
//                     }

//                     let unsharded_vars = unsharded_cpu_variables(&vs_tp, Some(comm.clone()))?;
//                     if rank == 0 {
//                         *final_weights_tp_rank0.lock().unwrap() = unsharded_vars;
//                     }

//                     Ok(())
//                 },
//             );
//         }

//         let final_weights_tp = final_weights_tp_rank0.lock().unwrap();

//         assert_eq!(
//             final_weights_non_tp.len(),
//             final_weights_tp.len(),
//             "Number of parameters differs between TP and non-TP runs."
//         );

//         for (name, non_tp_tensor) in &final_weights_non_tp {
//             let tp_tensor = final_weights_tp
//                 .get(name)
//                 .ok_or_else(|| anyhow::anyhow!("Parameter '{}' missing in TP results", name))?;

//             assert_eq!(
//                 non_tp_tensor.size(),
//                 tp_tensor.size(),
//                 "Shape mismatch for parameter '{}': Non-TP {:?}, TP {:?}",
//                 name,
//                 non_tp_tensor.size(),
//                 tp_tensor.size()
//             );

//             assert!(
//                 non_tp_tensor.allclose(tp_tensor, 1e-5, 1e-4, false),
//                 "Parameter '{}' differs significantly between TP and non-TP runs.\nNon-TP:\n{}\nTP:\n{}", name, non_tp_tensor, tp_tensor
//             );
//         }

//         Ok(())
//     }
// }

fn leviathan_robust_aggregate_median(mut values: Vec<f32>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid] as f64
    } else {
        (values[mid - 1] as f64 + values[mid] as f64) / 2.0
    }
}

pub fn leviathan_robust_aggregate(
    deltas: &Tensor,
    excision_multiplier: f64,
    iterations: usize,
) -> Tensor {
    let _no_grad = tch::no_grad_guard();
    let sizes = deltas.size();
    let n = sizes[0];
    let dim = sizes[1];
    let device = deltas.device();
    let cpu = Device::Cpu;
    let mut center = Tensor::zeros([dim], (Kind::Float, device));

    let initial = deltas
        .pow_tensor_scalar(2)
        .sum_dim_intlist(1, false, Kind::Float)
        .sqrt();
    let initial_cpu = Vec::<f32>::try_from(initial.to_device(cpu)).unwrap();
    let limit = excision_multiplier * leviathan_robust_aggregate_median(initial_cpu.clone());
    let mut mask = deltas
        .pow_tensor_scalar(2)
        .sum_dim_intlist(1, false, Kind::Float)
        .sqrt()
        .le(limit)
        .to_kind(Kind::Float);
    if mask.sum(Kind::Float).double_value(&[]) == 0.0 {
        mask = Tensor::ones([n], (Kind::Float, device));
    }
    let mask_cpu = Vec::<f32>::try_from(mask.to_device(cpu)).unwrap();

    for _ in 0..iterations {
        let diff = deltas - &center;
        let norms = diff
            .pow_tensor_scalar(2)
            .sum_dim_intlist(1, false, Kind::Float)
            .sqrt();
        let norms_cpu = Vec::<f32>::try_from(norms.to_device(cpu)).unwrap();
        let kept: Vec<f32> = norms_cpu
            .iter()
            .zip(mask_cpu.iter())
            .filter(|(_, m)| **m > 0.5)
            .map(|(value, _)| *value)
            .collect();
        let radius = leviathan_robust_aggregate_median(kept);
        let factor = norms
            .clamp_min(1e-12)
            .pow_tensor_scalar(-1.0)
            .f_mul_scalar(radius)
            .unwrap()
            .clamp_max(1.0);
        let weighted = &diff * (&factor * &mask).unsqueeze(1);
        let accum = weighted.sum_dim_intlist(0, false, Kind::Float);
        let n_kept = mask.sum(Kind::Float).double_value(&[]);
        center = center + accum.f_div_scalar(n_kept).unwrap();
    }
    center
}

#[cfg(test)]
mod leviathan_robust_tests {
    use super::*;

    #[test]
    fn robust_aggregate_down_weights_a_sign_flip_coalition() {
        tch::manual_seed(0);
        let dim = 512i64;
        let cpu = Device::Cpu;
        let honest_dir = Tensor::ones([dim], (Kind::Float, cpu));
        let mut rows: Vec<Tensor> = Vec::new();
        for _ in 0..11 {
            let noise = Tensor::randn([dim], (Kind::Float, cpu))
                .f_mul_scalar(0.05)
                .unwrap();
            rows.push(&honest_dir + &noise);
        }
        for _ in 0..5 {
            rows.push(honest_dir.f_mul_scalar(-1.0).unwrap());
        }
        let deltas = Tensor::stack(&rows, 0);

        let naive = deltas.sum(Kind::Float).double_value(&[]) / (dim as f64 * 16.0);
        let center = leviathan_robust_aggregate(&deltas, 3.0, 3);
        let robust = center.sum(Kind::Float).double_value(&[]) / dim as f64;

        assert!(
            robust > naive + 0.1,
            "robust {robust} did not pull away from the coalition (naive {naive})"
        );
        assert!(robust > 0.5, "robust {robust} still dominated by the coalition");
    }
}
