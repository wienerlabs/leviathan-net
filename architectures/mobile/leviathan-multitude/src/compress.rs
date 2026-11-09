//! SparseLoCo transport: chunked top-k, one sign bit per coefficient, and
//! error feedback.
//!
//! The parameters are cut into chunks of `chunk`. In each chunk the phone keeps
//! the `topk` coefficients of largest magnitude in `error_decay * error +
//! delta` and sends only their positions and signs. The transmitted
//! coefficients are zeroed in the error buffer and everything else carries over
//! to the next round, so nothing the phone learned is dropped, only delayed.
//! At the defaults (64, 8) that is 8 bits of position plus 1 bit of sign per 8
//! coefficients of 64: about 1.1 bits per parameter.
//!
//! The encoding is canonical (positions ascending within a chunk, padding bits
//! zero), so one update has exactly one byte string and one hash.

use serde::{Deserialize, Serialize};

use crate::config::CompressionConfig;
use crate::rng::sha256_parts;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompressError {
    #[error("delta and error buffer have different lengths ({delta} vs {error})")]
    LengthMismatch { delta: usize, error: usize },
    #[error("accumulated value at index {index} is not finite")]
    NonFinite { index: usize },
    #[error("chunk {chunk} must be in 1..=256 and topk {topk} in 1..=chunk")]
    BadShape { chunk: usize, topk: usize },
    #[error("update covers {got} parameters, expected {expected}")]
    WrongLength { got: usize, expected: usize },
    #[error("update carries {got} positions, expected {expected}")]
    WrongCount { got: usize, expected: usize },
    #[error("update carries {got} sign bytes, expected {expected}")]
    WrongSignBytes { got: usize, expected: usize },
    #[error("padding bits after the last sign are not zero")]
    NonCanonicalPadding,
    #[error("position {index} is out of range in chunk {chunk} of length {len}")]
    IndexOutOfRange { chunk: usize, index: u8, len: usize },
    #[error("positions in chunk {chunk} are not strictly ascending")]
    NotCanonical { chunk: usize },
}

/// How many coefficients an update over `len` parameters selects.
pub fn selected_count(len: usize, chunk: usize, topk: usize) -> usize {
    if chunk == 0 {
        return 0;
    }
    (len / chunk) * topk.min(chunk) + topk.min(len % chunk)
}

fn check_shape(chunk: usize, topk: usize) -> Result<(), CompressError> {
    if chunk == 0 || chunk > 256 || topk == 0 || topk > chunk {
        return Err(CompressError::BadShape { chunk, topk });
    }
    Ok(())
}

/// One phone's vote for one round.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressedUpdate {
    pub len: u32,
    pub chunk: u16,
    pub topk: u16,
    /// Position of each selected coefficient inside its chunk, ascending
    /// within every chunk.
    pub idx: Vec<u8>,
    /// One bit per entry of `idx`, least significant bit first. A set bit
    /// means the coefficient was negative.
    pub signs: Vec<u8>,
}

impl CompressedUpdate {
    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn selected(&self) -> usize {
        self.idx.len()
    }

    /// Payload size: positions plus packed signs.
    pub fn payload_bytes(&self) -> usize {
        self.idx.len() + self.signs.len()
    }

    fn negative(&self, pos: usize) -> bool {
        (self.signs[pos / 8] >> (pos % 8)) & 1 == 1
    }

    fn global_index(&self, pos: usize) -> usize {
        let chunk = self.chunk as usize;
        let per_chunk = (self.topk as usize).min(chunk);
        (pos / per_chunk) * chunk + self.idx[pos] as usize
    }

    /// `(parameter index, +1 or -1)` for every coefficient this update votes
    /// on. Only meaningful on an update that passed [`Self::validate`].
    pub fn votes(&self) -> impl Iterator<Item = (usize, i8)> + '_ {
        (0..self.idx.len()).map(move |pos| {
            (
                self.global_index(pos),
                if self.negative(pos) { -1 } else { 1 },
            )
        })
    }

    /// The update as a dense vector of -1, 0 and +1: the form Leviathan's
    /// replay verifier compares.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0f32; self.len as usize];
        for (index, vote) in self.votes() {
            dense[index] = vote as f32;
        }
        dense
    }

    /// Flip every sign. Padding bits stay zero, so the result is canonical.
    pub fn flip_signs(&mut self) {
        let selected = self.idx.len();
        for pos in 0..selected {
            self.signs[pos / 8] ^= 1 << (pos % 8);
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        let encoded = postcard::to_stdvec(self).expect("an update always serializes");
        sha256_parts(&[b"multitude/update/v1", &encoded])
    }

    /// Check that this update is the canonical encoding of a vote over `len`
    /// parameters with the run's compression settings.
    pub fn validate(&self, len: usize, cfg: &CompressionConfig) -> Result<(), CompressError> {
        check_shape(cfg.chunk, cfg.topk)?;
        if self.len as usize != len {
            return Err(CompressError::WrongLength {
                got: self.len as usize,
                expected: len,
            });
        }
        if self.chunk as usize != cfg.chunk || self.topk as usize != cfg.topk {
            return Err(CompressError::BadShape {
                chunk: self.chunk as usize,
                topk: self.topk as usize,
            });
        }
        let expected = selected_count(len, cfg.chunk, cfg.topk);
        if self.idx.len() != expected {
            return Err(CompressError::WrongCount {
                got: self.idx.len(),
                expected,
            });
        }
        let sign_bytes = expected.div_ceil(8);
        if self.signs.len() != sign_bytes {
            return Err(CompressError::WrongSignBytes {
                got: self.signs.len(),
                expected: sign_bytes,
            });
        }
        if !expected.is_multiple_of(8) && self.signs[sign_bytes - 1] >> (expected % 8) != 0 {
            return Err(CompressError::NonCanonicalPadding);
        }
        let mut pos = 0;
        for (chunk_number, start) in (0..len).step_by(cfg.chunk).enumerate() {
            let chunk_len = (len - start).min(cfg.chunk);
            let k = cfg.topk.min(chunk_len);
            let positions = &self.idx[pos..pos + k];
            for (offset, &index) in positions.iter().enumerate() {
                if index as usize >= chunk_len {
                    return Err(CompressError::IndexOutOfRange {
                        chunk: chunk_number,
                        index,
                        len: chunk_len,
                    });
                }
                if offset > 0 && positions[offset - 1] >= index {
                    return Err(CompressError::NotCanonical {
                        chunk: chunk_number,
                    });
                }
            }
            pos += k;
        }
        Ok(())
    }
}

/// Compress a pseudo-gradient. Returns the update and the new error buffer.
pub fn compress(
    delta: &[f32],
    error: &[f32],
    cfg: &CompressionConfig,
) -> Result<(CompressedUpdate, Vec<f32>), CompressError> {
    if delta.len() != error.len() {
        return Err(CompressError::LengthMismatch {
            delta: delta.len(),
            error: error.len(),
        });
    }
    check_shape(cfg.chunk, cfg.topk)?;
    let len = delta.len();
    let mut acc = Vec::with_capacity(len);
    for (index, (d, e)) in delta.iter().zip(error).enumerate() {
        let value = cfg.error_decay * e + d;
        if !value.is_finite() {
            return Err(CompressError::NonFinite { index });
        }
        acc.push(value);
    }

    let expected = selected_count(len, cfg.chunk, cfg.topk);
    let mut idx: Vec<u8> = Vec::with_capacity(expected);
    let mut signs = vec![0u8; expected.div_ceil(8)];
    let mut order: Vec<u8> = Vec::with_capacity(cfg.chunk);
    for start in (0..len).step_by(cfg.chunk) {
        let end = (start + cfg.chunk).min(len);
        let k = cfg.topk.min(end - start);
        let chunk = &acc[start..end];
        order.clear();
        order.extend((0..end - start).map(|j| j as u8));
        // Largest magnitude first; ties go to the lower position, so the
        // choice never depends on the sort algorithm.
        order.sort_unstable_by(|&a, &b| {
            chunk[b as usize]
                .abs()
                .total_cmp(&chunk[a as usize].abs())
                .then(a.cmp(&b))
        });
        let selected = &mut order[..k];
        selected.sort_unstable();
        for &j in selected.iter() {
            let pos = idx.len();
            if chunk[j as usize] < 0.0 {
                signs[pos / 8] |= 1 << (pos % 8);
            }
            idx.push(j);
        }
        for &j in &idx[idx.len() - k..] {
            acc[start + j as usize] = 0.0;
        }
    }
    Ok((
        CompressedUpdate {
            len: len as u32,
            chunk: cfg.chunk as u16,
            topk: cfg.topk as u16,
            idx,
            signs,
        },
        acc,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(chunk: usize, topk: usize) -> CompressionConfig {
        CompressionConfig {
            chunk,
            topk,
            error_decay: 1.0,
        }
    }

    #[test]
    fn keeps_the_largest_magnitudes_with_their_signs() {
        let delta = [0.1, -0.9, 0.3, 0.05, 0.7, -0.2, 0.0, -0.6];
        let (update, error) = compress(&delta, &[0.0; 8], &cfg(4, 2)).unwrap();
        update.validate(8, &cfg(4, 2)).unwrap();
        let votes: Vec<_> = update.votes().collect();
        assert_eq!(votes, vec![(1, -1), (2, 1), (4, 1), (7, -1)]);
        assert_eq!(error, vec![0.1, 0.0, 0.0, 0.05, 0.0, -0.2, 0.0, 0.0]);
    }

    #[test]
    fn error_feedback_carries_what_was_not_sent() {
        let c = cfg(4, 1);
        let (_, error) = compress(&[0.5, 0.4, 0.0, 0.0], &[0.0; 4], &c).unwrap();
        assert_eq!(error, vec![0.0, 0.4, 0.0, 0.0]);
        // Next round the carried 0.4 plus a small new delta wins the chunk.
        let (update, _) = compress(&[0.0, 0.1, 0.3, 0.0], &error, &c).unwrap();
        assert_eq!(update.votes().collect::<Vec<_>>(), vec![(1, 1)]);
    }

    #[test]
    fn handles_a_short_last_chunk() {
        let c = cfg(4, 3);
        let delta = [1.0, 2.0, 3.0, 4.0, -5.0, 6.0];
        let (update, _) = compress(&delta, &[0.0; 6], &c).unwrap();
        assert_eq!(update.selected(), selected_count(6, 4, 3));
        update.validate(6, &c).unwrap();
        assert_eq!(
            update.votes().collect::<Vec<_>>(),
            vec![(1, 1), (2, 1), (3, 1), (4, -1), (5, 1)]
        );
    }

    #[test]
    fn ties_are_broken_by_position() {
        let c = cfg(4, 2);
        let (a, _) = compress(&[0.5, 0.5, 0.5, 0.5], &[0.0; 4], &c).unwrap();
        assert_eq!(a.idx, vec![0, 1]);
    }

    #[test]
    fn rejects_non_finite_input() {
        let err = compress(&[0.0, f32::NAN], &[0.0; 2], &cfg(2, 1)).unwrap_err();
        assert_eq!(err, CompressError::NonFinite { index: 1 });
    }

    #[test]
    fn validate_rejects_tampering() {
        let c = cfg(4, 2);
        let (update, _) =
            compress(&[0.1, -0.9, 0.3, 0.05, 0.7, -0.2, 0.0, -0.6], &[0.0; 8], &c).unwrap();

        let mut swapped = update.clone();
        swapped.idx.swap(0, 1);
        assert!(matches!(
            swapped.validate(8, &c),
            Err(CompressError::NotCanonical { .. })
        ));

        let mut padded = update.clone();
        padded.signs[0] |= 0b1000_0000;
        assert_eq!(
            padded.validate(8, &c),
            Err(CompressError::NonCanonicalPadding)
        );

        let mut short = update.clone();
        short.idx.pop();
        assert!(matches!(
            short.validate(8, &c),
            Err(CompressError::WrongCount { .. })
        ));

        let mut out_of_range = update;
        out_of_range.idx[1] = 9;
        assert!(matches!(
            out_of_range.validate(8, &c),
            Err(CompressError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn flipping_signs_stays_canonical() {
        let c = cfg(4, 3);
        let (mut update, _) = compress(&[1.0, -2.0, 3.0, 4.0, 5.0], &[0.0; 5], &c).unwrap();
        let before = update.to_dense();
        update.flip_signs();
        update.validate(5, &c).unwrap();
        let after = update.to_dense();
        assert!(before.iter().zip(&after).all(|(a, b)| *a == -*b));
    }
}
