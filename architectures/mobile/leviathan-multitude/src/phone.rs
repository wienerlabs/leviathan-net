//! The phone side: what the mobile app embeds.

use std::time::Instant;

use candle_core::Device;
use ed25519_dalek::SigningKey;

use crate::adversary::{jitter, Behavior};
use crate::compress::{compress, selected_count, CompressedUpdate};
use crate::config::RunConfig;
use crate::data::{data_seed, Corpus};
use crate::protocol::{
    address, signing_key, weights_hash, Contribution, ContributionHeader, ErrorReveal,
};
use crate::rng::{sha256_parts, SplitMix64};
use crate::train::{pseudo_gradient, train_local};
use crate::Result;

/// Everything public about the round a phone is joining.
pub struct RoundJob<'a> {
    pub cfg: &'a RunConfig,
    pub run: [u8; 32],
    pub round: u32,
    pub round_seed: [u8; 32],
    pub global: &'a [f32],
    pub base_checkpoint: [u8; 32],
    pub corpus: &'a Corpus,
    pub device: &'a Device,
}

#[derive(Clone, Debug)]
pub struct LocalStats {
    pub first_loss: f32,
    pub mean_loss: f32,
    pub tokens: u64,
    pub upload_bytes: usize,
    pub seconds: f32,
}

/// One phone: its key and its error-feedback buffer.
pub struct PhoneNode {
    key: SigningKey,
    id: [u8; 32],
    error: Vec<f32>,
    round_start_error: Vec<f32>,
    last_round: Option<u32>,
}

impl PhoneNode {
    /// `secret` is the ed25519 seed. In the app it lives in the platform key
    /// store (Seed Vault on Solana phones, the Keychain or Keystore elsewhere).
    pub fn new(secret: [u8; 32], num_params: usize) -> Self {
        let key = signing_key(&secret);
        let id = key.verifying_key().to_bytes();
        Self {
            key,
            id,
            error: vec![0.0; num_params],
            round_start_error: vec![0.0; num_params],
            last_round: None,
        }
    }

    pub fn id(&self) -> [u8; 32] {
        self.id
    }

    pub fn address(&self) -> String {
        address(&self.id)
    }

    pub fn error_norm(&self) -> f32 {
        self.error.iter().map(|e| e * e).sum::<f32>().sqrt()
    }

    /// Train, compress and sign this phone's contribution to `job`.
    pub fn contribute(
        &mut self,
        job: &RoundJob,
        steps: u16,
        behavior: &Behavior,
    ) -> Result<(Contribution, LocalStats)> {
        let started = Instant::now();
        let start_error = self.error.clone();
        let commitment = weights_hash(&start_error);

        let (update, new_error, first_loss, mean_loss) = match behavior {
            Behavior::RandomVotes => {
                let update = random_votes(job, &self.id);
                (update, start_error.clone(), f32::NAN, f32::NAN)
            }
            _ => {
                let real_steps = match behavior {
                    Behavior::Lazy { real_steps } => (*real_steps).clamp(1, steps),
                    _ => steps,
                };
                let seed = data_seed(&job.round_seed, &self.id);
                let local = train_local(
                    job.cfg, job.global, job.corpus, &seed, real_steps, job.device,
                )?;
                let mut delta = pseudo_gradient(job.global, &local.params);
                if let Behavior::Jitter { scale } = behavior {
                    jitter(&mut delta, *scale, &self.id, job.round);
                }
                let (mut update, new_error) = compress(&delta, &start_error, &job.cfg.compression)?;
                if *behavior == Behavior::SignFlip {
                    update.flip_signs();
                }
                (update, new_error, local.first_loss, local.mean_loss)
            }
        };

        let header = ContributionHeader {
            run: job.run,
            round: job.round,
            phone: self.id,
            base_checkpoint: job.base_checkpoint,
            inner_steps: steps,
            tokens: steps as u64 * job.cfg.tokens_per_step(),
            mean_loss,
            error_commitment: commitment,
            update_hash: [0u8; 32],
        };
        let contribution = Contribution::sign(header, update, &self.key);

        self.round_start_error = start_error;
        self.error = new_error;
        self.last_round = Some(job.round);

        let stats = LocalStats {
            first_loss,
            mean_loss,
            tokens: steps as u64 * job.cfg.tokens_per_step(),
            upload_bytes: contribution.to_bytes().len(),
            seconds: started.elapsed().as_secs_f32(),
        };
        Ok((contribution, stats))
    }

    /// Answer an audit of `round` with the error buffer as it was when that
    /// round began.
    pub fn reveal(&self, round: u32) -> Option<ErrorReveal> {
        (self.last_round == Some(round)).then(|| ErrorReveal {
            round,
            phone: self.id,
            buffer: self.round_start_error.clone(),
        })
    }

    /// The contribution for `round` was not counted, so nothing it sent was
    /// applied: put the transmitted coefficients back into the error buffer.
    pub fn rollback(&mut self, round: u32) {
        if self.last_round == Some(round) {
            self.error = self.round_start_error.clone();
        }
    }
}

fn random_votes(job: &RoundJob, phone: &[u8; 32]) -> CompressedUpdate {
    let len = job.global.len();
    let chunk = job.cfg.compression.chunk;
    let topk = job.cfg.compression.topk;
    let mut rng = SplitMix64::from_digest(&sha256_parts(&[
        b"multitude/free-rider",
        phone,
        &job.round.to_le_bytes(),
    ]));
    let expected = selected_count(len, chunk, topk);
    let mut idx = Vec::with_capacity(expected);
    for start in (0..len).step_by(chunk) {
        let chunk_len = (len - start).min(chunk);
        let k = topk.min(chunk_len);
        let mut positions: Vec<u8> = (0..chunk_len).map(|j| j as u8).collect();
        for i in 0..k {
            let j = i + rng.below((chunk_len - i) as u64) as usize;
            positions.swap(i, j);
        }
        let mut chosen = positions[..k].to_vec();
        chosen.sort_unstable();
        idx.extend(chosen);
    }
    let mut signs = vec![0u8; expected.div_ceil(8)];
    for pos in 0..expected {
        if rng.next_u64() & 1 == 1 {
            signs[pos / 8] |= 1 << (pos % 8);
        }
    }
    CompressedUpdate {
        len: len as u32,
        chunk: chunk as u16,
        topk: topk as u16,
        idx,
        signs,
    }
}
