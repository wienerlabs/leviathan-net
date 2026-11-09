//! The global model and how a round advances it.

use crate::aggregate::Tally;
use crate::config::RunConfig;
use crate::model::genesis_params;
use crate::protocol::weights_hash;
use crate::rng::sha256_parts;
use crate::Result;

#[derive(Clone, Debug)]
pub struct GlobalModel {
    pub round: u32,
    pub seed: [u8; 32],
    pub params: Vec<f32>,
}

impl GlobalModel {
    pub fn genesis(cfg: &RunConfig, genesis_seed: u64) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            round: 0,
            seed: sha256_parts(&[b"multitude/genesis", &genesis_seed.to_le_bytes()]),
            params: genesis_params(&cfg.model, genesis_seed)?,
        })
    }

    pub fn checkpoint(&self) -> [u8; 32] {
        weights_hash(&self.params)
    }

    /// Apply a verified tally and move to the next round.
    ///
    /// The next seed chains the previous seed with the tally hash. That is
    /// enough for a devnet run, but a relay choosing which contributions to
    /// include could grind it, which is the risk the Leviathan roadmap lists
    /// for seeds built from timestamps; a production run takes its seed from a
    /// recent Solana blockhash or a VRF.
    pub fn advance(&mut self, tally: &Tally, outer_lr: f32) {
        tally.apply(&mut self.params, outer_lr);
        self.seed = sha256_parts(&[
            b"multitude/round-seed",
            &self.seed,
            &self.round.to_le_bytes(),
            &tally.hash(),
        ]);
        self.round += 1;
    }
}
