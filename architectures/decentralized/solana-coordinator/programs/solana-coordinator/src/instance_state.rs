use anchor_lang::prelude::*;
use bytemuck::Pod;
use bytemuck::Zeroable;
use psyche_coordinator::ClientState;
use psyche_coordinator::Coordinator;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::CoordinatorProgress;
use psyche_coordinator::HealthChecks;
use psyche_coordinator::RunState;
use psyche_coordinator::SOLANA_MAX_STRING_LEN;
use psyche_coordinator::TickResult;
use psyche_coordinator::Witness;
use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::Model;
use psyche_core::FixedString;
use psyche_core::NodeIdentity;
use psyche_core::SmallBoolean;
use serde::Deserialize;
use serde::Serialize;
use ts_rs::TS;

use crate::ProgramError;
use crate::client::Client;
use crate::clients_state::ClientsState;
use crate::round_seed::round_seed;
use crate::round_seed::slot_hash;

#[derive(
    Debug,
    Clone,
    Copy,
    Zeroable,
    AnchorSerialize,
    AnchorDeserialize,
    Serialize,
    Deserialize,
    TS,
    Default,
    PartialEq,
)]
#[repr(C)]
pub struct RunMetadata {
    pub name: FixedString<{ SOLANA_MAX_STRING_LEN }>,

    pub description: FixedString<280>,

    pub num_parameters: u64,
    pub vocab_size: u64,
}

impl RunMetadata {}

#[derive(
    Debug,
    Clone,
    Copy,
    Zeroable,
    AnchorSerialize,
    AnchorDeserialize,
    Serialize,
    Deserialize,
    TS,
)]
#[repr(C)]
pub struct CoordinatorInstanceState {
    pub metadata: RunMetadata,
    pub coordinator: Coordinator,
    pub clients_state: ClientsState,
    pub is_warmup_first_tick: SmallBoolean,
    pub is_training_first_tick: SmallBoolean,
    pub client_version: FixedString<96>,
    /// How many tie-breaker seats each round reserves, from the round that
    /// starts after it is set (`Coordinator::seat_tie_breakers`). The program
    /// running the run's appeals court sets it with its bench size, so that
    /// every party draws the same committee from the round. It sits in what
    /// was alignment padding before `seed_commit_slot`, so no other field
    /// moved, and an account written before it reads zero here: no seats, as
    /// before.
    pub tie_breaker_nodes: u16,
    /// Keeps `seed_commit_slot` where it was, and the `Pod` claim honest.
    pub tie_breaker_nodes_padding: [u8; 4],
    /// The slot the next round's seed is committed to: the slot of the first
    /// tick that found the round due, whose hash nobody knew when it was sent
    /// (wienerlabs/leviathan-net#62). Zero when none is.
    pub seed_commit_slot: u64,
}

unsafe impl Pod for CoordinatorInstanceState {}

/// `seed_commit_slot` when no round's seed is committed. Slot zero is the
/// genesis slot, which no transaction runs in, and it is what a new account
/// holds.
const NO_SEED_COMMITMENT: u64 = 0;

/// What a transaction can say of the next round's seed.
#[derive(Clone, Copy)]
enum SeedReveal {
    Seed(u64),
    /// Nothing is committed yet.
    Uncommitted,
    /// It is committed to this very slot, whose hash does not exist yet.
    Pending,
    /// The sysvar no longer holds the committed slot.
    Gone,
    /// This transaction has no sysvar to read the commitment from.
    Unknown,
}

impl CoordinatorInstanceState {
    /// Whether the next round's seed is committed and not yet revealed: until
    /// a tick in a later slot starts the round, the run takes no witness,
    /// health check or pause.
    pub fn seed_committed(&self) -> bool {
        self.seed_commit_slot != NO_SEED_COMMITMENT
    }

    /// What this transaction can say of the next round's seed, from the
    /// commitment and `slot_hashes`, the raw `SlotHashes` sysvar.
    fn reveal_round_seed(
        &self,
        clock: &Clock,
        slot_hashes: Option<&[u8]>,
    ) -> Result<SeedReveal> {
        let committed = self.seed_commit_slot;
        if committed == NO_SEED_COMMITMENT {
            return Ok(SeedReveal::Uncommitted);
        }
        // The committed slot's own hash exists only once the slot is done.
        if committed >= clock.slot {
            return Ok(SeedReveal::Pending);
        }
        let Some(slot_hashes) = slot_hashes else {
            return Ok(SeedReveal::Unknown);
        };
        match slot_hash(slot_hashes, committed)
            .map_err(|_| error!(ProgramError::SlotHashesUnreadable))?
        {
            Some(hash) => Ok(SeedReveal::Seed(round_seed(
                String::from(&self.coordinator.run_id).as_bytes(),
                self.coordinator.progress.step,
                committed,
                &hash,
            ))),
            None => Ok(SeedReveal::Gone),
        }
    }

    /// Settles the commitment after the run moved, or did not, from `before`.
    ///
    /// A round that started consumed it, and any other change of phase leaves
    /// the next round to commit afresh. A run left waiting for a seed commits
    /// it to this slot, when it has no commitment or the sysvar no longer
    /// holds the one it had. A transaction that cannot read the sysvar never
    /// replaces a commitment: it cannot tell whether one is gone, and
    /// replacing one would draw the round again.
    fn settle_round_seed(
        &mut self,
        before: RunState,
        reveal: SeedReveal,
        needs_seed: bool,
        clock: &Clock,
    ) {
        if self.coordinator.run_state != before {
            self.seed_commit_slot = NO_SEED_COMMITMENT;
        } else if needs_seed
            && matches!(reveal, SeedReveal::Uncommitted | SeedReveal::Gone)
        {
            msg!("Next round's seed committed to slot {}", clock.slot);
            self.seed_commit_slot = clock.slot;
        }
    }

    /// Refuses what would shape the next round once its seed is committed,
    /// and so about to be known: a witness, which decides whether the round
    /// starts; a health check, which drops clients from it; and a pause, which
    /// cancels it. A tick in a later slot starts the round, and all three are
    /// taken again.
    fn refuse_after_seed_commitment(&self) -> Result<()> {
        if self.seed_committed() {
            return err!(ProgramError::RoundSeedCommitted);
        }
        Ok(())
    }

    /// Advances the run. `slot_hashes` is the raw `SlotHashes` sysvar, which
    /// a round can only start with.
    pub fn tick(&mut self, slot_hashes: Option<&[u8]>) -> Result<()> {
        let active_clients_ids: Option<Vec<NodeIdentity>> =
            match self.coordinator.run_state {
                RunState::WaitingForMembers => {
                    // Reset state flags
                    self.is_warmup_first_tick = SmallBoolean::from(true);
                    self.is_training_first_tick = SmallBoolean::from(true);

                    let active_clients_ids =
                        self.clients_state.get_active_clients_ids();
                    msg!(
                        "Pending active clients ids: {}",
                        active_clients_ids.len()
                    );
                    Some(active_clients_ids.collect())
                },
                _ => None,
            };

        msg!("Pre-tick run state: {}", self.coordinator.run_state);

        let clock: Clock = Clock::get()?;
        let before = self.coordinator.run_state;
        let reveal = self.reveal_round_seed(&clock, slot_hashes)?;
        let random_seed = match reveal {
            SeedReveal::Seed(seed) => Some(seed),
            _ => None,
        };
        let ticked = self.coordinator.tick(
            active_clients_ids.as_ref().map(|v| v.iter()),
            clock.unix_timestamp as u64,
            random_seed,
        );
        self.settle_round_seed(
            before,
            reveal,
            matches!(ticked, Ok(TickResult::NeedsSeed)),
            &clock,
        );
        // A round starts only into `RoundTrain`, from the warmup or the last
        // round's witness phase. Nobody has drawn it yet: its seed was revealed
        // in this very tick.
        if before != RunState::RoundTrain
            && self.coordinator.run_state == RunState::RoundTrain
        {
            self.coordinator.seat_tie_breakers(self.tie_breaker_nodes);
        }
        match ticked {
            Ok(TickResult::NeedsSeed) => {
                msg!("A round is due and waits for its seed");
            },
            Ok(TickResult::Ticked) => {
                if self.coordinator.is_warmup_just_starting()
                    && self.is_warmup_first_tick.is_true()
                {
                    msg!("New epoch just starting, save epoch rewards rate");
                    self.clients_state.current_epoch_rates =
                        self.clients_state.future_epoch_rates;
                    self.is_warmup_first_tick = SmallBoolean::from(false);
                } else if self.coordinator.is_training_just_starting()
                    && self.is_training_first_tick.is_true()
                {
                    msg!("New epoch just starting, save epoch active clients");
                    self.clients_state.next_active += 1;
                    self.is_training_first_tick = SmallBoolean::from(false);
                }
            },
            Ok(TickResult::EpochEnd(success)) => {
                msg!("Epoch end, sucecsss: {}", success);

                // This is where a conviction becomes money. Every slash path in
                // the treasurer ends in `eject`, which only sets a state flag;
                // the flag is charged against the bond here, once per epoch.
                //
                // Match by identity. The permanent records are in join order and
                // `exited_clients` is in exit order - `move_clients_to_exited`
                // appends a fresh batch at the end of every round - so the two
                // lists disagree whenever a client that joined earlier exits
                // later. A single forward cursor over both walks past that
                // client and never comes back, and its slash is dropped in
                // silence (wienerlabs/leviathan#15, finding 1).
                let finished_clients = &self.coordinator.epoch_state.clients;
                let exited_clients =
                    &self.coordinator.epoch_state.exited_clients;

                let earned_share = match finished_clients.len() {
                    0 => 0,
                    finishers => self
                        .clients_state
                        .current_epoch_rates
                        .earning_rate_total_shared
                        .saturating_div(finishers as u64),
                };
                let slashing_rate = self
                    .clients_state
                    .current_epoch_rates
                    .slashing_rate_per_client;

                for client in self.clients_state.clients.iter_mut() {
                    if finished_clients.iter().any(|finished| {
                        finished.id == client.id
                            && finished.state == ClientState::Healthy
                    }) {
                        client.earned += earned_share;
                    }
                    if exited_clients.iter().any(|exited| {
                        exited.id == client.id
                            && exited.state == ClientState::Ejected
                    }) {
                        client.slashed += slashing_rate;
                    }
                }
            },
            Err(err) => return err!(ProgramError::from(err)),
        };

        msg!("Post-tick run state: {}", self.coordinator.run_state);
        Ok(())
    }

    pub fn set_paused(&mut self, paused: bool) -> Result<()> {
        let unix_timestamp = Clock::get()?.unix_timestamp as u64;
        msg!(
            "set_paused called: paused={}, state={}",
            paused,
            self.coordinator.run_state
        );
        if paused {
            // Whoever can pause could otherwise cancel a round whose seed it
            // dislikes, and draw again. A tick before the pause, in the same
            // transaction, starts the round, and the pause lets it finish.
            self.refuse_after_seed_commitment()?;
        }
        if let Err(err) = match paused {
            true => self.coordinator.pause(unix_timestamp),
            false => {
                if !self.coordinator.config.check() {
                    return err!(ProgramError::ConfigSanityCheckFailed);
                }
                if !self.coordinator.model.check() {
                    return err!(ProgramError::ModelSanityCheckFailed);
                }
                if !self.coordinator.check_cold_start_warmup_steps() {
                    return err!(ProgramError::ConfigSanityCheckFailed);
                }

                if self.coordinator.run_state == RunState::Uninitialized {
                    // this is the only way to get out of RunState::Uninitialized
                    // by doing this we force the sanity checks on the config and model
                    // pass before starting the first step
                    self.coordinator.run_state = RunState::Paused;
                    // step 1 is the first valid step
                    self.coordinator.progress.step = 1;
                }

                // check progress after potentially setting step to 1 (errors for default 0)
                if !self.coordinator.progress.check() {
                    return err!(ProgramError::ProgressSanityCheckFailed);
                }

                self.coordinator.resume(unix_timestamp)
            },
        } {
            return err!(ProgramError::from(err));
        }

        if paused {
            Ok(()) // do not tick when setting paused, tick() errors when paused
        } else {
            // clear all active joins -- require that everyone re-join
            self.clients_state.next_active += 1;

            // Resuming starts no round: the run waits for members first.
            self.tick(None)
        }
    }

    pub fn witness(
        &mut self,
        payer: &Pubkey,
        witness: Witness,
        slot_hashes: &[u8],
    ) -> Result<()> {
        let id = self.clients_state.find_signer(payer)?;

        self.refuse_after_seed_commitment()?;

        let clock: Clock = Clock::get()?;
        self.coordinator
            .witness(&id, witness, clock.unix_timestamp as u64)
            .map_err(|err| anchor_lang::error!(ProgramError::from(err)))?;

        self.tick(Some(slot_hashes))
    }

    pub fn warmup_witness(
        &mut self,
        payer: &Pubkey,
        witness: Witness,
        slot_hashes: &[u8],
    ) -> Result<()> {
        let id = self.clients_state.find_signer(payer)?;

        self.refuse_after_seed_commitment()?;

        // No seed is committed while witnesses are taken, so the last warmup
        // witness never starts the first round itself: the warmup's next tick
        // commits its seed.
        let clock: Clock = Clock::get()?;
        self.coordinator
            .warmup_witness(&id, witness, clock.unix_timestamp as u64, None)
            .map_err(|err| anchor_lang::error!(ProgramError::from(err)))?;

        self.tick(Some(slot_hashes))
    }

    /// Sets how many tie-breaker seats the rounds reserve, from the next round
    /// to start.
    ///
    /// The count moves the verifier seats' window, so it is refused once the
    /// next round's seed is committed: by then the committed slot's hash is
    /// knowable, and choosing the count afterwards would choose who verifies.
    pub fn set_tie_breaker_nodes(&mut self, tie_breaker_nodes: u16) -> Result<()> {
        self.refuse_after_seed_commitment()?;
        self.tie_breaker_nodes = tie_breaker_nodes;
        Ok(())
    }

    pub fn set_future_epoch_rates(
        &mut self,
        epoch_earning_rate_total_shared: Option<u64>,
        epoch_slashing_rate_per_client: Option<u64>,
    ) -> Result<()> {
        if let Some(epoch_earning_rate_total_shared) =
            epoch_earning_rate_total_shared
        {
            self.clients_state
                .future_epoch_rates
                .earning_rate_total_shared = epoch_earning_rate_total_shared;
        }
        if let Some(epoch_slashing_rate_per_client) =
            epoch_slashing_rate_per_client
        {
            self.clients_state
                .future_epoch_rates
                .slashing_rate_per_client = epoch_slashing_rate_per_client;
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        metadata: Option<RunMetadata>,
        config: Option<CoordinatorConfig>,
        model: Option<Model>,
        progress: Option<CoordinatorProgress>,
    ) -> Result<()> {
        if self.coordinator.run_state == RunState::Finished {
            return err!(ProgramError::UpdateConfigFinished);
        } else if !self.coordinator.halted()
            // these can't be updated without pausing
            // but metadata can be updated without pausing so it's not included here
            && (config.is_some() || model.is_some() || progress.is_some())
        {
            return err!(ProgramError::UpdateConfigNotHalted);
        }

        if let Some(metadata) = metadata {
            let _ = std::mem::replace(&mut self.metadata, metadata);
        }

        if let Some(config) = config {
            if !config.check() {
                return err!(ProgramError::ConfigSanityCheckFailed);
            }

            let _ = std::mem::replace(&mut self.coordinator.config, config);
        }

        if let Some(model) = model {
            if !model.check() {
                return err!(ProgramError::ModelSanityCheckFailed);
            }
            // Every node builds its optimizer from the run's definition when it
            // joins, and keeps it. A definition changed once the run has
            // started would reach only the nodes that join after the change,
            // and two nodes holding different optimizers, a different
            // aggregation rule above all, apply different updates to the same
            // results (wienerlabs/leviathan-net#60). Before the first start no
            // node has built one, and the run can be set up freely.
            let Model::LLM(current) = self.coordinator.model;
            let Model::LLM(proposed) = model;
            if self.coordinator.run_state != RunState::Uninitialized
                && proposed.optimizer != current.optimizer
            {
                return err!(ProgramError::ModelOptimizerFrozen);
            }

            let _ = std::mem::replace(&mut self.coordinator.model, model);
        }

        if (config.is_some() || model.is_some())
            && !self.coordinator.check_cold_start_warmup_steps()
        {
            return err!(ProgramError::ConfigSanityCheckFailed);
        }

        if let Some(progress) = progress {
            if !progress.check() {
                return err!(ProgramError::ModelSanityCheckFailed);
            }

            let _ = std::mem::replace(&mut self.coordinator.progress, progress);
        }

        Ok(())
    }

    pub fn join_run(&mut self, id: NodeIdentity) -> Result<()> {
        let existing =
            match self.clients_state.clients.iter_mut().find(|x| x.id == id) {
                Some(client) => {
                    // IMPORTANT
                    // NodeIdentity equality is on wallet key but includes the p2p key, so
                    // we must re-set this to ensure that the p2p key is up to date.
                    client.id = id;
                    client.active = self.clients_state.next_active;
                    msg!("Existing client {} re-joined", id);
                    true
                },
                None => false,
            };

        if !existing {
            let total_num_clients = self.clients_state.clients.len() as u16;
            if self.coordinator.config.global_batch_size_end
                == total_num_clients
            {
                return err!(ProgramError::MoreClientsThanBatches);
            }

            let new_client = Client {
                id,
                earned: 0,
                slashed: 0,
                active: self.clients_state.next_active,
                _unused: Default::default(),
            };

            if self.clients_state.clients.push(new_client).is_err() {
                return err!(ProgramError::ClientsFull);
            }
            msg!(
                "New client {} joined, {} total clients",
                id,
                self.clients_state.clients.len()
            );
        }

        // A join starts no round: joins are taken while the run waits for
        // members.
        if !self.coordinator.halted() {
            self.tick(None)
        } else {
            Ok(())
        }
    }

    pub fn health_check(
        &mut self,
        payer: &Pubkey,
        checks: HealthChecks,
        slot_hashes: &[u8],
    ) -> Result<()> {
        // O(n) on clients, reconsider
        let id = self.clients_state.find_signer(payer)?;
        self.refuse_after_seed_commitment()?;

        self.coordinator
            .health_check(&id, checks)
            .map_err(|err| anchor_lang::error!(ProgramError::from(err)))?;
        self.tick(Some(slot_hashes))
    }

    pub fn slash(&mut self, index: u64) -> Result<()> {
        self.coordinator
            .eject(index)
            .map_err(|err| anchor_lang::error!(ProgramError::from(err)))?;
        Ok(())
    }

    pub fn checkpoint(
        &mut self,
        payer: &Pubkey,
        repo: Checkpoint,
        slot_hashes: &[u8],
    ) -> Result<()> {
        // O(n) on clients, reconsider
        let id = self.clients_state.find_signer(payer)?;
        let index = self
            .coordinator
            .epoch_state
            .clients
            .iter()
            .position(|x| x.id == id)
            .ok_or(ProgramError::SignerNotAClient)?;

        self.coordinator
            .checkpoint(&id, index as u64, repo)
            .map_err(|err| anchor_lang::error!(ProgramError::from(err)))?;

        // Only tick if not halted (Paused/Uninitialized/Finished)
        // Checkpoint update itself doesn't require state transitions
        if !self.coordinator.halted() {
            self.tick(Some(slot_hashes))
        } else {
            msg!(
                "Checkpoint recorded while halted (state: {}), skipping tick",
                self.coordinator.run_state
            );
            Ok(())
        }
    }
}
