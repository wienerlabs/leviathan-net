//! What a run's accounts, read together at one slot, say about it: the values
//! `leviathan-indexer` exports.

use psyche_coordinator::ClientState;
use psyche_coordinator::CommitteeSelection;
use psyche_coordinator::Coordinator;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_treasurer::logic::MIN_VERIFIER_COMMITTEE;
use psyche_solana_treasurer::state::committee_matches_roster;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::Run;

use crate::economics::assess_security;
use crate::economics::audit_probability;
use crate::economics::chain_economics;
use crate::economics::healthy_clients;
use crate::economics::is_training;
use crate::economics::rates_in_force;
use crate::economics::SecurityAssessment;
use crate::verdicts::RoundPosition;
use crate::verdicts::VerdictLedger;
use crate::verdicts::VerdictTally;

/// Every state a client of an epoch can be in, in the order
/// [`RunObservation::epoch_clients`] counts them, and the name each is
/// exported under.
pub const CLIENT_STATES: [(ClientState, &str); 4] = [
    (ClientState::Healthy, "healthy"),
    (ClientState::Dropped, "dropped"),
    (ClientState::Withdrawn, "withdrawn"),
    (ClientState::Ejected, "ejected"),
];

/// The slot a snapshot was read at, from the Clock sysvar read in the same
/// request as the run's accounts, and the newest slot the RPC knew of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotClock {
    pub slot: u64,
    pub unix_timestamp: i64,
    pub tip_slot: u64,
}

impl SnapshotClock {
    /// How many slots the snapshot trails the RPC's newest slot.
    pub fn slot_lag(&self) -> u64 {
        self.tip_slot.saturating_sub(self.slot)
    }
}

/// A treasurer-managed run's accounts, read at the same slot as its
/// coordinator, and its verdict accounts.
pub struct TreasurerAccounts<'a> {
    pub run: &'a Run,
    /// The collateral the run's vault holds.
    pub vault_balance: u64,
    /// The verdict accounts of the run's registered clients that exist.
    pub verdicts: &'a [AuditVerdict],
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunObservation {
    pub run_state: u8,
    pub training: bool,
    pub epoch: u16,
    pub step: u32,
    /// Every client the coordinator ever registered for the run.
    pub registered_clients: u64,
    /// The clients of this epoch's roster, healthy or not, including those
    /// that exited it, by [`CLIENT_STATES`].
    pub epoch_clients: [u64; 4],
    pub min_clients: u16,
    pub audit_probability: f64,
    pub earned_points: u64,
    pub slashed_points: u64,
    pub convicted_clients: u64,
    pub clock: SnapshotClock,
    pub treasurer: Option<TreasurerObservation>,
}

impl RunObservation {
    pub fn active_clients(&self) -> u64 {
        self.epoch_clients[0]
    }

    pub fn epoch_roster_size(&self) -> u64 {
        self.epoch_clients.iter().sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TreasurerObservation {
    /// The verifier seats of the committee the round under way drew, or, with
    /// none standing over the roster, the seats the next draw would give.
    pub verifier_committee_size: u64,
    /// Whether a committee stands over the roster now, the condition the
    /// treasurer takes votes under (`committee_matches_roster`).
    pub committee_drawn: bool,
    /// The smallest committee the treasurer takes a verdict from.
    pub min_verifier_committee: u64,
    /// Points the epoch in force shares out among the clients that finish it:
    /// the epoch under way, or between epochs the next one.
    pub epoch_earning_rate: u64,
    /// Points a conviction charges in the epoch in force.
    pub epoch_slashing_rate: u64,
    pub bond_minimum: u64,
    /// The verdict at the probability a cheat is audited by a committee that
    /// can convict it: none, when the committee is smaller than the treasurer
    /// takes a verdict from.
    pub security: SecurityAssessment,
    pub vault_balance: u64,
    pub bonded_amount: u64,
    /// Points the coordinator credited that nobody has claimed yet.
    pub unclaimed_earnings: u64,
    pub claimed_collateral: u64,
    /// How far the vault falls short of the bonds posted into it.
    pub bond_shortfall: u64,
    /// How far the vault falls short of everything it owes: the bonds, and the
    /// earnings not yet claimed.
    pub obligation_shortfall: u64,
    pub verdicts: VerdictTally,
}

/// The committee a run's votes are cast by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitteeView {
    pub size: u64,
    pub drawn: bool,
}

/// The committee of the round under way, sized over the roster the round was
/// drawn over (`round.clients_len`), as the treasurer sizes it. With none
/// standing over the roster, in a warmup or once the roster moved under the
/// draw, the next draw's over the clients still healthy, and between epochs
/// over the smallest roster an epoch starts with.
pub fn committee(coordinator: &Coordinator, tie_breakers: u16) -> CommitteeView {
    let drawn = is_training(coordinator) && committee_matches_roster(coordinator);
    let roster = match coordinator.current_round() {
        Some(round) if drawn => round.clients_len as usize,
        _ if is_training(coordinator) => healthy_clients(coordinator),
        _ => coordinator.config.min_clients as usize,
    };
    CommitteeView {
        size: verifier_committee_size(coordinator, roster, tie_breakers),
        drawn,
    }
}

/// Whether cheating pays in a treasurer-managed run, from the epoch rates in
/// force, the run's bond minimum, and the probability a committee that can
/// convict audits a cheat. A committee smaller than the treasurer takes a
/// verdict from convicts nobody, so a run that has one is audited with
/// probability zero, whatever its `verification_percent`; the run authority's
/// own slash is not counted, since the security argument is the committee's.
pub fn treasurer_security(state: &CoordinatorInstanceState, run: &Run) -> SecurityAssessment {
    let coordinator = &state.coordinator;
    let rates = rates_in_force(
        coordinator,
        state.clients_state.current_epoch_rates,
        state.clients_state.future_epoch_rates,
    );
    let convicting =
        committee(coordinator, run.tie_breaker_committee_size).size >= MIN_VERIFIER_COMMITTEE;
    let probability = if convicting {
        audit_probability(coordinator)
    } else {
        0.0
    };
    assess_security(
        probability,
        &chain_economics(coordinator, &rates, run.bond_minimum_amount),
    )
}

/// The verifier seats the coordinator draws for a round over `roster`
/// clients, with the run's tie-breaker seats set aside: the count the
/// treasurer sizes a quorum by.
pub fn verifier_committee_size(coordinator: &Coordinator, roster: usize, tie_breakers: u16) -> u64 {
    CommitteeSelection::new(
        tie_breakers as usize,
        coordinator.config.witness_nodes as usize,
        coordinator.config.verification_percent,
        roster,
        0,
    )
    .map(|selection| selection.get_num_verifier_nodes())
    // A roster smaller than the tie-breaker seats has no verifier seat left.
    .unwrap_or(0)
}

fn epoch_clients(coordinator: &Coordinator) -> [u64; 4] {
    let mut counts = [0u64; 4];
    let roster = coordinator.epoch_state.clients.iter();
    let exited = coordinator.epoch_state.exited_clients.iter();
    for client in roster.chain(exited) {
        let position = CLIENT_STATES
            .iter()
            .position(|(state, _)| *state == client.state)
            .expect("CLIENT_STATES lists every state");
        counts[position] += 1;
    }
    counts
}

fn observe_treasurer(
    state: &CoordinatorInstanceState,
    accounts: &TreasurerAccounts,
    ledger: &mut VerdictLedger,
) -> TreasurerObservation {
    let coordinator = &state.coordinator;
    let run = accounts.run;
    let rates = rates_in_force(
        coordinator,
        state.clients_state.current_epoch_rates,
        state.clients_state.future_epoch_rates,
    );
    let committee = committee(coordinator, run.tie_breaker_committee_size);
    let earned: u64 = state
        .clients_state
        .clients
        .iter()
        .map(|client| client.earned)
        .sum();
    let unclaimed_earnings = earned.saturating_sub(run.total_claimed_earned_points);
    let obligations = run.total_bonded_amount as u128 + unclaimed_earnings as u128;
    let now = RoundPosition {
        epoch: coordinator.progress.epoch,
        round_height: coordinator.current_round().map_or(0, |round| round.height),
    };
    TreasurerObservation {
        verifier_committee_size: committee.size,
        committee_drawn: committee.drawn,
        min_verifier_committee: MIN_VERIFIER_COMMITTEE,
        epoch_earning_rate: rates.earning_rate_total_shared,
        epoch_slashing_rate: rates.slashing_rate_per_client,
        bond_minimum: run.bond_minimum_amount,
        security: treasurer_security(state, run),
        vault_balance: accounts.vault_balance,
        bonded_amount: run.total_bonded_amount,
        unclaimed_earnings,
        claimed_collateral: run.total_claimed_collateral_amount,
        bond_shortfall: run
            .total_bonded_amount
            .saturating_sub(accounts.vault_balance),
        obligation_shortfall: obligations
            .saturating_sub(accounts.vault_balance as u128)
            .try_into()
            .unwrap_or(u64::MAX),
        verdicts: ledger.observe(now, accounts.verdicts),
    }
}

/// What `state`, and the treasurer accounts of a treasurer-managed run, say
/// about the run at `clock`.
pub fn observe_run(
    state: &CoordinatorInstanceState,
    clock: SnapshotClock,
    treasurer: Option<TreasurerAccounts>,
    ledger: &mut VerdictLedger,
) -> RunObservation {
    let coordinator = &state.coordinator;
    let clients = &state.clients_state.clients;
    RunObservation {
        run_state: coordinator.run_state as u8,
        training: is_training(coordinator),
        epoch: coordinator.progress.epoch,
        step: coordinator.progress.step,
        registered_clients: state.clients_state.clients.len() as u64,
        epoch_clients: epoch_clients(coordinator),
        min_clients: coordinator.config.min_clients,
        audit_probability: audit_probability(coordinator),
        earned_points: clients.iter().map(|client| client.earned).sum(),
        slashed_points: clients.iter().map(|client| client.slashed).sum(),
        convicted_clients: clients.iter().filter(|client| client.slashed > 0).count() as u64,
        clock,
        treasurer: treasurer.map(|accounts| observe_treasurer(state, &accounts, ledger)),
    }
}
