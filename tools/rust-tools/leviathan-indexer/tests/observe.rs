//! A run's accounts, as the chain would hold them, read into what the indexer
//! exports and rendered the way Prometheus scrapes it.

use anchor_lang::prelude::Pubkey;
use leviathan_indexer::metrics::IndexerMetrics;
use leviathan_indexer::observe::observe_run;
use leviathan_indexer::observe::verifier_committee_size;
use leviathan_indexer::observe::RunObservation;
use leviathan_indexer::observe::SnapshotClock;
use leviathan_indexer::observe::TreasurerAccounts;
use leviathan_indexer::verdicts::VerdictLedger;
use leviathan_indexer::verdicts::VerdictTally;
use leviathan_telemetry::exposition::render;
use prometheus_client::registry::Registry;
use psyche_coordinator::Client as EpochClient;
use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_core::NodeIdentity;
use psyche_solana_coordinator::Client;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_treasurer::logic::MIN_VERIFIER_COMMITTEE;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::CommitmentEvidence;
use psyche_solana_treasurer::state::Run;
use psyche_solana_treasurer::state::VerdictStatus;

const EARNING_RATE: u64 = 900;
const SLASHING_RATE: u64 = 200;
/// What the run authority set for the next epoch, which the coordinator copies
/// into the current rates when that epoch's warmup starts.
const NEXT_EARNING_RATE: u64 = 1_200;
const NEXT_SLASHING_RATE: u64 = 400;
const BOND_MINIMUM: u64 = 500;

fn identity(seed: u8) -> NodeIdentity {
    NodeIdentity::new([seed; 32], Default::default())
}

fn epoch_client(seed: u8, state: ClientState) -> EpochClient {
    let mut client = EpochClient::new(identity(seed));
    client.state = state;
    client
}

/// A run in the middle of its third epoch: four of the six clients it started
/// with are on the roster (three healthy, one dropped this round), one was
/// ejected and one withdrew. Five clients ever registered have earned, and one
/// was slashed in an earlier epoch.
fn state() -> Box<CoordinatorInstanceState> {
    let mut state: Box<CoordinatorInstanceState> = bytemuck::zeroed_box();
    let coordinator = &mut state.coordinator;
    coordinator.run_state = RunState::RoundTrain;
    coordinator.progress.epoch = 2;
    coordinator.progress.step = 30;
    coordinator.config.min_clients = 2;
    coordinator.config.verification_percent = 50;
    coordinator.config.round_witness_time = 15;
    coordinator.config.cooldown_time = 10;
    coordinator.epoch_state.rounds_head = 1;
    coordinator.epoch_state.rounds[1].height = 7;
    // The round under way was drawn over all four on the roster: the dropped
    // client is still on it until the round ends.
    coordinator.epoch_state.rounds[1].clients_len = 4;
    for (seed, client_state) in [
        (1, ClientState::Healthy),
        (2, ClientState::Healthy),
        (3, ClientState::Healthy),
        (4, ClientState::Dropped),
    ] {
        coordinator
            .epoch_state
            .clients
            .push(epoch_client(seed, client_state))
            .unwrap();
    }
    for (seed, client_state) in [(5, ClientState::Ejected), (6, ClientState::Withdrawn)] {
        coordinator
            .epoch_state
            .exited_clients
            .push(epoch_client(seed, client_state))
            .unwrap();
    }
    for (seed, earned, slashed) in [
        (1, 300, 0),
        (2, 300, 0),
        (3, 300, 0),
        (4, 100, 0),
        (5, 0, SLASHING_RATE),
    ] {
        let mut client = <Client as bytemuck::Zeroable>::zeroed();
        client.id = identity(seed);
        client.earned = earned;
        client.slashed = slashed;
        state.clients_state.clients.push(client).unwrap();
    }
    state
        .clients_state
        .current_epoch_rates
        .earning_rate_total_shared = EARNING_RATE;
    state
        .clients_state
        .current_epoch_rates
        .slashing_rate_per_client = SLASHING_RATE;
    state
        .clients_state
        .future_epoch_rates
        .earning_rate_total_shared = NEXT_EARNING_RATE;
    state
        .clients_state
        .future_epoch_rates
        .slashing_rate_per_client = NEXT_SLASHING_RATE;
    state
}

fn treasurer_run() -> Run {
    Run {
        bump: 0,
        index: 7,
        main_authority: Pubkey::new_unique(),
        join_authority: Pubkey::new_unique(),
        coordinator_account: Pubkey::new_unique(),
        coordinator_instance: Pubkey::new_unique(),
        collateral_mint: Pubkey::new_unique(),
        total_claimed_collateral_amount: 250,
        total_claimed_earned_points: 250,
        total_bonded_amount: 2_000,
        bond_minimum_amount: BOND_MINIMUM,
        bond_withdraw_delay_seconds: 600,
        slash_bounty_bps: 0,
        challenge_window_seconds: 0,
        tie_breaker_committee_size: 0,
        appeal_window_seconds: 0,
    }
}

fn verdict(target: u8, epoch: u16, round_height: u32, status: VerdictStatus) -> AuditVerdict {
    AuditVerdict {
        bump: 0,
        run: Pubkey::new_unique(),
        target: Pubkey::new_from_array([target; 32]),
        epoch,
        round_height,
        status,
        verdict_count: 0,
        committed_hash: [0; 32],
        replayed_hash: [0; 32],
        target_index: 0,
        batch_start: 0,
        batch_end: 0,
        evidence: CommitmentEvidence::default(),
        pending_since_unix: 0,
        challenged_since_unix: 0,
        challenger: Pubkey::default(),
        overturn_count: 0,
        uphold_count: 0,
        settled_count: 0,
        released_voters: 0,
        bounty_owed: 0,
        bounty_paid: 0,
        charge: Default::default(),
        charge_epoch_start: 0,
        charge_rate: 0,
        appeal_upheld: false,
        voters: vec![],
        appeal_voters: vec![],
        vote_commitments: vec![],
    }
}

const CLOCK: SnapshotClock = SnapshotClock {
    slot: 1_000,
    unix_timestamp: 1_700_000_000,
    tip_slot: 1_042,
};

/// The run as the indexer sees it on a poll after its first, which only
/// learned that no dispute had been raised yet.
fn observe_treasurer_run(vault_balance: u64) -> RunObservation {
    let run = treasurer_run();
    let mut ledger = VerdictLedger::default();
    let quiet = observe_run(
        &state(),
        CLOCK,
        Some(TreasurerAccounts {
            run: &run,
            vault_balance,
            verdicts: &[],
        }),
        &mut ledger,
    );
    assert_eq!(quiet.treasurer.unwrap().verdicts, VerdictTally::default());
    let verdicts = [
        verdict(5, 2, 4, VerdictStatus::Upheld),
        verdict(4, 2, 6, VerdictStatus::Voting),
        verdict(3, 2, 7, VerdictStatus::Voting),
    ];
    observe_run(
        &state(),
        CLOCK,
        Some(TreasurerAccounts {
            run: &run,
            vault_balance,
            verdicts: &verdicts,
        }),
        &mut ledger,
    )
}

#[test]
fn the_roster_counts_who_started_the_epoch_and_who_is_still_healthy() {
    let observation = observe_run(&state(), CLOCK, None, &mut VerdictLedger::default());
    assert_eq!(observation.run_state, RunState::RoundTrain as u8);
    assert!(observation.training);
    assert_eq!((observation.epoch, observation.step), (2, 30));
    assert_eq!(observation.registered_clients, 5);
    // healthy, dropped, withdrawn, ejected
    assert_eq!(observation.epoch_clients, [3, 1, 1, 1]);
    assert_eq!(observation.active_clients(), 3);
    assert_eq!(observation.epoch_roster_size(), 6);
    assert_eq!(observation.earned_points, 1_000);
    assert_eq!(observation.slashed_points, SLASHING_RATE);
    assert_eq!(observation.convicted_clients, 1);
    assert!((observation.audit_probability - 0.5).abs() < 1e-12);
    assert_eq!(observation.clock.slot_lag(), 42);
    assert!(observation.treasurer.is_none());
}

/// A fraud in an epoch's final round can be convicted only while that round's
/// committee stands and its ejection will be charged: through the round's
/// witness phase and the epoch's cooldown (wienerlabs/leviathan-net#40).
#[test]
fn the_final_rounds_audit_window_is_its_witness_phase_and_the_cooldown() {
    let observation = observe_run(&state(), CLOCK, None, &mut VerdictLedger::default());
    assert_eq!(observation.final_round_audit_window_seconds, 25);
}

#[test]
fn a_treasurer_run_is_judged_on_its_own_rates_bond_and_vault() {
    let observation = observe_treasurer_run(2_500);
    let treasurer = observation.treasurer.expect("the run is treasurer-managed");
    assert_eq!(treasurer.epoch_earning_rate, EARNING_RATE);
    assert_eq!(treasurer.epoch_slashing_rate, SLASHING_RATE);
    assert_eq!(treasurer.bond_minimum, BOND_MINIMUM);
    // 900 shared over the three clients still healthy.
    assert_eq!(treasurer.security.reward, 300.0);
    assert_eq!(treasurer.security.effective_penalty, 200.0);
    // (1 - 0.5) * 300 - 0.5 * 200
    assert!((treasurer.security.expected_fraud_value - 50.0).abs() < 1e-9);
    assert!(!treasurer.security.economically_secure);
    assert_eq!(treasurer.security.break_even_penalty, Some(300.0));
    // floor(4 * 50 / 100) seats over the roster the round was drawn over.
    assert_eq!(treasurer.verifier_committee_size, 2);
    assert!(treasurer.committee_drawn);
    assert_eq!(treasurer.min_verifier_committee, MIN_VERIFIER_COMMITTEE);
    // 1000 earned, 250 claimed.
    assert_eq!(treasurer.unclaimed_earnings, 750);
    assert_eq!(treasurer.bonded_amount, 2_000);
    assert_eq!(treasurer.claimed_collateral, 250);
    assert_eq!(treasurer.bond_shortfall, 0);
    // 2500 held against 2000 bonded and 750 unclaimed.
    assert_eq!(treasurer.obligation_shortfall, 250);
    // upheld, the vote from round 6 lapsed, the vote in round 7 is open.
    assert_eq!(treasurer.verdicts.by_status, [2, 0, 0, 1, 0, 0]);
    assert_eq!(treasurer.verdicts.new_convictions, 1);
    assert_eq!(treasurer.verdicts.new_lapsed, 1);
}

#[test]
fn a_vault_below_its_bonds_shows_the_shortfall() {
    let treasurer = observe_treasurer_run(1_800).treasurer.unwrap();
    assert_eq!(treasurer.bond_shortfall, 200);
    assert_eq!(treasurer.obligation_shortfall, 950);
}

#[test]
fn the_committee_is_the_coordinators_own_draw() {
    let state = state();
    let coordinator = &state.coordinator;
    // A run at 50% draws floor((n - tie_breakers) / 2) verifiers.
    assert_eq!(verifier_committee_size(coordinator, 3, 0), 1);
    assert_eq!(verifier_committee_size(coordinator, 4, 0), 2);
    assert_eq!(verifier_committee_size(coordinator, 5, 1), 2);
    // Fewer clients than tie-breaker seats leaves no verifier seat.
    assert_eq!(verifier_committee_size(coordinator, 1, 2), 0);
    // The genesis rehearsal's 10% draws nobody below ten clients.
    let mut genesis = state.coordinator;
    genesis.config.verification_percent = 10;
    assert_eq!(verifier_committee_size(&genesis, 9, 0), 0);
    assert_eq!(verifier_committee_size(&genesis, 20, 0), 2);
}

#[test]
fn between_epochs_the_economics_are_the_next_epochs_over_the_smallest_roster() {
    let mut state = state();
    state.coordinator.run_state = RunState::WaitingForMembers;
    let run = treasurer_run();
    let observation = observe_run(
        &state,
        CLOCK,
        Some(TreasurerAccounts {
            run: &run,
            vault_balance: 0,
            verdicts: &[],
        }),
        &mut VerdictLedger::default(),
    );
    assert!(!observation.training);
    let treasurer = observation.treasurer.unwrap();
    assert_eq!(treasurer.epoch_earning_rate, NEXT_EARNING_RATE);
    assert_eq!(treasurer.epoch_slashing_rate, NEXT_SLASHING_RATE);
    // The next epoch's 1200 over min_clients = 2.
    assert_eq!(treasurer.security.reward, 600.0);
    assert_eq!(treasurer.security.slash_when_caught, 400.0);
    assert_eq!(treasurer.verifier_committee_size, 1);
}

fn scrape_of(observations: &[(&str, RunObservation)]) -> String {
    let mut registry = Registry::default();
    let metrics = IndexerMetrics::register(&mut registry);
    for (run_id, observation) in observations {
        metrics.observe(run_id, observation);
    }
    render(&registry).unwrap()
}

#[test]
fn a_treasurer_run_is_exported_with_every_series() {
    let scrape = scrape_of(&[("genesis", observe_treasurer_run(1_800))]);
    for line in [
        r#"leviathan_run_state{run_id="genesis"} 3.0"#,
        r#"leviathan_run_training{run_id="genesis"} 1.0"#,
        r#"leviathan_run_active_clients{run_id="genesis"} 3.0"#,
        r#"leviathan_run_epoch_roster_size{run_id="genesis"} 6.0"#,
        r#"leviathan_run_final_round_audit_window_seconds{run_id="genesis"} 25.0"#,
        r#"leviathan_run_epoch_clients{run_id="genesis",state="dropped"} 1.0"#,
        r#"leviathan_run_epoch_clients{run_id="genesis",state="ejected"} 1.0"#,
        r#"leviathan_run_treasurer_managed{run_id="genesis"} 1.0"#,
        r#"leviathan_run_economically_secure{run_id="genesis"} 0.0"#,
        r#"leviathan_run_expected_fraud_value{run_id="genesis"} 50.0"#,
        r#"leviathan_run_break_even_penalty{run_id="genesis"} 300.0"#,
        r#"leviathan_run_verifier_committee_size{run_id="genesis"} 2.0"#,
        r#"leviathan_run_committee_drawn{run_id="genesis"} 1.0"#,
        r#"leviathan_run_min_verifier_committee{run_id="genesis"} 2.0"#,
        r#"leviathan_run_vault_balance{run_id="genesis"} 1800.0"#,
        r#"leviathan_run_bonded_amount{run_id="genesis"} 2000.0"#,
        r#"leviathan_run_bond_shortfall{run_id="genesis"} 200.0"#,
        r#"leviathan_run_obligation_shortfall{run_id="genesis"} 950.0"#,
        r#"leviathan_run_verdicts{run_id="genesis",status="voting"} 2.0"#,
        r#"leviathan_run_verdicts{run_id="genesis",status="upheld"} 1.0"#,
        r#"leviathan_run_verdicts{run_id="genesis",status="overturned"} 0.0"#,
        r#"leviathan_run_convictions_total{run_id="genesis"} 1"#,
        r#"leviathan_run_lapsed_disputes_total{run_id="genesis"} 1"#,
        r#"leviathan_run_overturned_convictions_total{run_id="genesis"} 0"#,
        r#"leviathan_indexer_snapshot_slot{run_id="genesis"} 1000.0"#,
        r#"leviathan_indexer_slot_lag{run_id="genesis"} 42.0"#,
        r#"leviathan_indexer_snapshot_timestamp_seconds{run_id="genesis"} 1700000000.0"#,
        r#"leviathan_indexer_last_poll_success{run_id="genesis"} 1.0"#,
        r#"leviathan_indexer_poll_errors_total{run_id="genesis"} 0"#,
    ] {
        assert!(scrape.contains(line), "missing {line} in\n{scrape}");
    }
}

#[test]
fn a_coordinator_only_run_has_no_treasury_series() {
    let observation = observe_run(&state(), CLOCK, None, &mut VerdictLedger::default());
    let scrape = scrape_of(&[("flagship", observation)]);
    assert!(scrape.contains(r#"leviathan_run_treasurer_managed{run_id="flagship"} 0.0"#));
    assert!(scrape.contains(r#"leviathan_run_active_clients{run_id="flagship"} 3.0"#));
    for absent in [
        "leviathan_run_economically_secure{",
        "leviathan_run_vault_balance{",
        "leviathan_run_verdicts{",
        "leviathan_run_break_even_penalty{",
    ] {
        assert!(!scrape.contains(absent), "{absent} exported in\n{scrape}");
    }
}

#[test]
fn nothing_audited_has_no_break_even_penalty() {
    let mut state = state();
    state.coordinator.config.verification_percent = 0;
    let run = treasurer_run();
    let observation = observe_run(
        &state,
        CLOCK,
        Some(TreasurerAccounts {
            run: &run,
            vault_balance: 0,
            verdicts: &[],
        }),
        &mut VerdictLedger::default(),
    );
    let scrape = scrape_of(&[("unaudited", observation)]);
    assert!(
        !scrape.contains("leviathan_run_break_even_penalty{"),
        "{scrape}"
    );
    assert!(scrape.contains(r#"leviathan_run_economically_secure{run_id="unaudited"} 0.0"#));
}

#[test]
fn a_failed_poll_keeps_the_last_snapshot_and_counts_the_failure() {
    let mut registry = Registry::default();
    let metrics = IndexerMetrics::register(&mut registry);
    metrics.observe("genesis", &observe_treasurer_run(2_500));
    metrics.poll_failed("genesis");
    metrics.poll_failed("genesis");
    let scrape = render(&registry).unwrap();
    for line in [
        r#"leviathan_indexer_last_poll_success{run_id="genesis"} 0.0"#,
        r#"leviathan_indexer_poll_errors_total{run_id="genesis"} 2"#,
        r#"leviathan_indexer_snapshot_slot{run_id="genesis"} 1000.0"#,
    ] {
        assert!(scrape.contains(line), "missing {line} in\n{scrape}");
    }
}

#[test]
fn runs_are_exported_side_by_side() {
    let scrape = scrape_of(&[
        ("genesis", observe_treasurer_run(2_500)),
        (
            "flagship",
            observe_run(&state(), CLOCK, None, &mut VerdictLedger::default()),
        ),
    ]);
    assert!(scrape.contains(r#"leviathan_run_treasurer_managed{run_id="genesis"} 1.0"#));
    assert!(scrape.contains(r#"leviathan_run_treasurer_managed{run_id="flagship"} 0.0"#));
}

#[test]
fn the_committee_is_the_one_the_round_drew_and_the_next_draw_without_one() {
    use leviathan_indexer::observe::committee;
    use leviathan_indexer::observe::CommitteeView;
    let state = state();
    // While a committee stands over the roster, its seats are counted over
    // the clients the round was drawn over, the dropped one included, as the
    // treasurer counts them: two, not the one three healthy clients would give.
    assert_eq!(
        committee(&state.coordinator, 0),
        CommitteeView {
            size: 2,
            drawn: true,
        }
    );
    // In a warmup no committee stands; the next draw is over the healthy.
    let mut warmup = state.coordinator;
    warmup.run_state = RunState::Warmup;
    assert_eq!(
        committee(&warmup, 0),
        CommitteeView {
            size: 1,
            drawn: false,
        }
    );
    // Once the roster moves under the draw, the same.
    let mut moved = state.coordinator;
    moved.epoch_state.rounds[1].clients_len = 5;
    assert!(!committee(&moved, 0).drawn);
    // Between epochs, over the smallest roster an epoch starts with.
    let mut idle = state.coordinator;
    idle.run_state = RunState::WaitingForMembers;
    assert_eq!(
        committee(&idle, 0),
        CommitteeView {
            size: 1,
            drawn: false,
        }
    );
}

/// A drawn round is sized with the tie-breaker seats it reserves, whatever
/// the next draw will reserve; the next draw is sized with the count the
/// coordinator carries into it.
#[test]
fn the_committee_counts_the_tie_breaker_seats_of_its_own_draw() {
    use leviathan_indexer::observe::committee;
    let state = state();
    let head = state.coordinator.epoch_state.rounds_head as usize;
    let drawn = committee(&state.coordinator, 0).size;
    let mut reserving = state.coordinator;
    reserving.epoch_state.rounds[head].tie_breaker_tasks = 2;
    assert!(
        committee(&reserving, 0).size < drawn,
        "the round's own tie-breaker seats come out of the verifiers"
    );
    assert_eq!(
        committee(&state.coordinator, 2).size,
        drawn,
        "the next draw's count does not change a round already drawn"
    );
    let mut warmup = state.coordinator;
    warmup.run_state = RunState::Warmup;
    assert!(committee(&warmup, 2).size < committee(&warmup, 0).size);
}

#[test]
fn a_committee_too_small_to_convict_audits_nobody() {
    let mut state = state();
    // A quarter of four seats is one, and the treasurer takes no verdict from
    // a committee of one, whatever the run's verification percent says.
    state.coordinator.config.verification_percent = 25;
    let run = treasurer_run();
    let observation = observe_run(
        &state,
        CLOCK,
        Some(TreasurerAccounts {
            run: &run,
            vault_balance: 0,
            verdicts: &[],
        }),
        &mut VerdictLedger::default(),
    );
    assert!((observation.audit_probability - 0.25).abs() < 1e-12);
    let treasurer = observation.treasurer.unwrap();
    assert_eq!(treasurer.verifier_committee_size, 1);
    assert_eq!(treasurer.security.audit_probability, 0.0);
    assert_eq!(treasurer.security.break_even_penalty, None);
    // Never caught, a cheat takes its whole share.
    assert_eq!(treasurer.security.expected_fraud_value, 300.0);
    assert!(!treasurer.security.economically_secure);
    // The same run with a committee that can convict is judged at its
    // verification percent.
    state.coordinator.config.verification_percent = 50;
    let audited = observe_run(
        &state,
        CLOCK,
        Some(TreasurerAccounts {
            run: &run,
            vault_balance: 0,
            verdicts: &[],
        }),
        &mut VerdictLedger::default(),
    );
    assert_eq!(audited.treasurer.unwrap().security.audit_probability, 0.5);
}
