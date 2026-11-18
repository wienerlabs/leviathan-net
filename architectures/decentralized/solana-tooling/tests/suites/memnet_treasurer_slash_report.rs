//! A forfeit no verdict accounts for, from a `run_slash` or an equivocation,
//! pays its bounty to the reporter recorded when the conviction is made, never
//! to an account the convicted participant names when it leaves
//! (wienerlabs/leviathan-net#72).

use psyche_solana_tooling::get_accounts::get_participant;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_bond_finalize_withdraw_with_reporter;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_participant_create;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_equivocation_bounty;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash;
use psyche_solana_tooling::slash_reports::get_slash_report;
use psyche_solana_tooling::slash_reports::process_treasurer_run_charge_slash_report;
use psyche_solana_tooling::slash_reports::process_treasurer_run_pay_slash_report_bounty;
use psyche_solana_tooling::slash_reports::settle_slash_reports;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::state::ConvictionCharge;
use psyche_solana_treasurer::state::Participant;
use psyche_solana_treasurer::state::SlashKind;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

use super::appeals_harness::*;
use super::memnet_treasurer_equivocation::commit_then_reveal;
use super::memnet_treasurer_equivocation::resigned;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::payload_digest;
use super::signed_payload::replayed_digest;
use super::signed_payload::HONEST_PAYLOAD;

/// The bounty a charge of the run's slashing rate owes.
const BOUNTY: u64 =
    (SLASHING_RATE as u128 * BOUNTY_BPS as u128 / 10_000) as u64;

/// A token account of the run's collateral held by a second wallet of the
/// target's own, which no check at withdrawal can tell from a stranger's.
async fn second_wallet(h: &mut AppealsHarness) -> Pubkey {
    let wallet = Keypair::new();
    let payer = h.payer.insecure_clone();
    let mint = h.collateral_mint;
    h.endpoint
        .process_spl_associated_token_account_get_or_init(
            &payer,
            &wallet.pubkey(),
            &mint,
        )
        .await
        .unwrap()
}

/// The participant account of `key` in the run, which has to exist.
async fn participant_by_key(
    h: &mut AppealsHarness,
    key: &Pubkey,
) -> Participant {
    let address = find_participant(&h.run, key);
    get_participant(&mut h.endpoint, &address)
        .await
        .unwrap()
        .expect("the participant has an account")
}

async fn balance(h: &mut AppealsHarness, account: &Pubkey) -> u64 {
    h.endpoint
        .get_spl_token_account(account)
        .await
        .unwrap()
        .unwrap()
        .amount
}

/// The epoch the target is slashed in, which names its report.
fn epoch_start(h: &AppealsHarness) -> u64 {
    h.target_commitment.evidence.epoch_start
}

/// The run authority slashes the target on what it signed, naming `reporter`.
async fn authority_slashes(
    h: &mut AppealsHarness,
    reporter: Option<Pubkey>,
) -> anyhow::Result<()> {
    let payer = h.payer.insecure_clone();
    let authority = h.main_authority.insecure_clone();
    let commitment = h.target_commitment;
    let run_id = h.run_id.clone();
    let target = h.target_key;
    process_treasurer_run_slash(
        &mut h.endpoint,
        &payer,
        &authority,
        &h.run,
        &h.coordinator_account,
        &run_id,
        &target,
        reporter,
        h.target_index,
        &commitment,
        replayed_digest(),
    )
    .await
}

/// `reporter` proves the target signed two payloads for one batch: the fraud
/// it signed this round, and its honest payload for the same batch. The run
/// gives an equivocation's reporter the same share as a slash's, which it has
/// to set apart (wienerlabs/leviathan-net#92).
async fn equivocation_reported_by(
    h: &mut AppealsHarness,
    reporter: &Keypair,
) -> anyhow::Result<()> {
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let second = resigned(h, &p2p, 0, 0, 0, payload_digest(HONEST_PAYLOAD));
    let payer = h.payer.insecure_clone();
    let main_authority = h.main_authority.insecure_clone();
    let run_id = h.run_id.clone();
    let target = h.target_key;
    let (run, coordinator_account) = (h.run, h.coordinator_account);
    process_treasurer_run_set_equivocation_bounty(
        &mut h.endpoint,
        &payer,
        &main_authority,
        &run,
        BOUNTY_BPS,
    )
    .await?;
    commit_then_reveal(
        &mut h.endpoint,
        &payer,
        reporter,
        &run,
        &coordinator_account,
        &run_id,
        &target,
        h.target_index,
        &first,
        &second,
    )
    .await
}

/// A stranger charges the target's slash against its bond.
async fn charge(h: &mut AppealsHarness) -> anyhow::Result<()> {
    let stranger = h.payer.insecure_clone();
    let target = h.target_key;
    let epoch_start = epoch_start(h);
    process_treasurer_run_charge_slash_report(
        &mut h.endpoint,
        &stranger,
        &h.run,
        &h.coordinator_account,
        &target,
        epoch_start,
    )
    .await
}

/// A stranger credits the slash's reporter its bounty.
async fn pay(h: &mut AppealsHarness) -> anyhow::Result<()> {
    let stranger = h.payer.insecure_clone();
    let target = h.target_key;
    let epoch_start = epoch_start(h);
    process_treasurer_run_pay_slash_report_bounty(
        &mut h.endpoint,
        &stranger,
        &h.run,
        &target,
        epoch_start,
    )
    .await
}

/// The target asks for its whole bond back once its epoch is charged, and the
/// withdrawal delay elapses.
async fn ask_to_leave(h: &mut AppealsHarness) {
    h.finish_epoch_with_witnesses().await;
    let target = h.target_client_idx;
    h.request_withdraw(target, BOND).await;
    h.tick_after(WITHDRAW_DELAY as u64).await;
}

/// The target finalizes its withdrawal, naming `wallet` as a reporter.
async fn leave_naming(
    h: &mut AppealsHarness,
    wallet: &Pubkey,
) -> anyhow::Result<()> {
    let target = h.target_client_idx;
    let client = h.clients[target].insecure_clone();
    let collateral = h.clients_collateral[target];
    let payer = h.payer.insecure_clone();
    process_treasurer_participant_bond_finalize_withdraw_with_reporter(
        &mut h.endpoint,
        &payer,
        &client,
        &collateral,
        &h.collateral_mint,
        &h.run,
        &h.coordinator_account,
        wallet,
    )
    .await
}

/// Another client of the run, which reports the target.
fn reporter_of(h: &AppealsHarness) -> usize {
    (h.target_client_idx + 1) % h.clients.len()
}

/// The run authority slashes a target and names no reporter. The target's
/// withdrawal waits for the slash to be charged, anyone charges it, and the
/// target cannot take any of it back through a wallet it names: the bounty
/// stays in the vault.
#[tokio::test]
pub async fn a_run_slash_target_cannot_route_its_bounty_to_a_wallet_it_names() {
    let mut h = setup("Leviathan slash report authority", 720).await;
    authority_slashes(&mut h, None).await.unwrap();
    let target = h.target_key;
    assert_eq!(
        participant_by_key(&mut h, &target)
            .await
            .uncharged_slash_reports,
        1
    );

    let wallet = second_wallet(&mut h).await;
    ask_to_leave(&mut h).await;
    assert_eq!(
        refusal_code(leave_naming(&mut h, &wallet).await),
        treasurer_refusal(ProgramError::SlashReportsUncharged),
        "a slashed target left before its slash was charged"
    );

    let collateral = h.clients_collateral[h.target_client_idx];
    let before = balance(&mut h, &collateral).await;
    charge(&mut h).await.unwrap();
    assert_eq!(
        participant_by_key(&mut h, &target)
            .await
            .uncharged_slash_reports,
        0
    );
    leave_naming(&mut h, &wallet).await.unwrap();

    assert_eq!(
        balance(&mut h, &wallet).await,
        0,
        "a slashed target took back part of its forfeit through a wallet it named"
    );
    assert_eq!(
        balance(&mut h, &collateral).await - before,
        BOND - SLASHING_RATE
    );
    let epoch = epoch_start(&h);
    let report = get_slash_report(&mut h.endpoint, &h.run, &target, epoch)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(report.kind, SlashKind::AuthoritySlash);
    assert_eq!(report.reporter, None);
    assert_eq!(report.charge, ConvictionCharge::Charged);
    assert_eq!(report.bounty_owed, 0);
    // There is nobody to credit; the tooling says so before the chain could.
    assert!(pay(&mut h).await.is_err());
}

/// Another client proves the target equivocated. The target cannot take the
/// bounty back through a wallet it names; the reporter's bond is credited it,
/// once.
#[tokio::test]
pub async fn an_equivocator_cannot_route_its_bounty_to_a_wallet_it_names() {
    let mut h = setup("Leviathan slash report equiv", 721).await;
    let reporter = h.clients[reporter_of(&h)].insecure_clone();
    equivocation_reported_by(&mut h, &reporter).await.unwrap();
    let target = h.target_key;
    let epoch = epoch_start(&h);
    let report = get_slash_report(&mut h.endpoint, &h.run, &target, epoch)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(report.kind, SlashKind::Equivocation);
    assert_eq!(report.reporter, Some(reporter.pubkey()));
    assert_eq!(report.charge, ConvictionCharge::Ejected);
    assert_eq!(
        refusal_code(charge(&mut h).await),
        treasurer_refusal(ProgramError::EjectionNotCharged),
        "a slash was charged before the ledger charged its epoch"
    );

    let wallet = second_wallet(&mut h).await;
    ask_to_leave(&mut h).await;
    assert_eq!(
        refusal_code(leave_naming(&mut h, &wallet).await),
        treasurer_refusal(ProgramError::SlashReportsUncharged),
    );
    charge(&mut h).await.unwrap();
    // A fresh blockhash, so the second charge is a new transaction.
    h.endpoint.forward_clock_slot(1).await.unwrap();
    assert_eq!(
        refusal_code(charge(&mut h).await),
        treasurer_refusal(ProgramError::ConvictionNotChargeable),
        "a slash was charged twice"
    );
    assert_eq!(
        participant_by_key(&mut h, &target).await.bond_amount,
        BOND - SLASHING_RATE
    );
    leave_naming(&mut h, &wallet).await.unwrap();
    assert_eq!(balance(&mut h, &wallet).await, 0);

    let reporter_key = reporter.pubkey();
    let before = participant_by_key(&mut h, &reporter_key).await.bond_amount;
    pay(&mut h).await.unwrap();
    assert_eq!(
        participant_by_key(&mut h, &reporter_key).await.bond_amount - before,
        BOUNTY
    );
    h.endpoint.forward_clock_slot(1).await.unwrap();
    assert_eq!(
        refusal_code(pay(&mut h).await),
        treasurer_refusal(ProgramError::BountySeatPaid),
        "a reporter was credited twice"
    );
}

/// The run authority names who reported the slash; that reporter's bond is
/// credited the bounty, once.
#[tokio::test]
pub async fn a_run_slash_credits_the_reporter_it_names_once() {
    let mut h = setup("Leviathan slash report named", 722).await;
    let reporter = h.clients[reporter_of(&h)].pubkey();
    authority_slashes(&mut h, Some(reporter)).await.unwrap();
    h.finish_epoch_with_witnesses().await;
    charge(&mut h).await.unwrap();
    let before = participant_by_key(&mut h, &reporter).await.bond_amount;
    pay(&mut h).await.unwrap();
    assert_eq!(
        participant_by_key(&mut h, &reporter).await.bond_amount - before,
        BOUNTY
    );
    h.endpoint.forward_clock_slot(1).await.unwrap();
    assert_eq!(
        refusal_code(pay(&mut h).await),
        treasurer_refusal(ProgramError::BountySeatPaid),
    );
}

/// Neither slash takes the target as its reporter, and refusing it leaves the
/// target unslashed.
#[tokio::test]
pub async fn neither_slash_can_name_the_target_as_its_reporter() {
    let mut h = setup("Leviathan slash report self", 723).await;
    let target = h.target_key;
    assert_eq!(
        refusal_code(authority_slashes(&mut h, Some(target)).await),
        treasurer_refusal(ProgramError::ReporterIsTarget),
    );
    let itself = h.clients[h.target_client_idx].insecure_clone();
    assert_eq!(
        refusal_code(equivocation_reported_by(&mut h, &itself).await),
        treasurer_refusal(ProgramError::ReporterIsTarget),
    );
    assert_eq!(
        h.client_state(&target).await,
        Some(psyche_coordinator::ClientState::Healthy)
    );
    let epoch = epoch_start(&h);
    assert!(get_slash_report(&mut h.endpoint, &h.run, &target, epoch)
        .await
        .unwrap()
        .is_none());
}

/// A reporter with no participant account in the run cannot be credited yet,
/// and holds up nobody: the target is charged and leaves. Once the reporter
/// makes an account, it is credited.
#[tokio::test]
pub async fn a_reporter_with_no_participant_account_is_credited_once_it_makes_one(
) {
    let mut h = setup("Leviathan slash report outsider", 724).await;
    let outsider = Keypair::new();
    let payer = h.payer.insecure_clone();
    equivocation_reported_by(&mut h, &outsider).await.unwrap();

    ask_to_leave(&mut h).await;
    charge(&mut h).await.unwrap();
    assert!(
        pay(&mut h).await.is_err(),
        "a reporter with no account was credited"
    );
    let target = h.target_client_idx;
    h.try_finalize_withdraw(target).await.unwrap();

    // A sweep leaves the bounty for later rather than send what would fail,
    // even once someone has sent lamports to where the account will be.
    let (run, coordinator_account) = (h.run, h.coordinator_account);
    h.endpoint
        .process_system_transfer(
            &payer,
            &payer,
            &find_participant(&run, &outsider.pubkey()),
            1_000_000,
        )
        .await
        .unwrap();
    let waiting = settle_slash_reports(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert!(waiting.failures.is_empty(), "{:?}", waiting.failures);
    assert_eq!(
        (waiting.open, waiting.credited, waiting.awaiting_reporter),
        (1, 0, 1)
    );

    process_treasurer_participant_create(
        &mut h.endpoint,
        &payer,
        &outsider,
        &run,
    )
    .await
    .unwrap();
    let sweep = settle_slash_reports(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert!(sweep.failures.is_empty(), "{:?}", sweep.failures);
    assert_eq!(
        (sweep.open, sweep.credited, sweep.awaiting_reporter),
        (1, 1, 0)
    );
    assert_eq!(
        participant_by_key(&mut h, &outsider.pubkey())
            .await
            .bond_amount,
        BOUNTY
    );
}

/// The sweep a verifier daemon runs each pass charges a slash once its epoch
/// is charged and credits its reporter, and leaves nothing for a second
/// sweep.
#[tokio::test]
pub async fn a_sweep_charges_the_slash_and_credits_its_reporter() {
    let mut h = setup("Leviathan slash report sweep", 725).await;
    let reporter = h.clients[reporter_of(&h)].insecure_clone();
    equivocation_reported_by(&mut h, &reporter).await.unwrap();
    let payer = h.payer.insecure_clone();
    let (run, coordinator_account) = (h.run, h.coordinator_account);

    // While its epoch runs, the sweep leaves it alone.
    let early = settle_slash_reports(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert_eq!((early.open, early.charged, early.credited), (1, 0, 0));

    h.finish_epoch_with_witnesses().await;
    let before = participant_by_key(&mut h, &reporter.pubkey())
        .await
        .bond_amount;
    let sweep = settle_slash_reports(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert!(sweep.failures.is_empty(), "{:?}", sweep.failures);
    assert_eq!((sweep.open, sweep.charged, sweep.credited), (1, 1, 1));
    assert_eq!(
        participant_by_key(&mut h, &reporter.pubkey())
            .await
            .bond_amount
            - before,
        BOUNTY
    );
    let target = h.target_key;
    assert_eq!(
        participant_by_key(&mut h, &target)
            .await
            .uncharged_slash_reports,
        0
    );

    let again = settle_slash_reports(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert_eq!((again.open, again.charged, again.credited), (0, 0, 0));
}
