//! A proof of equivocation pays a bounty to whoever reports it, out of the
//! target's forfeit (wienerlabs/leviathan-net#72), and two routes let the
//! target take that bounty back (wienerlabs/leviathan-net#92):
//!
//! - **Copying.** A proof is public once sent, so anyone could copy an honest
//!   reporter's and land it first as its own. A reporter now commits to the
//!   proof and its own key first, and reveals it in a later slot; a copier
//!   cannot have committed before it saw the proof.
//! - **Reporting itself.** The target can report itself from a second wallet,
//!   which the chain cannot tell from an honest reporter. An equivocation's
//!   bounty is now a share the run sets apart from a slash's, always less
//!   than the whole forfeit, so reporting itself always costs the target the
//!   rest.

use anchor_lang::error::ErrorCode as AnchorErrorCode;
use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use psyche_coordinator::ClientState;
use psyche_coordinator::RunState;
use psyche_solana_coordinator::find_coordinator_instance;
use psyche_solana_tooling::equivocation::equivocation_commitment;
use psyche_solana_tooling::equivocation::process_treasurer_run_close_equivocation_commitment;
use psyche_solana_tooling::equivocation::process_treasurer_run_commit_equivocation;
use psyche_solana_tooling::equivocation::process_treasurer_run_slash_equivocation;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_set_equivocation_bounty;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use psyche_solana_tooling::slash_reports::get_slash_report;
use psyche_solana_tooling::slash_reports::settle_slash_reports;
use psyche_solana_treasurer::accounts::RunSlashEquivocationAccounts;
use psyche_solana_treasurer::find_equivocation_bounty;
use psyche_solana_treasurer::find_equivocation_commitment;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::find_slash_report;
use psyche_solana_treasurer::instruction::RunSlashEquivocation;
use psyche_solana_treasurer::logic::RunSlashEquivocationParams;
use psyche_solana_treasurer::state::EquivocationBounty;
use psyche_solana_treasurer::state::MAX_EQUIVOCATION_BOUNTY_BPS;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::system_program;
use solana_sdk::sysvar;

use super::appeals_harness::*;
use super::memnet_treasurer_equivocation::resigned;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;
use super::signed_payload::payload_digest;
use super::signed_payload::HONEST_PAYLOAD;

/// The target's two payloads for its batch this round: the one it sent, and
/// its honest payload for the same batch.
fn proof(h: &AppealsHarness) -> (SignedCommitment, SignedCommitment) {
    let p2p = h.p2ps[h.target_client_idx].insecure_clone();
    let first = h.target_commitment;
    let second = resigned(h, &p2p, 0, 0, 0, payload_digest(HONEST_PAYLOAD));
    (first, second)
}

async fn commit(
    h: &mut AppealsHarness,
    reporter: &Keypair,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> anyhow::Result<[u8; 32]> {
    let payer = h.payer.insecure_clone();
    let (run, target) = (h.run, h.target_key);
    process_treasurer_run_commit_equivocation(
        &mut h.endpoint,
        &payer,
        reporter,
        &run,
        &target,
        first,
        second,
    )
    .await
}

async fn reveal(
    h: &mut AppealsHarness,
    reporter: &Keypair,
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> anyhow::Result<()> {
    let payer = h.payer.insecure_clone();
    let run_id = h.run_id.clone();
    let (run, coordinator_account, target, target_index) =
        (h.run, h.coordinator_account, h.target_key, h.target_index);
    process_treasurer_run_slash_equivocation(
        &mut h.endpoint,
        &payer,
        reporter,
        &run,
        &coordinator_account,
        &run_id,
        &target,
        target_index,
        first,
        second,
    )
    .await
}

/// A wallet of no one in the run, funded to pay for its own transactions.
async fn stranger(h: &mut AppealsHarness) -> Keypair {
    let wallet = Keypair::new();
    h.endpoint
        .request_airdrop(&wallet.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    wallet
}

/// Who reported the slash that ejected the target this epoch.
async fn reporter_on_record(h: &mut AppealsHarness) -> Option<Pubkey> {
    let (run, target) = (h.run, h.target_key);
    let epoch_start = h.target_commitment.evidence.epoch_start;
    get_slash_report(&mut h.endpoint, &run, &target, epoch_start)
        .await
        .unwrap()
        .expect("the equivocation recorded a report")
        .reporter
}

async fn set_bounty(
    h: &mut AppealsHarness,
    bounty_bps: u16,
) -> anyhow::Result<()> {
    let payer = h.payer.insecure_clone();
    let main_authority = h.main_authority.insecure_clone();
    let run = h.run;
    process_treasurer_run_set_equivocation_bounty(
        &mut h.endpoint,
        &payer,
        &main_authority,
        &run,
        bounty_bps,
    )
    .await
}

/// A proof is taken only from a reporter that committed to it, in an earlier
/// slot than the one it reveals it in.
#[tokio::test]
pub async fn a_proof_is_taken_only_against_a_commitment_from_an_earlier_slot() {
    let mut h = setup("Leviathan equivocation commit", 760).await;
    let (first, second) = proof(&h);
    let reporter = stranger(&mut h).await;
    // No commitment at all.
    assert_eq!(
        refusal_code(reveal(&mut h, &reporter, &first, &second).await),
        u32::from(AnchorErrorCode::AccountNotInitialized),
    );
    // A commitment from this very slot.
    commit(&mut h, &reporter, &first, &second).await.unwrap();
    assert_eq!(
        refusal_code(reveal(&mut h, &reporter, &first, &second).await),
        treasurer_refusal(ProgramError::EquivocationCommittedTooRecently),
    );
    let target_key = h.target_key;
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );
    // A slot later the proof is taken, and its commitment closed.
    h.endpoint.forward_clock_slot(1).await.unwrap();
    reveal(&mut h, &reporter, &first, &second).await.unwrap();
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected)
    );
    assert_eq!(reporter_on_record(&mut h).await, Some(reporter.pubkey()));
    let commitment = equivocation_commitment(
        &h.run,
        &reporter.pubkey(),
        &target_key,
        &first,
        &second,
    );
    let address =
        find_equivocation_commitment(&h.run, &reporter.pubkey(), &commitment);
    assert!(h.endpoint.get_account(&address).await.unwrap().is_none());
}

/// The instruction revealing the proof as `reporter`, against the commitment
/// account at `commitment_account` named by `commitment`, whoever made it.
fn reveal_against(
    h: &AppealsHarness,
    reporter: &Pubkey,
    commitment_account: Pubkey,
    commitment: [u8; 32],
    first: &SignedCommitment,
    second: &SignedCommitment,
) -> Instruction {
    let accounts = RunSlashEquivocationAccounts {
        payer: h.payer.pubkey(),
        reporter: *reporter,
        run: h.run,
        coordinator_instance: find_coordinator_instance(&h.run_id),
        coordinator_account: h.coordinator_account,
        coordinator_program: psyche_solana_coordinator::ID,
        instructions: sysvar::instructions::ID,
        target_participant: find_participant(&h.run, &h.target_key),
        slash_report: find_slash_report(
            &h.run,
            &h.target_key,
            first.evidence.epoch_start,
        ),
        equivocation_commitment: commitment_account,
        system_program: system_program::ID,
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSlashEquivocation {
            params: RunSlashEquivocationParams {
                target: h.target_key,
                target_index: h.target_index,
                batch_start: first.batch_start,
                batch_end: first.batch_end,
                first: first.evidence,
                second: second.evidence,
                commitment,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// An honest reporter commits to the proof and, a slot later, reveals it. A
/// front-runner sees the proof only then: it cannot reveal it against a
/// commitment of its own from that slot, nor against the honest reporter's,
/// and once the proof has landed its own commitment is too late. The bounty
/// stays with the reporter that found the fraud.
#[tokio::test]
pub async fn a_copier_cannot_take_the_proof_from_the_reporter_that_committed_first(
) {
    let mut h = setup("Leviathan equivocation copier", 761).await;
    let (first, second) = proof(&h);
    let honest = stranger(&mut h).await;
    let copier = stranger(&mut h).await;
    let honest_commitment =
        commit(&mut h, &honest, &first, &second).await.unwrap();
    h.endpoint.forward_clock_slot(1).await.unwrap();

    // The honest reveal is public now. The copier commits to the same proof
    // under its own key and tries to land it in the same slot.
    commit(&mut h, &copier, &first, &second).await.unwrap();
    assert_eq!(
        refusal_code(reveal(&mut h, &copier, &first, &second).await),
        treasurer_refusal(ProgramError::EquivocationCommittedTooRecently),
    );
    // Nor can it reveal against the honest reporter's commitment: the
    // commitment names its reporter, both in its address and in its hash.
    let payer = h.payer.insecure_clone();
    let honest_account = find_equivocation_commitment(
        &h.run,
        &honest.pubkey(),
        &honest_commitment,
    );
    let copier_on_honest_account = find_equivocation_commitment(
        &h.run,
        &copier.pubkey(),
        &honest_commitment,
    );
    for (account, refusal) in [
        (honest_account, AnchorErrorCode::ConstraintSeeds),
        (
            copier_on_honest_account,
            AnchorErrorCode::AccountNotInitialized,
        ),
    ] {
        h.endpoint.forward_clock_slot(1).await.unwrap();
        let instruction = reveal_against(
            &h,
            &copier.pubkey(),
            account,
            honest_commitment,
            &first,
            &second,
        );
        assert_eq!(
            refusal_code(
                h.endpoint
                    .process_instructions_with_signers(
                        &payer,
                        &[
                            first.proof_instruction(),
                            second.proof_instruction(),
                            instruction,
                        ],
                        &[&copier],
                    )
                    .await
            ),
            u32::from(refusal),
        );
    }
    let target_key = h.target_key;
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Healthy)
    );

    // The honest reporter's proof lands, and it is the reporter on record.
    reveal(&mut h, &honest, &first, &second).await.unwrap();
    assert_eq!(
        h.client_state(&target_key).await,
        Some(ClientState::Ejected)
    );
    assert_eq!(reporter_on_record(&mut h).await, Some(honest.pubkey()));

    // The copier's commitment, from an earlier slot by now, comes too late.
    h.endpoint.forward_clock_slot(1).await.unwrap();
    assert_eq!(
        refusal_code(reveal(&mut h, &copier, &first, &second).await),
        treasurer_refusal(ProgramError::EquivocationNotLandable),
    );
    assert_eq!(reporter_on_record(&mut h).await, Some(honest.pubkey()));

    // It can take its commitment's rent back.
    let copier_commitment = equivocation_commitment(
        &h.run,
        &copier.pubkey(),
        &target_key,
        &first,
        &second,
    );
    let copier_account = find_equivocation_commitment(
        &h.run,
        &copier.pubkey(),
        &copier_commitment,
    );
    let rent = h
        .endpoint
        .get_account(&copier_account)
        .await
        .unwrap()
        .expect("the copier's commitment is open")
        .lamports;
    let before = h
        .endpoint
        .get_account(&copier.pubkey())
        .await
        .unwrap()
        .unwrap()
        .lamports;
    let run = h.run;
    process_treasurer_run_close_equivocation_commitment(
        &mut h.endpoint,
        &copier,
        &copier,
        &run,
        &copier_commitment,
    )
    .await
    .unwrap();
    assert!(h
        .endpoint
        .get_account(&copier_account)
        .await
        .unwrap()
        .is_none());
    let after = h
        .endpoint
        .get_account(&copier.pubkey())
        .await
        .unwrap()
        .unwrap()
        .lamports;
    // The rent, less the fee of the transaction that closed it.
    assert!(
        after > before && after <= before + rent,
        "{before} {after} {rent}"
    );
}

/// What the target and the wallet that reported it, as one owner, lose when
/// the target reports itself and its slash is charged, in a run that set
/// the equivocation bounty to `bounty_bps`, or set none.
async fn self_report_costs(index: u64, bounty_bps: Option<u16>) -> (u64, u64) {
    let mut h =
        setup(&format!("Leviathan equivocation self {index}"), index).await;
    if let Some(bounty_bps) = bounty_bps {
        set_bounty(&mut h, bounty_bps).await.unwrap();
    }
    let (first, second) = proof(&h);
    // A second wallet of the target's own, with a participant account of its
    // own, which the chain cannot tell from an honest reporter.
    let second_wallet_idx = (h.target_client_idx + 1) % h.clients.len();
    let second_wallet = h.clients[second_wallet_idx].insecure_clone();
    let target = h.target_client_idx;
    let target_before = h.participant_of(target).await.bond_amount;
    let wallet_before = h.participant_of(second_wallet_idx).await.bond_amount;

    commit(&mut h, &second_wallet, &first, &second)
        .await
        .unwrap();
    h.endpoint.forward_clock_slot(1).await.unwrap();
    reveal(&mut h, &second_wallet, &first, &second)
        .await
        .unwrap();
    assert_eq!(
        reporter_on_record(&mut h).await,
        Some(second_wallet.pubkey())
    );
    h.finish_epoch_with_witnesses().await;
    assert_eq!(h.run_state().await, RunState::WaitingForMembers);
    let payer = h.payer.insecure_clone();
    let (run, coordinator_account) = (h.run, h.coordinator_account);
    let sweep = settle_slash_reports(
        &mut h.endpoint,
        &payer,
        &run,
        &coordinator_account,
    )
    .await
    .unwrap();
    assert!(sweep.failures.is_empty(), "{:?}", sweep.failures);
    assert_eq!(sweep.charged, 1);

    let target_lost =
        target_before - h.participant_of(target).await.bond_amount;
    let wallet_gained =
        h.participant_of(second_wallet_idx).await.bond_amount - wallet_before;
    (target_lost, wallet_gained)
}

/// A target that reports itself from a second wallet loses the share of the
/// charge the run keeps from an equivocation's reporter. The run sets that
/// share, it is never the whole charge, and a run that set none keeps all of
/// it.
#[tokio::test]
pub async fn a_target_that_reports_itself_still_loses_the_share_the_run_keeps()
{
    assert_eq!(
        BOUNTY_BPS, 5_000,
        "the cases below assume this slash bounty"
    );
    for (index, bounty_bps, credited_bps) in [
        // What the run set.
        (762, Some(4_000), 4_000),
        // None set: the reporter is credited nothing.
        (763, None, 0),
        // More than a slash's bounty: held to the slash's.
        (764, Some(MAX_EQUIVOCATION_BOUNTY_BPS), BOUNTY_BPS),
    ] {
        let (lost, gained) = self_report_costs(index, bounty_bps).await;
        assert_eq!(lost, SLASHING_RATE, "{bounty_bps:?}");
        assert_eq!(
            gained,
            SLASHING_RATE * credited_bps as u64 / 10_000,
            "{bounty_bps:?}"
        );
        let net = lost - gained;
        assert_eq!(
            net,
            SLASHING_RATE * (10_000 - credited_bps as u64) / 10_000,
            "{bounty_bps:?}"
        );
        assert!(net > 0, "reporting itself was free at {bounty_bps:?}");
    }
}

/// The run's main authority sets the equivocation bounty, to less than the
/// whole forfeit; nobody else sets it.
#[tokio::test]
pub async fn a_run_cannot_give_an_equivocations_reporter_the_whole_forfeit() {
    let mut h = setup("Leviathan equivocation bounty", 765).await;
    for bounty_bps in [10_000, u16::MAX] {
        assert_eq!(
            refusal_code(set_bounty(&mut h, bounty_bps).await),
            treasurer_refusal(ProgramError::EquivocationBountyTooHigh),
        );
    }
    let payer = h.payer.insecure_clone();
    let run = h.run;
    let impostor = stranger(&mut h).await;
    assert_eq!(
        refusal_code(
            process_treasurer_run_set_equivocation_bounty(
                &mut h.endpoint,
                &payer,
                &impostor,
                &run,
                1_000,
            )
            .await
        ),
        u32::from(AnchorErrorCode::ConstraintRaw),
    );
    assert!(h
        .endpoint
        .get_account(&find_equivocation_bounty(&run))
        .await
        .unwrap()
        .is_none());

    for bounty_bps in [MAX_EQUIVOCATION_BOUNTY_BPS, 2_500] {
        h.endpoint.forward_clock_slot(1).await.unwrap();
        set_bounty(&mut h, bounty_bps).await.unwrap();
        let account = h
            .endpoint
            .get_account(&find_equivocation_bounty(&run))
            .await
            .unwrap()
            .unwrap();
        let bounty =
            EquivocationBounty::from_pda_account(&account.owner, &account.data)
                .unwrap()
                .unwrap();
        assert_eq!(bounty.run, run);
        assert_eq!(bounty.bounty_bps, bounty_bps);
    }
}
