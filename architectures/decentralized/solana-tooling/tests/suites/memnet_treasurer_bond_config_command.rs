//! The bond a treasurer run requires, set through the instruction
//! `run-manager set-bond-config` sends (wienerlabs/leviathan-net#51).

use psyche_solana_rpc::instructions::treasurer_run_bond_config_update;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::ProgramError;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::BOND;
use super::bonded_run::WITHDRAW_DELAY;
use super::refusal::refusal_code;
use super::refusal::treasurer_refusal;

const RUN_INDEX: u64 = 510;

async fn set_bond(
    harness: &mut Harness,
    signer: &Keypair,
    params: RunBondConfigUpdateParams,
) -> anyhow::Result<()> {
    let instruction =
        treasurer_run_bond_config_update(RUN_INDEX, &signer.pubkey(), params);
    harness
        .endpoint
        .process_instruction_with_signers(
            &harness.payer,
            instruction,
            &[signer],
        )
        .await
        .map(|_| ())
}

#[tokio::test]
pub async fn the_main_authority_sets_the_bond() {
    let mut harness =
        Harness::new(RUN_INDEX, "Leviathan bond config set", JoinGate::Bond)
            .await;
    let raised = RunBondConfigUpdateParams {
        bond_minimum_amount: 2 * BOND,
        bond_withdraw_delay_seconds: 2 * WITHDRAW_DELAY,
    };
    let authority = harness.main_authority.insecure_clone();
    set_bond(&mut harness, &authority, raised).await.unwrap();
    let run = get_run(&mut harness.endpoint, &harness.run)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.bond_minimum_amount, raised.bond_minimum_amount);
    assert_eq!(
        run.bond_withdraw_delay_seconds,
        raised.bond_withdraw_delay_seconds
    );
}

#[tokio::test]
pub async fn only_the_main_authority_sets_the_bond() {
    let mut harness = Harness::new(
        RUN_INDEX,
        "Leviathan bond config stranger",
        JoinGate::Bond,
    )
    .await;
    let stranger = Keypair::new();
    let lowered = RunBondConfigUpdateParams {
        bond_minimum_amount: 0,
        bond_withdraw_delay_seconds: 0,
    };
    assert_eq!(
        refusal_code(set_bond(&mut harness, &stranger, lowered).await),
        u32::from(anchor_lang::error::ErrorCode::ConstraintRaw),
    );
    let run = get_run(&mut harness.endpoint, &harness.run)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.bond_minimum_amount, BOND);
}

/// A bond with no withdrawal delay could leave before a conviction reaches
/// it, so the program refuses one.
#[tokio::test]
pub async fn a_bond_needs_a_withdrawal_delay() {
    let mut harness =
        Harness::new(RUN_INDEX, "Leviathan bond config window", JoinGate::Bond)
            .await;
    let authority = harness.main_authority.insecure_clone();
    assert_eq!(
        refusal_code(
            set_bond(
                &mut harness,
                &authority,
                RunBondConfigUpdateParams {
                    bond_minimum_amount: BOND,
                    bond_withdraw_delay_seconds: 0,
                },
            )
            .await
        ),
        treasurer_refusal(ProgramError::BondWindowRequired),
    );
}

#[tokio::test]
pub async fn a_negative_withdrawal_delay_is_refused() {
    let mut harness = Harness::new(
        RUN_INDEX,
        "Leviathan bond config negative",
        JoinGate::Bond,
    )
    .await;
    let authority = harness.main_authority.insecure_clone();
    assert_eq!(
        refusal_code(
            set_bond(
                &mut harness,
                &authority,
                RunBondConfigUpdateParams {
                    bond_minimum_amount: BOND,
                    bond_withdraw_delay_seconds: -WITHDRAW_DELAY,
                },
            )
            .await
        ),
        treasurer_refusal(ProgramError::InvalidParameter),
    );
}

/// A run can drop its bond altogether, as the bondless first mainnet stage
/// does, and then needs no withdrawal delay.
#[tokio::test]
pub async fn a_run_can_go_bondless() {
    let mut harness = Harness::new(
        RUN_INDEX,
        "Leviathan bond config bondless",
        JoinGate::Bond,
    )
    .await;
    let authority = harness.main_authority.insecure_clone();
    set_bond(
        &mut harness,
        &authority,
        RunBondConfigUpdateParams {
            bond_minimum_amount: 0,
            bond_withdraw_delay_seconds: 0,
        },
    )
    .await
    .unwrap();
    let run = get_run(&mut harness.endpoint, &harness.run)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (run.bond_minimum_amount, run.bond_withdraw_delay_seconds),
        (0, 0)
    );
}
