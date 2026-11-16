//! The coordinator account's layout, as the programs write it and as every
//! client reads it back.
//!
//! Version 2 added the run's aggregation rule to the DisTrO optimizer, in what
//! version 1 had as padding (wienerlabs/leviathan-net#60). The padding of a
//! real version 1 account is whatever the program's stack held, so a version 1
//! account has to be refused, not read with that as a setting.

use anchor_lang::Discriminator;
use psyche_coordinator::model::Model;
use psyche_coordinator::CoordinatorConfig;
use psyche_core::OptimizerDefinition;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_coordinator::logic::InitCoordinatorParams;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::DeserializeCoordinatorFromBytes;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_init;
use psyche_solana_tooling::process_coordinator_instructions::process_update;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

#[derive(serde::Deserialize)]
struct RunConfig {
    config: CoordinatorConfig,
    model: Model,
}

fn genesis_rehearsal() -> RunConfig {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../config/leviathan/genesis-rehearsal.toml"
    );
    toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Where `robust_aggregation` sits in a coordinator account, past the
/// discriminator: the one byte that changes when the field alone is set
/// through a reference to it.
fn robust_aggregation_offset() -> usize {
    let mut account: Box<CoordinatorAccount> = bytemuck::zeroed_box();
    let Model::LLM(llm) = &mut account.state.coordinator.model;
    llm.optimizer = OptimizerDefinition::Distro {
        clip_grad_norm: None,
        weight_decay: None,
        compression_decay: 0.0,
        compression_topk: 0,
        compression_chunk: 0,
        quantize_1bit: false,
        robust_aggregation: false,
    };
    let unset = bytemuck::bytes_of(&*account).to_vec();
    let Model::LLM(llm) = &mut account.state.coordinator.model;
    let OptimizerDefinition::Distro {
        robust_aggregation, ..
    } = &mut llm.optimizer
    else {
        unreachable!("set to DisTrO just above");
    };
    *robust_aggregation = true;
    let set = bytemuck::bytes_of(&*account).to_vec();
    let changed: Vec<usize> =
        (0..set.len()).filter(|&i| set[i] != unset[i]).collect();
    assert_eq!(changed.len(), 1, "setting one bool changed {changed:?}");
    CoordinatorAccount::DISCRIMINATOR.len() + changed[0]
}

#[tokio::test]
pub async fn a_version_one_account_is_refused_rather_than_misread() {
    // An account a real version 1 program wrote: the run it describes trains
    // with DisTrO, so the byte where version 2 keeps the aggregation rule was
    // that optimizer's padding.
    let bytes =
        include_bytes!("../fixtures/coordinator-account-v1.so").to_vec();
    let padding = bytes[robust_aggregation_offset()];
    assert!(
        padding > 1,
        "the version 1 padding holds {padding:#04x}, which happens to be a bool"
    );
    match coordinator_account_from_bytes(&bytes) {
        Err(DeserializeCoordinatorFromBytes::InvalidVersion {
            expected,
            actual,
        }) => {
            assert_eq!((expected, actual), (2, 1));
        },
        Err(other) => panic!("refused for the wrong reason: {other}"),
        Ok(_) => panic!("a version 1 account was read as version 2"),
    }
}

/// The aggregation rule a run set on chain is the one every client reads back,
/// through the real programs.
#[tokio::test]
pub async fn the_aggregation_rule_round_trips_through_the_programs() {
    let mut endpoint = create_memnet_endpoint().await;
    let payer = Keypair::new();
    endpoint
        .request_airdrop(&payer.pubkey(), 5_000_000_000)
        .await
        .unwrap();
    let main_authority = Keypair::new();
    let coordinator_account = endpoint
        .process_system_new_exempt(
            &payer,
            CoordinatorAccount::space_with_discriminator(),
            &psyche_solana_coordinator::ID,
        )
        .await
        .unwrap();
    let coordinator_instance = process_coordinator_init(
        &mut endpoint,
        &payer,
        &coordinator_account,
        InitCoordinatorParams {
            run_id: "Leviathan aggregation".to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: Keypair::new().pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();

    let rehearsal = genesis_rehearsal();
    for robust in [true, false] {
        let Model::LLM(mut llm) = rehearsal.model;
        let OptimizerDefinition::Distro {
            robust_aggregation, ..
        } = &mut llm.optimizer
        else {
            panic!("the genesis rehearsal trains with DisTrO");
        };
        *robust_aggregation = robust;
        process_update(
            &mut endpoint,
            &payer,
            &main_authority,
            &coordinator_instance,
            &coordinator_account,
            None,
            Some(rehearsal.config),
            Some(Model::LLM(llm)),
            None,
        )
        .await
        .unwrap();

        let bytes = endpoint
            .get_account_data(&coordinator_account)
            .await
            .unwrap()
            .unwrap();
        // The byte itself, as the program wrote it.
        assert_eq!(bytes[robust_aggregation_offset()], u8::from(robust));
        let account = coordinator_account_from_bytes(&bytes).unwrap();
        assert_eq!(account.version, CoordinatorAccount::VERSION);
        let Model::LLM(read) = account.state.coordinator.model;
        match read.optimizer {
            OptimizerDefinition::Distro {
                robust_aggregation,
                quantize_1bit,
                compression_topk,
                ..
            } => {
                assert_eq!(robust_aggregation, robust);
                // The fields beside it are the run's, untouched.
                assert!(quantize_1bit);
                assert_eq!(compression_topk, 8);
            },
            other => panic!("the run's optimizer read back as {other:?}"),
        }
    }
}
