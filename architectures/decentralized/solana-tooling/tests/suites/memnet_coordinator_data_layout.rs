//! The coordinator account's layout, as the programs write it and as every
//! client reads it back.
//!
//! Version 2 added the run's aggregation rule to the DisTrO optimizer, in what
//! version 1 had as padding (wienerlabs/leviathan-net#60). The padding of a
//! real version 1 account is whatever the program's stack held, so a version 1
//! account has to be refused, not read with that as a setting. Version 3
//! appended the commitment a round's seed is revealed from to the instance
//! state (wienerlabs/leviathan-net#62), which moved the account's `nonce` and
//! grew it by eight bytes.

use std::collections::HashMap;

use anchor_lang::AccountSerialize;
use anchor_lang::Discriminator;
use anchor_lang::Space;
use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HttpTrainingDataLocation;
use psyche_coordinator::model::LLMArchitecture;
use psyche_coordinator::model::LLMTrainingDataLocation;
use psyche_coordinator::model::LLMTrainingDataType;
use psyche_coordinator::model::Model;
use psyche_coordinator::CoordinatorConfig;
use psyche_coordinator::Round;
use psyche_coordinator::RunState;
use psyche_core::CosineLR;
use psyche_core::FixedString;
use psyche_core::FixedVec;
use psyche_core::LearningRateSchedule;
use psyche_core::OptimizerDefinition;
use psyche_core::Shuffle;
use psyche_core::SmallBoolean;
use psyche_core::TokenSize;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_coordinator::logic::InitCoordinatorParams;
use psyche_solana_coordinator::CoordinatorAccount;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_coordinator::DeserializeCoordinatorFromBytes;
use psyche_solana_coordinator::ProgramError;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint;
use psyche_solana_tooling::create_memnet_endpoint::create_memnet_endpoint_with_accounts;
use psyche_solana_tooling::get_accounts::ensure_coordinator_version;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_free;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_init;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_set_paused;
use psyche_solana_tooling::process_coordinator_instructions::process_update;
use solana_sdk::account::Account;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::rent::Rent;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use super::refusal::refusal_code;

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
    // Version 3 is longer, so the size alone refuses it before its version is
    // read; a version 2 account, the same size as version 1, is refused by
    // its version.
    match coordinator_account_from_bytes(&bytes) {
        Err(DeserializeCoordinatorFromBytes::IncorrectSize {
            expected,
            actual,
        }) => {
            assert_eq!(
                expected,
                CoordinatorAccount::space_with_discriminator()
            );
            assert_eq!(actual, bytes.len());
        },
        Err(other) => panic!("refused for the wrong reason: {other}"),
        Ok(_) => panic!("a version 1 account was read as the current version"),
    }
    let mut same_size = as_current_version(bytes);
    let version = CoordinatorAccount::DISCRIMINATOR.len()
        + std::mem::offset_of!(CoordinatorAccount, version);
    same_size[version..version + 8].copy_from_slice(&2u64.to_le_bytes());
    match coordinator_account_from_bytes(&same_size) {
        Err(DeserializeCoordinatorFromBytes::InvalidVersion {
            expected,
            actual,
        }) => assert_eq!((expected, actual), (CoordinatorAccount::VERSION, 2)),
        Err(other) => panic!("refused for the wrong reason: {other}"),
        Ok(_) => panic!("a version 2 account was read as the current version"),
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

const VERSION_ONE: &[u8] =
    include_bytes!("../fixtures/coordinator-account-v1.so");

/// What a version 1 account becomes in the current version: the same bytes,
/// the version raised, the byte that was padding set to the plain mean version
/// 1 nodes took by default, and an empty round seed commitment where version 3
/// appended one, before the `nonce`.
fn as_current_version(mut bytes: Vec<u8>) -> Vec<u8> {
    let version = CoordinatorAccount::DISCRIMINATOR.len()
        + std::mem::offset_of!(CoordinatorAccount, version);
    bytes[version..version + 8]
        .copy_from_slice(&CoordinatorAccount::VERSION.to_le_bytes());
    bytes[robust_aggregation_offset()] = 0;
    let commitment = CoordinatorAccount::DISCRIMINATOR.len()
        + std::mem::offset_of!(CoordinatorAccount, state)
        + std::mem::offset_of!(CoordinatorInstanceState, seed_commit_slot);
    bytes.splice(commitment..commitment, 0u64.to_le_bytes());
    bytes
}

/// Every field of the real version 1 account reads back as it did, once it is
/// brought to the current version: version 2 moved nothing, and version 3 only
/// appended.
#[tokio::test]
pub async fn the_current_version_moved_no_field_of_a_real_version_one_account()
{
    let coordinator_bytes = as_current_version(VERSION_ONE.to_vec());
    let coordinator_account =
        coordinator_account_from_bytes(&coordinator_bytes).unwrap();
    // Check the general layout for corruption
    assert_eq!(coordinator_account.version, CoordinatorAccount::VERSION);
    assert_eq!(coordinator_account.nonce, 2);
    let state = coordinator_account.state;
    assert_eq!(state.seed_commit_slot, 0);
    assert_eq!(state.is_warmup_first_tick, SmallBoolean::FALSE);
    assert_eq!(state.is_training_first_tick, SmallBoolean::FALSE);
    assert_eq!(state.client_version, fixed_str("test"));
    // Check infos on the coordinator run metadata
    let metadata = state.metadata;
    assert_eq!(metadata.name, fixed_str(""));
    assert_eq!(metadata.description, fixed_str(""));
    assert_eq!(metadata.num_parameters, 1100000000);
    assert_eq!(metadata.vocab_size, 32768);
    // Check on the on the coordinator datastructure
    let coordinator = state.coordinator;
    assert_eq!(coordinator.run_id, fixed_str("test"));
    assert_eq!(coordinator.run_state, RunState::Uninitialized);
    assert_eq!(coordinator.run_state_start_unix_timestamp, 0);
    assert_eq!(coordinator.pending_pause, SmallBoolean::FALSE);
    // Coordinator model
    match coordinator.model {
        Model::LLM(llm) => {
            assert_eq!(llm.max_seq_len, 2048);
            assert_eq!(llm.cold_start_warmup_steps, 0);
            assert_eq!(llm.architecture, LLMArchitecture::HfLlama);
            match llm.checkpoint {
                Checkpoint::Hub(hub) => {
                    assert_eq!(
                        hub.repo_id,
                        fixed_str("emozilla/llama2-1.1b-gqa-init")
                    );
                    assert_eq!(hub.revision, None);
                },
                _ => panic!("Expected Hub checkpoint"),
            };
            assert_eq!(llm.data_type, LLMTrainingDataType::Pretraining);
            match llm.data_location {
                LLMTrainingDataLocation::Http(http) => {
                    match http.location {
                        HttpTrainingDataLocation::Gcp {
                            bucket_name,
                            filter_directory,
                        } => {
                            assert_eq!(
                                bucket_name,
                                fixed_str("nous-pretraining-public-us")
                            );
                            assert_eq!(
                                filter_directory,
                                fixed_str("fineweb-edu-tokenized-llama2")
                            );
                        },
                        _ => panic!("Expected Gcp data location"),
                    };
                    assert_eq!(http.token_size_in_bytes, TokenSize::TwoBytes);
                    assert_eq!(http.shuffle, Shuffle::DontShuffle);
                },
                _ => panic!("Expected Http data location"),
            };
            match llm.lr_schedule {
                LearningRateSchedule::Cosine(learning_rate) => {
                    assert_eq!(
                        learning_rate,
                        CosineLR::new(0.0004, 250, 0.0, 25000, 0.00004)
                    );
                },
                _ => panic!("Expected Constant LR schedule"),
            };
            match llm.optimizer {
                OptimizerDefinition::Distro {
                    clip_grad_norm,
                    weight_decay,
                    compression_decay,
                    compression_topk,
                    compression_chunk,
                    quantize_1bit,
                    robust_aggregation,
                } => {
                    assert_eq!(clip_grad_norm, Some(1.0));
                    assert_eq!(weight_decay, None);
                    assert_eq!(compression_decay, 0.999);
                    assert_eq!(compression_topk, 2);
                    assert_eq!(compression_chunk, 64);
                    assert!(!quantize_1bit);
                    // What version 1 nodes did unless told otherwise.
                    assert!(!robust_aggregation);
                },
                _ => panic!("Expected Distro optimizer"),
            }
        },
    };
    // Coordinator config
    assert_eq!(coordinator.config.warmup_time, 15);
    assert_eq!(coordinator.config.cooldown_time, 30);
    assert_eq!(coordinator.config.max_round_train_time, 15);
    assert_eq!(coordinator.config.round_witness_time, 1);
    assert_eq!(coordinator.config.global_batch_size_warmup_tokens, 0);
    assert_eq!(coordinator.config.epoch_time, 60);
    assert_eq!(coordinator.config.total_steps, 25000);
    assert_eq!(coordinator.config.init_min_clients, 1);
    assert_eq!(coordinator.config.min_clients, 1);
    assert_eq!(coordinator.config.witness_nodes, 0);
    assert_eq!(coordinator.config.global_batch_size_start, 2048);
    assert_eq!(coordinator.config.global_batch_size_end, 2048);
    assert_eq!(coordinator.config.verification_percent, 0);
    assert_eq!(coordinator.config.waiting_for_members_extra_time, 3);
    // Coordinator progress
    assert_eq!(coordinator.progress.epoch, 0);
    assert_eq!(coordinator.progress.step, 0);
    assert_eq!(coordinator.progress.epoch_start_data_index, 0);
    // Coordinator epoch state
    let epoch_state = coordinator.epoch_state;
    assert_eq!(epoch_state.rounds, [Round::default(); 4]);
    assert_eq!(epoch_state.clients, FixedVec::default());
    assert_eq!(epoch_state.exited_clients, FixedVec::default());
    assert_eq!(epoch_state.rounds_head, 0);
    assert_eq!(epoch_state.start_step, 0);
    assert_eq!(epoch_state.last_step, 0);
    assert_eq!(epoch_state.start_timestamp, 0);
    assert_eq!(epoch_state.first_round, SmallBoolean::FALSE);
    assert_eq!(epoch_state.cold_start_epoch, SmallBoolean::FALSE);
    // Coordinator clients state
    let clients_state = state.clients_state;
    assert_eq!(clients_state.clients.len(), 0);
    assert_eq!(clients_state.next_active, 0);
    let current_epoch_rates = clients_state.current_epoch_rates;
    assert_eq!(current_epoch_rates.earning_rate_total_shared, 0);
    assert_eq!(current_epoch_rates.slashing_rate_per_client, 0);
    let future_epoch_rates = clients_state.future_epoch_rates;
    assert_eq!(future_epoch_rates.earning_rate_total_shared, 1000000);
    assert_eq!(future_epoch_rates.slashing_rate_per_client, 0);
}

/// A chain holding `bytes` as the coordinator account of the fixture's run,
/// with an instance naming it and `main_authority` as the run's authority, as
/// a version 1 deployment would have left them.
async fn chain_holding(
    bytes: Vec<u8>,
    main_authority: &Pubkey,
) -> (ToolboxEndpoint, Pubkey, Pubkey) {
    let coordinator_account = Pubkey::new_unique();
    let run_id = "test";
    let (coordinator_instance, bump) = Pubkey::find_program_address(
        &[CoordinatorInstance::SEEDS_PREFIX, run_id.as_bytes()],
        &psyche_solana_coordinator::ID,
    );
    let mut instance = Vec::new();
    CoordinatorInstance {
        bump,
        main_authority: *main_authority,
        join_authority: Pubkey::new_unique(),
        coordinator_account,
        run_id: run_id.to_string(),
    }
    .try_serialize(&mut instance)
    .unwrap();
    instance.resize(8 + CoordinatorInstance::INIT_SPACE, 0);
    let owned = |data: Vec<u8>| Account {
        lamports: Rent::default().minimum_balance(data.len()),
        data,
        owner: psyche_solana_coordinator::ID,
        executable: false,
        rent_epoch: 0,
    };
    let endpoint = create_memnet_endpoint_with_accounts(HashMap::from([
        (coordinator_account, owned(bytes)),
        (coordinator_instance, owned(instance)),
    ]))
    .await;
    (endpoint, coordinator_instance, coordinator_account)
}

/// The programs refuse to act on an account of another version, with a
/// constraint error, which is why a program is never upgraded in place across
/// an account version (docs/OPS_RUNBOOK.md). The same account brought to the
/// current version is accepted, so the version is what is refused. Its
/// authority can still close it and take its rent back, a power it already
/// has over any run of its own, which it can pause and then close.
#[tokio::test]
pub async fn the_programs_refuse_an_account_of_another_version() {
    let rehearsal = genesis_rehearsal();
    for (bytes, current) in [
        (VERSION_ONE.to_vec(), false),
        (as_current_version(VERSION_ONE.to_vec()), true),
    ] {
        let main_authority = Keypair::new();
        let (mut endpoint, coordinator_instance, coordinator_account) =
            chain_holding(bytes, &main_authority.pubkey()).await;
        let payer = Keypair::new();
        endpoint
            .request_airdrop(&payer.pubkey(), 5_000_000_000)
            .await
            .unwrap();

        let readable =
            ensure_coordinator_version(&mut endpoint, &coordinator_account)
                .await;
        assert_eq!(readable.is_ok(), current, "{readable:?}");

        let updated = process_update(
            &mut endpoint,
            &payer,
            &main_authority,
            &coordinator_instance,
            &coordinator_account,
            None,
            Some(rehearsal.config),
            None,
            None,
        )
        .await;
        if current {
            updated.unwrap();
        } else {
            assert_eq!(
                refusal_code(updated),
                anchor_lang::error::ErrorCode::ConstraintRaw as u32,
                "a version 1 account was updated"
            );
        }

        let spill = Keypair::new().pubkey();
        let freed = process_coordinator_free(
            &mut endpoint,
            &payer,
            &main_authority,
            &spill,
            &coordinator_instance,
            &coordinator_account,
        )
        .await;
        freed.unwrap();
        for closed in [coordinator_account, coordinator_instance] {
            assert!(
                endpoint.get_account(&closed).await.unwrap().is_none(),
                "{closed} was not closed"
            );
        }
        assert!(
            endpoint.get_account(&spill).await.unwrap().is_some(),
            "the rent went nowhere"
        );
    }
}

/// Once a run has started, its nodes hold optimizers built from its definition,
/// so the definition, the aggregation rule above all, is frozen: a change would
/// reach only the nodes that join after it. The rest of the run's settings can
/// still change while it is paused.
#[tokio::test]
pub async fn a_started_runs_optimizer_cannot_change() {
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
            run_id: "Leviathan frozen optimizer".to_string(),
            main_authority: main_authority.pubkey(),
            join_authority: Keypair::new().pubkey(),
            client_version: "latest".to_string(),
        },
    )
    .await
    .unwrap();
    let rehearsal = genesis_rehearsal();
    let with_rule = |robust: bool| {
        let Model::LLM(mut llm) = rehearsal.model;
        let OptimizerDefinition::Distro {
            robust_aggregation, ..
        } = &mut llm.optimizer
        else {
            panic!("the genesis rehearsal trains with DisTrO");
        };
        *robust_aggregation = robust;
        Model::LLM(llm)
    };

    // Before the first start, the rule can be set and set again.
    for robust in [false, true] {
        process_update(
            &mut endpoint,
            &payer,
            &main_authority,
            &coordinator_instance,
            &coordinator_account,
            None,
            Some(rehearsal.config),
            Some(with_rule(robust)),
            None,
        )
        .await
        .unwrap();
    }

    // Started, then paused again: halted, but no longer new.
    for paused in [false, true] {
        process_coordinator_set_paused(
            &mut endpoint,
            &payer,
            &main_authority,
            &coordinator_instance,
            &coordinator_account,
            paused,
        )
        .await
        .unwrap();
    }

    let refused = process_update(
        &mut endpoint,
        &payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        None,
        None,
        Some(with_rule(false)),
        None,
    )
    .await;
    assert_eq!(
        refusal_code(refused),
        u32::from(ProgramError::ModelOptimizerFrozen)
    );

    // The same optimizer is accepted with the rest of a changed config.
    let mut config = rehearsal.config;
    config.warmup_time += 1;
    process_update(
        &mut endpoint,
        &payer,
        &main_authority,
        &coordinator_instance,
        &coordinator_account,
        None,
        Some(config),
        Some(with_rule(true)),
        None,
    )
    .await
    .unwrap();
    let bytes = endpoint
        .get_account_data(&coordinator_account)
        .await
        .unwrap()
        .unwrap();
    let account = coordinator_account_from_bytes(&bytes).unwrap();
    assert_eq!(
        account.state.coordinator.config.warmup_time,
        config.warmup_time
    );
    assert_eq!(bytes[robust_aggregation_offset()], 1);
}

fn fixed_str<const L: usize>(value: &str) -> FixedString<L> {
    FixedString::from_str_truncated(value)
}
