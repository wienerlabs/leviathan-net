//! A client records the run's checkpoint on chain once its upload is done.
//! The run's last checkpoint is uploaded during the last cooldown, so it is
//! often sent after the run has finished. The coordinator records it then
//! too, from a client still on the ended epoch's roster, and does not tick
//! the halted run (wienerlabs/leviathan-net#85).

use psyche_coordinator::model::Checkpoint;
use psyche_coordinator::model::HubRepo;
use psyche_coordinator::model::Model;
use psyche_coordinator::RunState;
use psyche_core::FixedString;
use psyche_solana_coordinator::ProgramError;
use psyche_solana_tooling::process_coordinator_instructions::process_coordinator_checkpoint;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;

use super::bonded_run::Harness;
use super::bonded_run::JoinGate;
use super::bonded_run::NODES;
use super::refusal::refusal_code;

/// The repository a client uploaded the run's final model to.
fn final_model() -> HubRepo {
    HubRepo {
        repo_id: FixedString::from_str_truncated("leviathan/final-model"),
        revision: Some(FixedString::from_str_truncated("a1b2c3d")),
    }
}

/// The Hub repository behind the run's checkpoint. Once an epoch has run,
/// the coordinator serves the model peer to peer, and the Hub repository a
/// client uploads to stands behind that.
fn hub_repo_of(model: &Model) -> HubRepo {
    let Model::LLM(llm) = model;
    match llm.checkpoint {
        Checkpoint::P2P(repo) => repo,
        other => {
            panic!("the run's checkpoint is not served peer to peer: {other:?}")
        },
    }
}

#[tokio::test]
pub async fn a_checkpoint_sent_once_the_run_has_finished_is_recorded() {
    let mut h = Harness::with(
        750,
        "Leviathan checkpoint finished",
        JoinGate::Bond,
        NODES,
        |config| config.total_steps = 2,
    )
    .await;
    for node in 0..NODES {
        h.join(node).await.unwrap();
    }
    h.advance_to(RunState::Finished).await;
    let before = h.state().await.coordinator;
    assert_ne!(hub_repo_of(&before.model), final_model());

    // A node of the ended epoch records the model it uploaded.
    let roster: Vec<usize> = h.roster().await;
    assert!(
        !roster.is_empty(),
        "the ended epoch's roster is still there"
    );
    let node = roster[0];
    let wallet = h.nodes[node].wallet.insecure_clone();
    process_coordinator_checkpoint(
        &mut h.endpoint,
        &h.payer,
        &wallet,
        &h.coordinator_instance,
        &h.coordinator_account,
        Checkpoint::Hub(final_model()),
    )
    .await
    .unwrap();
    let after = h.state().await.coordinator;
    assert_eq!(hub_repo_of(&after.model), final_model());
    // The halted run was not ticked.
    assert_eq!(after.run_state, RunState::Finished);
    assert_eq!(after.progress.epoch, before.progress.epoch);
    assert_eq!(after.progress.step, before.progress.step);

    // A wallet that is not one of the run's clients records nothing.
    let stranger = Keypair::new();
    h.endpoint
        .request_airdrop(&stranger.pubkey(), 1_000_000_000)
        .await
        .unwrap();
    assert_eq!(
        refusal_code(
            process_coordinator_checkpoint(
                &mut h.endpoint,
                &h.payer,
                &stranger,
                &h.coordinator_instance,
                &h.coordinator_account,
                Checkpoint::Hub(HubRepo {
                    repo_id: FixedString::from_str_truncated("someone/else"),
                    revision: None,
                }),
            )
            .await
        ),
        u32::from(ProgramError::SignerNotAClient),
    );
    assert_eq!(
        hub_repo_of(&h.state().await.coordinator.model),
        final_model()
    );
}
