//! What the suites convict on: a payload the target signed with its p2p key,
//! as the treasurer requires (wienerlabs/leviathan-net#43).
//!
//! The payloads are real dumps produced by a real model
//! (`tests/fixtures/daemon`). A payload commitment's results digest is the
//! sha256 of the dump bytes (psyche-network's `results_digest`), so it is
//! computed here from the file itself. The step and batch come from chain
//! (`signed_commitment::payload_header`): the round the target is training,
//! or trained last, and the batch the coordinator assigned it there.

use std::path::PathBuf;

use psyche_coordinator::Coordinator;
use psyche_core::sha256;
#[cfg(feature = "daemon")]
use psyche_core::BatchId;
use psyche_core::NodeIdentity;
use psyche_solana_tooling::get_accounts::get_coordinator_account_state;
use psyche_solana_tooling::get_accounts::get_run;
use psyche_solana_tooling::process_treasurer_instructions::process_treasurer_run_slash;
#[cfg(feature = "daemon")]
use psyche_solana_tooling::signed_commitment;
use psyche_solana_tooling::signed_commitment::SignedCommitment;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

/// The payload a cheater broadcast: an honest one with a delta injected.
pub const FRAUD_PAYLOAD: &str = "fraud.vec-postcard";
/// The payload an honest trainer broadcast for the same batch, which is what
/// a verifier's replay reproduces.
pub const HONEST_PAYLOAD: &str = "honest.vec-postcard";

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/daemon")
        .join(name)
}

/// The results digest of the payload dump `name`.
pub fn payload_digest(name: &str) -> [u8; 32] {
    let bytes = std::fs::read(fixture(name)).expect("the fixture is readable");
    sha256(&bytes)
}

/// The digest a verifier records for its replay of the batch.
pub fn replayed_digest() -> [u8; 32] {
    payload_digest(HONEST_PAYLOAD)
}

/// A node's identity with its own p2p key, as a client joins with.
pub fn identity(wallet: &Keypair, p2p: &Keypair) -> NodeIdentity {
    NodeIdentity::new(wallet.pubkey().to_bytes(), p2p.pubkey().to_bytes())
}

/// The payload `node` committed to in the round the coordinator is in, with
/// the results of the dump `payload`, signed by `p2p`, the key `node` joined
/// with.
pub fn signed_payload(
    coordinator: &Coordinator,
    tie_breaker_committee_size: u16,
    node: &NodeIdentity,
    p2p: &Keypair,
    payload: &str,
) -> SignedCommitment {
    SignedCommitment::for_round(
        coordinator,
        tie_breaker_committee_size,
        node,
        p2p,
        payload_digest(payload),
    )
    .expect("the node trains a batch this round")
}

/// Which work a payload of `node` from the round the coordinator is in is.
#[cfg(feature = "daemon")]
pub fn payload_header(
    coordinator: &Coordinator,
    tie_breaker_committee_size: u16,
    node: &NodeIdentity,
) -> (u32, u32, BatchId) {
    signed_commitment::payload_header(
        coordinator,
        tie_breaker_committee_size,
        node,
    )
    .expect("the node trains a batch this round")
}

/// The name a client gives the dump of `wallet`'s payload for `batch_id` at
/// `step`.
#[cfg(feature = "daemon")]
pub fn dump_name(wallet: &Keypair, step: u32, batch_id: BatchId) -> String {
    format!(
        "result-{}-step{step}-batch{batch_id}.vec-postcard",
        wallet.pubkey()
    )
}

/// Keeps the payload dump at `source` in `dir` as a node keeps a peer's payload
/// it verified in the run `run_id`. It writes the record of what the committer signed for it, as
/// the client does: `p2p` signs the payload's commitment hash with the same
/// ed25519 key the client's iroh identity holds. Then it writes the dump
/// itself, under the name the client gives it.
#[cfg(feature = "daemon")]
pub fn keep_signed_payload(
    dir: &std::path::Path,
    source: &std::path::Path,
    wallet: &Keypair,
    p2p: &Keypair,
    run_id: &str,
    (step, trainer_nonce, batch_id): (u32, u32, BatchId),
) -> PathBuf {
    let distro_results = leviathan_verifier::read_dump(source).unwrap();
    let payload = psyche_network::TransmittableDistroResult {
        step,
        trainer_nonce,
        batch_id,
        distro_results,
    };
    let data_hash = payload.comptue_hash(run_id).unwrap();
    let key = psyche_network::SecretKey::from_bytes(&p2p.secret().to_bytes());
    assert_eq!(key.public().as_bytes(), &p2p.pubkey().to_bytes());
    let signature = key.sign(&data_hash).to_bytes();
    let record =
        psyche_network::CommitmentRecord::new(&payload, data_hash, signature);
    let path = dir.join(dump_name(wallet, step, batch_id));
    std::fs::write(
        psyche_network::commitment_record_path(&path),
        record.to_bytes(),
    )
    .unwrap();
    std::fs::write(
        &path,
        psyche_network::distro_results_to_bytes(&payload.distro_results)
            .unwrap(),
    )
    .unwrap();
    path
}

/// Keeps the payload dump at `source` in `dir` with no commitment record, as
/// a verifier keeps its own reference results.
#[cfg(feature = "daemon")]
pub fn keep_reference(
    dir: &std::path::Path,
    source: &std::path::Path,
    wallet: &Keypair,
    (step, _, batch_id): (u32, u32, BatchId),
) -> PathBuf {
    let path = dir.join(dump_name(wallet, step, batch_id));
    std::fs::copy(source, &path).unwrap();
    path
}

/// The run authority `authority` convicts the client at roster `index`, the
/// node of `wallet` and `p2p`, on the payload it signed in the round the
/// coordinator is in.
#[allow(clippy::too_many_arguments)]
pub async fn slash_on_signed_payload(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    authority: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    index: u64,
    wallet: &Keypair,
    p2p: &Keypair,
) -> anyhow::Result<()> {
    let tie_breakers = get_run(endpoint, run)
        .await?
        .expect("the run exists")
        .tie_breaker_committee_size;
    let coordinator =
        get_coordinator_account_state(endpoint, coordinator_account)
            .await?
            .expect("the coordinator exists")
            .coordinator;
    let commitment = signed_payload(
        &coordinator,
        tie_breakers,
        &identity(wallet, p2p),
        p2p,
        FRAUD_PAYLOAD,
    );
    process_treasurer_run_slash(
        endpoint,
        payer,
        authority,
        run,
        coordinator_account,
        run_id,
        index,
        &commitment,
        replayed_digest(),
    )
    .await
}
