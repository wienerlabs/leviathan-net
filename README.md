<p align="center">
    <img src="./assets/leviathan-banner.png" alt="Leviathan">
</p>

<h1 align="center">Leviathan</h1>

<p align="center"><b>Trustless training for the People's model.</b></p>

Frontier AI is produced inside five balance sheets. The mathematics of training over the open internet is already solved and shipping; what remains unsolved is trust. Nobody has made it safe to accept a gradient from a stranger and pay them for it.

Leviathan is a Solana-coordinated training network where anyone with a GPU joins by posting a bond, earns Proof of Gradient for contributions that survive verification, and loses the bond if they lie. The chain carries commitments, audits and money. The mesh carries compressed tensors. The model belongs to the network that trained it.

Hobbes drew Leviathan as a giant composed of thousands of individuals. This one is composed of thousands of GPUs.

> This repository is `leviathan-net`, the network substrate: a fork of [PsycheFoundation/nousnet](https://github.com/PsycheFoundation/nousnet) (Apache-2.0) that carries the layer upstream designed but left unimplemented. The vision, whitepaper, simulation and phase plan live in [wienerlabs/leviathan](https://github.com/wienerlabs/leviathan).

## The open slot

Every live decentralized-training network picked one column. Nobody picked both.

| | Verification guarantees | Live economics (bonds, slashing) |
|---|---|---|
| Bittensor training subnets | scoring gates, admitted gaps | emissions only, no bonds |
| Nous Psyche | witness liveness, verifier is a `todo!()` | no stake, dead slash code, whitelist |
| Gensyn Verde | strong, FP32 single-GPU determinism | pre-mainnet |
| OVIG | tolerance-band replay audits | a paper, not a network |
| **Leviathan** | **OVIG-style replay audits** | **bonds sized (1-p)/p, live slashing** |

The combination of bonded contributions, random replay audits and slashing is not run by anyone for live LLM training today. That is the entire moat, and every piece is individually proven in public work from 2024 to 2026.

## Security model

Three layers, each covering the others' gap:

1. **Robust aggregation bounds damage.** Centered clipping with far-outlier excision caps any contribution's influence radius in the round it happens. Re-validated on real transformer gradients: a 5/16 sign-flip coalition drives naive mean to divergence while clip plus excision holds the honest baseline and admits 3% of malicious contributions.
2. **Replay audits price lying.** Each contribution is audited with probability p. Local rounds are replayable pure functions of `(checkpoint, seed, data)`, so a mismatch beyond the calibrated tolerance band is a binary fraud proof. On the ALIE run at p = 0.1, all five stealth cheaters were caught at a mean of 9.8 rounds against the 1/p = 10 expectation.
3. **Bonds make sybil a cost.** Break-even bond = reward x (1-p)/p. A slashed identity re-enters only by posting a fresh bond, and slashed stake funds the verifiers hunting it. Security is self-funding under attack.

This is economic security with published parameters, never a cryptographic overclaim.

## Architecture

```mermaid
classDiagram
    class Participant {
        +GPU_Resources
        +post_bond()
        +submit_gradient()
    }
    class Treasurer {
        +participant_bond_deposit()
        +request_withdraw()
        +finalize_withdraw()
        +vault_management()
    }
    class P2P_Mesh {
        <<Iroh / DisTrO>>
        +compressed_tensors
        +sparse_loco_recipe
        +diloco_loop
    }
    class Solana_Blockchain {
        <<Network Substrate>>
        +Coordinator_Program
        +Treasurer_Program
        +Authorizer_Program
        +settle_slashing()
    }
    class Coordinator {
        +verification_percent
        +audit_selection(p_probability)
        +derive_assignments(seed)
    }
    class Audit_System {
        +deterministic_replay()
        +tolerance_band_check()
        +conviction_logic()
    }
    class Robust_Aggregation {
        +centered_clipping()
        +outlier_excision()
        +bound_damage()
    }
    class Security_Layers {
        <<Economic and Robustness>>
        +Robust_Aggregation
        +Replay_Audits
        +Bonded_Slashing
    }
    class Phase0_Results {
        +Honest_Loss_2_175
        +Byzantine_Resilience
        +Mean_Catch_Time_9_8_rounds
    }

    Participant "1" -- "1" Treasurer : bonded_participation
    Participant "N" -- "1" P2P_Mesh : mesh_communication
    Treasurer --> Solana_Blockchain : settle
    Treasurer --> Security_Layers : layer_3_sybil_cost
    Coordinator --* Solana_Blockchain
    Coordinator --> Audit_System : triggers_p_0_1
    Audit_System --> Robust_Aggregation : provides_verification
    Audit_System --> Security_Layers : layer_2
    Robust_Aggregation --> Security_Layers : layer_1
    Security_Layers --> Solana_Blockchain : execute_slash
    Phase0_Results ..> Security_Layers : validates_parameters
```

## What lives here

This fork carries the layer upstream left as dead code, wired to the reward engine and covered by the in-process `memnet` test harness (no validator required).

| Component | State |
|---|---|
| Bonded participation (`solana-treasurer`) | `participant_bond_deposit`, `request_withdraw`, `finalize_withdraw` with a challenge window; settlement forfeits slashed collateral into the run vault |
| Audit lottery (`shared/coordinator`) | `audit_selection` derives verifier-to-target assignments from the round seed, deterministic and replayable; audit pressure scales with `verification_percent` |
| Dispute and slash (`solana-coordinator`) | `slash_client` convicts a client, ejection carries into `exited_clients` where the slashing rate applies; the `slashed` counter upstream never read now has a live producer |
| Committee vote (`solana-treasurer`) | bonded verifiers submit verdicts; a two-thirds quorum fires the slash and the forfeited bond pays the voters (`slash_bounty_bps`) |
| Appeals court (`solana-treasurer`) | per run, a quorum opens a challenge window instead of slashing at once; a separately drawn tie-breaker bench re-audits, and an overturn makes the convicting verifiers forfeit their bonds ([docs/DISPUTE_DESIGN.md](docs/DISPUTE_DESIGN.md)) |
| Bond-gated claim (`solana-treasurer`) | a participant with no bond posted cannot claim its earnings, so nothing is taken without something left to slash |
| Replay verifier (`shared/verifier`, `leviathan-verifier-daemon`) | relative L2 distance against a tolerance band, sha256 fraud proofs matching the on-chain commitment; the daemon replays what peers broadcast and submits the slash |
| Band calibration (`tools/rust-tools/leviathan-verifier`) | `calibrate-band` measures honest drift per hardware class on the production replay path ([docs/BAND_CALIBRATION.md](docs/BAND_CALIBRATION.md)) |
| Indexer (`tools/rust-tools/leviathan-indexer`) | reads a coordinator account and prints run telemetry and the economic security verdict as JSON |

Upstream's verifier dispatch was `Committee::Verifier => todo!()`, its `verification_percent` was pinned to zero by an assert, `ClientState::Ejected` was never set by any path, and the `slashed` counter no program read. All four are now live.

## Devnet deployment

| Program | ID |
|---|---|
| coordinator | `GdHJHiQp7uMv8TanfpaCaKQ8nHm5suvEt9JvjpZFWZ19` |
| authorizer | `ECEmta24U9WCwh397N4diSc8JnAbyJTG3YiTUVL5umrb` |
| treasurer | `Fq1Mv8osXqHxiiXjm4yhvQGE5wgx9QMueK8n2qwbqovV` |

Programs build with `anchor build --no-idl`. The instruction layers are exercised end to end by the `memnet` suites:

```
cargo test -p psyche-solana-tooling
```

The whole security economics also runs against live devnet through the toolbox RPC endpoint:

```
cargo run -p psyche-solana-tooling --bin devnet-conviction-demo --features demo
```

A verified run: a participant posts a bond of 500, the run authority convicts it mid-epoch through the treasurer, the coordinator writes `slashed = 200` at epoch end, and the bond withdrawal returns 300 while the forfeited 200 stays in the run vault as reward liquidity. The complete cross-program loop, on live chain.

The same toolbox carries three more live drivers (`--features demo`):

| Binary | What it proves on devnet |
|---|---|
| `devnet-committee-demo` | bonded verifiers reach a two-thirds quorum and the target is slashed without the run authority |
| `devnet-appeals-demo` | two verifiers convict, the target appeals, two tie-breakers overturn; both verifiers forfeit 200 and the target keeps its full bond |
| `devnet-bond-gated-claim-demo` | a claim is refused before the bond exists and accepted after; a zero withdraw delay is refused |

And with a real model in the loop: on a bonded devnet run, three nodes posted 500 each and one broadcast a forged gradient through the client's `LEVIATHAN_FAKE_DELTA` test switch. `leviathan-verifier-daemon` replayed the first step: the honest nodes landed at 0.0037 and 0.0043, the cheater at 6.0046 against the 0.05 band. The daemon slashed it on chain with both hashes in the log, the coordinator ejected it, and its withdrawal returned nothing of its 500 bond (#29). That slash went through the run authority; the committee path above is separate.

## Run a node

```
curl -fsSL https://raw.githubusercontent.com/wienerlabs/leviathan-net/main/scripts/install.sh | bash
```

[DEVNET.md](./DEVNET.md) covers the flagship run, join authorization, the container image and the env file; [docs/NOSANA.md](docs/NOSANA.md) covers renting GPU hosts; [docs/FIRST_RUN.md](docs/FIRST_RUN.md) lists what is still to decide before a public run.

## Where it stands

The protocol is live on devnet. Not there yet: the model is a test model (Nano-Llama, 30-token vocabulary); joining needs a per-node authorization from the run's join authority; bonds with real value wait on the external audit (wienerlabs/leviathan#4), so the planned first mainnet stage is bondless ([config/leviathan/mainnet-genesis-bondless.toml](config/leviathan/mainnet-genesis-bondless.toml)); the programs are not deployed on mainnet-beta; the verifier daemon replays a cold-start step; robust aggregation sits behind `LEVIATHAN_ROBUST_AGG` and is off by default. The internal security review is public in [docs/SECURITY_REVIEW_INTERNAL.md](docs/SECURITY_REVIEW_INTERNAL.md).

## Lineage

Built on the shoulders of, and differentiated from, PsycheFoundation/nousnet (Apache-2.0): the Solana coordinator, iroh P2P mesh, and Rust DisTrO compression are inherited; the bond, audit, slash, committee and appeals layers are ours. The compression recipe follows SparseLoCo over a DiLoCo outer loop; the inference path follows vLLM workers verified by TOPLOC. This repository remains Apache-2.0; see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

