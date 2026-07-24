# Leviathan on devnet

The programs and a funded, joinable training run are live on Solana devnet.
This is a working devnet deployment, not mainnet: bonds and rewards use a
devnet SPL collateral token with no real value.

## Programs (devnet)

| program | id |
|---|---|
| coordinator | `JD9rHTiqBFgHjViWZc7gFZX74LvKKysbLbqFRaFvtmmN` |
| authorizer | `2Kg5ERG6ubuzyPmQ24axsws7V2ja2EvWp5CHMKFCrTxv` |
| treasurer | `9A1kc8Dr9dFJW9t1npAk7EHrADm6TAyFeVLH27CDdvv8` |

Build with `anchor build --no-idl`.

## Flagship run

| field | value |
|---|---|
| run id | `leviathan-devnet` |
| coordinator account | `FyACSfZFC2oRiJqx7vYakrrMzm46AqTYTSgBW7DHxCHY` |
| collateral mint | `BWLv1Fj5RKJbcr3ZMLVKhviFq1i3tq6afgVS2ngyot3X` |
| model | Nano-Llama (nano CI model) |
| rewards funded | 500,000 collateral in the run vault |
| epoch rates | earning 1,000 per epoch (shared), slashing 100 per client |
| access | permissionless (authorizer sentinel `111...1`) |

The run is treasurer-managed and unpaused; it sits in WaitingForMembers until a
client connects, then advances through its epochs as clients tick it. Healthy
clients accrue earned points at each epoch boundary, redeemable for collateral
through the treasurer's `participant_claim`.

## Run a node

If you have nothing set up yet, one line fetches everything, checks your
prerequisites, and guides you the rest of the way:

```
curl -fsSL https://raw.githubusercontent.com/wienerlabs/leviathan-net/main/scripts/install.sh | bash
```

Set `WALLET=<keypair path>` (and optionally `BOND=<amount>`) before that line to
go straight from clone to a running, bonded node.

If you already have the repository, one command sets up the libtorch toolchain
(PyTorch 2.9.1 for the tch fork), builds the client, and joins the flagship run:

```
./scripts/leviathan-node.sh --wallet <path/to/devnet-keypair.json> [--bond <amount>]
```

`--bond <amount>` posts collateral through the treasurer before joining, so a
bonded node is one command. Inspect and manage the bond separately with
`run-manager bond-status`, `bond-deposit`, `bond-withdraw-request` and
`bond-withdraw-finalize`.

The wallet needs a little devnet SOL for transaction fees. Override `RUN_ID`,
`RPC`, `WS_RPC`, `TORCH_VENV` or `AUTHORIZER` via env if needed.

### Dashboard telemetry

`leviathan-indexer --features live` reads a coordinator account and prints run
telemetry as JSON. It is libtorch-free, so it runs anywhere. Publish it for the
dashboard:

```
OUT=telemetry.json ./scripts/publish-telemetry.sh \
  --coordinator-account <coordinator-pubkey> --run-id <run> --rpc <rpc> \
  --reward-per-round 0.324 --bond 10.55 --slash-when-caught 10.55
```

The economics flags add the on-chain economic security verdict. The
`publish telemetry` GitHub Action runs this every 15 minutes and commits
`telemetry.json`; point the leviathan-web dashboard's `VITE_TELEMETRY_URL` at the
raw file to make it live.

### Configure once

A dedicated RPC is strongly recommended. The shared public endpoint rate-limits
under a single node's transaction load, which stalls joins and ticks. Save your
settings once and every command picks them up:

```
mkdir -p ~/.leviathan && cp scripts/leviathan-env.example ~/.leviathan/env
# then edit ~/.leviathan/env with your RPC, wallet and run
```

`leviathan-node.sh` and `install.sh` source `~/.leviathan/env` (override the path
with `LEVIATHAN_ENV`). Explicit env and flags still win over the file. The
verifier daemon reads its RPC from `--rpc-url` or `SOLANA_RPC_URL`; `run-manager`
reads `--rpc` or the `RPC` env, so exporting the same values covers all three. The first build
links libtorch and takes a few minutes; subsequent runs are instant.

`LEVIATHAN_JOIN_TIMEOUT_SECS` (default 45 in the script, 30 in the client) sets
the join-transaction confirmation deadline; the public devnet RPC routinely
exceeds the client's original 5s, so this is what makes sustained multi-epoch
re-joining work.

## What is live vs what is next

Live on devnet: the three programs, the funded flagship run, sustained
multi-epoch training by a real client on a real model, and the full conviction
loop (bond, dispute, slash, forfeit) proven end to end by
`devnet-conviction-demo` and the memnet suites.

Reward accrual is proven deterministically in the memnet suites (a client that
stays Healthy through a full epoch earns its share of the epoch rate). Smooth
per-epoch reward cycling on the live run wants either several connected nodes or
a dedicated RPC: the public devnet endpoint is slow enough that single-node epoch
completion and run-manager operations are unreliable, which is an infrastructure
constraint rather than a protocol one.

Next (Phase 2): a dedicated devnet RPC for reliable operations, bond deposit
enforced at join so every training node is bonded by the protocol rather than by
convention, a verifier daemon that audits live training contributions and slashes
on a fraud verdict, and a multi-volunteer swarm behind an iroh relay.
