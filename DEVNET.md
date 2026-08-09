# Leviathan on devnet

The programs and a funded, joinable training run are live on Solana devnet.
This is a working devnet deployment, not mainnet: bonds and rewards use a
devnet SPL collateral token with no real value.

## Programs (devnet)

| program | id |
|---|---|
| coordinator | `GdHJHiQp7uMv8TanfpaCaKQ8nHm5suvEt9JvjpZFWZ19` |
| authorizer | `ECEmta24U9WCwh397N4diSc8JnAbyJTG3YiTUVL5umrb` |
| treasurer | `Fq1Mv8osXqHxiiXjm4yhvQGE5wgx9QMueK8n2qwbqovV` |

Build with `anchor build --no-idl`.

## Flagship run

| field | value |
|---|---|
| run id | `leviathan-devnet` |
| coordinator instance | `HXpzk5aUxUiRVs12DJZBPQMwtdoZXZhkheHqEBXyqXzc` |
| coordinator account | `GMSnhB7W4cVUr4kstVB7DgHDsoCYNk5jgHmZQwoSiDFP` |
| model | Nano-Llama (nano CI model) |
| authorities | main and join both `33qU3JFkrehB2HkgdHzcpj9gDkFk8c2okQC51REWhjKh` |
| access | per-node join authorization, granted by the join authority |
| bonds | none: this run is coordinator-only |

The run is unpaused and sits in WaitingForMembers until clients connect, then
advances through its epochs as clients tick it.

This one is **coordinator-only**: no treasurer, so no bonds, no reward accrual
and no slash settlement. It exists to carry training on the redeployed programs.
A treasurer-managed run has to be created separately, and its economics — the
epoch rates in particular — are set per run.

### Joining

`join_run` requires an authorization whose grantor is the run's join authority,
whose grantee is the node, and whose scope is `CoordinatorJoinRun`. The join
authority creates one per node:

```
run-manager join-authorization-create --rpc <rpc> --ws-rpc <ws> \
  --wallet-private-key-path <join-authority> --authorizer <node pubkey>
```

The three nodes currently authorized:

| node | authorization |
|---|---|
| `8tcQfLmW1ucG5ohoDf3fci8vRz3Kspviu5bKsTm628Un` | `5nqfj7T2mqHKWztKmVmXagJzvopQeB7CXAebDiKKuzYH` |
| `Cgh78poQ5uwqkzxz4Qt5e7jAodqWtkYAekzU3JPWVijD` | `CwLedrkTsR3Nc3jP83xQ2HT1wxZspAgtHYCapHnx6QDn` |
| `DW2boohj6G1NMwr2w3NtLd3da7QFQ6Gkp9xD1s6gozqs` | `CEaRbDetAfa5vfeP8wwRyRnHd5DTFy4KZP89GZA9pj7r` |

### Why the program ids changed

The internal security review (wienerlabs/leviathan#15) added fields to
`AuditVerdict`, `Run` and `Participant`, so accounts written by the old programs
cannot be read by the new ones. The old deployment's upgrade authority
(`HYXmvGi8SFn7GdGLA2m7YVUxqqwv3rYy7wYhwZ4EoaYn`) is not held here, so the
programs were redeployed under ids this repository controls rather than upgraded
in place. The previous deployment and its runs are abandoned, not migrated.

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

### Run a node from the container image

`nousresearch/psyche-client` runs the same client on a rented GPU host. The
entrypoint takes its configuration from the environment:

| variable | value | why |
|---|---|---|
| `RPC` / `WS_RPC` | your devnet endpoints | required; the entrypoint exits without them |
| `RUN_ID` | `leviathan-devnet` | required; the run to join |
| `RAW_WALLET_PRIVATE_KEY` | base58 key or a JSON byte array | the client reads the wallet from the environment, so no keypair file has to be mounted |
| `IROH_RELAY` | `n0` | **required off our own infrastructure.** The default `psyche` relays refused the TLS handshake from all three networks we tried, and a node that cannot reach a relay never submits its join: it sits at `role=NotInRound` with no error line |
| `NVIDIA_VISIBLE_DEVICES` | `all` | only needed on images built before this was added to the image itself |

The wallet still needs devnet SOL, and the run only advances while a client is
connected, so bring several nodes up together inside the same
`WaitingForMembers` window rather than one at a time.

`docs/NOSANA.md` covers renting the hosts to run this on: a job definition that
works, the settings that fail silently without it, the client limit a run has,
and what three nodes actually cost.

### Dashboard telemetry

`leviathan-indexer --features live` reads a coordinator account and prints run
telemetry as JSON. It is libtorch-free, so it runs anywhere. Publish it for the
dashboard:

```
OUT=telemetry.json ./scripts/publish-telemetry.sh \
  --coordinator-account <coordinator-pubkey> --run-id <run> --rpc <rpc> \
  --reward-per-round 0.324 --bond 10.55 --slash-when-caught 10.55
```

The economics flags add the on-chain economic security verdict. This is for the
command line only. The dashboard reads the coordinator and treasurer accounts
itself, so it needs nothing published and there is no action keeping a file in
sync any more.

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
