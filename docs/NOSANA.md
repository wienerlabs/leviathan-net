# Running Leviathan nodes on rented GPUs (Nosana)

How to bring up a Leviathan swarm on rented capacity, and the things that bite
when you do. Everything here was verified on 2026-08-03: three nodes joined one
run on Nosana and trained through 50 epochs, on three separate hosts.

Nosana runs containers on GPUs rented from third-party hosts. You submit a job
definition, it schedules the container, and you pay per hour.

## What participated

| Node | Card | Driver | CUDA |
|---|---|---|---|
| 1 | GeForce RTX 3060 Ti (8 GB) | 595.71.05 | 13.2 |
| 2 | GeForce RTX 3060 Ti (8 GB) | 595.71.05 | 13.2 |
| 3 | GeForce RTX 3060 (12 GB) | 560.35.03 | 12.6 |

All three came from the same 3060 market. The market names a class, not a card:
we got two different models and **two different CUDA versions**. Anything that
calibrates a tolerance band per hardware class (leviathan-net#7) should treat
heterogeneity as the normal case rather than the exception.

## Cost

The 3060 market is $0.048/hour, so three nodes cost $0.144/hour. At 50 nodes
that is roughly $2.40/hour on 3060s, or $9.60/hour on 3090s.

Availability binds harder than price. The A100, H100, A40, A6000 and 6000 Ada
markets had no free hosts at all — every deployment queues. The 3090 class had 23
of 59 hosts free, and the 3060/3080 classes were comparably open. If a run needs
24 GB, the 3090 is the class that can actually be reserved.

## The job definition

There is no published image that works today (see "The published image" below),
so the job builds the client from source on the host. Paste this into the Nosana
dashboard, filling in your own endpoints and key:

```json
{
  "version": "0.1",
  "type": "container",
  "meta": { "trigger": "dashboard", "system_requirements": { "vram_total_mb": 8192 } },
  "ops": [{
    "id": "leviathan-node",
    "type": "container/run",
    "args": {
      "image": "docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04",
      "gpu": true,
      "cmd": ["bash", "-c", "set -euo pipefail\nexport DEBIAN_FRONTEND=noninteractive\napt-get update -qq\napt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential pkg-config libssl-dev\ncurl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable\n. \"$HOME/.cargo/env\"\ncurl -LsSf https://astral.sh/uv/install.sh | sh\nexport PATH=\"$HOME/.local/bin:$PATH\"\ngit clone --depth 1 https://github.com/wienerlabs/leviathan-net /opt/leviathan-net\ncd /opt/leviathan-net\necho \"[boot] commit: $(git rev-parse --short HEAD)\"\nif ! grep -q \"$EXPECT_COORDINATOR_PROGRAM\" architectures/decentralized/solana-coordinator/programs/solana-coordinator/src/lib.rs; then\n  echo \"[boot] FATAL: this checkout does not carry coordinator id $EXPECT_COORDINATOR_PROGRAM\"\n  echo \"[boot] Stopping rather than deriving the coordinator under a different program and joining the wrong run.\"\n  exit 1\nfi\necho \"[boot] coordinator program confirmed: $EXPECT_COORDINATOR_PROGRAM, run: $RUN_ID\"\nnvidia-smi || echo 'WARNING: nvidia-smi failed'\nwhile true; do\n  ./scripts/leviathan-node.sh --wallet /opt/wallet.json || true\n  echo '[loop] client exited, retrying in 15s'\n  sleep 15\ndone\n"],
      "env": {
        "RPC": "https://devnet.helius-rpc.com/?api-key=YOUR_KEY",
        "WS_RPC": "wss://devnet.helius-rpc.com/?api-key=YOUR_KEY",
        "RUN_ID": "leviathan-devnet",
        "EXPECT_COORDINATOR_PROGRAM": "GdHJHiQp7uMv8TanfpaCaKQ8nHm5suvEt9JvjpZFWZ19",
        "RAW_WALLET_PRIVATE_KEY": "YOUR_BASE58_DEVNET_KEY",
        "IROH_RELAY": "n0",
        "IROH_DISCOVERY": "n0",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "CARGO_PROFILE_DEV_DEBUG": "0",
        "CARGO_INCREMENTAL": "0"
      }
    }
  }]
}
```

Set the container timeout to at least three hours: the first build takes 35 to 45
minutes on a rented host, and the container has to outlive it by enough to train.

For N nodes, submit this N times. **Each node needs its own keypair** — two nodes
sharing a wallet share an on-chain identity and collide. Each also needs a little
devnet SOL for transaction fees.

### Why the job checks the program id before it builds

`EXPECT_COORDINATOR_PROGRAM` is compared against the checkout before anything is
compiled, and the job stops if it does not match.

The client derives the coordinator account as
`PDA(["coordinator", RUN_ID], coordinator_program_id)`, and the program id comes
from `declare_id!` in the source it just cloned. If that source is at a revision
with a different id, the client builds cleanly, derives a different coordinator,
and either fails with `AccountNotFound` after a 45 minute build or — worse —
finds a run that happens to exist there and joins it. Nothing in the logs looks
wrong in the second case.

Devnet was redeployed under new program ids (see `DEVNET.md`, "Why the program
ids changed"), so both outcomes are live possibilities for anyone running an
older checkout. The check costs a second and turns a silent wrong-run into a
loud stop.

### The run is not permissionless

`join_run` needs an authorization whose grantor is the run's join authority,
whose grantee is the node, and whose scope is `CoordinatorJoinRun`. The join
authority creates one per node before that node starts:

```
run-manager join-authorization-create --rpc <rpc> --ws-rpc <ws> \
  --wallet-private-key-path <join-authority-key> --authorizer <node pubkey>
```

The direction matters and is easy to get backwards: the **join authority signs**
and the **node is the argument**. Signing as the node and passing the authority
creates the mirror-image authorization, which is accepted on chain, costs rent,
and does nothing for joining.

## Four settings that are not optional

**`IROH_RELAY=n0`.** The default `psyche` relays refuse the TLS handshake from
outside our own infrastructure — verified from two consumer ISPs and one
unrelated third-party host, all returning `access_denied` (alert 49). A node that
cannot reach a relay never submits its join and sits at `role=NotInRound`
**without printing an error**, which is the worst possible failure shape. The
public n0 relays work.

**`NVIDIA_VISIBLE_DEVICES=all`.** Without it the NVIDIA container runtime injects
no driver and the client reports no GPU on a host that has one. The base image
here sets it, so this matters mainly when overriding the environment wholesale.

**`RAW_WALLET_PRIVATE_KEY` in base58, not as a JSON byte array.** The client
accepts both, but `[116, 231, …]` does not survive Nosana's environment delivery
intact — the client fails with `invalid type: integer 116, expected a sequence`.
The base58 form is a single token with no commas, brackets or spaces.

**A restart loop around the client.** The client exits when the coordinator does
not select it for a round, which is what happens to anything joining outside a
`WaitingForMembers` window. If the container dies there, the scheduler tears it
down and builds the whole thing again — 45 minutes to redo what a 15 second sleep
does. The loop above keeps the build cached and retries in seconds.

## A run has a hard client limit

`global_batch_size_end` is the maximum number of clients a run will accept. The
coordinator rejects the join that would make client count equal it:

```
AnchorError: MoreClientsThanBatches (6023)
There are more clients than total number of batches to assign.
```

`leviathan-devnet` sets this to 4, and all four slots were taken — one by a live
node, three by registrations whose clients were long gone. Nothing else could
join, and the only signal was that error at join time.

Two things follow. A run intended for 50 volunteer nodes needs
`global_batch_size_end` sized for them, or it closes its doors at the fourth. And
**registrations outlive the clients that made them**, so a run can fill up with
clients that are not training — cheap to do on purpose in a permissionless
network, and a quiet way to lock out honest volunteers.

Raising it on an existing run needs that run's `main_authority`.

## Standing up your own run

To size a run for a swarm, create one. `config/leviathan/genesis-rehearsal.toml`
allows 8 clients on the same Nano-Llama model with 90 second epochs:

```
KEY_FILE=<keypair> \
RPC=<https endpoint> WS_RPC=<wss endpoint> \
RUN_ID=<your run id> \
CONFIG_FILE=./config/leviathan/genesis-rehearsal.toml \
./scripts/create-permissionless-run.sh
```

The wallet you pass becomes `main_authority`, so you can change the config later.
Short epochs help: a `WaitingForMembers` window comes around every 90 seconds, so
nodes that finish building at different times still converge quickly.

## Watching without the dashboard

`leviathan-indexer` reads run state straight off the coordinator account and is
libtorch-free, so it runs anywhere and does not depend on scraping container
logs:

```
cargo build -p leviathan-indexer --features live --bin leviathan-indexer
./target/debug/leviathan-indexer \
  --coordinator-account <pubkey> --run-id <run> --rpc <endpoint>
```

It prints run state, epoch, registered and active client counts and the
leaderboard. The coordinator account for a run id is on its
`CoordinatorInstance` PDA, derived from `["coordinator", run_id]`.

The other check that needs nothing at all is the chain: a node that joined is
sending transactions, so signature activity on its wallet is proof of life, and
its absence is proof the node never got on.

## The published image

`nousresearch/psyche-client:latest` cannot join a current run. It dates from
2026-03-20, before the program ids changed twice, and derives a coordinator
instance that does not exist:

```
Error: all RPCs exhausted: AccountNotFound: pubkey=Hne3n5aAZJC9i3aA76t1f8rqAZV9eTeoMnGakd7RTH1V
```

That address is reproducible: it is `PDA(["coordinator", "leviathan-devnet"])`
under `4SHugWqSXwKE5fqDchkJcPEqnoZE22VYKtSTVm7axbT7`, a coordinator id retired two
deploys ago. It is stale because the workflow that publishes it waited on a
Garnix check suite that this org has no installation for, so it timed out before
pushing (fixed in #13). The nix image still cannot be built in a GitHub runner —
its closure includes CUDA built from source, which no public cache carries — so
`docker/Dockerfile.client-cuda` builds an equivalent client with plain Docker
instead, taking CUDA from NVIDIA's base image.
