#!/usr/bin/env bash
set -uo pipefail

# macOS launcher for a local Leviathan swarm that reuses an already-built client
# and an existing py3.13 libtorch venv, instead of the py3.11/uv path the generic
# swarm script assumes. Nodes join one run, each writes its own gradient dumps,
# and one node is the sign-flip cheater so leviathan-verifier can catch it.

RUN_ID="${RUN_ID:-leviathan-devnet}"
RPC="${RPC:-https://api.devnet.solana.com}"
WS_RPC="${WS_RPC:-wss://api.devnet.solana.com}"
WALLETS="${WALLETS:-$HOME/leviathan-wallets}"
NODES="${NODES:-3}"
CHEATER="${CHEATER:-2}"
FAKE="${FAKE:-sign_flip}"
AUTHORIZER="${AUTHORIZER:-11111111111111111111111111111111}"
LOGDIR="${LOGDIR:-$HOME/leviathan-swarm/logs}"
DUMPROOT="${DUMPROOT:-$HOME/leviathan-swarm/dumps}"

TORCH_LIB="$HOME/.leviathan-torch/lib/python3.13/site-packages/torch/lib"
PYLIB="/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib"
export DYLD_LIBRARY_PATH="$TORCH_LIB:$PYLIB"
export PYTORCH_ENABLE_MPS_FALLBACK=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BIN="target/debug/psyche-solana-client"
[[ -x "$BIN" ]] || { echo "error: build $BIN first" >&2; exit 1; }

mkdir -p "$LOGDIR" "$DUMPROOT"
pids=()
cleanup() { echo "[swarm] stopping ${#pids[@]} nodes"; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

for ((i = 0; i < NODES; i++)); do
  wallet="$WALLETS/node$i.json"
  [[ -f "$wallet" ]] || { echo "error: missing wallet $wallet" >&2; exit 1; }
  dump="$DUMPROOT/node$i"
  mkdir -p "$dump"
  fake_env=""
  role="honest"
  if [[ "$i" == "$CHEATER" ]]; then fake_env="$FAKE"; role="cheater($FAKE)"; fi
  echo "[swarm] node$i $role -> $LOGDIR/node$i.log, dumps in $dump"
  LEVIATHAN_JOIN_TIMEOUT_SECS=90 LEVIATHAN_FAKE_DELTA="$fake_env" RUST_LOG="info,psyche=info" \
    "$BIN" train \
    --wallet-private-key-path "$wallet" \
    --rpc "$RPC" --ws-rpc "$WS_RPC" \
    --run-id "$RUN_ID" \
    --data-parallelism 1 --tensor-parallelism 1 --micro-batch-size 1 \
    --authorizer "$AUTHORIZER" --logs console \
    --iroh-discovery local \
    --write-gradients-dir "$dump" \
    > "$LOGDIR/node$i.log" 2>&1 &
  pids+=("$!")
  sleep 4
done

echo "[swarm] ${#pids[@]} nodes running (pids: ${pids[*]}). Audit with:"
echo "  leviathan-verifier --submitted $DUMPROOT/node$CHEATER --reference $DUMPROOT/node0"
wait
