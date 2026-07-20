#!/usr/bin/env bash
set -euo pipefail

# Launch a local multi-node Leviathan swarm against one run, optionally with one
# cheater node broadcasting fraudulent gradients. Each node runs its own client
# process with its own wallet, and writes its gradient dumps to its own dir so
# leviathan-verifier can replay-audit a node against the honest reference.
#
# Usage:
#   ./scripts/leviathan-swarm.sh --run-id <id> --wallets <dir> [--nodes N] [--cheater I] [--fake sign_flip]
#
# --wallets <dir> holds one funded devnet keypair per node named node0.json, node1.json, ...
# --cheater I marks node I as the cheater (default: none). --fake sets the fraud mode.

RUN_ID="${RUN_ID:-leviathan-devnet}"
RPC="${RPC:-https://api.devnet.solana.com}"
WS_RPC="${WS_RPC:-wss://api.devnet.solana.com}"
TORCH_VENV="${TORCH_VENV:-/tmp/leviathan-torch-venv}"
AUTHORIZER="${AUTHORIZER:-11111111111111111111111111111111}"
JOIN_TIMEOUT="${LEVIATHAN_JOIN_TIMEOUT_SECS:-45}"
NODES=2
CHEATER=-1
FAKE="sign_flip"
WALLETS=""
LOGDIR="${LOGDIR:-/tmp/leviathan-swarm}"
DUMPROOT="${DUMPROOT:-/tmp/leviathan-swarm/dumps}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --wallets) WALLETS="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --cheater) CHEATER="$2"; shift 2 ;;
    --fake) FAKE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$WALLETS" ]]; then
  echo "error: --wallets <dir> is required (node0.json .. nodeN.json, each funded on devnet)" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "$TORCH_VENV/bin/python" ]]; then
  uv venv --python 3.11 "$TORCH_VENV"
fi
if ! "$TORCH_VENV/bin/python" -c "import torch, numpy, setuptools" >/dev/null 2>&1; then
  uv pip install --python "$TORCH_VENV/bin/python" torch==2.9.1 numpy setuptools
fi

TORCH_LIB="$TORCH_VENV/lib/python3.11/site-packages/torch/lib"
export LIBTORCH_USE_PYTORCH=1
export PYO3_PYTHON="$TORCH_VENV/bin/python"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH="$TORCH_LIB"
export LD_LIBRARY_PATH="$TORCH_LIB"
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [[ ! -x "target/debug/psyche-solana-client" ]]; then
  echo "[swarm] building client"
  cargo build -p psyche-solana-client
fi

mkdir -p "$LOGDIR" "$DUMPROOT"
pids=()
cleanup() { echo "[swarm] stopping ${#pids[@]} nodes"; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

for ((i = 0; i < NODES; i++)); do
  wallet="$WALLETS/node$i.json"
  if [[ ! -f "$wallet" ]]; then
    echo "error: missing wallet $wallet" >&2
    exit 1
  fi
  dump="$DUMPROOT/node$i"
  mkdir -p "$dump"
  fake_env=""
  role="honest"
  if [[ "$i" == "$CHEATER" ]]; then
    fake_env="$FAKE"
    role="cheater($FAKE)"
  fi
  echo "[swarm] node$i $role -> $LOGDIR/node$i.log, dumps in $dump"
  LEVIATHAN_JOIN_TIMEOUT_SECS="$JOIN_TIMEOUT" LEVIATHAN_FAKE_DELTA="$fake_env" RUST_LOG="info,psyche=info" \
    ./target/debug/psyche-solana-client train \
    --wallet-private-key-path "$wallet" \
    --rpc "$RPC" --ws-rpc "$WS_RPC" \
    --run-id "$RUN_ID" \
    --data-parallelism 1 --tensor-parallelism 1 --micro-batch-size 1 \
    --authorizer "$AUTHORIZER" --logs console \
    --write-gradients-dir "$dump" \
    > "$LOGDIR/node$i.log" 2>&1 &
  pids+=("$!")
  sleep 3
done

echo "[swarm] ${#pids[@]} nodes running. Ctrl-C to stop. Audit a node with:"
echo "  leviathan-verifier --submitted $DUMPROOT/node<cheater> --reference $DUMPROOT/node<honest>"
wait
