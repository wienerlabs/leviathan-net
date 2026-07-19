#!/usr/bin/env bash
set -euo pipefail

# Run a Leviathan training node against the devnet flagship run.
# One command: sets up the libtorch toolchain, builds the client, and joins.
#
# Usage:
#   ./scripts/leviathan-node.sh --wallet <path/to/keypair.json>
#
# Env overrides:
#   RUN_ID      (default leviathan-devnet)
#   RPC         (default https://api.devnet.solana.com)
#   WS_RPC      (default wss://api.devnet.solana.com)
#   TORCH_VENV  (default /tmp/leviathan-torch-venv)
#   AUTHORIZER  (default 11111111111111111111111111111111, permissionless)

RUN_ID="${RUN_ID:-leviathan-devnet}"
RPC="${RPC:-https://api.devnet.solana.com}"
WS_RPC="${WS_RPC:-wss://api.devnet.solana.com}"
TORCH_VENV="${TORCH_VENV:-/tmp/leviathan-torch-venv}"
AUTHORIZER="${AUTHORIZER:-11111111111111111111111111111111}"
JOIN_TIMEOUT="${LEVIATHAN_JOIN_TIMEOUT_SECS:-45}"
WALLET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wallet) WALLET="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$WALLET" ]]; then
  echo "error: --wallet <path> is required (a funded devnet keypair)" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. libtorch toolchain: the tch fork pins PyTorch 2.9.1.
if [[ ! -x "$TORCH_VENV/bin/python" ]]; then
  echo "[node] creating torch venv at $TORCH_VENV"
  uv venv --python 3.11 "$TORCH_VENV"
fi
if ! "$TORCH_VENV/bin/python" -c "import torch, numpy, setuptools" >/dev/null 2>&1; then
  echo "[node] installing torch 2.9.1 + numpy + setuptools"
  uv pip install --python "$TORCH_VENV/bin/python" torch==2.9.1 numpy setuptools
fi

TORCH_LIB="$TORCH_VENV/lib/python3.11/site-packages/torch/lib"
export LIBTORCH_USE_PYTORCH=1
export PYO3_PYTHON="$TORCH_VENV/bin/python"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH="$TORCH_LIB"
export LD_LIBRARY_PATH="$TORCH_LIB"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 2. build the client if needed.
if [[ ! -x "target/debug/psyche-solana-client" ]]; then
  echo "[node] building psyche-solana-client (first build links libtorch, takes a while)"
  cargo build -p psyche-solana-client
fi

# 3. join the run and train.
# Note: launch the binary directly, never through /usr/bin/env, because macOS SIP
# strips DYLD_* variables when exec-ing through a protected interpreter, which would
# hide libtorch from the client.
echo "[node] joining run '$RUN_ID' on $RPC as $(solana-keygen pubkey "$WALLET" 2>/dev/null || echo wallet)"
export LEVIATHAN_JOIN_TIMEOUT_SECS="$JOIN_TIMEOUT"
export RUST_LOG="${RUST_LOG:-info,psyche=info}"
exec ./target/debug/psyche-solana-client train \
  --wallet-private-key-path "$WALLET" \
  --rpc "$RPC" --ws-rpc "$WS_RPC" \
  --run-id "$RUN_ID" \
  --data-parallelism 1 --tensor-parallelism 1 --micro-batch-size 1 \
  --authorizer "$AUTHORIZER" --logs console
