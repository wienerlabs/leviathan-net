#!/usr/bin/env bash
# Entrypoint for the plain-Docker client image (docker/Dockerfile.client-cuda).
#
# Everything is configured through the environment, because that is all a rented
# GPU host gives you.

set -uo pipefail

: "${RPC:?RPC is required (a devnet or mainnet http endpoint)}"
: "${WS_RPC:?WS_RPC is required (the matching websocket endpoint)}"
: "${RUN_ID:?RUN_ID is required (e.g. leviathan-devnet)}"

# The wallet arrives either as a base58 key in the environment or as a mounted
# keypair file. Prefer base58: a JSON byte array carries commas and brackets, and
# some job schedulers do not deliver those intact.
WALLET_ARGS=()
if [[ -n "${RAW_WALLET_PRIVATE_KEY:-}" ]]; then
  echo "[entrypoint] wallet from RAW_WALLET_PRIVATE_KEY (${#RAW_WALLET_PRIVATE_KEY} chars)"
elif [[ -n "${WALLET_PATH:-}" ]]; then
  [[ -r "$WALLET_PATH" ]] || { echo "[entrypoint] WALLET_PATH=$WALLET_PATH is not readable" >&2; exit 1; }
  WALLET_ARGS=(--wallet-private-key-path "$WALLET_PATH")
  echo "[entrypoint] wallet from $WALLET_PATH"
else
  echo "[entrypoint] set RAW_WALLET_PRIVATE_KEY (base58) or WALLET_PATH (keypair file)" >&2
  exit 1
fi

nvidia-smi || echo "[entrypoint] WARNING: nvidia-smi failed, the client will not see a GPU"

ARGS=(
  train
  --rpc "$RPC" --ws-rpc "$WS_RPC"
  --run-id "$RUN_ID"
  --data-parallelism "${DATA_PARALLELISM:-1}"
  --tensor-parallelism "${TENSOR_PARALLELISM:-1}"
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
  --logs "${LOGS:-console}"
)
[[ -n "${AUTHORIZER:-}" ]] && ARGS+=(--authorizer "$AUTHORIZER")

# The client exits when the coordinator does not select it for a round, which
# happens to anything that joins outside a WaitingForMembers window. Restart it
# here rather than letting the container die: the scheduler would otherwise pull
# and set up the whole image again to do what a 15 second sleep does.
RETRY_DELAY="${RETRY_DELAY_SECS:-15}"
while true; do
  psyche-solana-client "${WALLET_ARGS[@]}" "${ARGS[@]}"
  echo "[entrypoint] client exited with $?, retrying in ${RETRY_DELAY}s"
  sleep "$RETRY_DELAY"
done
