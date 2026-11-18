#!/usr/bin/env bash
# Checks that measure-error-feedback trains a run the way real clients do
# (wienerlabs/leviathan-net#48). Two real clients train a short run against a
# local centralized server, one of them dumping every node's payload, and the
# simulation has to send each payload byte for byte from the same checkpoint,
# data and rounds.
#
#   cargo build --bin psyche-centralized-server --bin psyche-centralized-client \
#     --bin check-error-feedback-fidelity
#   ./scripts/error-feedback-fidelity.sh
#
# Requires the libtorch environment (see docs/OPS_RUNBOOK.md) and the network,
# for the model and data on the hub. Takes a few minutes: the run trains
# several epochs on the CPU.
set -euo pipefail

RUN_CONFIG="${RUN_CONFIG:-config/leviathan/error-feedback-fidelity.toml}"
WORK="${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/leviathan-error-feedback-fidelity.XXXXXX")}"
BIN="${BIN:-./target/debug}"
PORT="${PORT:-39871}"
# The run trains a few minutes; past this it is stuck, not slow.
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"

# macOS strips DYLD_* when it execs a protected shell, so a caller's exported
# library path never reaches these binaries. Rebuild it here instead.
TORCH_VENV="${TORCH_VENV:-$HOME/.leviathan-torch}"
if [[ -d "$TORCH_VENV" ]]; then
    TORCH_LIB=$(find "$TORCH_VENV" -maxdepth 6 -type d -path "*site-packages/torch/lib" | head -1)
    PYTHON_LIB="${PYTHON_LIB:-/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib}"
    if [[ -n "$TORCH_LIB" ]]; then
        export DYLD_LIBRARY_PATH="$TORCH_LIB:$PYTHON_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    fi
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

echo "[fidelity] work dir $WORK"
"$BIN/psyche-centralized-server" run --state "$RUN_CONFIG" --server-port "$PORT" --tui false \
    > "$WORK/server.log" 2>&1 &
SERVER=$!
PIDS+=($SERVER)
# The clients connect once the server listens, not after a guess at how long
# that takes; a port someone else holds stops the server here.
ready=$((SECONDS + 60))
until grep -aq "Server listening on" "$WORK/server.log"; do
    if ! kill -0 "$SERVER" 2>/dev/null || ((SECONDS > ready)); then
        echo "[fidelity] the server did not start on port $PORT; logs in $WORK" >&2
        exit 1
    fi
    sleep 1
done

# A client stops with an error once the run is finished (wienerlabs/leviathan-net#82),
# so how it exited says nothing; what counts is that it saw the run through to
# the end. The console log colours its fields, so the escapes go first.
finished() {
    tr -d '\033' < "$WORK/client$1.log" | grep -aq 'new_state[^ ]*Finished'
}

CLIENTS=()
for node in 0 1; do
    # A fresh identity per client: the run admits each by its own key.
    RAW_IDENTITY_SECRET_KEY=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n') \
        "$BIN/psyche-centralized-client" train \
        --run-id "$(sed -n 's/^run_id = "\(.*\)"$/\1/p' "$RUN_CONFIG")" \
        --server-addr "localhost:$PORT" --device cpu --logs console \
        --iroh-relay disabled --iroh-discovery local \
        --write-gradients-dir "$WORK/dumps$node" \
        > "$WORK/client$node.log" 2>&1 &
    PIDS+=($!)
    CLIENTS+=($!)
done

echo "[fidelity] two clients training $RUN_CONFIG"
deadline=$((SECONDS + TIMEOUT_SECS))
while true; do
    running=0
    for node in 0 1; do
        if kill -0 "${CLIENTS[$node]}" 2>/dev/null; then
            running=$((running + 1))
        elif ! finished "$node"; then
            echo "[fidelity] client $node stopped before the run finished; logs in $WORK" >&2
            exit 1
        fi
    done
    if ((running == 0)); then
        break
    fi
    if ((SECONDS > deadline)); then
        echo "[fidelity] the run did not finish in ${TIMEOUT_SECS}s; logs in $WORK" >&2
        exit 1
    fi
    if ! kill -0 "$SERVER" 2>/dev/null; then
        echo "[fidelity] the server stopped before the run finished; logs in $WORK" >&2
        exit 1
    fi
    # A client dropped from the run leaves the other waiting for members for
    # good, and its rounds can no longer be accounted for.
    if grep -aq "Dropped .* clients" "$WORK/server.log"; then
        echo "[fidelity] the server dropped a client; logs in $WORK" >&2
        exit 1
    fi
    sleep 5
done

echo "[fidelity] replaying the run in the simulation"
"$BIN/check-error-feedback-fidelity" --run-config "$RUN_CONFIG" --dumps "$WORK/dumps0"
echo "[fidelity] passed: the simulation sent every payload the clients sent"
