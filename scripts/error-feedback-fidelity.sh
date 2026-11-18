#!/usr/bin/env bash
# Checks that measure-error-feedback trains a run the way real clients do
# (wienerlabs/leviathan-net#48). Two real clients train a short run against a
# local centralized server, one of them dumping every node's payload, and the
# simulation has to send each payload byte for byte from the same checkpoint,
# data and rounds.
#
# The clients also keep the state each round trains from, and the same run
# checks that they kept and serve the state each payload of its last epoch
# commits to, and forgot those of earlier epochs (wienerlabs/leviathan-net#89).
#
#   cargo build --bin psyche-centralized-server --bin psyche-centralized-client \
#     --bin check-error-feedback-fidelity --bin check-round-states
#   ./scripts/error-feedback-fidelity.sh
#
# Requires the libtorch environment (see docs/OPS_RUNBOOK.md) and the network,
# for the model and data on the hub. Takes a few minutes: the run trains
# several epochs on the CPU.
set -euo pipefail

RUN_CONFIG="${RUN_CONFIG:-config/error-feedback-fidelity/state.toml}"
WORK="${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/leviathan-error-feedback-fidelity.XXXXXX")}"
BIN="${BIN:-./target/debug}"
PORT="${PORT:-39871}"
# The run trains a few minutes; past this it is stuck, not slow.
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"

LABEL="[fidelity]"
NODES=2
# shellcheck source=lib/real-client-run.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/real-client-run.sh"
run_real_clients

echo "[fidelity] replaying the run in the simulation"
"$BIN/check-error-feedback-fidelity" --run-config "$RUN_CONFIG" --dumps "$WORK/dumps0"
echo "[fidelity] passed: the simulation sent every payload the clients sent"

echo "[fidelity] checking the states the clients kept"
"$BIN/check-round-states" --run-config "$RUN_CONFIG" \
    --dumps "$WORK/dumps0" "$WORK/dumps1" --states "$WORK/states0" "$WORK/states1"
echo "[fidelity] passed: each client kept and serves the state each payload of the last epoch commits to"
