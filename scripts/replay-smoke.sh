#!/usr/bin/env bash
# Proves the replay verifier end to end without a swarm: build a dataset a small
# model can train on, produce one honest contribution, audit it with no honest
# reference dumps anywhere, then forge the same contribution and audit again.
#
#   ./scripts/replay-smoke.sh
#
# Requires the libtorch environment (see docs/OPS_RUNBOOK.md); the nano model is
# pulled from the hub cache on first run.
set -euo pipefail

MODEL="${MODEL:-pefontana/Nano-Llama}"
VOCAB="${VOCAB:-30}"
WORK="${WORK:-$(mktemp -d -t leviathan-replay-smoke)}"
BIN="${BIN:-./target/debug}"

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
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

echo "[smoke] work dir $WORK"
mkdir -p "$WORK/data" "$WORK/submitted" "$WORK/cheat"

echo "[smoke] building a dataset the model can actually index"
"$BIN/nano-dataset" --out "$WORK/data/nano.ds" --vocab-size "$VOCAB" --sequences 64

echo "[smoke] producing one honest contribution"
"$BIN/nano-contribute" \
    --model "$MODEL" \
    --data-dir "$WORK/data" \
    --out-dir "$WORK/submitted" \
    --committer node0

DUMP_NAME="result-node0-step1-batchB[0, 0].vec-postcard"

echo "[smoke] auditing the honest contribution with no reference dumps"
"$BIN/leviathan-verifier" \
    --submitted "$WORK/submitted" \
    --replay-model "$MODEL" \
    --replay-data-dir "$WORK/data"

echo "[smoke] forging the same contribution"
"$BIN/forge-cheater" \
    --input "$WORK/submitted/$DUMP_NAME" \
    --output "$WORK/cheat/$DUMP_NAME"

echo "[smoke] auditing the forged contribution, expecting a conviction"
set +e
"$BIN/leviathan-verifier" \
    --submitted "$WORK/cheat" \
    --replay-model "$MODEL" \
    --replay-data-dir "$WORK/data"
STATUS=$?
set -e

if [[ $STATUS -ne 2 ]]; then
    echo "[smoke] FAILED: the forged contribution should have exited 2, got $STATUS"
    exit 1
fi

echo "[smoke] passed: honest cleared, forged convicted, no reference dumps involved"
