#!/usr/bin/env bash
# Checks that a run which seats verifiers keeps every honest client. Three
# real clients train a short run against a local centralized server; each
# round draws one of them to verify, so the seats move from round to round. A
# client that verified one round and trains the next used to be judged by the
# health check as a trainer the witnesses never heard, and dropped. Every
# client has to see the run through to its end, none dropped, and at least
# one has to have trained the round after one it verified.
#
# Each round's verifier also fetches, from the payload's trainer, the state
# the payload it audits commits to (wienerlabs/leviathan-net#89): every one
# has to be revealed, and be the state its payload commits to.
#
#   cargo build --bin psyche-centralized-server --bin psyche-centralized-client \
#     --bin check-revealed-states
#   ./scripts/verifier-seats.sh
#
# Requires the libtorch environment (see docs/OPS_RUNBOOK.md) and the network,
# for the model and data on the hub. Takes a few minutes on the CPU.
set -euo pipefail

RUN_CONFIG="${RUN_CONFIG:-config/verifier-seats/state.toml}"
WORK="${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/leviathan-verifier-seats.XXXXXX")}"
BIN="${BIN:-./target/debug}"
PORT="${PORT:-39872}"
# The run trains a few minutes; past this it is stuck, not slow.
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"

LABEL="[verifier-seats]"
NODES=3
# shellcheck source=lib/real-client-run.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/real-client-run.sh"
run_real_clients

# The seat each client was drawn for each step, from its own log. The console
# log colours its fields, so the escapes go first.
seats() {
    tr -d '\033' < "$WORK/client$1.log" |
        sed -n 's/.*Got training assignment for step \([0-9]*\) .*committee=\([A-Za-z ]*\) witness position.*/\1 \2/p'
}
moved=0
for ((node = 0; node < NODES; node++)); do
    # A step this client verified, followed by one it trained.
    if seats "$node" | awk 'prev == "Verifier" && $2 == "Trainer" && $1 == step + 1 { found = 1 } { prev = $2; step = $1 } END { exit !found }'; then
        moved=$((moved + 1))
    fi
done
if ((moved == 0)); then
    echo "$LABEL no client trained the round after one it verified, so nothing was checked; logs in $WORK" >&2
    exit 1
fi
echo "$LABEL every client saw the run through, and $moved trained a round after verifying the one before"

# What the verifiers say about the states they asked for.
said() {
    cat "$WORK"/client*.log | tr -d '\033' | grep -ac "$1" || true
}
not_revealed=$(said "was not revealed")
no_state=$(said "commits to no state, so none can be revealed")
revealed=$(said "Revealed the state an audited payload was trained from")
if ((not_revealed > 0 || no_state > 0)); then
    echo "$LABEL $not_revealed audited states were not revealed and $no_state audited payloads committed to none; logs in $WORK" >&2
    exit 1
fi

echo "$LABEL checking the $revealed states the verifiers say they revealed"
dumps=()
for ((node = 0; node < NODES; node++)); do
    dumps+=("$WORK/dumps$node")
done
# With three clients and a third of the roster verifying, every round that
# trains seats a verifier, so every step a payload was trained at has one.
"$BIN/check-revealed-states" --run-config "$RUN_CONFIG" --dumps "${dumps[@]}" --revealed "$revealed" \
    --every-trained-step
echo "$LABEL passed: every verifier fetched the state each payload it audited commits to"
