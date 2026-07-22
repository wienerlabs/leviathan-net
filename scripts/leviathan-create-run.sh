#!/usr/bin/env bash
set -euo pipefail

# Create a fresh coordinator-only devnet run that starts in WaitingForMembers with
# verification on, so a local swarm can all join in the same window and the audit
# lottery assigns verifiers. No treasurer/bond layer: this demo exercises the
# coordinator eject/slash and the offline verifier, not the bond path.

RUN_ID="${RUN_ID:-leviathan-demo}"
RPC="${RPC:-https://api.devnet.solana.com}"
WS_RPC="${WS_RPC:-wss://api.devnet.solana.com}"
WALLET="${WALLET:-$HOME/.config/solana/id.json}"
CFG="${CFG:-config/leviathan/genesis-rehearsal.toml}"
CLIENT_VERSION="${CLIENT_VERSION:-0.2.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RM="target/debug/run-manager"
[[ -x "$RM" ]] || { echo "build run-manager first" >&2; exit 1; }
common=(--rpc "$RPC" --ws-rpc "$WS_RPC" --wallet-private-key-path "$WALLET")

echo "[1/4] CreateRun $RUN_ID (authority=$(solana-keygen pubkey "$WALLET"))"
"$RM" create-run "${common[@]}" --run-id "$RUN_ID" --client-version "$CLIENT_VERSION"

echo "[2/4] UpdateConfig from $CFG"
"$RM" update-config "${common[@]}" --run-id "$RUN_ID" --config-path "$CFG" \
  --num-parameters 1000 --vocab-size 32

echo "[3/4] SetFutureEpochRates"
"$RM" set-future-epoch-rates "${common[@]}" --run-id "$RUN_ID" \
  --earning-rate-total-shared 1000 --slashing-rate-per-client 100

echo "[4/4] SetPaused --resume"
"$RM" set-paused "${common[@]}" --run-id "$RUN_ID" --resume

echo "DONE. Run '$RUN_ID' should now be WaitingForMembers."
