#!/usr/bin/env bash
set -euo pipefail

# Read a run's coordinator account and write the dashboard telemetry JSON.
# leviathan-indexer is libtorch-free, so this is cheap and runs anywhere.
#
# Usage:
#   ./scripts/publish-telemetry.sh --coordinator-account <pubkey> [options] > telemetry.json
#
# Or with the operator settings file (~/.leviathan/env) providing RPC:
#   OUT=path/to/telemetry.json ./scripts/publish-telemetry.sh --coordinator-account <pubkey>
#
# Options are passed straight through to leviathan-indexer: --run-id, --rpc,
# --leaderboard, --reward-per-round, --bond, --slash-when-caught. The economics
# flags add the security verdict.

LEVIATHAN_ENV="${LEVIATHAN_ENV:-$HOME/.leviathan/env}"
if [[ -f "$LEVIATHAN_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LEVIATHAN_ENV"
  set +a
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIN="target/debug/leviathan-indexer"
if [[ ! -x "$BIN" ]]; then
  echo "[telemetry] building leviathan-indexer" >&2
  cargo build -p leviathan-indexer --features live --bin leviathan-indexer >&2
fi

# Default the indexer RPC to whatever the operator configured, if present.
args=("$@")
if [[ -n "${RPC:-}" ]] && [[ ! " ${args[*]} " == *" --rpc "* ]]; then
  args+=(--rpc "$RPC")
fi

telemetry="$("$BIN" "${args[@]}")"

# Stamp the read time so the dashboard can show freshness. Uses python for a
# portable ISO-8601 UTC timestamp and to keep the JSON valid.
stamped="$(printf '%s' "$telemetry" | python3 -c '
import json, sys, datetime
data = json.load(sys.stdin)
data["generated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
print(json.dumps(data, indent=2))
')"

OUT="${OUT:-}"
if [[ -n "$OUT" ]]; then
  printf '%s\n' "$stamped" > "$OUT"
  echo "[telemetry] wrote $OUT" >&2
else
  printf '%s\n' "$stamped"
fi
