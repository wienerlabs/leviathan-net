#!/usr/bin/env bash
set -euo pipefail

# Leviathan node installer.
#
# One line for a first-time operator:
#   curl -fsSL https://raw.githubusercontent.com/wienerlabs/leviathan-net/main/scripts/install.sh | bash
#
# It checks the prerequisites, fetches the repository, builds the client, and
# hands off to leviathan-node.sh. Re-running it is safe and only does the work
# that is still missing.
#
# Env overrides:
#   LEVIATHAN_DIR   where to place the checkout (default ~/.leviathan/leviathan-net)
#   LEVIATHAN_REPO  git remote (default https://github.com/wienerlabs/leviathan-net.git)
#   LEVIATHAN_REF   branch or tag to check out (default main)
#   RUN_ID          run to join (default leviathan-devnet)
#   WALLET          keypair path; if unset the installer stops after setup and prints next steps
#   BOND            optional bond amount to post before joining

LEVIATHAN_DIR="${LEVIATHAN_DIR:-$HOME/.leviathan/leviathan-net}"
LEVIATHAN_REPO="${LEVIATHAN_REPO:-https://github.com/wienerlabs/leviathan-net.git}"
LEVIATHAN_REF="${LEVIATHAN_REF:-main}"
RUN_ID="${RUN_ID:-leviathan-devnet}"
WALLET="${WALLET:-}"
BOND="${BOND:-}"

say() { printf '[install] %s\n' "$1"; }
die() { printf '[install] error: %s\n' "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

case "$(uname -s)" in
  Darwin) os="macos" ;;
  Linux) os="linux" ;;
  *) die "unsupported OS $(uname -s), Leviathan nodes run on macOS or Linux" ;;
esac
say "detected $os"

# 1. prerequisites. The installer never installs anything with sudo; it points
# the operator at the official one-line installers instead.
missing=()
have git || missing+=("git")
have curl || missing+=("curl")
if ! have cargo; then
  say "rust toolchain not found"
  say "install it with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  missing+=("rust (cargo)")
fi
if ! have solana; then
  say "solana cli not found"
  say "install it with: sh -c \"\$(curl -sSfL https://release.anza.xyz/stable/install)\""
  missing+=("solana")
fi
if ! have uv; then
  say "uv (python launcher for the libtorch toolchain) not found"
  say "install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  missing+=("uv")
fi
if [[ ${#missing[@]} -gt 0 ]]; then
  die "missing prerequisites: ${missing[*]}. Install them with the commands above, then re-run this installer."
fi
say "all prerequisites present"

# 2. fetch or update the repository.
if [[ -d "$LEVIATHAN_DIR/.git" ]]; then
  say "updating existing checkout at $LEVIATHAN_DIR"
  git -C "$LEVIATHAN_DIR" fetch --quiet origin "$LEVIATHAN_REF"
  git -C "$LEVIATHAN_DIR" checkout --quiet "$LEVIATHAN_REF"
  git -C "$LEVIATHAN_DIR" pull --quiet --ff-only origin "$LEVIATHAN_REF" || \
    say "could not fast-forward, staying on the current commit"
else
  say "cloning $LEVIATHAN_REPO into $LEVIATHAN_DIR"
  mkdir -p "$(dirname "$LEVIATHAN_DIR")"
  git clone --quiet --branch "$LEVIATHAN_REF" "$LEVIATHAN_REPO" "$LEVIATHAN_DIR" || \
    die "clone failed. If the repository is private, clone it manually with your credentials, then re-run with LEVIATHAN_DIR pointing at it."
fi

cd "$LEVIATHAN_DIR"

# 3. hand off. Without a wallet we stop after setup and print the exact next step,
# so the operator can create and fund a keypair before spending time on a build.
if [[ -z "$WALLET" ]]; then
  say "setup complete. Next steps:"
  cat <<EOF

  1. Create a keypair:      solana-keygen new -o ~/leviathan-node.json
  2. Fund it on devnet:     solana airdrop 2 -k ~/leviathan-node.json -u devnet
  3. Join the run:          cd $LEVIATHAN_DIR && ./scripts/leviathan-node.sh --wallet ~/leviathan-node.json

  To post a bond and join a bonded run in one step, add --bond <amount>.
EOF
  exit 0
fi

say "handing off to leviathan-node.sh"
node_args=(--wallet "$WALLET" --run-id "$RUN_ID")
[[ -n "$BOND" ]] && node_args+=(--bond "$BOND")
exec ./scripts/leviathan-node.sh "${node_args[@]}"
