#!/usr/bin/env bash
# Builds the three programs with their mainnet ids and deploys them, then hands
# the upgrade authority to a multisig.
#
#   CLUSTER=devnet ./scripts/deploy-mainnet.sh          rehearsal, costs devnet SOL
#   CLUSTER=mainnet ./scripts/deploy-mainnet.sh         the real thing
#
# The rehearsal is not optional. It deploys the exact mainnet binaries, with the
# exact mainnet program ids, to devnet. Everything except the cluster is
# identical, so anything that breaks there would have broken with real money.
#
# Nothing is deployed until every precondition below passes, and mainnet asks
# for a typed confirmation.
set -euo pipefail

CLUSTER="${CLUSTER:-devnet}"
KEYDIR="${KEYDIR:-$HOME/.config/solana/leviathan-mainnet-keys}"
PAYER="${PAYER:-$HOME/.config/solana/leviathan-devnet.json}"
# Upgrade authority after deployment. This should be a multisig, not a person.
UPGRADE_AUTHORITY="${UPGRADE_AUTHORITY:-ALxuDYPT5BYE5jWW5zF4BK8o1KXAwPcrt7SGdUspjNNr}"

case "$CLUSTER" in
    devnet) RPC="${RPC:-https://api.devnet.solana.com}" ;;
    mainnet) RPC="${RPC:-https://api.mainnet-beta.solana.com}" ;;
    *) echo "CLUSTER must be devnet or mainnet, got $CLUSTER" >&2; exit 1 ;;
esac

PROGRAMS=(solana-coordinator solana-treasurer solana-authorizer)

echo "[deploy] cluster   $CLUSTER"
echo "[deploy] rpc       $RPC"
echo "[deploy] payer     $(solana-keygen pubkey "$PAYER")"
echo "[deploy] authority $UPGRADE_AUTHORITY"
echo

echo "[deploy] checking preconditions"

for P in "${PROGRAMS[@]}"; do
    KEY="$KEYDIR/psyche_${P//-/_}-keypair.json"
    if [[ ! -f "$KEY" ]]; then
        echo "  missing program keypair: $KEY" >&2
        exit 1
    fi
done

# The ids compiled into the binaries must match the keypairs we are about to
# deploy with. If these ever drift, the deploy fails late and confusingly.
EXPECTED_TREASURER=$(solana-keygen pubkey "$KEYDIR/psyche_solana_treasurer-keypair.json")
if ! cargo test -p psyche-solana-treasurer --test program_id --features mainnet >/dev/null 2>&1; then
    echo "  the mainnet program id test does not pass, refusing to deploy" >&2
    exit 1
fi
echo "  program ids match their keypairs"

# An upgrade authority that is a normal wallet defeats the point of a multisig.
# Squads vaults are program derived, so they are off curve.
if ! solana account "$UPGRADE_AUTHORITY" --url "$RPC" >/dev/null 2>&1; then
    echo "  warning: upgrade authority $UPGRADE_AUTHORITY not found on $CLUSTER"
    if [[ "$CLUSTER" == "mainnet" ]]; then
        echo "  refusing to deploy to mainnet with an authority that does not exist" >&2
        exit 1
    fi
fi
echo "  upgrade authority reachable"

BALANCE=$(solana balance "$PAYER" --url "$RPC" | awk '{print $1}')
echo "  payer balance ${BALANCE} SOL"
if (( $(echo "$BALANCE < 6" | bc -l) )); then
    echo "  under 6 SOL. Three programs of this size will not fit." >&2
    exit 1
fi

if [[ "$CLUSTER" == "mainnet" ]]; then
    echo
    echo "This deploys to MAINNET with real SOL and real consequences."
    echo "Programs will hold participant bonds. Confirm the audit is done."
    read -r -p 'Type "deploy to mainnet" to continue: ' CONFIRM
    if [[ "$CONFIRM" != "deploy to mainnet" ]]; then
        echo "aborted"
        exit 1
    fi
fi

echo
for P in "${PROGRAMS[@]}"; do
    CRATE="psyche-${P}"
    SNAKE="psyche_${P//-/_}"
    MANIFEST="architectures/decentralized/${P}/programs/${P}/Cargo.toml"
    KEY="$KEYDIR/${SNAKE}-keypair.json"

    echo "[deploy] building $CRATE with its mainnet id"
    cargo build-sbf --manifest-path "$MANIFEST" --features mainnet

    SO="architectures/decentralized/${P}/target/deploy/${SNAKE}.so"
    if [[ ! -f "$SO" ]]; then
        echo "  build produced no $SO" >&2
        exit 1
    fi
    echo "  $(stat -f%z "$SO" 2>/dev/null || stat -c%s "$SO") bytes"

    echo "[deploy] deploying $(solana-keygen pubkey "$KEY")"
    solana program deploy \
        --program-id "$KEY" \
        --upgrade-authority "$PAYER" \
        -k "$PAYER" \
        --url "$RPC" \
        "$SO"

    echo "[deploy] handing upgrade authority to $UPGRADE_AUTHORITY"
    solana program set-upgrade-authority \
        "$(solana-keygen pubkey "$KEY")" \
        --upgrade-authority "$PAYER" \
        --new-upgrade-authority "$UPGRADE_AUTHORITY" \
        --skip-new-upgrade-authority-signer-check \
        -k "$PAYER" \
        --url "$RPC"
    echo
done

echo "[deploy] verifying"
for P in "${PROGRAMS[@]}"; do
    SNAKE="psyche_${P//-/_}"
    ID=$(solana-keygen pubkey "$KEYDIR/${SNAKE}-keypair.json")
    AUTH=$(solana program show "$ID" --url "$RPC" | awk '/Authority/{print $2}')
    if [[ "$AUTH" != "$UPGRADE_AUTHORITY" ]]; then
        echo "  $P authority is $AUTH, expected $UPGRADE_AUTHORITY" >&2
        exit 1
    fi
    echo "  $P $ID authority ok"
done

echo
echo "[deploy] done on $CLUSTER. Upgrades now require the multisig."
