#!/usr/bin/env bash

set -o errexit
set -e
set -m

# Parse command line arguments
DEPLOY_TREASURER=false
EXTRA_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "--treasurer" ]]; then
        DEPLOY_TREASURER=true
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# use the agenix provided wallet if you have it
if [[ -n "${devnet__keypair__wallet_PATH}" && -f "${devnet__keypair__wallet_PATH}" ]]; then
    DEFAULT_WALLET="${devnet__keypair__wallet_PATH}"
else
    DEFAULT_WALLET="$HOME/.config/solana/id.json"
fi
WALLET_FILE=${KEY_FILE:-"$DEFAULT_WALLET"}
RPC=${RPC:-"http://127.0.0.1:8899"}
WS_RPC=${WS_RPC:-"ws://127.0.0.1:8900"}

# Detect if we're deploying to devnet
IS_DEVNET=false
if [[ "$RPC" == *"devnet.solana.com"* ]]; then
    IS_DEVNET=true
fi

echo -e "\n[+] deploy info:"
echo -e "[+] WALLET_FILE = $WALLET_FILE"
echo -e "[+] RPC = $RPC"
echo -e "[+] WS_RPC = $WS_RPC"
echo -e "[+] RUN_ID = $RUN_ID"
echo -e "[+] CONFIG_FILE = $CONFIG_FILE"
echo -e "[+] IS_DEVNET = $IS_DEVNET"
echo -e "[+] DEPLOY_TREASURER = $DEPLOY_TREASURER"
echo -e "[+] -----------------------------------------------------------"

if [[ "$IS_DEVNET" == "true" ]]; then
    echo -e "\n[+] - generating new keypairs for devnet..."
    solana-keygen new -o architectures/decentralized/solana-coordinator/target/deploy/psyche_solana_coordinator-keypair.json -f --no-bip39-passphrase
    solana-keygen new -o architectures/decentralized/solana-authorizer/target/deploy/psyche_solana_authorizer-keypair.json -f --no-bip39-passphrase
    if [[ "$DEPLOY_TREASURER" == "true" ]]; then
        solana-keygen new -o architectures/decentralized/solana-treasurer/target/deploy/psyche_solana_treasurer-keypair.json -f --no-bip39-passphrase
    fi
    cd architectures/decentralized/solana-coordinator && anchor keys sync && cd -
    cd architectures/decentralized/solana-authorizer && anchor keys sync && cd -
    if [[ "$DEPLOY_TREASURER" == "true" ]]; then
        cd architectures/decentralized/solana-treasurer && anchor keys sync && cd -
    fi
fi

# Deploy Coordinator
echo -e "\n[+] Starting coordinator deploy"
pushd architectures/decentralized/solana-coordinator

echo -e "\n[+] - building..."
anchor build --no-idl

echo -e "\n[+] - deploying..."
anchor deploy --provider.cluster ${RPC} --provider.wallet ${WALLET_FILE} -- --max-len 500000
sleep 1

echo -e "\n[+] Coordinator program deployed successfully!"
popd

# Deploy Authorizer
echo -e "\n[+] Starting authorizer deploy"
pushd architectures/decentralized/solana-authorizer

echo -e "\n[+] - building..."
anchor build --no-idl

echo -e "\n[+] - deploying..."
anchor deploy --provider.cluster ${RPC} --provider.wallet ${WALLET_FILE}
sleep 1

echo -e "\n[+] Authorizer program deployed successfully!"
popd

# Deploy Treasurer (if flag is set)
TREASURER_ARGS=""
if [[ "$DEPLOY_TREASURER" == "true" ]]; then
    echo -e "\n[+] Starting treasurer deploy"
    pushd architectures/decentralized/solana-treasurer

    echo -e "\n[+] - building..."
    anchor build --no-idl

    echo -e "\n[+] - deploying..."
    anchor deploy --provider.cluster ${RPC} --provider.wallet ${WALLET_FILE}
    sleep 1

    echo -e "\n[+] Treasurer program deployed successfully!"
    popd

    # Create token
    echo -e "\n[+] Creating token"
    WALLET_PUBKEY=$(solana-keygen pubkey ${WALLET_FILE})
    TOKEN_ADDRESS=$(spl-token create-token --decimals 0 --url ${RPC} --fee-payer ${WALLET_FILE} --mint-authority ${WALLET_PUBKEY} | grep "Address:" | awk '{print $2}')
    spl-token create-account ${TOKEN_ADDRESS} --url ${RPC} --fee-payer ${WALLET_FILE} --owner ${WALLET_PUBKEY}
    spl-token mint ${TOKEN_ADDRESS} 1000000 --url ${RPC} --fee-payer ${WALLET_FILE} --mint-authority ${WALLET_FILE} --recipient-owner ${WALLET_PUBKEY}

    TREASURER_ARGS="--treasurer-collateral-mint ${TOKEN_ADDRESS}"
fi
