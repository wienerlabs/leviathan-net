# Training swarm smoke (1.9)

A real client training a real model on this machine, coordinated by our devnet programs.

## Toolchain

The client links libtorch through the NousResearch tch fork, which pins PyTorch 2.9.1. A dedicated venv supplies it:

```
uv venv --python 3.11 /tmp/leviathan-torch-venv
uv pip install --python /tmp/leviathan-torch-venv/bin/python torch==2.9.1 numpy setuptools
```

Build and run the client with that torch on the path:

```
export LIBTORCH_USE_PYTORCH=1
export PYO3_PYTHON=/tmp/leviathan-torch-venv/bin/python
export LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH=/tmp/leviathan-torch-venv/lib/python3.11/site-packages/torch/lib
export PYTORCH_ENABLE_MPS_FALLBACK=1
cargo build -p psyche-solana-client
```

`setuptools` and `numpy` are required by torch-sys's build probe (it imports `torch.utils.cpp_extension`).

## Run

Create a permissionless devnet run with the nano CI model and launch a client:

```
run-manager join-authorization-create --authorizer 11111111111111111111111111111111 --rpc <devnet>
run-manager create-run --run-id <id> --client-version demo --rpc <devnet>
run-manager update-config --run-id <id> --config-path config/solana-test/nano-config.toml --num-parameters 1000 --vocab-size 30 --rpc <devnet>
run-manager set-paused --run-id <id> --resume --rpc <devnet>

psyche-solana-client train \
  --wallet-private-key-path <wallet> \
  --rpc https://api.devnet.solana.com --ws-rpc wss://api.devnet.solana.com \
  --run-id <id> --data-parallelism 1 --tensor-parallelism 1 --micro-batch-size 1 \
  --authorizer 11111111111111111111111111111111 --logs console
```

## Verified run (run lev-swarm-1784488974, coordinator 13MpkwruSvt8F8iJwsR96xPo9dnsVjmmUqnkhgizLMKb)

The client joined on-chain, downloaded pefontana/Nano-Llama, and trained a full epoch on MPS:

- 17 training rounds, loss 3.404 to 3.400 (ln(30) is the random-init baseline for the 30-token vocab, so it started at initialization and began to descend)
- 15 pseudo-gradients compressed with DisTrO and broadcast over the iroh mesh
- on-chain: 1 Join, 17 Witness, 33 Tick transactions, all coordinated by our devnet coordinator

Known limit: after the epoch closed the client hit a `join_run` re-join timeout. Psyche requires clients to re-join each epoch (the active vs next_active gate); multi-epoch resilience is Phase 2 work. This smoke proves the trainer, the libtorch/MPS toolchain, and the on-chain coordination path end to end.
