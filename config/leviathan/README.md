# Leviathan run recipes

These are run configs tuned to exercise the Leviathan security layer, not just
train. The difference from an inherited Psyche config is one line:
`verification_percent` is non-zero, so the audit lottery assigns verifiers and
the replay-audit path is live.

## genesis-rehearsal.toml

A real, joinable rehearsal run on the proven nano model, with verification on at
the published operating point.

- `verification_percent = 10` sets the audit probability p = 0.1, the same p the
  economics are calibrated to. Expected rounds to catch a persistent cheater is
  1/p = 10.
- `min_clients = 2` so the swarm actually aggregates across nodes.
- The DisTrO block is the SparseLoCo transport: chunked top-k at chunk 64, topk 8,
  1-bit sign, decay 0.999. `max_round_train_time` sets the DiLoCo outer cadence.
- The model and data are the same real, HTTP-fetchable ones a client trained on
  live devnet, so remote volunteers can join.

Stand it up and join it with the standard flow:

```
run-manager create-run --run-id leviathan-genesis --treasurer-collateral-mint <mint> ...
run-manager update-config --run-id leviathan-genesis --config-path config/leviathan/genesis-rehearsal.toml --num-parameters 1000 --vocab-size 30 ...
run-manager set-future-epoch-rates --run-id leviathan-genesis --earning-rate-total-shared 1000 --slashing-rate-per-client 100 ...
run-manager treasurer-top-up-rewards --run-id leviathan-genesis --collateral-amount 500000 ...
run-manager set-paused --run-id leviathan-genesis --resume ...
./scripts/leviathan-swarm.sh --run-id leviathan-genesis --wallets <dir> --nodes 3 --cheater 2
```

The cheater node broadcasts fraudulent gradients (LEVIATHAN_FAKE_DELTA); audit
its dump against an honest node's with `leviathan-verifier`, and watch the
telemetry with `leviathan-indexer --features live`.

Pass the run economics to the indexer and it also reports whether the run is
economically secure at its audit probability, using the break-even penalty
`reward * (1 - p) / p`:

```
leviathan-indexer --coordinator-account <pubkey> \
  --reward-per-round 1000 --bond 500000 --slash-when-caught 9000
```

The verdict is secure when the effective penalty, capped at the posted bond,
meets or exceeds the break-even penalty, so a persistent cheater loses money in
expectation.

## fineweb-rehearsal.toml

The first rehearsal on real pretraining text. Security and transport parameters
are the genesis rehearsal's; the model and the data change.

- Model: `emozilla/llama2-20m-init` pinned to a commit, 19,140,864 parameters,
  the Llama 2 tokenizer (vocabulary 32,000), `max_seq_len = 512`.
- Data: the first shard of `emozilla/fineweb-10bt-tokenized-datatrove-llama2`,
  FineWeb's 10B-token sample already tokenized with the Llama 2 tokenizer in the
  datatrove `.ds` format the HTTP provider reads, pinned to commit `cd477ae`. The
  shard holds 1,953,125 sequences of 512 tokens (about 1B tokens); 25,000 steps at
  a global batch of 8 read 200,000 of them.
- One shard on purpose. Trainers read through `HttpDataProvider`, which orders
  files by URL; the verifier replays through `LocalDataProvider`, which takes the
  directory's `read_dir` order and is pinned to `DontShuffle`. With a single file
  both sides index the same sequences. A multi-shard run waits for the two
  providers to agree on file order. `NumberedFiles` is also out for Hugging Face:
  its `url_template` is a 64-byte field on chain, and a longer value is truncated
  without an error (`shared/coordinator/tests/leviathan_configs.rs` now fails on
  that for every config here).

Stand it up like the genesis rehearsal, with this model's metadata:

```
run-manager update-config --run-id leviathan-fineweb --config-path config/leviathan/fineweb-rehearsal.toml --num-parameters 19140864 --vocab-size 32000 ...
```

The verifier replays from the same shard on local disk. Calibrate the band for
this model on the GPU class the nodes rent (see `docs/BAND_CALIBRATION.md`) and
keep the daemon in `--dry-run` until that band is set, so no honest node is
slashed against the nano model's band:

```
mkdir -p data/fineweb-rehearsal
curl -L -o data/fineweb-rehearsal/00000_00000_shuffled.ds \
  https://huggingface.co/datasets/emozilla/fineweb-10bt-tokenized-datatrove-llama2/resolve/cd477aed05fd5c2efd4b0c6c2459f1537fbd1a30/00000_00000_shuffled.ds
./target/debug/calibrate-band --model emozilla/llama2-20m-init \
  --data-dir data/fineweb-rehearsal --sequence-length 512 --samples 32 --class <gpu>
leviathan-verifier-daemon --run-id leviathan-fineweb ... \
  --replay-model emozilla/llama2-20m-init --replay-data-dir data/fineweb-rehearsal \
  --replay-sequence-length 512 --replay-step <first step> --band <calibrated> --dry-run
```

## Scaling to 350M and 1B

The parameter shape is the same; three things change, and all three need real
artifacts the run owner provides:

- `[model.LLM.checkpoint.Hub] repo_id` points at a real init checkpoint at the
  target size (a 350M or 1B HfLlama init on the Hub).
- `[model.LLM.data_location]` points at the real pretraining data mix, either an
  HTTP-served shard set or the run's TCP data server; `max_seq_len` rises to 2048.
- `global_batch_size_*` grows so the target global batch is at least the number of
  trainer nodes, and `total_steps` and the cosine schedule are set for the token
  budget.

Keep `verification_percent = 10` and the DisTrO block unchanged: those are the
security and transport parameters, and they do not depend on model size. Calibrate
the tolerance band per hardware class before mainnet (see the Phase 2 calibration
harness task); the band is the adversary's undetectable budget and is a published
dashboard metric.
