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
run-manager set-bond-config --run-id leviathan-genesis --bond-minimum-amount 500 --bond-withdraw-delay-seconds 600 ...
run-manager treasurer-top-up-rewards --run-id leviathan-genesis --collateral-amount 500000 ...
run-manager set-paused --run-id leviathan-genesis --resume ...
./scripts/leviathan-swarm.sh --run-id leviathan-genesis --wallets <dir> --nodes 3 --cheater 2
```

The cheater node broadcasts fraudulent gradients (LEVIATHAN_FAKE_DELTA); audit
its dump against an honest node's with `leviathan-verifier`, and watch the
telemetry with `leviathan-indexer --features live`.

The indexer also reports whether the run is economically secure at its audit
probability, using the break-even penalty `reward * (1 - p) / p`. It reads every
input off chain: the reward is a client's share of the epoch's earnings rate, the
slash is the epoch's slashing rate, and the bond is the run's bond minimum.

```
leviathan-indexer --run-id leviathan-genesis
```

The verdict is secure when the effective penalty, capped at the bond, meets or
exceeds the break-even penalty, so a cheater loses money in expectation. A
committee smaller than the treasurer takes a verdict from convicts nobody, so a
run with one is judged as audited with probability zero. The rates above are a
rehearsal's and are not secure. At 10% a round seats no verifier until the run
has ten nodes, so on three nodes no committee can convict and a cheat keeps its
whole 333 share; only the run authority can convict, and
`LeviathanCommitteeCannotConvict` fires. With a committee that can convict, a
slash of 100 audited one time in ten still pays a cheater about 290 an epoch. A
run meant to be secure needs enough nodes for a committee, and a slashing rate
and a bond of at least the break-even penalty, about 3,000 at that share.

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
./target/debug/calibrate-band --run-config config/leviathan/fineweb-rehearsal.toml \
  --data-dir data/fineweb-rehearsal --samples 32 --class <gpu>
leviathan-verifier-daemon --run-id leviathan-fineweb ... \
  --replay-data-dir data/fineweb-rehearsal --band <calibrated> --dry-run
```

The daemon reads everything else from the run on chain: the checkpoint and its
revision, the learning-rate schedule, the optimizer, the sequence length and
the data layout. It also reads the one step a replay can reproduce, which is the
run's first, so there is nothing to keep in step with the run by hand
(wienerlabs/leviathan-net#47):

```
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
