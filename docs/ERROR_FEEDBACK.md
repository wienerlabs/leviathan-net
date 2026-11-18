# What error feedback costs a replay audit (issue #48)

A replay audit convicts a node when an honest recomputation of its
contribution lands outside the tolerance band. Under DisTrO that recomputation
works for a run's first step only. After it, a node's result also depends on
two pieces of state that the node holds privately:

- **Its error-feedback buffer.** This holds the part of its update that it has
  computed but not yet sent:

  ```
  D_s = decay * (D_{s-1} - T_{s-1}) + lr * g_s        result_s = topk(encode(D_s))
  ```

  The client zeroes `D` only at the first round of an epoch that warms up from
  a cold start, and every Leviathan config sets `cold_start_warmup_steps` to
  zero. A node that joins, or restarts, starts from a zeroed buffer. The
  buffer is kept in the training dtype. On a bf16 node, as the client is, a
  0.999 decay leaves every value unchanged, and small increments are absorbed
  (#90). So on a real node the formula holds only up to that rounding.
- **Its weights.** A running node keeps its model through cooldown, warmup and
  pauses (the #50 comment on #48). Replicas that apply the same updates on
  different hardware drift apart.

Issue #48 set out three options and asked for the choice to rest on
measurements rather than on paper:

1. Reset the buffer at a public boundary.
2. Commit to the buffer and reveal it on audit.
3. Leave replay at the first step.

It named three measurements:

- the convergence cost of a reset every N steps;
- multi-step drift across hardware;
- the size of a reveal.

This document reports all three and draws the decision from them.

## The decision

**Option 2, extended to the weights.**

Each node commits to the state it trains a round from: its weights, its buffer, and the values behind its previous round's public indices and signs. An audited node reveals that state, and the verifier replays that one round.

- **Training is unchanged**, so there is no convergence cost.
- **On a device that repeats itself, the verifier replays the round exactly.** The CPU does, across processes, and the M1's GPU does within one process. Across processes the GPU does not repeat itself: two runs of the same nodes diverge within 4 to 48 rounds. So whether a verifier on its own GPU can replay a node's round exactly is still open, and #89 has to measure it.
- **A verifier of another class drifts by one round's worth.** Nothing carries from one checked round to the next, so the drift cannot compound. But at 0.17 to 0.41 it is far above any band in force, so a check across classes waits on #49.

**Option 1 is rejected.** A reset costs convergence whatever the verifier's hardware: 1.6% to 12.5% of the loss by round 1000, depending on how often it resets, and no gap is closing. On top of that, across hardware classes a replay of the window drifts further with every round, even with the weights pinned at its start.

**Option 3 stays in force until option 2 is built.** Until then, the verifier replays a run's first step only.

The implementation is tracked in #89.

## What is measured

`measure-error-feedback` is a binary in `tools/rust-tools/leviathan-verifier`. It trains a run's nodes through the client's own trainer, in the order the client and the coordinator run them.

**Round order.** A round at height `h` of its epoch:

1. applies what every node sent at height `h - 2`, in the order of the batch;
2. then trains.

Each node therefore trains one update behind. DisTrO's lookahead, which subtracts `sign(D) * lr` before the gradient, stands in for the update the node has not applied yet.

**Epoch boundaries.**

- An epoch's first two rounds apply nothing.
- When the epoch's time is up, two closing rounds train nothing and apply what is left.
- The next epoch starts with no previous result to take out of a node's buffer.
- An epoch can also end without closing rounds: at the run's last step, or when it runs short of clients or witnesses, or on a pause. Its last two rounds are then never applied. The tool closes every epoch with two rounds.

**Data.** Every round, training or not, takes its step number and its slice of the data.

**Who trains.** Every node trains every round, and nothing zeroes a buffer but the reset policy. The tool refuses two kinds of run it would train wrongly:
- a run that seats verifiers among its nodes, since a verifier seat trains nothing that round;
- a run that warms up from a cold start, which zeroes every buffer at the first round of each cold-start epoch.

**Epoch length.** A run's configuration does not fix how many rounds an epoch trains: rounds run until the epoch's time is up. `--epoch-rounds` sets it.

**Reset policy.** `--reset-every N` zeroes every node's buffer every N training rounds. `0` never does, which is what the client does.

**Shadow verifier.** `--shadow-device` and `--shadow-dtype` add a shadow verifier that replays node 0 on its own device and in its own dtype. `--shadow-start` sets where the shadow starts each round:

| `--shadow-start` | Where the shadow starts a round |
|---|---|
| `own` | Its own weights, built by applying every public update since the run's start on its own hardware, and its own buffer, rebuilt since the last reset. |
| `resync` | As `own`, except at every reset, where it takes node 0's weights as a checkpoint. This is a canonical checkpoint at a public boundary, and it is what option 1 needs. Within the window it applies the public updates on its own hardware again. |
| `reveal` | At every round, node 0's weights, buffer and previous result values, as they stand when the round starts. This is the state a node would reveal for an audit of that one round, and it is what option 2 needs. The values follow from the buffer: a result is the buffer's top-k, and nothing changes the buffer between rounds. |

**Recorded per training round:**

| Column | What it is |
|---|---|
| `loss` | The nodes' mean training loss on the round's batch. No batch is trained twice, so this is a loss on data the model has not yet seen. |
| `distance` | The relative L2 distance between what node 0 sent and what the shadow replayed, both decompressed. The band is compared with this quantity. |
| `churn` | The share of the coordinates node 0 sent that the replay did not send at all, because its top-k kept other ones. |
| `flips` | Of the coordinates both sent, the share the replay sent with the other sign. |

Over 1-bit payloads, where the replay sends as many coordinates as node 0, the three are tied:

```
distance² = 2 · churn + 4 · (1 − churn) · flips
```

The `1 − churn` is there because `flips` is a share of the coordinates both sent, not of all node 0 sent. It holds to within 2e-7 over every round measured here. A distance can therefore come from top-k churn alone, with no sign wrong. That is the difference #49 has to price.

**Sizes.** The tool also prints the size of one node's buffer and of a round's result values.

## Proof that it trains a run the way real clients do

A measurement of this kind is only as good as its claim to follow the client. The client's order is not obvious from any one file, and a first version of this tool got it wrong: it applied the previous round's results instead of those from two rounds back. A test that compares the simulation with itself cannot catch that.

So the simulation is checked against real clients, byte for byte. `scripts/error-feedback-fidelity.sh` does the following:

1. It starts a local centralized server and two real `psyche-centralized-client` processes on `config/error-feedback-fidelity/state.toml`. That is the genesis rehearsal's model, a nano Llama at sequence length 64 (not the measured run's 20M model), with its data and 1-bit DisTrO, short epochs, and a learning rate that changes at every step. The clients and the checker run on the CPU in bf16.
2. One client dumps every node's payload of every round with `--write-gradients-dir`.
3. `check-error-feedback-fidelity` rebuilds the rounds from those dumps: the epochs, the closing rounds, and which node trained which share.
4. The checker trains those rounds in the simulation and compares every payload's wire bytes with the dump.

It requires at least two epochs, so that the closing rounds and an epoch's start are covered.

**Result.** On six independent runs (release and debug builds, fresh identities, different shuffles of the data between nodes), all 70 payloads of 35 training rounds over 3 epochs matched byte for byte.

**What it covers.** It covers the logic the measurement shares with the client:
- the round order;
- the epochs' openings and closings;
- the order results are applied in;
- the step each round trains and applies at.

The checker takes each round's batches and which node trained which from the dumps. The measurement's own assignment of them (`round_batch`, `even_shares`) is therefore checked by unit tests and by reading the code, not by the real clients.

**Mutations.** Each mutation below edited one line of the simulation, and the checker was run again on one run's dumps. The check fails when the simulation departs from the client in any of these ways:

| Departure from the client | Payloads that differ |
|---|---|
| Applying the previous round instead of two back (the old order) | 68 of 70 |
| Keeping the previous result across an epoch's start | 48 of 70 |
| Applying results in node order instead of batch order | 38 of 70 |
| Applying at the next step's learning rate | 66 of 70 |
| Training at the next step | 68 of 70 |

It runs in the CI libtorch job.

`tests/error_feedback.rs` covers what the real clients cannot:

- **Self-replay.** A shadow on the nodes' own hardware replays every round exactly, with and without resets, across an epoch's close.
- **Resets.** A reset changes the run from its round on, and not before.
- **Resync.** It hands the shadow node 0's weights. A shadow started from other weights replays node 0 exactly from the first resync on.
- **Reveal.** It hands the shadow everything a round depends on. A shadow on other weights replays node 0 exactly at every round.
- **State transfer.** A trainer takes another's weights and buffer across dtypes. A state that does not fit is refused and changes nothing.

## Running it

Needs the libtorch environment (`docs/OPS_RUNBOOK.md`).

```
cargo build --release -p leviathan-verifier --bin measure-error-feedback
./target/release/measure-error-feedback \
  --run-config config/leviathan/fineweb-rehearsal.toml \
  --nodes 2 --epoch-rounds 10 --steps 600 \
  --device mps --dtype bf16 --reset-every 3,10 \
  --shadow-device cpu --shadow-dtype fp32 --shadow-start resync \
  --window 100 --csv-out drift.csv
```

- `--steps` counts training rounds. The closing rounds come on top: 600
  training rounds in epochs of 10 are steps 1 to 718.
- Without `--data-dir` it reads the run's own HTTP data location, as a node
  does.
- Leave out `--shadow-device` to measure convergence alone, which is much
  faster.

The fidelity check runs on its own:

```
cargo build --bin psyche-centralized-server --bin psyche-centralized-client \
  --bin check-error-feedback-fidelity
bash scripts/error-feedback-fidelity.sh
```

## Setup

All measurements use `config/leviathan/fineweb-rehearsal.toml`:

- **Model:** `emozilla/llama2-20m-init`, 19,140,864 parameters, randomly
  initialized.
- **Data:** the first FineWeb shard, read from its HTTP location.
- **Batch:** sequence length 512, global batch 8, split between 2 nodes of 4
  sequences each.
- **Committee:** `verification_percent = 10` seats no verifier among 2 nodes,
  so both train every round.
- **DisTrO:** 1-bit, chunk 64, top-k 8, decay 0.999, with robust aggregation.
- **Schedule:** Cosine, base rate 4e-4, warmed up over 250 steps, decaying to
  4e-5 at step 25000.
- **Epochs:** 10 training rounds, then the 2 that close the epoch
  (`--epoch-rounds 10`).

**Hardware.** One Apple M1 Pro (16 GB). The nodes train on its GPU (`mps`) in
bf16, as the client trains. A shadow runs either on its CPU in fp32, a
verifier of another class, or on the same GPU in bf16, a verifier of the
nodes' own class.
- The CPU shadow differs from the nodes in device and dtype at once. A CPU
  shadow in bf16, which would separate the two, was not measured.
- The GPU shadow runs in the nodes' own process.
- Each node trains with no data or tensor parallelism and one micro-batch, as
  the replay trainer does.

**Data.** Each setup ran once. The nodes train on the GPU in bf16 in every run, and the CSVs are in `docs/error-feedback/`:

| Data | Shadow | Shadow starts | Training rounds | Policies (`--reset-every`) |
|---|---|---|---|---|
| `convergence-mps-bf16.csv` | none | | 1000 | 0, 3, 10, 100 |
| `drift-own-cpu-fp32.csv` | CPU, fp32 | `own` | 600 | 3 |
| `drift-resync-cpu-fp32.csv` | CPU, fp32 | `resync` | 600 | 3, 10 |
| `drift-resync-mps-bf16.csv` | GPU, bf16 | `resync` | 600 | 10 |
| `reveal-cpu-fp32.csv` | CPU, fp32 | `reveal` | 600 | 0 |
| `reveal-mps-bf16.csv` | GPU, bf16 | `reveal` | 600 | 0 |

Every range in the tables below is a mean, or a mean and a maximum, over the CSV's rows in that range.

## Results

### 1. A reset costs convergence, and the cost grows

Each policy trains the same 1000 training rounds, steps 1 to 1198, from the same checkpoint on the same data. The table gives the nodes' mean loss per range of training rounds, and each policy against never resetting:

| Training rounds | never | every 3 | every 10 | every 100 | every 3 vs never | every 10 vs never | every 100 vs never |
|---|---|---|---|---|---|---|---|
| 1-50 | 10.1352 | 10.1463 | 10.1411 | 10.1351 | +0.11% | +0.06% | -0.00% |
| 51-100 | 9.4383 | 9.2964 | 9.3621 | 9.4371 | -1.50% | -0.81% | -0.01% |
| 101-200 | 8.1065 | 7.8497 | 7.8862 | 8.0378 | -3.17% | -2.72% | -0.85% |
| 201-300 | 7.2765 | 7.5131 | 7.3118 | 7.2082 | +3.25% | +0.48% | -0.94% |
| 301-400 | 6.8354 | 7.3306 | 7.0323 | 6.8051 | +7.24% | +2.88% | -0.44% |
| 401-500 | 6.6777 | 7.2730 | 6.9483 | 6.6740 | +8.92% | +4.05% | -0.06% |
| 501-600 | 6.5359 | 7.1833 | 6.8784 | 6.5848 | +9.90% | +5.24% | +0.75% |
| 601-700 | 6.3691 | 7.0523 | 6.7435 | 6.4273 | +10.73% | +5.88% | +0.91% |
| 701-800 | 6.2234 | 6.9372 | 6.6327 | 6.3000 | +11.47% | +6.58% | +1.23% |
| 801-900 | 6.2052 | 6.9710 | 6.6559 | 6.3013 | +12.34% | +7.26% | +1.55% |
| 901-1000 | 6.0770 | 6.8371 | 6.5191 | 6.1723 | +12.51% | +7.28% | +1.57% |

- **During warmup, a reset is free or even helps.** The learning rate climbs from 0 over the first 250 steps, about the first 208 training rounds. A run measured over its first 300 training rounds would conclude that a reset costs little.
- **Then every policy falls behind, and the more often it resets, the sooner.** A reset every 3 or every 10 rounds is behind from rounds 201–300 on. A reset every 100 rounds is behind from rounds 501–600 on.
- **No gap is closing by round 1000.** Each is at its widest in the last range. Over the last two ranges, though, each moves by less than the run-to-run spread below: every 10, for one, goes from 7.26% to 7.28%.
- **Put in rounds:** never resetting reaches the final loss of a reset every 3 rounds around round 350, of every 10 around round 525, and of every 100 around round 872. That is the round at which its trailing 100-round mean first gets there.
- **The shorter the window, the larger the cost.** Over the last 100 training rounds:

  | Policy | Cost |
  |---|---|
  | Reset every 3 rounds | 12.5% |
  | Reset every 10 rounds | 7.3% |
  | Reset every 100 rounds | 1.6% |

**These gaps are not the GPU's noise.** The M1's GPU is not deterministic from run to run, so three of the policies were trained again in other runs:
- in the two reveal runs, the nodes never reset;
- in the two drift runs that reset every 3 rounds, the nodes reset every 3;
- in the two drift runs that reset every 10 rounds, the nodes reset every 10.

A shadow beside the nodes changes nothing they do. The repeated runs stay close to the first in every range:

| Policy | Largest gap between runs, in any range | Its cost at rounds 501–600 |
|---|---|---|
| Never reset | 0.08% | none: it is the baseline |
| Reset every 3 rounds | 0.29% | 9.90% |
| Reset every 10 rounds | 0.21% | 5.24% |

The runs differ only in the GPU's nondeterminism. The data order and the initial model are the same in all of them, and varying those was not measured.

Never resetting:

| Training rounds | first run | reveal, cpu fp32 | reveal, gpu bf16 | largest gap from first run |
|---|---|---|---|---|
| 1-100 | 9.7867 | 9.7887 | 9.7861 | 0.02% |
| 101-200 | 8.1065 | 8.1057 | 8.1067 | 0.01% |
| 201-300 | 7.2765 | 7.2709 | 7.2751 | 0.08% |
| 301-400 | 6.8354 | 6.8365 | 6.8405 | 0.07% |
| 401-500 | 6.6777 | 6.6769 | 6.6778 | 0.01% |
| 501-600 | 6.5359 | 6.5379 | 6.5387 | 0.04% |

Resetting every 3 rounds:

| Training rounds | first run | resync, cpu fp32 | own, cpu fp32 | largest gap from first run |
|---|---|---|---|---|
| 1-100 | 9.7214 | 9.7202 | 9.7213 | 0.01% |
| 101-200 | 7.8497 | 7.8547 | 7.8488 | 0.06% |
| 201-300 | 7.5131 | 7.5130 | 7.5051 | 0.11% |
| 301-400 | 7.3306 | 7.3396 | 7.3105 | 0.27% |
| 401-500 | 7.2730 | 7.2896 | 7.2564 | 0.23% |
| 501-600 | 7.1833 | 7.2043 | 7.1840 | 0.29% |

Resetting every 10 rounds:

| Training rounds | first run | resync, cpu fp32 | resync, gpu bf16 | largest gap from first run |
|---|---|---|---|---|
| 1-100 | 9.7516 | 9.7536 | 9.7584 | 0.07% |
| 101-200 | 7.8862 | 7.8897 | 7.8882 | 0.04% |
| 201-300 | 7.3118 | 7.3030 | 7.3015 | 0.14% |
| 301-400 | 7.0323 | 7.0179 | 7.0234 | 0.21% |
| 401-500 | 6.9483 | 6.9407 | 6.9387 | 0.14% |
| 501-600 | 6.8784 | 6.8910 | 6.8863 | 0.18% |

**Where policy 100 comes from.** Policies 0, 3 and 10 ran in one process. Policy 100 ran in a second process after the first was stopped. Each policy is a simulation of its own from the same checkpoint, so it does not matter which process ran it.

### 2. Replaying a window from public data drifts with every round, across classes

Under option 1 a verifier rebuilds a node's state from public data: it replays every round since the last reset. Two variants:

- **`own`.** The shadow keeps its own weights from the run's start, applying the same public updates on its own hardware. A reset zeroes only its buffer.
- **`resync`.** At every reset it takes node 0's weights as a canonical checkpoint. Within the window it applies the public updates on its own hardware again, so its weights drift from node 0's as well as its buffer.

With `--epoch-rounds 10`, a 10-round window is exactly one epoch, so its first two rounds apply nothing.

**A verifier of another class** (CPU fp32), by the round's position in its window:

`own`, a reset every 3 rounds:

| Round of the window | distance mean / max | churn mean / max | flips mean / max | rounds |
|---|---|---|---|---|
| 1 | 0.3930 / 0.5240 | 0.0792 / 0.1353 | 0.000102 / 0.002260 | 200 |
| 2 | 0.5466 / 1.0173 | 0.1551 / 0.4994 | 0.000665 / 0.027175 | 200 |
| 3 | 0.6698 / 1.1510 | 0.2314 / 0.5822 | 0.001970 / 0.105907 | 200 |

`resync`, a reset every 3 rounds:

| Round of the window | distance mean / max | churn mean / max | flips mean / max | rounds |
|---|---|---|---|---|
| 1 | 0.1523 / 0.1922 | 0.0117 / 0.0185 | 0.000000 / 0.000000 | 200 |
| 2 | 0.2265 / 0.6028 | 0.0265 / 0.1817 | 0.000000 / 0.000027 | 200 |
| 3 | 0.2998 / 0.6178 | 0.0471 / 0.1909 | 0.000002 / 0.000060 | 200 |

`resync`, a reset every 10 rounds:

| Round of the window | distance mean / max | churn mean / max | flips mean / max | rounds |
|---|---|---|---|---|
| 1 | 0.1440 / 0.1857 | 0.0104 / 0.0172 | 0.000000 / 0.000000 | 60 |
| 2 | 0.2037 / 0.2920 | 0.0210 / 0.0426 | 0.000000 / 0.000027 | 60 |
| 3 | 0.2613 / 0.6274 | 0.0361 / 0.1968 | 0.000002 / 0.000054 | 60 |
| 4 | 0.3193 / 0.7095 | 0.0543 / 0.2517 | 0.000005 / 0.000125 | 60 |
| 5 | 0.3666 / 0.7095 | 0.0709 / 0.2517 | 0.000004 / 0.000035 | 60 |
| 6 | 0.4129 / 0.8124 | 0.0903 / 0.3300 | 0.000008 / 0.000144 | 60 |
| 7 | 0.4649 / 0.8597 | 0.1144 / 0.3692 | 0.000018 / 0.000249 | 60 |
| 8 | 0.5167 / 0.9385 | 0.1420 / 0.4384 | 0.000085 / 0.001772 | 60 |
| 9 | 0.5639 / 1.0196 | 0.1701 / 0.5148 | 0.000232 / 0.005183 | 60 |
| 10 | 0.6041 / 1.0948 | 0.1971 / 0.5980 | 0.000334 / 0.015262 | 60 |

What the tables show:

- **A window's first round already drifts, and more without resync.**
  - Both buffers have just been zeroed. Under `resync` the weights are node 0's too, and the distance, 0.15, is what one round on another device and dtype costs.
  - Under `own` it is 0.39 on average, and it grows with the run: 0.30 in rounds 1–120, 0.45 in rounds 481–600. The extra comes from the replica's own weights, which drift further from node 0's as the run goes on.
  - So any design has to pin the weights as well as the buffer, as the #50 comment on #48 anticipated.
- **Even with the weights pinned at the start, each round of the window adds drift.** The distance grows round by round, from 0.14 in the first round of a 10-round window to 0.60 in the last. The growth mixes two drifts. The verifier adds its own slightly different result to its buffer, so the top-k sets diverge further. And it applies the window's updates on its own hardware, so its weights drift too.
- **Without resync, signs start to go wrong.** Under `own`, flips reach 10.6% of the shared coordinates in the worst round. With the weights pinned they reach at most 1.5%.

**A verifier of the nodes' own class** (the same GPU, bf16, in the nodes' process, resync every 10 rounds) replays the whole window exactly:

All 600 training rounds match node 0 exactly, in every position of every window: distance, churn and flips are 0 in each.

So on a device that repeats itself, as this GPU does within one process, replaying a window is exact. Across processes this GPU does not repeat itself (see the limits). Option 1's convergence cost, in section 1, applies whatever the verifier's hardware.

### 3. A reveal turns every later round into a single step

Under `reveal`, the shadow starts each round from what node 0 would reveal for an audit of that round: its weights, its buffer and its previous round's values, as they stand when the round starts. It then replays the one round. Nothing carries from one checked round to the next, so whatever drift appears belongs to that one round alone.

**On the nodes' own GPU, in their process, a verifier replays it exactly.** The shadow ran on the nodes' GPU in bf16. All 600 training rounds of the run matched node 0 exactly: distance, churn and flips are 0 in every round. Across processes this GPU does not repeat itself, so this shows what a device that repeats itself allows, not yet what a verifier on another machine of the class gets (see the limits).

**A verifier of another class drifts by one round's worth.** The shadow ran on the CPU in fp32, which differs from the nodes in device and dtype at once:

| Training rounds | distance mean / max | churn mean / max | flips mean / max |
|---|---|---|---|
| 1-100 | 0.2608 / 0.4084 | 0.0345 / 0.0834 | 0.000000 / 0.000000 |
| 101-200 | 0.2785 / 0.3908 | 0.0390 / 0.0764 | 0.000000 / 0.000000 |
| 201-300 | 0.2870 / 0.3377 | 0.0414 / 0.0570 | 0.000000 / 0.000000 |
| 301-400 | 0.3223 / 0.3521 | 0.0520 / 0.0620 | 0.000000 / 0.000000 |
| 401-500 | 0.3442 / 0.3787 | 0.0593 / 0.0717 | 0.000000 / 0.000000 |
| 501-600 | 0.3509 / 0.3778 | 0.0616 / 0.0714 | 0.000000 / 0.000000 |
| all | 0.3073 / 0.4084 | 0.0480 / 0.0834 | 0.000000 / 0.000000 |

- **It cannot compound, but it grows with the run.** Nothing carries from one checked round to the next. Still, the mean rises in every range, from 0.26 in rounds 1–100 to 0.35 in rounds 501–600. How it behaves later in a run, or at a decayed learning rate, is not measured.
- **It is more than a round from a zeroed buffer costs.** In round 1, where the buffer is still zero, the distance is 0.17, about what a window's first round drifts under `resync`. From a filled buffer it is larger.
- **It is all top-k churn.** No sign was wrong in any of the 600 rounds. Up to 8% of the coordinates node 0 sent were ones the verifier did not pick, and every coordinate both picked had the same sign.
- **The dtype may account for part of it.** The nodes' buffer is bf16, where the decay does nothing and small increments are absorbed (#90). The fp32 shadow decays and absorbs nothing.
- **It is far above every band in force.** All 600 rounds exceed the shipped default of 0.05, and the band calibrated over unquantized values, 0.029 (`docs/BAND_CALIBRATION.md`). The same document measured a band of 1.104 over 1-bit signs. Under today's metric, an honest cross-class check would therefore convict. Even a run's first step is checked against a band today, not exactly. Checking across classes waits on #49's band and distance for 1-bit payloads.

### 4. What a reveal sends

From the tool's printout, for one node of this run in bf16:

| Part of the state | Size | In this run |
|---|---|---|
| Weights | one value per parameter, in the training dtype | 36.5 MiB |
| Error-feedback buffer | one value per parameter, in the training dtype | 38,281,728 bytes (36.5 MiB) |
| Values of the previous round's result | one per coordinate sent | 76,352 bytes (0.07 MiB) |

The values follow from the buffer, so a verifier can derive them rather than take them; they are a rounding error in the size either way. A reveal is therefore twice the model in its training dtype. For the sizes the roadmap names, trained in bf16:

| Model | A reveal |
|---|---|
| 19.1M (this run) | 73.1 MiB |
| about 350M | about 1.3 GiB |
| about 1B | about 3.7 GiB |

- **It needs a window of its own.** At 100 Mbit/s of upload, a 1B reveal takes over five minutes. The final round's audit window is sized in tens of seconds (`docs/FIRST_RUN.md`, section 3).
- **Every node pays for the commitment every round.** Committing to the state means copying the weights and the buffer off the device and hashing about twice the model, each round.
- **A node has to keep its state** for every round still inside its audit window: that many reveals' worth of storage, about 3.7 GiB a round at 1B.
- **A buffer kept in fp32**, should #90 move it there, makes a reveal three times the model in bf16: about 5.6 GiB at 1B.

## What it means for the options

### Option 1, a reset at a public boundary: rejected

- **It costs convergence on any hardware.** Resetting every 100 rounds costs 1.6% of the loss by round 1000, and every 3 rounds costs 12.5%. No gap has started to close.
- **It needs a canonical checkpoint at every boundary.** Without one, a verifier's own replica of the weights drifts, and even a window's first round is off by 0.39 across classes.
- **Across classes, its windows drift further every round**, even with the checkpoint: from 0.14 to 0.60 over 10 rounds.
- **On a device that repeats itself it replays exactly.** So does option 2, without the convergence cost.

### Option 2, commit and reveal: chosen, extended to the weights

- **Training is unchanged.**
- **Each check is one round from a revealed state:**
  - exact on a device that repeats itself;
  - across classes, one round's drift (0.31 mean, 0.41 max) that cannot compound, and that is all top-k churn. It is still far above any band in force, so it waits on #49.
- **The commitment has to cover the weights.** A verifier's own replica drifts from them, as option 1's first rounds show.
- **What it costs:**
  - **The reveal.** Twice the model, with a window of its own.
  - **Data availability.** The chain cannot see that a node refused to reveal (#48).
  - **Continuity.** A revealed state has to be the one the node actually reached. Otherwise a node could carry a poisoned update in its weights or its buffer and send results that are honest with respect to it.
    - Where a transition replays exactly, a verifier can check that one revealed state leads to the next committed one. That takes the state of both rounds, so it checks the audited transition only. Like any contribution, a poisoned transition is caught with the audit probability. The poison then stays in the buffer until it is sent, since a bf16 buffer does not decay (#90).
    - Across classes the next state cannot be matched by hash and needs a bound. #48 sketches the argument; it is not shown.
  - **A class has to be pinned down.** For a transition to replay exactly, a verifier has to match not only the device and dtype but also the micro-batch size and the data and tensor parallelism. The simulation used one micro-batch and no parallelism.
- **What it leaves to #49: the band for a check across classes.** The drift there is top-k churn with no sign wrong: 0 flips in 600 rounds. A band that counts flips on the coordinates both sides sent apart from churn could therefore hold flips far tighter than the distance alone allows. The churn it tolerates, up to about 8% here, is a share of coordinates a cheater could choose freely, signs included, so #49 has to price that too.

### Option 3, leave replay at the first step: in force until option 2 is built

The README, `docs/FIRST_RUN.md` and the verifier daemon already say that only a run's first step is replayed.

## Limits of these measurements

- **One machine.** Everything ran on an Apple M1 Pro, on its CPU and its GPU.
  - No CUDA class was measured.
  - **Exactness on the GPU was measured within one process.** On the CPU, the fidelity check reproduces whole epochs byte for byte across processes. The M1's GPU does not repeat itself across processes:
    - Runs of the same nodes on the same data agree for the first rounds. They then diverge, at training round 4, 21, 33 or 48, depending on the pair compared.
    - Within one process, identical work agrees.
    - So a verifier in its own process on this GPU would most likely not replay a node's round exactly. Whether CUDA classes repeat themselves across machines is the measurement #89 has to make before it relies on exact checks.
  - **The cross-class pair differs in device and dtype at once.** A CPU shadow in bf16 would separate the two, and was not measured.
- **One small model, early in its run.**
  - The model has 19.1M parameters and was randomly initialized.
  - The 1000 training rounds are 4.1M tokens trained, steps 1 to 1198 of the run's 25000. The first 250 steps warm the learning rate up from 0, and the rest are near its peak.
  - The cost of a reset late in a run, at a decayed rate, or on the 350M and 1B models, is not measured. Over these 1000 rounds it only grows.
- **Two nodes, one epoch length, no parallelism.** More nodes average more results per round, and epochs of other lengths close at other places. Neither is measured, and neither are micro-batches or data or tensor parallelism.
- **The fidelity check ran a smaller model.** It ran the genesis rehearsal's nano Llama at sequence length 64 on the CPU. It covers the logic the measurement shares with the client, not the measured model or device.
- **The distance metric is under revision.** The distance is the relative L2 over 1-bit signs, which #49 is redesigning. Churn and flips are reported beside it, so the drift reads the same whatever #49 picks.
