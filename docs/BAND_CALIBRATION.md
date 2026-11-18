# Tolerance-band calibration across hardware classes (issue #7)

The replay verifier convicts a target when its honest replay differs from the
submission by more than the **tolerance band**. An honest node is not a cheater,
yet its DisTrO delta still differs from a verifier's recomputation, because the
two ran on different hardware and — above all — different precision: the network
trains in **bf16** while the reference replay is **fp32**. That gap is the band's
floor. Set the band below the honest drift and honest nodes get convicted; set it
far above and a cheater's undetectable budget grows. So the band has to be
measured per hardware class, not guessed.

`calibrate_band(drift_distances, safety_factor)` (in `psyche-verifier`) already
turns a set of honest-drift samples into a band (`worst × safety_factor`). What
was missing was the harness that produces those samples on real hardware. That is
`calibrate-band`.

## What it measures

`calibrate-band` computes the same one-step delta twice — once in a **reference**
config (default fp32 on CPU, the canonical deterministic ground truth a verifier
is assumed to replay in) and once in a **target** config (default bf16 on the
node's accelerator) — for several distinct batches, each from the same fresh
checkpoint. For each batch it records `relative_l2_distance(target, reference)`,
the honest drift, and finally reports `calibrate_band` of the whole set.

The delta computation reuses the exact production path (`build_replay_trainer` →
`TrainerReplayEngine`), so what it measures is what the verifier actually sees.
The only difference between the two runs is device and dtype, so the distance is
pure honest drift.

## Running it locally

Needs the libtorch environment (see `docs/OPS_RUNBOOK.md`). A quick precision-only
check on CPU:

```
cargo build -p leviathan-verifier --bin calibrate-band
./target/debug/calibrate-band \
  --run-config config/leviathan/genesis-rehearsal.toml --samples 8 \
  --reference-device cpu --reference-dtype fp32 \
  --target-device cpu --target-dtype bf16 \
  --class local-cpu-bf16
```

If reference and target are identical the drift is ~0 and the tool says so — vary
`--target-device` or `--target-dtype` to measure a real class.

A word of caution about the CPU path, worth recording: fp32-vs-bf16 drift
measured with bf16 **on CPU** is misleading. It sits near `3e-3` for most batches
but a single batch can spike past `0.2`, because CPU bf16 is emulated and the
rounding occasionally flips which coefficients DisTrO's top-k keeps, so a
different sparse support decompresses to a very different dense vector. On a real
accelerator (see the RTX 3060 Ti row below) native bf16 does not do this — the
drift stays tight around `4e-3`. So calibrate on the actual hardware class, not on
a CPU bf16 stand-in, and use a healthy sample count (32+) either way.

## Running it per hardware class on Nosana

One job per GPU market; the class label is taken from the card. This measures the
real training config (bf16 on the GPU) against the canonical fp32-CPU reference.
Paste into the Nosana dashboard, one market at a time:

```json
{
  "version": "0.1",
  "type": "container",
  "meta": { "trigger": "dashboard", "system_requirements": { "vram_total_mb": 8192 } },
  "ops": [{
    "id": "calibrate-band",
    "type": "container/run",
    "args": {
      "image": "docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04",
      "gpu": true,
      "env": {
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "SAMPLES": "32"
      },
      "cmd": ["bash", "-c", "set -euo pipefail\nexport DEBIAN_FRONTEND=noninteractive\napt-get update -qq\napt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential pkg-config libssl-dev python3 python3-pip\ncurl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable\n. \"$HOME/.cargo/env\"\npip3 install --quiet torch==2.9.1\nexport LIBTORCH_USE_PYTORCH=1\nTORCH_LIB=$(python3 -c \"import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))\")\nexport LD_LIBRARY_PATH=\"$TORCH_LIB:${LD_LIBRARY_PATH:-}\"\ngit clone --depth 1 https://github.com/wienerlabs/leviathan-net /opt/leviathan-net\ncd /opt/leviathan-net\ncargo build --release -p leviathan-verifier --bin calibrate-band\nHW=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')\necho \"[calibrate] hardware class: $HW\"\n./target/release/calibrate-band --run-config config/leviathan/genesis-rehearsal.toml --samples \"${SAMPLES:-32}\" --reference-device cpu --reference-dtype fp32 --target-device cuda --target-dtype bf16 --class \"$HW\" --json-out /tmp/band.json\necho '=== RESULT JSON ==='\ncat /tmp/band.json\n"]
    }
  }]
}
```

The clone above pulls the default branch, which is correct once this lands on
`main`. **Before it is merged**, point the clone at the PR branch instead, or
`cargo build` fails with `no bin target named calibrate-band`:

```
git clone --depth 1 --branch feat/band-calibration-harness-issue-7 \
  https://github.com/wienerlabs/leviathan-net /opt/leviathan-net
```

The first build takes 10–20 minutes on a rented host; set the container timeout
to at least 40 minutes. The last line printed is the JSON to collect. Availability
binds harder than price — see `docs/NOSANA.md` for which GPU markets can actually
be reserved.

## Per-class results

Fill in as each class is measured (paste the `band` from each job's JSON). The
band feeds the `verification_percent` economics — a larger honest band means a
larger audit probability is needed to keep a cheater's expected value negative.

| Hardware class | dtype | samples | drift max | safety | **band** | default 0.05 adequate? |
|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 3060 Ti | bf16 | 32 | 0.005804 | 5.0 | **0.029019** | yes (0.029 < 0.05) |
| _(more classes as measured)_ | bf16 | 32 | | 5.0 | | |

The 3060 Ti drift was tight across all 32 batches (min 0.00258, mean 0.00366,
max 0.00580), so its calibrated band (0.029) sits comfortably under the shipped
default (0.05): honest 3060 Ti nodes will not be falsely convicted, and the band
is not so wide that it hands a cheater room. The 3060 class is the swarm's most
common and cheapest-to-rent hardware (see `docs/NOSANA.md`), so this covers the
dominant case; 24 GB classes (3090/4090) are worth adding before mainnet.

The 3060 Ti row was measured before `calibrate-band` read the run's
configuration (wienerlabs/leviathan-net#47). It replayed unquantized values at
a constant learning rate, while every Leviathan configuration sends 1-bit signs
under a Cosine warmup. Drift over 1-bit signs is a different quantity: an
honest node and the verifier disagree only where a value near zero lands on
the other side of it, or a borderline index enters or leaves the top-k. Every
class has to be measured again with `--run-config` before its band is trusted.

A first measurement under the genesis rehearsal shows why. It was taken on a
CPU, comparing fp32 with bf16 (`--samples 8`). Seven of eight samples drifted
by exactly 0, and one by 0.2209. The calibrated band came out at 1.104, against
the 0.05 default. Over 1-bit signs, honest drift is not small and smooth. It is
zero until one coordinate changes, and then it jumps. A band wide enough to
cover that jump also lets a node turn over a share of its signs, about
(band/2)² of them, without being caught. The next section measures what the
jump is made of, and gives 1-bit payloads a band of their own
(wienerlabs/leviathan-net#49).

Reference config for all rows: `cpu / fp32`, micro batch 1. A node's micro batch
size and whether it accumulates gradients in fp32 are its own settings, not
the run's, so they add to the honest drift the band has to cover as well. If the verifier policy changes to
replay on a GPU instead, re-run with `--reference-device cuda` and record that
here — the reference is a policy choice and the band is only meaningful relative
to it.

## 1-bit payloads: churn and flips (#49)

Over 1-bit payloads a replay can differ from what a node sent in two ways:

- **churn:** it picks other coordinates for its top-k, so a coordinate the node
  sent is one the replay did not send;
- **flips:** it picks the same coordinate with the other sign.

The relative L2 distance mixes the two:
`distance² = 2 · churn + 4 · (1 − churn) · flips`, where churn is a share of
the coordinates the node sent and flips a share of those both sent. So the
distance is the square root of the share of coordinates in question. A safety
factor of k on a band on the distance multiplies by k² the share of its
coordinates a cheater may then choose. The same factor on a band on churn and
flips multiplies that share by k.

**The genesis jump, taken apart.** `calibrate-band` now reports churn and flips
for a 1-bit run. Run again on this repository's M1 CPU, fp32 against bf16, with
the issue's 8 samples
(`one-bit/genesis-first-step-cpu-fp32-vs-cpu-bf16.json`):

| Sample | Distance | Churn | Flips |
|---|---|---|---|
| 2 | 0.1562 | 0.0122 | 0 |
| 5 | 0.2209 | 0 | 0.0122 |
| the other six | 0 | 0 | 0 |

The genesis model sends 82 coordinates, so 0.0122 is one of them:
- the 0.2209 the issue measured is a single sign that crossed zero;
- on this machine one further sample churned a single coordinate.

Calibrated with the same factor of 5:
- the distance band is 1.104, which lets a cheater flip about 30% of its signs;
- the sign band is 6.1% churn and 6.1% flips.

**Later rounds, from a revealed state.** `measure-error-feedback`'s `reveal`
mode starts a shadow verifier from node 0's weights and buffer at every round
of the FineWeb rehearsal, and replays that one round (`docs/ERROR_FEEDBACK.md`).
This is the check #89 builds. The nodes train on the M1's GPU in bf16. Over
each run's first 100 training rounds (40 for the CPU-trained nodes, which are
slow):

| Nodes | Verifier | What differs | Churn mean / max | Flips max | Distance mean / max |
|---|---|---|---|---|---|
| GPU bf16 | GPU bf16, same process | nothing | 0 / 0 | 0 | 0 / 0 |
| GPU bf16 | CPU bf16 | the device | 0.0281 / 0.0378 | 0 | 0.236 / 0.275 |
| CPU bf16 | CPU fp32 | the dtype, on the CPU | 0.0284 / 0.0386 | 0 | 0.237 / 0.278 |
| GPU bf16 | GPU fp32 | the dtype, on the GPU | 0.0346 / 0.1183 | 0 | 0.261 / 0.487 |
| GPU bf16 | CPU fp32 | both | 0.0345 / 0.0834 | 0 | 0.261 / 0.408 |

The data is in `one-bit/` and in `docs/error-feedback/reveal-*.csv`.
- **In a round's first step, from a zeroed buffer,** each pair churns 1.5–1.7%
  and flips nothing.
- **Over longer runs, the fp32 replays' tails grow.** The GPU fp32 verifier
  reaches 14.3% churn within 300 rounds, with one round of 0.15% flips. The CPU
  fp32 verifier reaches 8.3% within 600.

What this shows:
- **Across hardware classes, an honest single-step replay churns and almost
  never flips.** A band on the flips can be held near zero, and with it the
  cheapest cheat over signs, turning them over. It cannot be held at zero,
  because a sign near zero can cross it, as the genesis sample did.
- **Replaying in the run's dtype narrows the churn.** The bf16 replay's worst
  churn in 100 rounds was 3.8%, against 8.3% and 11.8% for the fp32 replays.
  Each is a single run, so the tails are indicative rather than established.
- **A verifier of the nodes' own class replays exactly**, within one process.
  Whether another machine of the same GPU class does is #89's to measure. Where
  it does, both bands can be 0.

**What the verifier does now:**
- **1-bit runs.** `leviathan-verifier` and the verifier daemon take
  `--churn-band` and `--flip-band`, and hold a payload to both.
- **Values under a sign band.** A band on the signs cannot see magnitudes. So
  under it, a payload that carries values rather than packed signs is unfit,
  and fraud. The daemon tests' `fraud` fixture is such a payload: it moves no
  sign and no coordinate, only magnitudes. Nodes still apply such payloads
  (wienerlabs/leviathan-net#95).
- **The daemon refuses a mismatched band** unless it only watches
  (`--dry-run`): a band on the distance for a run that sends 1-bit signs, and a
  band on the signs for one that sends values.
- **`--replay-dtype`** chooses the daemon's replay dtype, fp32 by default. A
  bf16 replay churns less, but is emulated on a CPU and runs several times
  slower (`measure-error-feedback` took about 45 s a round with a CPU bf16
  shadow). Its band has to be calibrated for the dtype chosen.
- **Calibration.** `calibrate-band` reports churn and flips per sample, with
  the band they call for and the flags to pass. Calibrate per pair of node
  class and verifier class, with more samples than the 8 above, and use the
  safety factor knowing it is now linear in the cheater's share.

**What is not settled here:**
- **The price of a cheat.** Under a band of churn c and flips f, a cheater
  chooses at most c of the coordinates it sends, signs included, and turns over
  f of the rest. What that buys against robust aggregation, and so how large a
  band the audit economics can afford, is the band-sweep damage analysis's to
  price. It is not in this repository.
- **The GPU classes nodes rent** (CUDA) are not measured, and only the CPU and
  the M1's GPU are.
- **The tails above are one run each.**

## Notes

- The harness is deterministic given the model, dataset seed, and configs; two
  runs on the same hardware produce the same band.
- A per-class band far above `DEFAULT_BAND` (0.05) is expected for bf16 and is the
  point of this exercise: it tells us the shipped default is a placeholder and the
  real economics must use the measured, per-class value.
- The band covers one step replayed from a known state. A replay of later steps
  drifts with every step of its window, so they are to be checked one
  transition at a time instead, from the state an audited node reveals
  (`docs/ERROR_FEEDBACK.md`, #48). Across classes such a transition still
  drifts: 0.31 mean and 0.41 max between bf16 on an M1 GPU and fp32 on its CPU,
  all of it top-k churn with no sign wrong. That is far above the default band
  and the 0.029 above, which is why 1-bit payloads are now held to a band on
  churn and flips instead (the section above).
