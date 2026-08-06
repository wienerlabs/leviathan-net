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
  --model pefontana/Nano-Llama --samples 8 \
  --reference-device cpu --reference-dtype fp32 \
  --target-device cpu --target-dtype bf16 \
  --class local-cpu-bf16
```

If reference and target are identical the drift is ~0 and the tool says so — vary
`--target-device` or `--target-dtype` to measure a real class.

An early observation worth recording: bf16 honest drift on the nano model is
**highly variable** — most batches sit near `3e-3`, but a single batch can spike
past `0.2`, because bf16 rounding flips which coefficients DisTrO's top-k keeps,
and a different sparse support decompresses to a very different dense vector. A
band picked from a handful of samples will swing wildly, so calibrate from a
healthy sample count (32+).

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
      "cmd": ["bash", "-c", "set -euo pipefail\nexport DEBIAN_FRONTEND=noninteractive\napt-get update -qq\napt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential pkg-config libssl-dev python3 python3-pip\ncurl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable\n. \"$HOME/.cargo/env\"\npip3 install --quiet torch==2.9.1\nexport LIBTORCH_USE_PYTORCH=1\nTORCH_LIB=$(python3 -c \"import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))\")\nexport LD_LIBRARY_PATH=\"$TORCH_LIB:${LD_LIBRARY_PATH:-}\"\ngit clone --depth 1 https://github.com/wienerlabs/leviathan-net /opt/leviathan-net\ncd /opt/leviathan-net\ncargo build --release -p leviathan-verifier --bin calibrate-band\nHW=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')\necho \"[calibrate] hardware class: $HW\"\n./target/release/calibrate-band --model pefontana/Nano-Llama --samples \"${SAMPLES:-32}\" --reference-device cpu --reference-dtype fp32 --target-device cuda --target-dtype bf16 --class \"$HW\" --json-out /tmp/band.json\necho '=== RESULT JSON ==='\ncat /tmp/band.json\n"]
    }
  }]
}
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
| _(fill from Nosana runs)_ | bf16 | 32 | | 5.0 | | |

Reference config for all rows: `cpu / fp32`. If the verifier policy changes to
replay on a GPU instead, re-run with `--reference-device cuda` and record that
here — the reference is a policy choice and the band is only meaningful relative
to it.

## Notes

- The harness is deterministic given the model, dataset seed, and configs; two
  runs on the same hardware produce the same band.
- A per-class band far above `DEFAULT_BAND` (0.05) is expected for bf16 and is the
  point of this exercise: it tells us the shipped default is a placeholder and the
  real economics must use the measured, per-class value.
