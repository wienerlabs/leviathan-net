# Leviathan Multitude

Distributed LLM training for phones, on the Leviathan security model.

> "A Multitude of men, are made One Person, when they are by one man, or one Person, Represented."
> Thomas Hobbes, *Leviathan*, chapter XVI

The main network builds the giant out of bonded GPUs. Multitude builds it out of phones that train while they charge, with the same rules: every contribution is attributable, replayable and auditable, and lying costs a bond.

## Why a phone run of its own

A phone cannot be a trainer on the 1B flagship run, and the reasons are in this repository:

- every client holds the full model, its gradients and the DisTrO buffer at once (`shared/modeling/src/distro.rs`); for 1B parameters that is more than 6 GB in bf16 before activations, and `docs/FIRST_RUN.md` puts the floor at an 8 to 12 GB RTX 3060;
- `max_round_train_time` is set by the slowest card and a node that misses it is dropped every round;
- the replay band is calibrated per hardware class, and a heterogeneous phone fleet widens it;
- every node downloads every peer's `DistroResult` (about 4 MB per peer per round at 1B), which does not scale to thousands of phones.

So Multitude is a phone-class run with its own shape: a model sized for a phone, local rounds long enough to amortize communication, a relay that aggregates for everyone, and an aggregate that anyone can recount exactly.

## How a round works

1. **Round.** The global weights, their hash (the checkpoint) and a round seed are public.
2. **Etiquette.** The phone trains only on a charger, on an unmetered network, idle, cool and with memory to spare (`device.rs`). It plans how many inner steps fit in the round from its measured throughput; a phone too slow for the minimum sits the round out.
3. **Local round.** A DiLoCo inner loop: AdamW from zero state on windows drawn from `sha256(round seed, phone key)` (`train.rs`, `data.rs`). The round is a pure function of its inputs, and the tests check that a replay is bit-identical on the same device.
4. **Vote.** The pseudo-gradient plus the error-feedback buffer is cut into chunks of 64 and each chunk sends its 8 largest coefficients as one sign bit each: SparseLoCo transport at about 1.1 bits per parameter (`compress.rs`). The phone commits to the hash of its error buffer and signs the contribution with its ed25519 key, which is also its Solana address (`protocol.rs`).
5. **Tally.** A relay counts the votes as integers, per parameter: net vote and number of voters (`aggregate.rs`). It publishes the tally, the counted phones in key order, and a Merkle root over their contributions, so every phone can prove it was counted (`merkle.rs`).
6. **Recount.** Any watcher recomputes the tally from the contributions it saw. Integers make this exact: a relay that miscounts is caught bit for bit, with no tolerance band and nothing to appeal. Leaving a valid contribution out is reported, not convicted, since it can arrive late. Signing two different contributions for one round is equivocation, provable from the two signatures.
7. **Audit.** A lottery drawn from the round seed, with the coordinator's `"audit"` salt, picks verifiers and targets (`lottery.rs`). The target reveals its error buffer, the verifier checks it against the commitment, replays the round, and compares the two votes with Leviathan's own `psyche_verifier::audit_contribution` (`audit.rs`).
8. **Step.** Every replica moves each parameter by the outer learning rate times its mean vote. The arithmetic is fixed-order IEEE-754, so every honest replica lands on bit-identical weights and the checkpoint hash means the same thing on every phone (`round.rs`).
9. **Settle.** Counted phones share the round's reward equally; a convicted phone forfeits its bond, and part of it pays whoever proved the fraud. A phone with no bond left cannot claim anything.

Because each phone casts at most one vote of plus or minus one per parameter and the step is the mean vote, no phone moves a parameter by more than the outer learning rate in a round, and where n phones vote its share is 1/n. Influence is bounded by construction, the property signSGD with majority vote relies on (Bernstein et al., 2019).

## Where it meets the main network

| Multitude | Leviathan |
|---|---|
| Replay comparison | `shared/verifier`: the same crate and the same `audit_contribution`, relative L2 against a band with a sha256 fraud proof |
| Audit lottery | `shared/coordinator/src/audit_selection.rs`, same `"audit"` salt, `verification_percent` as the audit rate |
| Transport | the DisTrO settings in `config/leviathan/genesis-rehearsal.toml`: chunk 64, top-k 8, one-bit signs |
| Tally | `CompressDCT::batch_decompress` takes the scatter-mean of the same plus-minus-one votes; Multitude keeps it as integers |
| Equal share | `instance_state.rs`: `earning_rate_total_shared / finishers` |
| Bounty | the treasurer's `slash_bounty_bps` |
| Bond-gated claim | `participant_claim` refuses a participant whose bond is below the minimum |
| Cheating economics | `leviathan-indexer`'s `assess_security`, mirrored in `economics.rs` |
| Band calibration | `calibrate-band`, with `calibrate_band` and `DEFAULT_SAFETY_FACTOR` from `psyche-verifier` |
| Cheater switch | the client's `LEVIATHAN_FAKE_DELTA`, here `adversary::Behavior` |

## Run it

```
cd architectures/mobile
cargo test
cargo run --release -p leviathan-multitude-sim -- --cheat 3:sign-flip --cheat 5:random --cheat 6:lazy --evil-relay-round 12
```

The simulator runs eight phones of four emulated hardware classes for 30 rounds on the Shakespeare prompts Leviathan's client samples from (`shared/client/src/state/prompt_texts/index.json`). With the flags above, on one machine in about 40 seconds:

```
  transport  top-8 of 64, 1-bit signs, error feedback: 19053 bytes of votes per round, 1.12 bits per parameter
  band       calibrated on the replay path, worst drift x 5:
               flagship-apple   worst honest drift 0.0000
               flagship-adreno  worst honest drift 0.0407
               mid-range        worst honest drift 0.0670
               budget           worst honest drift 0.0784
               band = 0.3918
...
r02 | 7/8 trained | steps 9-12 | local 4.187 | global 4.616 | 18.8 KB up | coverage  50% | audits 1
      convicted EW2Q…qES9 (lazy, distance 0.9640 against band 0.3918)
r05 | 7/7 trained | steps 9-12 | local 3.674 | global 4.318 | 18.8 KB up | coverage  52% | audits 4
      convicted Gpj2…YwR2 (random votes, distance 1.4101 against band 0.3918)
      convicted CEaN…uhnK (sign-flip, distance 2.0000 against band 0.3918)
r12 | 4/5 trained | steps 12-12 | local 3.056 | global 3.432 | 18.8 KB up | coverage  36% | audits 1
      relay: FRAUD caught by recount (tally hash 06c37b233c70e6cf does not match the recomputed 8e35c698770096fe); relay bond forfeited, honest tally used
...
Result
  global loss 5.550 -> 2.877 nats/byte over 30 rounds
  honest phones slashed: 0 (largest honest audit distance 0.0643 against band 0.3918)
  relay fraud caught by recount in round(s) [12]
  economics: reward share 0.221/round, bond 10.0, p 0.225 -> expected value of cheating -2.077/round (secure)
```

`--cheat` also takes `withhold` (a phone that refuses its audit), `--json` writes the full report, and `--corpus` trains on any text file. The demo audits at `verification_percent = 25` so a short run catches its cheaters; Leviathan's published operating point is 10.

The core crate is pure Rust (candle for tensors, no libtorch) and type-checks for phone targets:

```
rustup target add aarch64-linux-android aarch64-apple-ios
cargo check -p leviathan-multitude --target aarch64-linux-android
cargo check -p leviathan-multitude --target aarch64-apple-ios
```

Linking an app still needs the Android NDK or Xcode. `--features metal` trains on Apple's GPU.

## What the simulator does not prove

The network, the clock, the sensors and the hardware are simulated. Phone classes plan their steps from estimated throughputs for a 135M model, but the training runs on the host CPU. Hardware drift is a model (each pseudo-gradient value scaled by `1 + scale * u`), not a measurement. The model is 135K parameters on 2.5 KB of text: a test of the protocol, not a useful model.

## Next

1. **Measure a real phone.** Run `train_local` on an iPhone over Metal and an Android phone on CPU: tokens per second, temperature, battery, and the drift of each against an x86 replay, which sets the band per phone class. Until then a band-based slash stays in dry run, as the main network's rule requires.
2. **Mobile bindings.** UniFFI over `PhoneNode` and the device policy for Swift and Kotlin, with the key in Seed Vault, the Keychain or the Keystore.
3. **Transport.** iroh, as on the main network, for contributions and tallies, and a relay service.
4. **On chain.** Join through the authorizer behind the admission service of `docs/FIRST_RUN.md` section 0, with one admission per attested device (App Attest, Play Integrity) and delegates counted; bonds and claims through the treasurer; round seeds from a recent blockhash or a VRF; convictions through the committee vote.
5. **A real model.** Scale to 100M to 150M parameters once step 1 shows the memory and throughput fit.

Known limits: the round seed chains the previous seed with the tally hash, which a relay choosing what to include could grind (production takes it from the chain); audits run after the tally is applied, as on the main network, so an uncaught round's damage is bounded by the mean vote rather than undone; the error-buffer reveal is the size of the model, which wants per-chunk commitments at 100M parameters; and iOS limits background compute, so overnight training on an iPhone likely means the app stays open on the charger.
