# Checking every round of a run (issue #89)

A replay audit used to work for a run's first step only. After it, a node's
result depends on two pieces of state it holds privately: its error-feedback
buffer and its weights. `docs/ERROR_FEEDBACK.md` measured the options and chose
**commit and reveal, extended to the weights**: each round a node commits to
the state it trains from, and an audited node reveals that state, from which a
verifier replays that one round. This document is the design that is built,
in the order it is built, and what it leaves open.

## The state a node commits to

At round `h` a node trains from:

- **its weights**, after applying every peer's result from round `h - 2`;
- **its error-feedback buffer** `D`, as it stands before round `h`'s step adds
  its gradient;
- **the values of its own round `h - 1` result**, which the step takes out of
  `D` first. They follow from the buffer and the result's public indices, so
  they are not part of the state: a verifier derives them.

That is `S_h = (weights, D)`. Its encoding is canonical: the model's variables
by name, then the buffer's, each as its dtype, shape and little-endian bytes.
The buffer keeps its own dtype (the parameter's, `docs/ERROR_FEEDBACK.md`
section 5).

**Its digest is BLAKE3**, which is the hash iroh names a blob by. The
commitment then names a blob any peer can fetch, from the node, and check by
its address alone. Nothing on chain ever hashes a state: it only carries the
32 bytes.

## The commitment, version 3

The hash a node signs for its payload (`payload_commitment_hash`) gains the
state digest:

```
v3 = sha256("leviathan/payload-commitment/v3" || run || epoch_start || step
            || trainer_nonce || batch_start || batch_end || results_digest
            || state_digest)
```

- `TransmittableDistroResult` carries `state_digest`, so every receiver checks
  the signature over it as it does now.
- The treasurer's `CommitmentEvidence` carries it, so a conviction still
  rebuilds the hash from the account alone (#43). `AuditVerdict` grows by 32
  bytes, a new generation for the dashboard's decoders.
- An equivocation is two different v3 hashes for one step and batch, so two
  payloads that differ only in the state they claim are equivocation, as they
  should be: the node told peers two different things about one round.
- **It is a flag day**, like #67's v2: every node, verifier and treasurer of a
  run moves together, from the run's start.
- **A node may commit to no state.** The hash then binds a tag saying so,
  which no digest can stand in for. A trainer driven from Python does this:
  its ranks there do not join the gathers a digest of a sharded model needs.
  Its rounds past the first cannot be replayed, and an audit of one finds no
  state to fetch.

## When the digest is taken, and what it costs

The trainer thread takes it between applying round `h - 2`'s results and the
forward pass of round `h`: it copies the weights and the buffer off the device
and hashes them. That is about twice the model, every round, on every node.

| Model | State, bf16 | Copied and hashed every round |
|---|---|---|
| 19.1M (the rehearsal) | 73 MiB | 73 MiB |
| about 350M | about 1.3 GiB | about 1.3 GiB |
| about 1B | about 3.7 GiB | about 3.7 GiB |

Measured with `measure-round-state` on the FineWeb rehearsal's 19.1M Llama in
bf16, on an M1 Pro, after a real step filled the buffer (20 rounds each):

| Where the node trains | State | Copied off the device, encoded and hashed |
|---|---|---|
| CPU | 73.0 MiB | 0.045 s per round |
| M1 GPU (`mps`) | 73.0 MiB | 0.082 s per round |

That is about 0.9 GiB/s from the GPU. The trainer takes the digest on its
thread before the round's forward pass, so the round waits for it. At that
rate a 350M state would take about 1.5 s a round and a 1B state about 4 s;
those are extrapolated, not measured, and a CUDA node's copy rate is its
own.

## Retention, and serving a state

A node writes `S_h` to a state store of its own, one file per round named by
its digest. It keeps every round still inside the audit window, which is
`audit_rounds`, a run parameter:

- **The disk this takes:** `audit_rounds` times the state, about 3.7 GiB a
  round at 1B.
- **When a round leaves the window** its file is deleted.

The node imports each file into its iroh blob store when it writes it. It
serves it on request, never by broadcast: everyone downloading every state
would cost the run twice the model per node per round.

## Revealing to a verifier

A verifier seated for round `h` downloads, for each payload it audits, the
state the payload's commitment names, from the payload's trainer, by the
digest. The trainer is the roster key that signed the payload, and iroh
reaches it at that key.

- **The fetch checks itself.** iroh verifies a blob against its hash as it
  arrives, and the hash is the digest the trainer signed. A state that does
  not match cannot be served under it.
- **Where it goes.** The verifier's node writes it beside the payload dump, as
  `<dump>.state`, for the verifier daemon.
- **The reveal window.** The download has `reveal_seconds`, a run parameter,
  sized as `state_bytes / reveal_rate + slack`. It is not the final round's
  audit window: a 1B state at 100 Mbit/s takes over five minutes.

## Replaying a later round

The daemon audits a payload past a run's first step when its state is beside
it:

1. It reads `<dump>.state` and checks its digest against the payload's
   commitment.
2. It builds a replay trainer of the run's model, loads the state into it
   (`Trainer::load`, which takes weights and a buffer across dtypes, #48), and
   derives the previous round's values from the buffer and the node's round
   `h - 1` payload, which every node holds.
3. It replays round `h` on the node's batch, and judges the payload against
   the replay as it judges a first step: by the run's band, on the distance
   or, for 1-bit runs, on churn and flips (#49).

`Coordinator::is_replayable` then no longer limits what is judged; the state's
presence does.

## Continuity

A revealed state has to be the one the node actually reached. Otherwise a
node could hide a poisoned update in its buffer, and send results honest with
respect to it.

- **Where a class repeats itself, it is checked.** On the CPU, and wherever a
  verifier matches the node's device, dtype, micro-batch and parallelism, a
  replay of round `h` from `S_h` is exact. The verifier then derives `S_{h+1}`
  from it and the public results, and compares its digest with the one the
  node committed to at `h + 1`. A mismatch convicts the node of a forged state.
- **Across classes it is not.** The next state cannot be matched by hash
  there, and no bound is shown to hold (#48). A verifier of another class
  checks the round, not the transition.
- **Which GPU classes repeat themselves is not measured.** The M1's GPU does
  not across processes (`docs/ERROR_FEEDBACK.md`). The two-machine
  measurement on the CUDA classes a run admits needs those machines, and is
  left to the operators. Until it is made, a GPU class is checked by the band,
  as a class that does not repeat itself.

## A node that does not reveal

The chain cannot see that a node refused to reveal, and a rule here must not
let verifiers claim a non-reveal falsely.

- **Not revealing costs what fraud costs.** It is convicted at the slashing
  rate. A cheaper penalty would make refusing to reveal the way out of every
  audit.
- **Verifiers vote it as they vote fraud.** It needs a quorum of the round's
  committee, within the reveal window. The verdict records that the vote is
  for a missing state, not for a replay.
- **The target answers by revealing.** It challenges the verdict and serves
  the state to the bench. A bench that fetches the state within its own
  window overturns the verdict, and the state then goes through the ordinary
  audit, so a cheater gains nothing by revealing late.
- **An overturned non-reveal costs its voters nothing.** A non-reveal cannot
  be proven either way, and a target could otherwise withhold from the
  verifiers and serve the bench, to charge them. The losing-verifier crank
  passes those votes, as it passes a dissent.

So a false non-reveal claim needs a colluding quorum, as a false fraud vote
does, and it fails against a target that serves its state on appeal.

## What is built, PR by PR

1. **This design** (docs).
2. **The state commitment:** v3, the digest taken in the trainer thread, the
   payload's `state_digest`, the treasurer's evidence, and the cost measured.
3. **Retention and serving:** the state store, the audit window, and serving
   a state by its digest.
4. **The verifier:** the reveal fetch, and the daemon's replay of a later
   round from the revealed state, with continuity on classes that repeat
   themselves. Tests run real clients on the CPU, at a round past the first:
   an honest node cleared, a forged result convicted, a forged state caught.
5. **A node that does not reveal:** the verdict's kind, the reveal window,
   and the crank passing an overturned non-reveal. Then `ERROR_FEEDBACK.md`,
   the README and `FIRST_RUN.md` say later rounds are checked, and #89 and #48
   close.
