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
bf16, on an M1 Pro, after a real step filled the buffer, as the workspace's
dev profile builds it (20 rounds each, means):

| Where the node trains | State | Copied off the device, encoded and hashed | Also written to disk | The client's import |
|---|---|---|---|---|
| CPU | 73.0 MiB | 0.044 s per round | 0.080 s per round | 0.082 s |
| M1 GPU (`mps`) | 73.0 MiB | 0.094 s per round | 0.117 s per round | 0.079 s |

That is about 0.8 GiB/s from the GPU, and writing the state as it is hashed
adds 0.02 to 0.04 s. The trainer takes the digest on its thread before the
round's forward pass, so the round waits for it. The client's import runs
beside the round (below). At the GPU's rates, a node keeping its states would
spend about 2 s a round at 350M and about 6 s at 1B, and about 1.7 s and 5 s
on the digest alone; those are extrapolated, not measured, and a CUDA node's
copy rate and disk are its own.

## Retention, and serving a state

A node started with `--round-state-dir` keeps every state it commits to
there. Without it, the node commits to each round's state but keeps none,
and cannot reveal one; it says so when it starts.

- **Where.** The trainer writes `S_h` into `rounds/`, in a file named by its
  digest, in the pass that hashes it. The encoding goes under a temporary
  name first and is renamed once whole, so a file under a digest's name holds
  that state. A state that cannot be written is logged, and still committed
  to: it is the state the round trains from.
- **Served from its own store.** The client imports the file, in place, into
  an iroh blob store of the node's own under `store/`, on disk. The store the
  run's results go through is in memory, and a state is about twice the model.
  The store hashes the file as it imports it and refuses one that is not the
  state it is named by, as a disk would leave one it damaged.
- **For how long: the epoch.** The treasurer convicts only on a commitment
  from the epoch under way (`CommitmentEvidence::is_from_current_epoch`). So
  an audit can name any round of the current epoch, and none of an earlier
  one. A state is kept until the node keeps the first state of a later epoch,
  which forgets every earlier one: its tag, so the store drops it, and its
  file. This replaces the `audit_rounds` parameter this design first had: the
  bound is already on chain, and the coordinator's config has no room for one.
- **The disk this takes:** one state per batch the node trains, for every
  round of an epoch: `epoch_time - warmup_time` divided by how long a round
  takes. At the bondless mainnet config's 300 s epochs and 75 s rounds, that
  is four rounds: about 15 GiB at 1B.
- **Across a restart.** The store is on disk, so a node that restarts still
  serves the states of the epoch under way. When it opens them, it removes
  any file it wrote and never kept, or never made whole.

The node serves a state on request, never by broadcast: everyone downloading
every state would cost the run twice the model per node per round. It serves
over iroh's blob protocol under its own ALPN, `/leviathan/round-state/1`, and
only to the peers on its roster, as it serves everything else.

## Revealing to a verifier

A verifier seated for round `h` downloads, for each payload it audits, the
state the payload's commitment names, from the payload's trainer, by the
digest. The trainer is the roster key that signed the payload, and iroh
reaches it at that key.

- **Only what it audits.** The node computes its own assignments when the
  round starts, with the audit lottery the verifier daemon and the treasurer
  draw (`select_audits_for_current_round`), and draws none while the roster
  has moved under the round's draw, when the treasurer refuses every vote. It
  fetches the state of a payload it is assigned once it keeps that payload,
  and of no other: every state of every payload would be the run's states
  once more on every verifier.
- **Asking again.** A trainer keeps a state beside its payload's broadcast,
  not before it, so the first request can find it missing. The verifier asks
  again every round witness time, the span a round's results are broadcast
  in, until the payload's epoch is over: after that the treasurer convicts
  nobody on it. A state of another length is refused at once.

- **The fetch checks itself.** iroh verifies a blob against its hash as it
  arrives, and the hash is the digest the trainer signed. A state that does
  not match cannot be served under it.
- **Its size first.** The verifier works out the length of a state of the
  run's model from its shapes and dtypes alone (`round_state_len`), and
  checks it against the size the trainer proves before anything else is
  downloaded. A trainer cannot make a verifier download more than that.
- **Where it goes.** The verifier's node writes it beside the payload dump, as
  `<dump>.state` (`revealed_state_path`), for the verifier daemon, and only
  whole. It writes it only if the commitment record beside the dump names that
  state: a payload that lost its name to another the committer signed for the
  batch is kept as an equivocation, not under that name.
- **The reveal window.** How long a trainer has before verifiers may vote
  that it did not reveal is `reveal_seconds`, sized as
  `state_bytes / reveal_rate + slack`. It is not the final round's audit
  window: a 1B state at 100 Mbit/s takes over five minutes. The coordinator's
  config has no room for it, so it is the treasurer's, beside the run, like
  the equivocation bounty: the treasurer is what takes the vote (part 5).

## Replaying a later round

The daemon audits a payload past a run's first step when its state is beside
it:

1. **The state.** It reads `<dump>.state` (`read_round_state`), whose digest
   the reveal already checked against the payload's commitment.
2. **Loading it.** It loads the state into its replay trainer (`Trainer::load`,
   which takes weights and a buffer across dtypes, #48).
3. **The previous round's values.** A node's step at round `h` first takes
   out of its buffer the values it sent at round `h - 1`. Those values are the
   buffer encoded as the node encodes it to send, read at the indices of its
   round `h - 1` payload (`Distro::results_at`). Nothing changes the buffer
   between the step that sent them and this one (`apply` touches only the
   weights), so they follow from the revealed state and the payload's public
   indices, and are never revealed.
4. **What the node did the round before.** The verifier keeps beside the dump
   what the node did the round before (`<dump>.previous`), from its own view
   of that round's draw:
   - **nothing:** it verified there, or the round opens its epoch, which every
     node starts afresh; its step takes nothing out;
   - **trained:** it trained a batch there, whose payload the verifier kept
     with every other.
5. **The replay.** It replays round `h` on the node's batch, and judges the
   payload against the replay as it judges a first step: by the run's band,
   on the distance or, for 1-bit runs, on churn and flips (#49).

`Coordinator::is_replayable` then no longer limits what is judged; the state's
presence does.

**What the verifier cannot replay is its own gap, not the node's:**
- it did not see the round before;
- the node trained more than one batch there, and only the last one's values
  are the buffer's;
- it did not keep the node's payload of that round.

The payload is skipped, not convicted.

**On the same class the replay is exact.** On the CPU, a replay from the state
a node committed to at round 3 sends exactly what the node sent there, and
commits to the same state. Three real clients, one forging every result,
check it end to end: replayed from the states the verifiers fetched, every
honest payload is cleared at a band of zero and every forged one convicted.

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
3. **Retention and serving:** the state store, kept for the epoch under way,
   serving a state by its digest, and the fetch a verifier makes.
4. **The verifier**, in three:
   - **the reveal fetch:** each verifier fetches the state of each payload
     it audits, beside its dump, checked against three real clients on the
     CPU, one of them verifying each round;
   - **the replay:** the daemon replays a later round from the revealed
     state, and judges it by the band. Tests run real clients on the CPU, at
     a round past the first: an honest node cleared, a forged result
     convicted;
   - **continuity** on classes that repeat themselves: a forged state caught.
5. **A node that does not reveal:** the verdict's kind, the reveal window,
   and the crank passing an overturned non-reveal. Then `ERROR_FEEDBACK.md`,
   the README and `FIRST_RUN.md` say later rounds are checked, and #89 and #48
   close.
