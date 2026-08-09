# The first public run

What has to be decided before anyone can be invited, and what each choice is
forced by. Every number here is either measured, derived from
`sim/leviathan_sim/committee.py` (wienerlabs/leviathan `docs/COMMITTEE_ECONOMICS.md`),
or marked as still open.

The run this describes is not the devnet test runs. It is the one that gets a
public invitation, a finish line and a loss curve anyone can watch.

## 0. The gate that has to move first

**An outsider cannot join today, and no amount of documentation fixes it.**

`join_run` requires an `Authorization` account with `grantor == join_authority`,
`grantee == the node`, scope `CoordinatorJoinRun`, and `active == true`. Creating
one takes `pub grantor: Signer` — the join authority has to sign, per node. There
is no sentinel, no open-join flag, no self-service path. `authorization_grantor_update`
then has to flip `active` to true, which is a second signature.

So a "generate a keypair, airdrop, bond, join" one-liner cannot be written: step
four blocks on a human with the join authority key. Three options, and this is
the week's real decision:

| Option | What it costs | What it buys | Risk |
|---|---|---|---|
| **A. Admission service** | ~150 lines, a host, the join authority key online | Join works today, no protocol change, no new audit surface | The key is online. Compromise means an attacker admits nodes, not that it steals bonds |
| **B. Pre-granted batch** | An afternoon | Nothing to build | Does not scale, and handing out keys is worse than an online service |
| **C. Open-join mode** | Protocol change, program redeploy, audit | Genuinely permissionless | Removes the only sybil gate the network has. Every committee is priced in the fraction of identities an attacker holds |

**Recommendation: A.** It is honest to say "the join authority admits anyone who
asks, automatically, for now" and it makes the three-minute claim true. C is the
right end state but it must not ship before leviathan#4, because open join plus
bonds off means an attacker mints identities for free and the committee sampling
argument collapses. Note that option A does not weaken anything: the join
authority already has exactly this power, the service only automates its policy.

Delegation does not rescue this. One `Authorization` carries up to 64 delegates
(capped by wienerlabs/leviathan#15 finding 19), but adding a delegate is still a
grantor signature.

## 1. Hardware envelope

Measured on Nosana, from `docs/NOSANA.md`:

| Class | VRAM | Price | Availability |
|---|---|---|---|
| RTX 3060 / 3060 Ti | 8–12 GB | $0.048/hr | open |
| RTX 3090 | 24 GB | ~$0.19/hr | 23 of 59 hosts free |
| A100 / H100 / A40 / A6000 / 6000 Ada | 40 GB+ | n/a | **no free hosts at all** |

Availability binds harder than price. Anything that needs a datacentre card
cannot be reserved, so the model has to fit the 3060 or 3090 class or the run
cannot be filled.

This is also why the model is chosen for the cards rather than the cards for the
model: a run nobody can join is not a run.

## 2. Model and dataset

Not yet chosen. What constrains the choice:

- **Architecture** must be one the client already builds: `HfLlama`, `HfDeepseek`,
  `HfAuto` or `Torchtitan`.
- **The dataset's vocabulary must match the model's.** This is the trap the
  genesis config already warns about: Nano-Llama has 30 tokens, so a corpus
  tokenized for a larger vocabulary indexes past the embedding table and *the
  replay verifier cannot recompute anything*. A vocab mismatch does not degrade
  the run, it disables verification.
- **Data location** is `LLMTrainingDataLocation::Http` with a token size and a
  URL. A public dataset has to be pre-tokenized and hosted somewhere with enough
  bandwidth for every node to stream it.
- **VRAM** has to hold weights, gradients and the DisTrO accumulator at once, plus
  activations. The rule of thumb is not good enough here: measure it on a rented
  3060 before committing, the same way the band was measured. `calibrate-band`
  already runs the production path on real hardware and is the natural place to
  add a peak-memory report.

Candidates worth pricing against the 8 GB and 24 GB classes, in that order. **Do
not lock a repo id and revision into the config without pulling it first** and
confirming the tokenizer matches the hosted corpus.

Open: model, revision, dataset URL, measured peak VRAM per class.

## 3. Security parameters

### The band is measured, and it is the one parameter that is settled

| Class | dtype | samples | drift max | safety | band | vs default 0.05 |
|---|---|---|---|---|---|---|
| RTX 3060 Ti | bf16 | 32 | 0.005804 | 5.0 | **0.029019** | comfortably under |

Honest drift on the dominant class is tight (min 0.00258, mean 0.00366). The
shipped default of 0.05 does not falsely convict 3060 Ti nodes. **The 3090 class
has not been measured** and should be before the run opens, because a 24 GB class
is the fallback if the model does not fit in 8 GB.

Two cautions already recorded in `BAND_CALIBRATION.md` and worth repeating: CPU
bf16 is not a stand-in for accelerator bf16 (it spikes past 0.2 when rounding
flips which coefficients top-k keeps), and the band is only meaningful relative
to the reference config, which is currently fp32 on CPU.

### Audit rate and bond are one decision, not two

From `assess_security` in the indexer:

```
break_even_penalty        = reward_per_round * (1 - p) / p
effective_penalty         = min(slash_when_caught, bond)
expected_fraud_per_round  = (1 - p) * reward_per_round - p * effective_penalty
```

At `verification_percent = 10`, deterrence alone needs a bond of nine times the
per-round reward. But deterrence is not the binding constraint. From the
economics doc, at `p = 0.1` with a 50% bounty:

| Committee | Quorum | Deterrence bond | Verifier-sustainable bond | Required | Collusion capital |
|---|---|---|---|---|---|
| 3 | 2 | $2.91 | $10.55 | **$10.55** | $21 |
| 6 | 4 | $2.91 | $21.10 | **$21.10** | $84 |
| 9 | 6 | $2.91 | $31.65 | **$31.65** | $190 |
| 21 | 14 | $2.91 | $73.85 | **$73.85** | $1034 |

Verifier pay dominates everywhere. A verifier pays the replay cost on every audit
and is only paid when it catches someone, so a bond sized only for deterrence
leaves a rational verifier declining to audit, and a security layer nobody runs
is not a security layer.

**The collusion column is the one to read for a first run.** A three-verifier
committee can be bought for $21. That is not a reason to avoid a small committee
at genesis, it is a reason to say the number out loud rather than let someone
else discover it.

One correction to the framing: the bond is a **floor**, not a ceiling. The
mechanism has `bond_minimum_amount` and no maximum. Sizing it is choosing how
much a cheat costs, and the answer is the larger of the two floors above.

### Stage one is still bondless

`config/leviathan/mainnet-genesis-bondless.toml` sets `bond_minimum_amount = 0`
until leviathan#4 clears. That is the right call and it has a consequence that
must be said everywhere the run is described: **an ejected cheater forfeits its
earned rewards for the epoch and nothing else.** The audit works, the committee
works, the appeals court works, the economic argument does not. Stage two is one
`run_bond_config_update`, no redeploy, no migration.

Settled: `verification_percent = 10`, band `0.05` default (3060 Ti measured at
0.029). Open: committee size, and whether the first public run is stage one
(bondless) or waits for the audit.

## 4. The finish line

A run people can join needs an end. Tokens are fully determined by three numbers
already in the config:

```
total tokens = total_steps * global_batch_size * max_seq_len
```

The genesis draft is `25000 * 8 * 64` = 12.8M tokens, which is a smoke test, not
a run. Announcing a target means picking `total_steps` and the batch schedule so
that the product is a number worth watching, and then not changing it. The
schedule can warm the batch size up (`global_batch_size_start` to `_end`), so
state the target in tokens rather than steps.

Round length follows from the hardware: `max_round_train_time` has to fit the
slowest card in the swarm doing one micro-batch, or slow nodes are dropped every
round. Measure it on a 3060 before setting it.

Open: token target, batch schedule, `max_round_train_time`.

## 5. What the dashboard has to show

The panels exist and read the chain directly. What the run needs on top:

- live participant count, round number, current loss
- total bonded, audits run, **convictions**
- the finish line as a fraction of the token target

The conviction counter is the one number nobody else in this space can show, and
it is already wired: the dashboard decodes `AuditVerdict` accounts and shows the
appeal bench per dispute. Loss is the gap. `WitnessMetadata` carries
`tokens_per_sec`, `bandwidth_per_sec` and `loss` and is sent to the chain every
round as instruction data, but it is never stored in an account and never emitted
as an event, so nothing downstream reads it. Surfacing the loss curve means
indexing transaction history, which is an indexer job, not a browser job.

Open: indexer support for `WitnessMetadata`, without which there is no public
loss curve.

## Decision order

1. Join (section 0). Nothing else can be invited until this moves.
2. Model, dataset and measured VRAM (section 2). Sets which cards can join.
3. Round length and token target (section 4). Needs the model.
4. Committee size and bond stage (section 3). Needs the reward rate.
5. Indexer loss curve (section 5). Parallel, no dependency.
