# The first public run

What has to be decided before anyone can be invited, and what each choice is
forced by. Every number here is either measured, derived from
`sim/leviathan_sim/committee.py` (wienerlabs/leviathan `docs/COMMITTEE_ECONOMICS.md`),
or marked as still open.

The run this describes is not the devnet test runs. It is the one that gets a
public invitation, a finish line and a loss curve anyone can watch.

## 0. The gate that has to move first

**An outsider can join a run only if its join authority has opened it. The
protocol already supports opening one; whether the first public run should be
open is the decision, not whether it can be.**

`join_run` accepts an `Authorization` when `is_valid_for(join_authority, node,
CoordinatorJoinRun)` holds (`solana-authorizer/src/state/authorization.rs`): the
grantor is the run's join authority, the scope matches, `active == true`, and the
node is the grantee, one of its delegates, or **the grantee is the default pubkey**
(`11111111111111111111111111111111`), which admits every signer.
`authorization_create` accepts that grantee and creates it inactive, so opening
costs the join authority two signatures once, not two per node.
`run-manager join-authorization-create --authorizer 11111111111111111111111111111111`
sends the create and the activating `authorization_grantor_update` in one
command, and `scripts/create-permissionless-run.sh` does exactly that. The
client looks up that same grant when `--authorizer` is left unset
(`find_join_authorization` in `solana-common/src/backend.rs` defaults to the
system program id, the default pubkey), so a node joins an open run with no extra
flag and no human in the loop.

The grant is per join authority, not per run: the account is seeded by grantor,
grantee and scope, with no run in it. Opening one run opens every run that shares
its join authority, so a public run needs a join authority key of its own.

So a "generate a keypair, airdrop, bond, join" one-liner can be written today
against an open run. What blocks it is policy, not code, and this is the week's
real decision:

| Option | What it costs | What it buys | Risk |
|---|---|---|---|
| **A. Admission service** | ~150 lines, a host, the join authority key online | Join works today, and every admission passes a policy (rate, allowlist, bond) | The key is online. Compromise means an attacker admits nodes, not that it steals bonds. Each admission can grow by delegation, see below |
| **B. Pre-granted batch** | An afternoon | Nothing to build | Does not scale, and handing out keys is worse than an online service |
| **C. Open run** | One `join-authorization-create` with the default grantee. No protocol change, no redeploy | Genuinely permissionless, today | Removes the only sybil gate the network has. Every committee is priced in the fraction of identities an attacker holds |

**Recommendation: A.** It is honest to say "the join authority admits anyone who
asks, automatically, for now" and it makes the three-minute claim true. C is the
right end state, and it is a switch rather than a build, which is exactly why it
must not be flipped before leviathan#4: open join plus bonds off means an attacker
mints identities for free and the committee sampling argument collapses. Note
that option A does not weaken anything: the join authority already has exactly
this power, the service only automates its policy.

Delegation cuts the other way. `authorization_grantee_update` is signed by the
grantee, not the grantor, and every key it adds passes `is_valid_for`, so each
admitted node can add up to 64 delegates of its own (`MAX_DELEGATES`, the cap
from wienerlabs/leviathan#15 finding 19). One admission is up to 65 identities,
and with bonds off they cost nothing. An admission service that meters
identities has to count delegates as well, or price every grant as 65.

A run that is its own join authority (the treasurer `Run` PDA) does not have
this problem. Its nodes join through the treasurer's `participant_join_run`,
which pins each join to the signer's own authorization and checks that signer's
own committed bond, so every identity posts a bond of its own and delegates do
not ride on one (wienerlabs/leviathan-net#34). The delegation arithmetic above
applies to runs admitted by an outside join authority.

## 1. Hardware envelope

Measured on Nosana, from `docs/NOSANA.md`:

| Class | VRAM | Price | Availability |
|---|---|---|---|
| RTX 3060 / 3060 Ti | 8 to 12 GB | $0.048/hr | open |
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

### The band: a band on churn and flips, still to be calibrated per class

| Class | dtype | samples | drift max | safety | band | vs default 0.05 |
|---|---|---|---|---|---|---|
| RTX 3060 Ti | bf16 | 32 | 0.005804 | 5.0 | **0.029019** | comfortably under |

That row was measured on unquantized values at a constant learning rate. Every
Leviathan run sends 1-bit signs, and over signs a replay differs by
coordinates picked differently (churn) or signs turned (flips), not by a
small, smooth distance.
- **The verifier daemon's band.** For a 1-bit run the daemon is held to
  `--churn-band` and `--flip-band`, and refuses a band on the distance unless
  it only watches.
- **Where the bands come from.** `calibrate-band` reports them, per pair of node
  class and verifier class (`BAND_CALIBRATION.md`, wienerlabs/leviathan-net#49).
- **What is measured so far.** The CPU and the M1's GPU:
  - In a round's first step and over the first rounds, honest single-step
    replays across classes churn 1.5–4% and almost never flip.
  - Later rounds with an fp32 verifier reach 8–14% churn.
- **What is not.** The GPU classes nodes rent (3060 Ti, 3090) have to be
  calibrated this way before the run opens.

Two cautions already recorded in `BAND_CALIBRATION.md` and worth repeating: CPU
bf16 is not a stand-in for accelerator bf16 (it spikes past 0.2 when rounding
flips which coefficients top-k keeps), and the band is only meaningful relative
to the reference config, which is currently fp32 on CPU.

### Audit rate and bond are one decision, not two

From `assess_security` in the indexer:

```
break_even_penalty    = reward * (1 - p) / p
effective_penalty     = min(slash_when_caught, bond)
expected_fraud_value  = (1 - p) * reward - p * effective_penalty
```

The indexer reads every input off chain: `reward` is a client's share of the
epoch's earnings rate, the most one cheat can take from an epoch;
`slash_when_caught` is the epoch's slashing rate; `bond` is the run's bond
minimum. At `verification_percent = 10`, deterrence alone needs a penalty of
nine times the reward. But deterrence is not the binding constraint. From the
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

### The collusion column means something narrower than it looks

`collusion_capital_usd(quorum, bond) = quorum * bond` in the sim, so the $21 is
the cost of *bonding* two identities. It is not the cost of *controlling* a
committee, because the sim does not model the draw: seats come from
`CommitteeSelection` over `Round.random_seed`, so an attacker cannot choose which
of its identities is seated. What it actually needs is a fraction of the whole
pool, and both that fraction and the bond scale with pool size.

At `verification_percent = 10`, quorum `max(2, ceil(2v/3))`, and bond at the
sim's $5.275 per quorum seat:

| Pool | Seats | Quorum | Bond | Sim figure | Identities for a quorum in half the rounds | Real cost | 20% attacker, per round | Rounds until it lands one |
|---|---|---|---|---|---|---|---|---|
| 30 | 3 | 2 | $10.55 | $21 | 15 (50%) | **$158** | 9.4% | **11** |
| 50 | 5 | 4 | $21.10 | $84 | 35 (70%) | $739 | 0.41% | 245 |
| 100 | 10 | 7 | $36.93 | $258 | 65 (65%) | $2,400 | 0.04% | 2,553 |
| 200 | 20 | 14 | $73.85 | $1,034 | 135 (68%) | $9,970 | ~0 | 3.1M |
| 500 | 50 | 34 | $179.35 | $6,098 | 335 (67%) | **$60,082** | ~0 | >10⁹ |

Three things follow, and they matter more than the headline number.

**Pool size is a security parameter, and it is not in the sim.** The fraction an
attacker needs stays roughly constant at two thirds, but the absolute cost rises
380x from a pool of 30 to a pool of 500, because the pool and the bond both grow.

**Reliable control is the wrong threat model. One lucky round is enough.** An
attacker holding a fifth of a 30-node pool draws a quorum every eleven rounds.
The same attacker in a 100-node pool waits two and a half thousand. The exposure
is concentrated almost entirely in small pools and falls off a cliff: between 30
and 100 nodes the per-round risk drops by a factor of 234. **The first few dozen
participants are the vulnerable period, not the mechanism.**

**The appeals court is the backstop the sim ignores, and it is partial.** A
colluding quorum that convicts an innocent can be challenged, and if the bench
overturns, the colluding verifiers forfeit. So a lucky round is not a free win.
But tie-breakers are drawn from the same shuffle, so an attacker large enough to
hold the verifier quorum reliably is also large enough to hold the bench. The
appeal defends against a small attacker that got lucky. It does not defend
against a majority.

Say all of this rather than the $21 alone. Unqualified, that number reads as a
fatal flaw and understates the real attack cost by an order of magnitude; with
the sampling, it is a bounded and manageable parameter regime whose one genuine
danger is launching with too few participants.

One correction to the framing: the bond is a **floor**, not a ceiling. The
mechanism has `bond_minimum_amount` and no maximum. Sizing it is choosing how
much a cheat costs, and the answer is the larger of the two floors above.

### The final round of an epoch has a measured audit window

A fraud in an epoch's final round has to be convicted while that round's
committee is still in force and an ejection is still charged. That is through
the round's witness phase and the epoch's cooldown: `round_witness_time +
cooldown_time` (wienerlabs/leviathan-net#40). Two things used to cut that
window short:

- **One client leaving the final round shut the cooldown.** The end of the
  round compacted the roster, so a committee seat read by roster index no
  longer matched the round's draw, and every vote in the cooldown was refused.
  The round that ends an epoch now keeps its roster as drawn until the
  cooldown ends. The cooldown's end then compacts it, at the same height,
  before the epoch is charged.
- **A target the coordinator could not eject undid the vote that convicted
  it.** This happens when the target was withdrawn because the run paused or
  lost its witnesses, or was already ejected by another conviction. Such a
  conviction is now recorded as due and settled against the bond.

The window closes at the end of the cooldown:

- The epoch is charged.
- A target whose withdrawal delay has run out may leave.
- The next warmup replaces the roster.
- The next epoch refuses a commitment from this one.

Until the warmup, a vote may still land if nobody left, and is recorded as
due. Nothing guarantees it reaches a target that has already left, so it does
not count toward the window.

Carrying a dispute into the next epoch was weighed and set aside. The
coordinator keeps neither the ended epoch's first step nor its end time once
the next warmup starts. A cheater can also join each epoch with a new p2p key,
so its old signature no longer checks against the roster.

**The decision.**

- **The window is at least twice the slowest verifier's worst audit latency.**
  A verifier daemon polls every `--poll-secs`. A payload that lands just after
  a pass began waits out that pass, the interval, and the next pass, so its
  vote can come as late as `poll + 2 × pass`. A pass covers preparing the
  replay, judging and submitting.
- **Measured here, as a lower bound:** building the replay engine and running
  a pass over the two contributions of the genesis rehearsal's 1-bit
  Nano-Llama step took 0.233, 0.238 and 0.265 s on an Apple M1 Pro CPU.
  `memnet_verifier_daemon_replay` prints it on every run.
  - In memnet the vote lands in-process.
  - The daemon binary also reads its model from disk and scans its payload
    directory.
  - A pass grows with the number of payloads it judges.
  - On a cluster, the vote's confirmation adds to the pass.
- **Rehearsal sizing:** with the default 8 s poll, the rehearsal's old window
  of 10 s (5 + 5) was short of twice the latency for any model. Its
  `cooldown_time` is now 20, a 25 s window. That covers a pass of up to
  2.25 s, which is far above the measured lower bound and leaves room for what
  it leaves out.
- **The flagship's window** is sized the same way once its model is chosen
  (section 2), from the pass time the daemon reports on that model.

**How it is held live.**

- The daemon exports `leviathan_verifier_slowest_pass_seconds` and
  `leviathan_verifier_poll_interval_seconds`.
- The indexer exports `leviathan_run_final_round_audit_window_seconds` from the
  run's config on chain.
- `LeviathanAuditWindowTooShort` fires when the window is less than twice a
  verifier's interval plus two of its slowest passes in the last hour.
- It needs both exporters up. The alerts on a down or stale indexer or verifier
  cover that.

The final round's window binds. The verifier replays every round whose state
its trainer revealed, not only a run's first step (wienerlabs/leviathan-net#48,
#89, `docs/ROUND_AUDIT.md`), and the alert says whether the run is ready for
it. The reveal needs a window of its own: for a 1B model trained in bf16 the
state is about 3.7 GiB. Set it with `run_set_reveal_window`, shorter than the
run's `round_witness_time`, and give a run with a large state a witness phase
long enough to fetch it (`docs/OPS_RUNBOOK.md`).

### Stage one is still bondless

`config/leviathan/mainnet-genesis-bondless.toml` sets `bond_minimum_amount = 0`
until leviathan#4 clears. That is the right call and it has a consequence that
must be said everywhere the run is described: **an ejected cheater forfeits its
earned rewards for the epoch and nothing else.** The audit works, the committee
works, the appeals court works, the economic argument does not. Stage two is one
`run_bond_config_update`, no redeploy, no migration.

Settled: `verification_percent = 10`, and the final round's audit window: at least twice a verifier's worst
latency, measured, with the rehearsals at 25 s. Open: committee size, the sign
band for each class pair nodes rent (wienerlabs/leviathan-net#49), the
flagship's `cooldown_time` once its model's pass is measured, and whether the
first public run is stage one (bondless) or waits for the audit.

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

1. Join policy (section 0): admission service or open run, on a join authority
   key used for this run alone. Nothing else can be invited until it is decided.
2. Model, dataset and measured VRAM (section 2). Sets which cards can join.
3. Round length and token target (section 4). Needs the model.
4. Committee size and bond stage (section 3). Needs the reward rate.
5. Indexer loss curve (section 5). Parallel, no dependency.
