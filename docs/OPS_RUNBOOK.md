# Operations runbook

How to run a Leviathan run and what to watch. This is the devnet operator's
guide; the same shape carries to mainnet once the audit and legal gates
(leviathan#4, leviathan#5) clear.

## What is live

- Three programs on devnet: coordinator, authorizer, treasurer.
- A flagship run (`leviathan-devnet`) with a funded, joinable training run.
- The treasurer carries the bonded committee vote and the multi-verifier bounty
  split.

Program ids and the flagship coordinator account are in [DEVNET.md](../DEVNET.md).

## Daily operation

### Keep the run healthy

The coordinator advances on ticks. Training nodes tick as they participate; a
lightweight ticker is enough to keep an idle run cycling. Watch for:

- `run_state` stuck outside `WaitingForMembers` with no active clients. Restart a
  node or a ticker.
- `active_clients` at zero for several epochs. Nobody is training.

Read state without a wallet:

```
run-manager json-dump-run --run-id leviathan-devnet --rpc <rpc>
```

### Read a run without a node

The dashboard no longer consumes a published file. It decodes the coordinator
and treasurer accounts in the browser, so there is nothing to refresh and no
cross-repo write token to keep alive. The hourly `publish telemetry` action that
used to do this is gone.

`leviathan-indexer` still prints a run as JSON when you want one on the command
line rather than in a browser. It finds the run's accounts from its id, and for
a treasurer-managed run computes the economic verdict from the run's own rates
and bond:

```
OUT=telemetry.json ./scripts/publish-telemetry.sh --run-id leviathan-devnet --rpc <rpc>
```

### Publish the fleet summary

Hardware is not on chain, so the dashboard's fleet panel is fed out of band.
Each node reports itself, and the operator folds the reports into counts:

```
leviathan-fleet report --out node-01.json
leviathan-fleet summarize node-*.json --out fleet.json
```

Publish `fleet.json` to the site and point `VITE_FLEET_URL` at it. Only the
summary is published, never the per-node reports: `random_seed` is public, so
the verifier seats for a round are already computable by anyone, and naming the
machine behind each identity turns that into a target list.

### A dedicated RPC is not optional

The public devnet endpoint rate-limits under a single node's transaction load,
which stalls joins and ticks. Configure a dedicated endpoint once in
`~/.leviathan/env` (see [DEVNET.md](../DEVNET.md)); the node script, the daemon
(`SOLANA_RPC_URL`) and run-manager (`RPC`) all read it.

## Checking the verifier before you trust it

Run this before a genesis run, after any change to the replay path, and whenever
a conviction looks wrong:

```
./scripts/replay-smoke.sh
```

It builds a dataset sized for the model, produces one honest contribution,
audits it with no honest reference dumps in the loop, then forges that same
contribution and audits again. Passing means the verifier clears honest work and
convicts a forgery using a reference it recomputed itself. Failing means do not
run a bonded run: either the verifier convicts honest nodes or it cannot see a
cheat, and both cost real money.

Two failure modes it exists to catch, both of which look like success from the
outside. If the replay is configured with a zero learning rate, every recomputed
delta is identically zero, so a forgery of zero also compares equal and the audit
silently passes everything. And if the dataset was tokenized for a different
vocabulary than the model, the replay cannot run at all, which shows up as
skipped contributions rather than a loud error.

## Signals to watch

The kill switches are Prometheus alerts over what `leviathan-indexer
--metrics-listen` reads off chain and what `leviathan-verifier-daemon
--metrics-listen` judged. `telemetry/README.md` says how to stand them up
(leviathan-net#57).

| Alert | Meaning | Action |
|---|---|---|
| `LeviathanEconomicInsecurity` | at the run's own epoch rates and bond minimum, a cheat makes money in expectation: the penalty is below break-even at the audit probability, or a round seats too few verifiers to convict anyone | raise the slashing rate and the bond, or the audit percent, or lower the earning rate; grow the roster if the committee is the cause |
| `LeviathanTreasuryDrain` | the vault holds less than the bonds posted into it, which no instruction the treasurer allows can cause | halt the run and treat it as an exploit |
| `LeviathanTreasuryUnderfunded` | the vault covers the bonds but not the earnings still to be claimed | top up the run's rewards |
| `LeviathanMeshPartition` | less than half of the epoch's roster is still healthy while it trains | check the relay and the nodes' connectivity; `leviathan_run_epoch_clients` says whether they dropped, withdrew or were ejected |
| `LeviathanHonestFalsePositive` | the appeals court overturned a conviction: a quorum convicted an innocent target | see "Suspected wrongful conviction" below |
| `LeviathanDisputeOutcomeUnseen` | a conviction's verdict account was handed to a new dispute before the indexer saw how its appeal ended | read the verdict account's transaction history; an overturn there is a wrongful conviction |
| `LeviathanUncaughtFraudSuspect` | a bonded verifier voted fraud, with the target's signed commitment, and the quorum never came. This includes a dispute whose voters withdrew their votes during its round, for instance with `bond-withdraw-finalize --withdraw-live-votes`: a fraud claim that was retracted is still one that never became a conviction | replay the batch the verdict account names: either a cheat escaped or a verifier accused an honest node |
| `LeviathanFraudNotSubmitted` | a verifier holds a fraud proof the chain does not hold: refused for a reason other than already having it, or never received | read that verifier's log for the refused submission; the daemon retries every pass |
| `LeviathanAuditWindowTooShort` | the final round's audit window, `round_witness_time + cooldown_time`, is less than twice a verifier's worst latency (its polling interval plus two of its slowest passes), so a fraud in an epoch's final round may outlast it | raise `cooldown_time` through `run-manager update-config`, poll more often, or move the verifier to faster hardware; see `docs/FIRST_RUN.md` section 3 |
| `LeviathanCommitteeCannotConvict` | verification is on, but a round seats fewer verifiers than the treasurer takes a verdict from | grow the roster or raise `verification_percent`; until then only the run authority can convict |
| `LeviathanIndexerLag`, `LeviathanIndexerStale`, `LeviathanIndexerNotReading`, `LeviathanIndexerDown`, `LeviathanVerifierDown`, `LeviathanVerifierFailing` | the alerts above are reading old state or none, or a verifier audits nothing | fix the RPC, the indexer or the verifier before trusting anything else |

A signed payload that does not fit the run's model is fraud. It may not decode,
not hold together as an encoding, or not have the honest result's length. The
verifier convicts it at an infinite distance like any other fraud, and counts
it in `leviathan_verifier_fraud_found_total` (wienerlabs/leviathan-net#59). The
verifier checks a payload's size and encoding before it decompresses it, so no
payload can take it down. A payload or reference it cannot read at all is its
own trouble: it counts in `leviathan_verifier_contributions_unreplayable_total`
and is looked at again on the next pass.

The length rule, like the band, trusts the verifier's reference to be the
run's own model. A verifier replaying the wrong model convicts honest nodes
whether their lengths disagree or their values do. If one verifier's fraud
count jumps across every trainer at once, stop it and check its model before
its votes reach a quorum. `LeviathanHonestFalsePositive` fires on any
conviction the appeals court overturns.

`leviathan_verifier_contributions_malformed_total` counts equivocation records
that prove nothing. A node writes one only after checking it, so one that
fails was damaged or tampered with on its disk. The verifier sets each aside
and goes on with the rest.

A verifier counts the peers it caught signing two different payloads for one
batch (`leviathan_verifier_equivocations_found_total`). Each is a proof the
chain takes on its own, and the verifier submits it whatever it is assigned
(wienerlabs/leviathan-net#46). It does so in every pass, including a
replaying verifier's passes past the run's first epoch, where nothing can be
replayed, and passes whose replay could not be prepared. The daemon refuses to
start with neither `--reference-dir` nor `--replay-data-dir`. The submission is counted like a fraud proof's,
so `LeviathanFraudNotSubmitted` covers it as well. A proof the epoch can no
longer charge, because the epoch ended or a pause withdrew the peer before it
landed, fires that alert once, and the verifier's log names the peer. Records
from epochs that are over are set aside without an alert.

## Bond and slash economics

The bond has two floors. The deterrence floor makes cheating expected negative:
`bond = reward * (1 - p) / p`. The verifier-sustainability floor makes auditing
profitable: `bond = audit_cost * quorum / (fraud_rate * bounty_bps / 10000)`. The
run must require the larger of the two. At genesis scale the verifier floor
dominates. See COMMITTEE_ECONOMICS.md in the private repo.

Practical consequence: do not set the bond from the whitepaper break-even figure
alone. Size it from the committee table for the committee size you are running.

Set the equivocation bounty apart from `slash_bounty_bps`, with
`run_set_equivocation_bounty`. An equivocation proof is permissionless, and an
equivocator can report itself from a second wallet, which the chain cannot tell
from an honest reporter. Equivocating then costs it
`slashing_rate * (1 - bounty_bps / 10000)`. The chain refuses a share of the
whole forfeit, and holds the share to `slash_bounty_bps` at the charge, so that
cost is never nothing; a run that set none gives an equivocation's reporter
nothing. Copying an honest reporter's proof no longer works: a reporter commits
to its proof a slot before it reveals it (`docs/DISPUTE_DESIGN.md`,
wienerlabs/leviathan-net#72, #92).

Set a reveal window with `run_set_reveal_window` once a run audits later
rounds (`docs/ROUND_AUDIT.md`, wienerlabs/leviathan-net#89). It is how long a
trainer has, into a round's witness phase, to reveal the state a payload
commits to before verifiers may vote that it did not, sized as
`state_bytes / reveal_rate + slack`. It has to be shorter than the run's
`round_witness_time`, which the chain checks when it is set: give a run with a
large state a witness phase long enough. A run that set none takes no
non-reveal vote, and an overturned non-reveal costs its voters nothing.

## Incident response

### A node cannot join

- Confirm the run is joinable: `run-manager can-join` / `list-runs`.
- Confirm a valid join authorization exists. A fresh run needs one:
  `run-manager join-authorization-create --authorizer 11111111111111111111111111111111`.
- If joins time out, the RPC is the usual cause. Raise
  `LEVIATHAN_JOIN_TIMEOUT_SECS` and use a dedicated RPC.

### A slash did not fire

- Under the committee vote a single verdict does not convict; a quorum does.
  Confirm enough bonded verifiers voted (`run-manager bond-status` per verifier).
- The target must be in the current epoch roster and Healthy or Dropped. A target
  already Withdrawn or Ejected cannot be slashed again.

### Suspected wrongful conviction

A target that believes it was convicted wrongly opens a challenge, and a
separately drawn tie-breaker bench re-audits. An overturn spares the target and
makes the convicting verifiers forfeit their bonds once the losing-verifier
crank reaches them (`docs/DISPUTE_DESIGN.md`). `LeviathanHonestFalsePositive`
fires on every overturn. Crank the losing verifiers, then pull the verdict
account (it names the batch and holds what the target signed), replay that batch
off chain, and find out why the quorum disagreed: a band too tight for the
hardware, a verifier on the wrong data, or a colluding committee.

### Redeploy a program

In place, keeping the id, with the upgrade authority:

```
solana program deploy \
  --program-id <program-keypair> --upgrade-authority <authority> \
  -k <fee-payer> --url <rpc> <program.so>
```

Build the program first with `cargo build-sbf --manifest-path <program>/Cargo.toml`
(not `anchor build`, which expects an Anchor workspace). Check the stack warnings:
a frame-size warning on a new instruction means it may not fit the BPF stack.
Never re-run the create-run deploy path, which mints a new id.

Check `CoordinatorAccount::VERSION` before upgrading in place. A coordinator
program refuses accounts of any other version, and the treasurer's claim and
withdrawal check it too, so upgrading a program whose accounts are an older
version leaves every run on it unable to pay out or release a bond, until a
migration instruction exists. Versions 2 and 3 (wienerlabs/leviathan-net#60,
#62) have none: deploy them under new ids, as `DEVNET.md` describes, where no
older account exists.

The payload commitment is a wire format too, and its change in
wienerlabs/leviathan-net#67 is a flag day.
- **What changed.** A payload now carries the epoch it was trained in, and
  the commitment hash covers it under a new domain (`v2`). A kept commitment
  record grew from 120 to 128 bytes, and a dump's name carries the epoch.
- **Who has to upgrade at once.** Every node, verifier and treasurer of a run
  moves together. A node on the old build and one on the new drop each
  other's payloads. A verifier on the new build reads no record the old one
  kept, and the treasurer refuses an old signature.
- **How.** Bring a run up on the new build from its start. Do not upgrade it
  mid-run.

wienerlabs/leviathan-net#89 is a flag day of the same kind. The hash a node
signs moves to `v3` and binds the digest of the state the node trained the
payload from: its weights and its error-feedback buffer as the round began
(`docs/ROUND_AUDIT.md`). A payload carries that digest, a kept commitment
record grows from 128 to 161 bytes, and the treasurer's evidence carries it
too. Every node, verifier and treasurer of a run moves together, from the
run's start.

The treasurer's own accounts grow the same way:

| Change | Account | Size (bytes) |
|---|---|---|
| wienerlabs/leviathan-net#39: gained `open_verdict_votes` | `Participant` | 69 |
| wienerlabs/leviathan-net#39: gained `released_voters`, `bench_bounty_owed` and `bench_paid` | `AuditVerdict` | 4392 |
| wienerlabs/leviathan-net#67: evidence gained the epoch it was trained in | `AuditVerdict` | 4400 |
| wienerlabs/leviathan-net#64: bounty fields renamed `bounty_owed` and `bounty_paid`; gained `charge`, `charge_epoch_start`, `charge_rate` and `appeal_upheld` | `AuditVerdict` | 4418 |
| Votes counted per signed payload: gained `vote_commitments`, the commitment hash each vote convicts on | `AuditVerdict` | 6470 |
| wienerlabs/leviathan-net#72: gained `uncharged_slash_reports` | `Participant` | 73 |
| wienerlabs/leviathan-net#72: new, one per slash no verdict accounts for | `SlashReport` | 134 |
| wienerlabs/leviathan-net#89: evidence gained the state the target trained from (`state_committed`, `state_digest`) | `AuditVerdict` | 6503 |
| wienerlabs/leviathan-net#92: new, one per proof of equivocation a reporter committed to and has not revealed | `EquivocationCommitment` | 113 |
| wienerlabs/leviathan-net#92: new, one per run that set an equivocation bounty | `EquivocationBounty` | 43 |
| wienerlabs/leviathan-net#89: gained `kind`, whether the votes convict on a replay or on a state not revealed | `AuditVerdict` | 6504 |
| wienerlabs/leviathan-net#89: new, one per run that set a reveal window | `RevealWindow` | 49 |

An older, shorter account does not deserialize with the new layout, so the treasurer that carries them deploys under a new id too.

#64 also changes what a withdrawal pays.

- **A conviction's bounty.**
  - It is no longer paid at the convicted target's withdrawal. It is recorded on the verdict when the target's bond is charged.
  - That charge is made either by `run_charge_conviction` once the epoch has been charged, or by the withdrawal itself.
  - A conviction recorded as `Due` is charged by `run_charge_conviction`'s counterpart, `run_charge_due_conviction`, once the target is on no roster an epoch will still charge, or by the withdrawal. It used to wait for the target to come back or withdraw (wienerlabs/leviathan-net#74).
  - Each recipient's bond is then credited through `run_pay_verdict_bounty`.
  - The verifier daemon does all of it, for every open dispute in the run, at the end of each pass, after its own judging, so a backlog never costs it a vote. A daemon in dry-run mode sends nothing.
  - `run-manager bond-withdraw-finalize` does both for every seat the wallet is owed before it withdraws, whether or not the wallet still has a vote open.
- **A slash no verdict accounts for.** Since wienerlabs/leviathan-net#72, the withdrawal no longer names who is paid for it. A `run_slash` or an equivocation records its reporter on a `SlashReport` when it ejects its target:
  - A `run_slash` takes the target and an optional reporter. `run_slash_instruction` and `process_treasurer_run_slash` in the tooling take both.
  - An equivocation's reporter is whoever sent the proof, after committing to it a slot before (wienerlabs/leviathan-net#92). Its bounty is the run's equivocation share, not `slash_bounty_bps`.
  - Once the epoch is charged, `run_charge_slash_report` charges the target's bond, and `run_pay_slash_report_bounty` credits the reporter's bond.
  - The verifier daemon sends both for every open report at the end of each pass. It leaves a bounty whose reporter has no participant account yet for a later pass. A daemon in dry-run mode sends nothing.
  - `bond-withdraw-finalize` sends both for the wallet's own reports and for those it reported, before it withdraws. A step the daemon took first is skipped. It has no `--reporter` option any more.
  - A bond with a report not yet charged is held (`SlashReportsUncharged`).

The appeals bench size now reaches the coordinator (wienerlabs/leviathan#15,
finding 4). `run_set_challenge_config` names the coordinator's accounts and
calls its new `set_tie_breaker_nodes`, which reserves the seats in every round
from the next one to start. Upgrade the coordinator program before the
treasurer: a treasurer on the new build cannot set a bench on a coordinator
without the instruction. Neither account changes size or moves a field (the
count sits in what was padding before `seed_commit_slot`, and an older account
reads zero there), so both upgrade in place. A run that already has a bench has
to send `run_set_challenge_config` once more after the upgrade, or its rounds
keep reserving none. The call is refused while a round's seed is committed
(`RoundSeedCommitted`); send it again a slot later.

## Going to mainnet

Mainnet arrives in two stages, because the audit gates money and nothing else.

**Stage one, bonds off.** Deploy the programs and run a real training run with
`bond_minimum_amount` at zero (`config/leviathan/mainnet-genesis-bondless.toml`).
Everything works: verifiers are drawn, contributions are replay audited, a
committee convicts, an ejected client forfeits its epoch rewards, and the
appeals court can reverse a wrongful conviction. What is absent is a bond, so a
bug in the slashing path cannot take anyone's principal. Describe the run this
way in public. A cheater still loses its earnings, so cheating is not free, but
the economic security argument in `docs/COMMITTEE_ECONOMICS.md` does not hold
until bonds are on.

**Stage two, bonds on.** After the external audit clears (leviathan#4), a single
`run_bond_config_update` sets a real bond and a real withdraw delay. No
redeploy, no migration, no new program ids.

### Deploying

The programs carry a different id on mainnet, selected at build time by the
`mainnet` Cargo feature rather than by editing source before a deploy. The
feature propagates to every program a program calls, so a mainnet treasurer can
never CPI into a devnet coordinator. `cargo test -p psyche-solana-treasurer
--test program_id --features mainnet` asserts exactly that, and the deploy
script refuses to run if it fails.

Rehearse first. This deploys the real mainnet binaries, with the real mainnet
ids, to devnet:

```
CLUSTER=devnet ./scripts/deploy-mainnet.sh
```

Then the real thing, which asks for a typed confirmation:

```
CLUSTER=mainnet ./scripts/deploy-mainnet.sh
```

Both hand upgrade authority to the multisig at the end and verify on chain that
it landed. After this, upgrading a program takes a multisig proposal, which is
the point: a single compromised key cannot replace the code holding the bonds.

Program keypairs live outside the repo, in
`~/.config/solana/leviathan-mainnet-keys`. The devnet keys ended up in git
history, which was acceptable for devnet and is not for mainnet. Back these up
somewhere that survives the loss of this machine, because losing them means the
programs can never be upgraded again.

### Before the first mainnet run

- `./scripts/replay-smoke.sh` passes. A verifier that cannot tell honest work
  from a forgery is worse than no verifier.
- `verification_percent` is above zero. At zero no verifier is ever drawn and
  the run is unverified regardless of everything else.
- The dataset and the model share a vocabulary. A mismatch turns every audit
  into a silent skip rather than a loud error.
- The learning rate is real. A zero rate makes every delta identically zero, so
  the audit passes everything including a forgery.

## What is not yet operational

- An always-on Prometheus with on-call routing for the kill switches. The
  exporters, the rules and the scrape targets exist; nothing pages anyone yet.
- Multi-node relay for a large swarm (leviathan-net#9).
