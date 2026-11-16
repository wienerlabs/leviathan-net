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
| `LeviathanUncaughtFraudSuspect` | a bonded verifier voted fraud, with the target's signed commitment, and the quorum never came | replay the batch the verdict account names: either a cheat escaped or a verifier accused an honest node |
| `LeviathanFraudNotSubmitted` | a verifier holds a fraud proof the chain does not hold: refused for a reason other than already having it, or never received | read that verifier's log for the refused submission; the daemon retries every pass |
| `LeviathanCommitteeCannotConvict` | verification is on, but a round seats fewer verifiers than the treasurer takes a verdict from | grow the roster or raise `verification_percent`; until then only the run authority can convict |
| `LeviathanIndexerLag`, `LeviathanIndexerStale`, `LeviathanIndexerNotReading`, `LeviathanIndexerDown`, `LeviathanVerifierDown`, `LeviathanVerifierFailing` | the alerts above are reading old state or none, or a verifier audits nothing | fix the RPC, the indexer or the verifier before trusting anything else |

A verifier also counts signed payloads it cannot judge at all
(`leviathan_verifier_contributions_malformed_total`): they do not decode, or do
not fit the replayed model. It sets each aside and goes on with the rest, so one
such payload cannot stop it judging everyone else.

## Bond and slash economics

The bond has two floors. The deterrence floor makes cheating expected negative:
`bond = reward * (1 - p) / p`. The verifier-sustainability floor makes auditing
profitable: `bond = audit_cost * quorum / (fraud_rate * bounty_bps / 10000)`. The
run must require the larger of the two. At genesis scale the verifier floor
dominates. See COMMITTEE_ECONOMICS.md in the private repo.

Practical consequence: do not set the bond from the whitepaper break-even figure
alone. Size it from the committee table for the committee size you are running.

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
