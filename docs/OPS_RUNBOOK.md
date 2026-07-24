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

### Publish telemetry

The dashboard consumes `telemetry.json`. Refresh it from the coordinator account:

```
OUT=telemetry.json ./scripts/publish-telemetry.sh \
  --coordinator-account <coordinator> --run-id leviathan-devnet --rpc <rpc> \
  --reward-per-round <r> --bond <b> --slash-when-caught <s>
```

The `publish telemetry` GitHub Action does this every 15 minutes. Point the
dashboard's `VITE_TELEMETRY_URL` at the committed file.

### A dedicated RPC is not optional

The public devnet endpoint rate-limits under a single node's transaction load,
which stalls joins and ticks. Configure a dedicated endpoint once in
`~/.leviathan/env` (see [DEVNET.md](../DEVNET.md)); the node script, the daemon
(`SOLANA_RPC_URL`) and run-manager (`RPC`) all read it.

## Signals to watch

The economic security verdict and the kill-switch signals come out of the
telemetry and the sim. The prometheus alert rules that pair with these live in
the telemetry stack (leviathan-net#12).

| Signal | Where | Meaning | Action |
|---|---|---|---|
| `economically_secure: false` | telemetry `security` | expected fraud value per round is positive | raise the bond or the audit percent, or lower the reward, until it flips |
| `audit_probability` 0 | telemetry | verification is off, no audits happen | set `verification_percent` above 0 |
| uncaught fraud > slash events | derived (fraud proofs vs slashes) | the double-sell kill switch, cheats not being convicted | halt the run, investigate the verifier path |
| convicted count climbing fast | telemetry | either real fraud or a broken verifier | sample the verdicts, confirm they are true positives |
| `total_slashed` rising with no convictions | telemetry inconsistency | accounting drift | dump the run and reconcile |

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

There is no on-chain penalty yet for verifiers who convict an innocent target
(leviathan-net#4, the losing-side penalty, is open). Until that lands, treat a
disputed conviction as a manual review: pull the verdict hashes from the slash
log, replay the target's batch off chain, and if the conviction was wrong, do not
reuse those verifiers.

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

## What is not yet operational

- Alerting rules wired to a live prometheus (the rules exist, the scrape target
  does not).
- Multi-node relay for a large swarm (leviathan-net#9).
- The losing-side penalty for wrongful convictions (leviathan-net#4).
