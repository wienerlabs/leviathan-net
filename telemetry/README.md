# Local telemetry setup

To spawn a local telemetry stack with OpenTelemetry for telemetry collection, Prometheus for metrics, Loki for logs and
Grafana for visualization, run

```bash
docker compose -f telemetry/docker-compose.yml up
```

from the root of the repository.

## Kill switches

Prometheus loads `leviathan-killswitch.rules.yml`, the mainnet kill-switch
alerts. Two exporters feed them, and `prometheus.yml` scrapes both on the host:

```
leviathan-indexer --run-id <run> --rpc <rpc> --metrics-listen 0.0.0.0:9464
leviathan-verifier-daemon ... --metrics-listen 0.0.0.0:9465
```

The indexer reads every value off chain from the run id, polling every
`--poll-secs`: run state and roster, the economic verdict and its inputs, the
vault and what it owes, the verdict accounts, and how fresh its own view is.
Repeat `--run-id` to export several runs. Its counters count what it saw move
after it started; its first poll only learns what is there, so a restart never
reports an old dispute as a new one. The verifier daemon exports what it judged,
the peers it caught signing two payloads for one batch, what the chain already
held, what it could not get on chain, payloads it could not judge at all, and
how long a pass takes against how long it waits between passes. Neither endpoint authenticates a scrape, so bind it to
an interface only Prometheus reaches.

| Alert | Fires when |
|---|---|
| `LeviathanEconomicInsecurity` | a cheat makes money in expectation at the run's own rates and bond: the penalty is below break-even at its audit probability, or a round seats too few verifiers to convict |
| `LeviathanTreasuryDrain` | the vault holds less than the bonds posted into it |
| `LeviathanTreasuryUnderfunded` | the vault covers the bonds but not the earnings still to be claimed |
| `LeviathanMeshPartition` | less than half of the epoch's roster is still healthy while it trains |
| `LeviathanHonestFalsePositive` | a conviction was overturned on appeal |
| `LeviathanDisputeOutcomeUnseen` | a conviction's verdict account was handed to a new dispute before the indexer saw whether an appeal overturned it |
| `LeviathanIndexerLag`, `LeviathanIndexerStale`, `LeviathanIndexerNotReading`, `LeviathanIndexerDown` | the indexer's view trails the chain, stops moving, cannot be read at all, or cannot be scraped |
| `LeviathanUncaughtFraudSuspect` | a fraud vote lapsed without a quorum |
| `LeviathanFraudNotSubmitted` | a verifier proved fraud and the chain does not hold it |
| `LeviathanVerifierDown`, `LeviathanVerifierFailing` | a verifier cannot be scraped, or its passes keep failing |
| `LeviathanAuditWindowTooShort` | an epoch's final round leaves less than twice the time a verifier could need to vote on a fraud before the epoch is charged |
| `LeviathanCommitteeCannotConvict` | verification is on and a round draws too few verifier seats to convict |

`docs/OPS_RUNBOOK.md` says what to do about each. `telemetry/tests/` holds a
promtool test of every alert, firing and not firing, and CI runs it:

```
docker run --rm -v "$PWD/telemetry:/telemetry" --entrypoint promtool \
  prom/prometheus:v3.5.0 test rules /telemetry/tests/leviathan-killswitch.test.yml
```

Once the telemetry stack is up, start your local training setup as usual, but remember to add the OTLP arguments when
running the Psyche client:

```
OTLP_METRICS_URL = "http://localhost:4318/v1/metrics" # OpenTelemetry collector metrics endpoint
OTLP_LOGS_URL = "http://localhost:4318/v1/logs"       # OpenTelemetry collector logs endpoint
```

For convenience, you can run `just dev start-training-localnet-light-client-telemetry` to start the Psyche client with
the arguments already set for telemetry collection
