# Losing-side penalty: design

Issue leviathan-net#4, the open half. The committee vote can convict, but nothing
punishes a verifier that convicts an innocent target. This document frames the
options, because the choice touches the trust model and is worth locking before
building.

## The problem

A verifier votes only when it claims to have found fraud: its verdict asserts
that an honest replay of the target's batch differs from what the target
committed, beyond the tolerance band. A wrongful verdict means the verifier lied
about that replay.

To penalise a wrongful verdict, the protocol must know the verdict was wrong. And
that is the hard part: the chain cannot recompute a gradient. Whether the target
actually cheated is an off-chain fact. So any losing-side penalty needs a source
of ground truth, and the whole design question is where that comes from without
reintroducing a single trusted party, which is the thing the committee vote just
removed.

## Options

### A. Optimistic challenge with a tie-breaker committee (recommended)

The slash does not finalise immediately. When a quorum is reached the target is
marked slashed-pending and a challenge window opens. The target can escalate by
posting a challenge bond, which convenes the tie-breaker committee, a larger and
higher-bond set drawn by the same lottery. The tie-breaker re-audits and votes.

- Tie-breaker overturns: the slash is reversed, the original voters forfeit their
  bonds (the losing-side penalty), and the target recovers its challenge bond plus
  a share of the forfeited verifier bonds.
- Tie-breaker upholds: the slash finalises and the target forfeits its challenge
  bond to the vault.

Trust: adjudication is a committee, not one key, so it stays in the same trust
model as the vote. The recursion (who watches the tie-breaker) is bounded by
economics: the tie-breaker is larger and its members bond more, so buying it is
strictly more expensive than buying the original committee. This is the appeals
court a court system already uses: you can appeal once, to a bigger bench.

Cost: this is the largest change. It delays finalisation, needs the tie-breaker
committee wired (the coordinator has a `TieBreaker` variant, currently `todo!()`),
a challenge-bond account, and a second-level quorum. It touches the coordinator,
so it needs a fresh SBF build and redeploy.

### B. Symmetric verifier audit by hash comparison (does not work, recorded so it is not retried)

The tempting cheap version: a verifier's verdict carries its `replayed_hash`; if a
second verifier assigned to the same target submits a replay that matches the
target within band, the chain declares the first verifier wrong and slashes its
bond, all treasurer-only, no appeals court.

This does not work, and the reason is the crux of the whole problem. The chain
holds two claims, verifier 1 says the target is beyond band and verifier 2 says it
is within band, but it cannot recompute the gradient to tell which replay is
honest. A neural-net forward pass does not fit the Solana compute limit. So the
chain cannot adjudicate a gradient dispute from hashes alone. Any resolution needs
someone to establish the off-chain truth, which is a committee vote, which is
option A. B collapses into A. There is no cheap treasurer-only losing-side
penalty that is actually trustless.

### C. Defer, document the gap

Leave conviction as is, keep the manual-review note in the runbook, and price the
gap honestly in the economics (framing an innocent target costs only a quorum of
bonds). Revisit after the genesis run shows whether wrongful convictions happen at
all in practice.

Trust: unchanged, but the hole is real and known.

Cost: none.

## Trade-off summary

| | Trust model | Collusion resistance | Touches coordinator | Ship cost |
|---|---|---|---|---|
| A appeals committee | committee, staged | strong (bigger bench) | yes, redeploy | large |
| B hash comparison | none, it cannot adjudicate | n/a, does not work | no | n/a |
| C defer | unchanged | none, documented | no | none |

## Recommendation

The honest conclusion is that there is no cheap version. Because the chain cannot
recompute a gradient, a losing-side penalty needs an appeals committee (A), which
is a large change: a delayed finalisation, the tie-breaker committee wired into
the coordinator, a challenge bond, and a second-level quorum, plus a redeploy.

So the recommendation is C now, A as a dedicated effort later. Defer the penalty,
keep the gap documented and priced (framing an innocent target costs a quorum of
bonds, and the runbook flags disputed convictions for manual review), and build A
properly when the tie-breaker committee is being wired anyway. This is not the
critical path: the genesis run and the token launch are, and neither depends on
this. A wrongful conviction on devnet is a manual review, not a fund loss.

The decision to lock is only A-later versus A-now. There is no third option that
is both cheap and trustless.
