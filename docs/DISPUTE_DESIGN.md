# Losing-side penalty: design

Status: BUILT. Option A (the optimistic challenge with a tie-breaker committee)
is implemented, treasurer-only, and verified live on devnet. It turned out not to
need a coordinator redeploy, because the tie-breaker committee already exists in
the coordinator's `CommitteeSelection` lottery; the treasurer draws it with
`from_coordinator_with_tie_breakers`. The rest of this document is the design
record that led there, kept because the trust-model reasoning still matters.

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

Cost as built: this was the largest change, but smaller than feared. It delays
finalisation and adds a second-level quorum, but it did not touch the coordinator.
The `TieBreaker` committee is already a first-class arm of the coordinator's
`CommitteeSelection` lottery; only `start_round_train` never populated a non-zero
count. Rather than migrate the coordinator's zero-copy state, the treasurer draws
its own selection with `from_coordinator_with_tie_breakers`, overriding the
tie-breaker count from a per-run config (`tie_breaker_committee_size`). One
selection partitions the epoch into disjoint tie-breaker / verifier / trainer sets,
so a client is never both a verifier and its own appeals judge. The whole appeals
court is five treasurer instructions (`run_set_challenge_config`,
`run_open_challenge`, `run_submit_appeal_verdict`, `run_finalize_slash`,
`run_slash_losing_verifier`) plus a lifecycle on the existing `AuditVerdict`
account. Only the treasurer was rebuilt and redeployed. The losing side forfeits
its bond through the same slash-then-settle path the target already used: an
overturned verdict ejects the convicting verifiers, and their bonds settle at the
epoch boundary exactly as a cheater's would. Verified live on devnet: two
verifiers convict, the target appeals, two tie-breakers overturn, both verifiers
forfeit 200 while the innocent target keeps its full bond.

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
| A appeals committee (BUILT) | committee, staged | strong (bigger bench) | no, treasurer-only | medium |
| B hash comparison | none, it cannot adjudicate | n/a, does not work | no | n/a |
| C defer | unchanged | none, documented | no | none |

## Outcome

The honest conclusion held: there is no cheap version, because the chain cannot
recompute a gradient, so a losing-side penalty needs an appeals committee. What
changed is the cost estimate. A did not require touching the coordinator; the
tie-breaker lottery was already there, dormant. So A was built now instead of
deferred, treasurer-only, with a per-run switch (`challenge_window_seconds` and
`tie_breaker_committee_size`, both default 0). When they are zero the run behaves
exactly as before: a verifier quorum slashes immediately, no appeals. When they
are set, a quorum only opens a challenge window, and the optimistic-finality path
takes over.

The trust model is unchanged from the committee vote: adjudication is a bonded
committee, not a key, and the appeals bench is drawn by the same lottery but from
a disjoint, separately sized set, so buying it is a second independent purchase.
The recursion is bounded by economics, not by another layer of code. Backward
compatibility is preserved, so the genesis run and token launch remain unblocked
whether or not a given run turns appeals on.

The appeal bounty is now built too, and built to be symmetric. Without a reward
the appeals court had the same economic hole the committee-economics sim found for
verifiers: a tie-breaker pays the cost of a re-audit but earns nothing, so a
rational tie-breaker never votes, and a bench nobody staffs is not a bench.

The subtlety is that a one-sided bounty is worse than none. If tie-breakers were
paid only when they overturn, a rational tie-breaker would vote to overturn just to
get paid, which is exactly the bias that frees guilty targets. So the bounty pays
the tie-breakers on both outcomes, each funded by the side that lost: an overturn
pays them out of the convicting verifiers' forfeited bonds, an upheld appeal pays
them out of the target's forfeited bond. Because a tie-breaker earns either way,
the bounty does not push its vote in either direction, so it votes its honest read
and the Schelling point stays put. It reuses the existing `slash_bounty_bps` knob
and the same split-and-verify settlement the target-cheater path already used: at
settlement the forfeited bond's bounty share is split among the recorded appeal
voters, each recipient's token account owner checked against a recorded voter, and
the tokens come from the vault where the forfeited bond already sits. memnet
confirms both directions: a tie-breaker earns its share whether the verdict is
overturned or upheld, while a cleared target still recovers its full bond and a
convicted one still forfeits.

One refinement remains, optional and priced: an explicit challenge bond so a
frivolous appeal has a cost of its own. It is additive on top of the shipped
lifecycle and needs an economic parameter, so it is a deliberate choice rather
than a gap. A second, smaller call left open: on an upheld appeal the target's
bounty goes to the tie-breakers who did the decisive re-audit rather than the
original verifiers; splitting it between them is a one-line change if the verifier
incentive on appealed convictions ever needs propping up.
