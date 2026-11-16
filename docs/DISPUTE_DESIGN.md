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
account. Only the treasurer was rebuilt and redeployed. Verified live on devnet:
two verifiers convict, the target appeals, two tie-breakers overturn, both
verifiers forfeit 200 while the innocent target keeps its full bond.

The losing side first forfeited its bond through the slash-then-settle path the
target used: the crank ejected each convicting verifier, the epoch's end charged
the ejection, and the bond paid at the loser's own withdrawal. That reached a
loser only while it sat on the roster of an epoch that would charge it, and
nothing held a verifier's bond while its verdict could still be overturned. A
verifier could vote, request its bond, and leave during the appeal, or between
the overturn and the crank. A loser that left the run, or a run that halted, was
out of reach for good (wienerlabs/leviathan-net#39). Two rules now close that:

- **A vote holds the bond until it can no longer cost the voter.** Each vote is
  counted on the voter's `Participant` (`open_verdict_votes`), and a withdrawal
  is refused while any is counted (`VerdictVotesOpen`).
  `participant_release_verdict_vote` stops a vote counting once the verdict says
  it can no longer cost the voter:
  - the conviction stands (`Upheld`, `Due`);
  - or the verdict never reached quorum, and either its round is over or the
    voter withdraws its own vote.

  Anyone may send it. A withdrawn vote comes off the verdict's count, so it can
  never be part of a later quorum. The verdict marks each released vote in a
  bitmap (`released_voters`), so no vote is released twice. Its account is taken
  over for a new dispute only once every vote on the old one is released or
  charged, so a reset never strands a count. Before it votes, the verifier
  daemon settles the last dispute against the target in everything that needs
  nobody's consent: it finalises a conviction whose windows have closed,
  releases and withdraws votes, charges an overturn's losers and credits the
  bench. `run-manager bond-withdraw-finalize` does the same for a wallet's own
  votes and bench seats before it withdraws.
- **The crank charges a loser's bond directly; the bench is credited seat by
  seat.** `run_slash_losing_verifier` no longer ejects. It forfeits the epoch's
  slashing rate from the loser's bond, out of what convictions on the ledger do
  not already claim, the way an unlanded conviction is settled at a target's
  withdrawal. It closes the loser's vote and adds what the forfeit owes the
  bench to the verdict (`bench_bounty_owed`). It settles every loser wherever it
  is, so nothing waits for an epoch, and it depends on no other account, so no
  payout and no size of bench can stop it. The loser keeps its seat. If the
  forfeit leaves its bond below the run minimum, it cannot vote, join or claim
  again, starting at once rather than at the next epoch's charge. Once every
  loser is charged, `run_pay_bench_bounty` credits each tie-breaker's bond with
  its even share, one seat per transaction, marked in `bench_paid`. Anyone may
  send it. The share never leaves the vault and no token account is involved,
  so nothing a tie-breaker or a mint does can stop it. An overturn is settled,
  and its account free for a new dispute, once every loser is charged and every
  seat credited.
- **Switching appeals off ends the appeals in flight.** A run with appeals on
  always has an appeal deadline, so a zero window means the run turned appeals
  off. A challenge still open then has no deadline left to wait for, and
  `run_finalize_slash` finalises it at once. It used to mean the challenge never
  timed out, which held the conviction, and now every vote on it, for good. A
  run authority could switch appeals off to end an appeal before the bench
  reaches quorum, and so spare its verifiers an overturn. That sits within
  the authority's existing trust: it can already shorten the appeal window to
  a second, or convict directly through `run_slash`. A bonded run's authority
  is a key its operator holds, and the overturn is final once the bench
  reaches quorum.

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
and the same split-and-verify settlement the target-cheater path already used: the
forfeited bond's bounty share is split among the recorded appeal voters, each
recipient's token account owner checked against a recorded voter, and the tokens
come from the vault where the forfeited bond already sits. An overturn's share is
credited to each tie-breaker's bond once the crank has charged every loser; an
upheld appeal pays it at the target's settlement. memnet
confirms both directions: a tie-breaker earns its share whether the verdict is
overturned or upheld, while a cleared target still recovers its full bond and a
convicted one still forfeits.

A conviction reaches the bond in two steps: the coordinator charges the
target's ejection when the epoch ends, and settlement forfeits it when the
target withdraws (wienerlabs/leviathan-net#36). Three rules keep the first
step from being skipped:

- **Ejections only land where the charge is certain.** Between epochs the
  roster of the epoch that just ended is still on the account, so an ejection
  there used to succeed and was then wiped, uncharged, by the next warmup. A
  warmup that is paused skips the charge as well. The coordinator now ejects
  only in `RoundTrain`, `RoundWitness` and `Cooldown`, every exit from which
  passes through the charge. Votes are still cast whenever the committee is
  drawn, so the bench can rule between epochs in either direction. A ruling
  that needs an ejection where none would be charged (an upheld appeal, an
  immediate quorum, or a finalisation) is recorded as `Due`: the conviction
  stands and waits to be charged. A vote is taken only while the committee in
  force was drawn over the roster as it stands: not in a warmup, whose roster
  is already the next epoch's, and not once the roster has been compacted
  under the last round's draw, because a seat read through a shifted index
  belongs to someone else.
- **A pending conviction holds the bond.** While the target's verdict is inside
  its challenge window or appeal, its withdrawal is held however long ago its
  delay elapsed. Without that, a free challenge was a free exit: open it, let
  the delay run out, and leave with the whole bond. The epoch itself holds the
  bond the same way.
- **A conviction that can no longer land is settled at the bond.** Once its
  windows have closed, or once it is `Due`, a conviction finalises through
  `run_finalize_slash` in any epoch where the target is on the roster and an
  ejection is certain to be charged. A target that walked away (its epoch
  ended, the run halted, or it simply never came back) would otherwise hold its
  bond forever with nothing able to finalise against it. Its withdrawal charges
  the conviction directly, at the run's slashing rate at that time, pays the
  bounty to whoever ruled it, and records the verdict as upheld, so it is
  charged exactly once. For the same reason a run with a challenge window must
  give appeals a deadline (`appeal_window_seconds > 0`). The windows are read
  from the run as they stand, so the run authority changing them moves the
  deadlines of disputes in flight, as it always has. The run authority's
  `run_slash` and a committee verdict are separate convictions, each charged
  once: an authority that slashes a target the committee also convicts
  charges it twice.

### What a conviction proves on chain

The chain cannot recompute a gradient, so whether a payload was honest remains
the verifiers' claim, and the appeals court is what contests it. What the
target committed to does not have to be a claim. A trainer signs its payload
hash with its p2p key before it broadcasts the payload, and the coordinator
roster holds that key. So every conviction now carries that signature, and the
treasurer checks it (wienerlabs/leviathan-net#43):

- **The hash is rebuilt on chain.** A payload's commitment hash is
  `sha256(domain, sha256(run_id), step, nonce, batch_start, batch_end,
  results_digest)`. The `results_digest` is the sha256 of the payload's wire
  bytes, the same bytes a payload dump holds. A conviction names the step,
  the nonce and the digest. The treasurer rebuilds the hash from them, the
  batch the conviction records, and the run id its coordinator holds. Binding
  the run means a signature a key made in one run convicts it of nothing in
  another.
- **The target's key has to have signed it.** The same transaction carries an
  ed25519 precompile instruction with the key, the signature and the message
  inside it. The runtime verifies the signature before any program runs. The
  treasurer reads that instruction through the instructions sysvar and accepts
  it only if the key is the target's roster p2p key and the message is exactly
  the rebuilt hash.
- **The payload is from this epoch.** Its step lies between the epoch's first
  step and the current one. A commitment from an earlier epoch names work that
  epoch already charged.
- **The record is self-checking.** The verdict stores the evidence alongside
  the hash, so anyone can rebuild the hash from the account and its run.
  Holding the dump, anyone can check its sha256 against the digest.

Before this, a quorum could record a hash the target never committed to, and
the record an appeal bench or an observer reads would have been fiction. A
conviction can still cite a genuine payload and lie about its replay. That is
the claim the appeals court exists for, and the record now names exactly which
payload to re-audit.

The client keeps a peer's payload together with the record of what the peer
signed for it. It writes the record first and then renames the dump into
place, so a dump is never seen without its record. It also keeps only the first
payload a peer signed for a batch. A peer that signs two payloads for one batch
therefore cannot leave one payload's record beside the other's dump, which
would leave nothing to convict on. Signing two payloads for one batch is itself
provable from the two signatures, though no rule charges it yet.

The coordinator's own `slash_client` stays the owner's instrument. For a
treasurer run the owner is the run PDA, so every slash goes through the checks
above. A run without a treasurer trusts its owner key, as it always has.

One refinement remains, optional and priced: an explicit challenge bond so a
frivolous appeal has a cost of its own. It is additive on top of the shipped
lifecycle and needs an economic parameter, so it is a deliberate choice rather
than a gap. A second, smaller call left open: on an upheld appeal the target's
bounty goes to the tie-breakers who did the decisive re-audit rather than the
original verifiers; splitting it between them is a one-line change if the verifier
incentive on appealed convictions ever needs propping up.
