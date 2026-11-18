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
finalisation and adds a second-level quorum. The `TieBreaker` committee is
already a first-class arm of the coordinator's `CommitteeSelection` lottery; only
`start_round_train` never populated a non-zero count. The treasurer first drew
its own selection with the per-run count (`tie_breaker_committee_size`) and left
the coordinator's rounds at zero. The clients, though, train what the round's
own draw assigns them, so the treasurer's verifiers were not the clients', and
every audit the lottery handed out named a batch its target never trained
(wienerlabs/leviathan#15, finding 4). Now `run_set_challenge_config` carries
the count into the coordinator (`set_tie_breaker_nodes`), which seats it in
every round it starts (`Round::tie_breaker_tasks`), and everyone draws the
round as it is: the clients, the audit lottery and the treasurer. The count
sits in what was alignment padding in the coordinator's account, so no field
moved. It is refused while a round's seed is committed, since the count moves
the verifier seats and the committed slot's hash is about to be known. One
selection partitions each round into disjoint tie-breaker / verifier / trainer sets.
An appeal is judged over many rounds, each drawn afresh, so disjointness within
one draw is not enough: the treasurer also refuses an appeal vote from a verifier
who voted on the verdict (`AppellateVotedOnVerdict`) and from the target itself
(`AppellateIsTarget`), the two parties no draw makes neutral. The whole appeals
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
  nobody's consent. It:
  - finalises a conviction whose windows have closed;
  - charges a conviction whose ejection the ledger has charged;
  - releases and withdraws votes;
  - charges an overturn's losers;
  - credits every recipient its share.

  `run-manager bond-withdraw-finalize` does the same for a wallet's own votes
  and seats before it withdraws.
- **The crank charges a loser's bond directly; the bench is credited seat by
  seat.** `run_slash_losing_verifier` no longer ejects. It forfeits the epoch's
  slashing rate from the loser's bond, out of what convictions on the ledger do
  not already claim, the way an unlanded conviction is settled at a target's
  withdrawal. It closes the loser's vote and adds what the forfeit owes the
  bench to the verdict (`bounty_owed`). It settles every loser wherever it
  is, so nothing waits for an epoch, and it depends on no other account, so no
  payout and no size of bench can stop it. The loser keeps its seat. If the
  forfeit leaves its bond below the run minimum, it cannot vote, join or claim
  again, starting at once rather than at the next epoch's charge. Once every
  loser is charged, `run_pay_verdict_bounty` credits each tie-breaker's bond
  with its even share, one seat per transaction, marked in `bounty_paid`. Anyone
  may send it. The share never leaves the vault and no token account is
  involved, so nothing a tie-breaker or a mint does can stop it. An overturn is
  settled, and its account free for a new dispute, once every loser is charged
  and every seat credited.
- **A conviction that stands pays its recipients the same way.** The target's
  forfeit used to pay its bounty at the target's own withdrawal, to whoever the
  verdict account named at that moment, with every recipient's token account
  in the one transaction (wienerlabs/leviathan-net#64). That broke three ways:
  - a later dispute could take the account over first, and receive the bounty;
  - an overturned verdict named its losers, so a cleared target paid them the
    bounty of an unrelated `run_slash`;
  - a committee larger than one transaction holds, or a recipient with a closed
    or frozen token account, left the target unable to withdraw.

  Now each conviction owes exactly its own charge. The ledger charges a client
  at most once per epoch, so the charge for the epoch a conviction ejected the
  target in is that conviction's. The verdict records it when it lands:
  - `charge` becomes `Ejected`;
  - `charge_epoch_start` records the start of that epoch, which unlike the
    epoch number no authority can rewrite;
  - `charge_rate` records the slashing rate the epoch's end charges.

  Once that epoch has been charged, anyone may send `run_charge_conviction`. It
  takes the charge from the target's bond as settled points, records what the
  forfeit owes on the verdict (`bounty_owed`), and marks it `Charged`. A target
  that never created a participant account, which a run with its own join
  authority admits, has no bond and is charged nothing, so its verdict still
  settles. The target's own withdrawal does the same first if nobody has. A conviction that
  never reached the ledger is charged at the withdrawal as before, and recorded
  the same way. `run_pay_verdict_bounty` then credits each recipient's bond
  with its share, one seat per transaction:
  - the voters whose quorum convicted the target;
  - or, if the bench reached its quorum to uphold it (`appeal_upheld`), the
    bench. An appeal that ran out its window was finalised as the verifiers
    reached it, so its bounty is theirs: a tie-breaker that voted before the
    deadline used to take it from them.

  A conviction that stands is settled, and its account free for a new dispute,
  only once it is charged and every recipient credited. The target cannot hold
  an ejected conviction open, because its charge needs nothing of it. Nor can
  it hold a `Due` one open by staying away: once the target is on no roster an
  epoch will still charge, anyone may send `run_charge_due_conviction`, which
  charges the bond as the withdrawal would and records the bounty the same way
  (wienerlabs/leviathan-net#74). Nor does a voter have to
  wait for someone to dispute the target again: the verifier daemon settles
  every open dispute in the run, as far as nobody's consent is needed, at the
  end of each pass, after its own judging, and `bond-withdraw-finalize`
  credits every seat a wallet is owed before it withdraws.
- **A slash no verdict accounts for records its reporter when it is made.**
  A `run_slash` or an equivocation conviction ejects its target like a verdict
  does. It writes a `SlashReport`, keyed by the target and the start of the
  epoch the ejection is charged in, naming who reported it:
  - for a `run_slash`, the reporter the run authority names, or none;
  - for an equivocation, whoever sent the proof.

  Once that epoch is charged, `run_charge_slash_report` charges the target's
  bond and records the reporter's share of the forfeit. Then
  `run_pay_slash_report_bounty` credits that share to the reporter's bond.
  Anyone may send either, and the verifier daemon and `bond-withdraw-finalize`
  do. A slash naming no reporter leaves its bounty in the vault. The target's
  bond is held until its reports are charged, and the withdrawal pays no
  reporter of its own.

  It used to pay the bounty to whatever token account the withdrawal named.
  The withdrawal is signed by the slashed participant itself, so the bounty
  could go to a second wallet it controlled (wienerlabs/leviathan-net#72).
- **An equivocation's reporter commits before it reveals, and its bounty is
  the run's own share.** An equivocation proof is permissionless, and its
  reporter is paid out of the target's forfeit. That left two routes for the
  target to take the bounty back (wienerlabs/leviathan-net#92):
  - **Copying.** The proof is public once anyone sends it: both payloads and
    both signatures are in the transaction. Anyone could copy an honest
    reporter's, put its own wallet in as the reporter, and land it first,
    taking the bounty from whoever found the fraud. Now a reporter first sends
    `run_commit_equivocation` with a hash of the proof and its own key
    (`EquivocationClaim::commitment`). It reveals the proof through
    `run_slash_equivocation` in a later slot, against that commitment
    (`EquivocationCommittedTooRecently` in the same slot,
    `EquivocationNotCommitted` against another proof or another reporter's).
    A copier sees only the hash until the reveal, and by then it is too late
    to commit: the honest reporter's proof is already in flight and lands
    first. A reporter that loses the race closes its commitment
    (`run_close_equivocation_commitment`) and takes its rent back; a reveal
    closes its own.
  - **Reporting itself.** A target can still commit and reveal its own proof
    from a second wallet, which the chain cannot tell from an honest
    reporter. So an equivocation's bounty is a share of its own, set by the
    run's main authority with its bond economics
    (`run_set_equivocation_bounty`). It is refused at the whole forfeit
    (`EquivocationBountyTooHigh`, at most 9999 bps), and at the charge it is
    also held to the run's `slash_bounty_bps`. A run that set none gives an
    equivocation's reporter nothing.

  What remains: a target that reports itself loses
  `slashing_rate * (1 - bounty_bps / 10000)` of the charge, more than
  nothing whatever the run sets. A higher share pays honest reporters more and
  self-reporting targets more alike. It is kept in an account of its own,
  `EquivocationBounty`, rather than in `Run`, whose length tells the
  dashboard's decoders its generations apart. A `run_slash` is not exposed to
  either route: the run authority names its reporter.
- **A target with no participant account when it is slashed owes its reporter
  nothing.** It has no bond then, so its report is charged at nothing. If it
  posts a bond later, the ledger's charge is taken from that bond at its
  withdrawal, and the vault keeps all of it. This happens only in a run that
  does not gate joining on a bond, where bounties have nothing to pay from.
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
a disjoint, separately sized set, and never seats the verdict's own voters or its
target, so buying it is a second independent purchase.
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
and the Schelling point stays put. It reuses the existing `slash_bounty_bps` knob:
the forfeited bond's bounty share is split evenly among the recorded appeal
voters, and each one's share is credited to its bond, out of the vault where the
forfeit already sits, by seat. An overturn's shares are credited once the crank
has charged every loser; an upheld appeal's once the target's forfeit is charged
(wienerlabs/leviathan-net#39, #64). memnet
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
  the conviction directly, at the run's slashing rate at that time, records
  what that owes whoever ruled it, who are credited by seat, and records the
  verdict as upheld, so it is charged exactly once. A target that never
  withdraws would still leave a `Due` conviction, and the bounty its voters
  are owed, waiting for good. So anyone may charge a `Due` conviction through
  `run_charge_due_conviction` while the target is on no roster an epoch will
  still charge: between epochs, or once it has left. It is the withdrawal's
  charge, made without the target: the slashing rate in force, out of what the
  ledger does not already claim, not counted as a settled point since the
  ledger never saw it, and the verdict recorded as upheld and charged. While
  the target is on such a roster it is refused, since the conviction can still
  land there as an ejection. A target that comes back afterwards is not
  charged again: the verdict is no longer due (wienerlabs/leviathan-net#74).
  For the same reason a run with a challenge window must
  give appeals a deadline (`appeal_window_seconds > 0`). The windows are read
  from the run as they stand, so the run authority changing them moves the
  deadlines of disputes in flight, as it always has. The run authority's
  `run_slash` and a committee verdict are separate convictions, each charged
  once: an authority that slashes a target the committee also convicts
  charges it twice. A committee conviction of a target the authority already
  ejected this epoch is recorded as due, because the coordinator refuses a
  second ejection, and is settled at the target's withdrawal; that refusal used
  to undo the vote that reached the quorum (wienerlabs/leviathan-net#40). The
  other way round, `run_slash` still has nothing to record a refusal in: an
  authority slash of a target the committee already ejected this epoch is
  refused, and the target is charged once. A proof of equivocation
  (`run_slash_equivocation`, below) lands the same way as `run_slash`.

### A payload that does not fit the model

A signed payload that does not fit the run's model is fraud
(wienerlabs/leviathan-net#59). Its sender did none of the work it claims for
its batch, since every honest node's `Distro::apply` refuses the payload. Yet
it would finish the epoch and earn unless something else caught it.

"Does not fit" means any of the following:
- the payload does not decode;
- it does not hold together as a DisTrO encoding of its own shapes: a `totalk`
  that is not the one its shape gives, or indices or values that are not what
  that encoding scatters;
- it does not have the length of the honest result for its batch. The run's
  own model gives that result, replayed or trained by the verifier's node.

The first two are properties of the signed bytes alone. The third rests on
the reference, as the band does. A verifier on the wrong model would find
honest payloads too long or too short, as it would find them out of band if
the lengths happened to agree. The appeals court and the honest
false-positive kill switch exist for that case.

The verifier judges such a payload at an infinite distance, as it judges one
that is not made of numbers. A committee convicts it like any band miss, with
the target's signature over the payload as the committed side and the hash of
the honest replay as the replayed side. An appeal bench re-auditing the dump
with `leviathan-verifier` against the same replay reaches the same verdict,
because the daemon and the offline tool share one judgement
(`judge_payload`). A payload that has the honest length and holds together,
but lays its values out otherwise, is left to the band, which finds it nothing
like the honest result.

Before decompressing a payload, the verifier makes two checks:
- **Its size.** The values its shapes claim must number exactly the honest
  result's, counted without allocating any of them. A payload of a few bytes
  can claim shapes of billions of values, every index inside them.
- **Its encoding.** Each result must hold together as an encoding of its own
  shape (`Distro::check_encodings`).

Decompression allocates every value the payload claims and indexes with what
it holds, using libtorch's infallible operations. Before these checks, a signed
payload could make the verifier exhaust its memory, or panic it with indices
past its shape. Either took the verifier down again on every restart while the
payload was kept.

### What a conviction proves on chain

The chain cannot recompute a gradient, so whether a payload was honest remains
the verifiers' claim, and the appeals court is what contests it. What the
target committed to does not have to be a claim. A trainer signs its payload
hash with its p2p key before it broadcasts the payload, and the coordinator
roster holds that key. So every conviction now carries that signature, and the
treasurer checks it (wienerlabs/leviathan-net#43):

- **The hash is rebuilt on chain.** A payload's commitment hash is
  `sha256(domain, sha256(run_id), epoch_start, step, nonce, batch_start,
  batch_end, results_digest)`. The `results_digest` is the sha256 of the
  payload's wire bytes, the same bytes a payload dump holds. `epoch_start` is
  when the epoch the payload was trained in started, the coordinator's
  `epoch_state.start_timestamp`. A conviction names the epoch, the step, the
  nonce and the digest. The treasurer rebuilds the hash from them, the batch
  the conviction records, and the run id its coordinator holds. Binding the run
  means a signature a key made in one run convicts it of nothing in another.
- **The target's key has to have signed it.** The same transaction carries an
  ed25519 precompile instruction with the key, the signature and the message
  inside it. The runtime verifies the signature before any program runs. The
  treasurer reads that instruction through the instructions sysvar and accepts
  it only if the key is the target's roster p2p key and the message is exactly
  the rebuilt hash.
- **The payload is from this epoch.** The epoch it names is the one the
  coordinator is in, and its step lies between the epoch's first step and the
  current one. A commitment from an earlier epoch names work that epoch
  already charged. The step alone would not say so. A run restarted from an
  earlier step (`run-manager update-config --restart-from-step`) trains that
  step again, and a run id freed and taken again starts at step 1. Both
  times the old commitment's step is back inside the range. The epoch's start
  is not. A restart needs the run paused, and a paused run resumes only
  through a new warmup, which starts a new epoch. The coordinator starts an
  epoch only in a later second than the waiting it began from, and a run id
  taken again starts its epochs later still. On a cluster clock that never
  runs back, no two epochs of a run id share a start
  (wienerlabs/leviathan-net#67).
- **The record is self-checking.** The verdict stores the evidence alongside
  the hash, so anyone can rebuild the hash from the account and its run.
  Holding the dump, anyone can check its sha256 against the digest.
- **Votes agree on the payload, not on the replay.** Votes are counted per
  commitment the target signed: a quorum is two thirds of the committee voting
  for one payload, and the verdict records the payload most votes are for. The
  first vote used to fix the evidence for the rest of the round, its replayed
  hash included. Every payload is broadcast, so one seat could cite another
  payload the target signed and veto what two thirds agreed on, and honest
  verifiers whose replays differ in the last bits, which is what the band is
  for, never reached a quorum. A vote for evidence the verdict does not record
  convicted nobody: it is no seat of the bounty, an overturn charges it
  nothing, and it is released once the verdict resolves. The target never
  votes on its own verdict (`VerifierIsTarget`).

Before this, a quorum could record a hash the target never committed to, and
the record an appeal bench or an observer reads would have been fiction. A
conviction can still cite a genuine payload and lie about its replay. That is
the claim the appeals court exists for, and the record now names exactly which
payload to re-audit.

The client keeps a peer's payload together with the record of what the peer
signed for it. It keeps only a payload whose header names the work it was
broadcast for: the round's epoch and step, and the batch the peer was assigned.
The sender signs its header with the results, so a header naming other work
passes the hash check. Kept, it would be a record of work outside the round,
which convicts nobody, for a payload every peer trained on. Dropping it closes
that escape (wienerlabs/leviathan-net#67). The client writes the record first
and then renames the dump into place, so a dump is never seen without its
record. The dump's name carries the epoch, step and batch, so a step trained
again after a restart is kept under a name of its own. It also keeps only the
first payload a peer signed for a batch. A peer that signs two payloads for one batch
therefore cannot leave one payload's record beside the other's dump, which
would leave nothing to convict on.

### Equivocation

A trainer signs one payload per batch it trains. Two different commitment
hashes for the same run, round and batch, both signed by the target's roster
p2p key, are equivocation: the target told different peers different things
about the same work. The round is the epoch and the step the commitment names
(wienerlabs/leviathan-net#67). An honest client never does that:
- It sends the first rank's result for a batch and drops the rest.
- It trains a batch once per step, and a restarted client joins only a later
  epoch.
- A run restarted from an earlier step trains the step again, but in a new
  epoch, so that is another round.
- Sending the same payload twice gives the same bytes, so the same hash, and
  two copies of one commitment are not equivocation.

The nonce is a trainer-local counter. Only the round and the batch say which
work a payload is, so two commitments to the same round and batch that differ
in nonce or results are equivocation (wienerlabs/leviathan-net#46).

- **The client keeps the second commitment.** When a payload arrives under a
  name whose record holds a different commitment, the client keeps the first
  payload and drops the second, as before. Beside the first it now writes an
  equivocation record: the second commitment's header and signature, and the
  digest of its results. That is everything the chain needs to rebuild the
  second hash and check its signature, without the payload. The first
  contradiction is kept, and a repeat of the kept commitment writes nothing.
  Each record goes to a temporary name, reaches the disk, and is linked into
  place, which fails if the name is taken. A payload arriving while the first
  record is written then reads it whole, and a crash leaves no partial record
  holding the name. Either used to lose the proof.
- **The proof convicts on its own.** `run_slash_equivocation` takes the two
  commitments, the batch and the target's roster index, and reveals them
  against the reporter's commitment from an earlier slot (above). The same
  transaction carries an ed25519 precompile instruction for each, both signed
  by the target's roster p2p key. The treasurer rebuilds both hashes as above
  and refuses the proof unless:
  - both are from the current epoch (`CommitmentOutsideEpoch`);
  - the steps are the same and the hashes differ (`EquivocationNotProven`);
  - both signatures are the key of the client at that roster index
    (`CommitmentNotSigned`).

  A seat past the roster names nobody (`TargetMismatch`). A signature over another batch's
  commitment does not prove one over this batch's. It needs no replay, so it
  needs no committee and leaves nothing for an appeal to contest, and anyone
  may send it.
- **It lands only as an ejection.** It has no verdict account in which to
  record a conviction to charge later, so it lands only where an ejection is
  certain to be charged. That means `conviction_landing` must give an
  ejection. Otherwise the proof is refused (`EquivocationNotLandable`): when
  the target is already ejected, has withdrawn, or the epoch is not charging.
  The ejection is charged with the epoch, at the run's slashing rate, like any
  other. A target already ejected this epoch is charged once. A committee
  conviction after the proof is recorded as due and charged as well, as after
  `run_slash`. The window is ample. A peer sees the second payload in the
  round it arrives, and the final round's roster stands through the cooldown
  (wienerlabs/leviathan-net#40).
- **Every verifier submits it.** Each pass, the daemon looks beside every kept
  payload for an equivocation record, whatever batches its committee assigns
  it. It checks both commitments against the target's roster key, commits to
  the proof, and reveals it on a later pass, once the slot has moved on. The
  first proof to land ejects the target, and the next daemon finds it ejected
  (`already_on_chain`) and closes its own commitment.
  - **Refused.** A proof the chain did not take counts as a failed
    submission, which the uncaught-fraud kill switch fires on, and is sent
    again every pass.
  - **No longer landable.** Before sending, the daemon asks
    `conviction_landing` whether the proof can land. One it can never land in
    this epoch is counted once as a failed submission and dropped: the target
    withdrew or left the roster, or the epoch stopped charging before the
    proof landed.
  - **From an ended epoch.** A record from an epoch that is over is set aside
    without being counted, so a daemon restarted with old records on disk
    does not report them again.
  - **Ahead of the daemon.** A record from a round the daemon's view of the
    run has not reached is looked at again on the next pass.

What this does not charge:
- **An equivocation whose epoch ends before any verifier sends the proof.**
  For instance, a pause withdraws everyone first. As with `run_slash`,
  nothing records it to charge later. The kill switch reports it. A client
  cannot withdraw itself from an epoch on chain: only a pause withdraws, and
  it withdraws everyone.
- **One no single peer sees.** The proof needs one node to hold both
  payloads. A trainer broadcasts its commitment over gossip, which floods it
  to every peer. Only a trainer that sent each commitment to peers who never
  relay the other could keep both from meeting.

A daemon that finds an equivocation record that proves nothing counts it as
malformed. A node writes a record only after checking the second payload's
signature, so such a record was damaged or tampered with on disk.

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
