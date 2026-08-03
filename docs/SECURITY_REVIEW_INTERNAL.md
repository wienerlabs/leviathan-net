# Internal security review of the on-chain programs

Scope and method for wienerlabs/leviathan#15. This is the internal pass that
precedes the external audit (#4). It does not close #4 and does not clear
mainnet bonds.

Reviewed at commit `2a24bad9`, against `psyche-solana-treasurer`,
`psyche-solana-coordinator` and the parts of `psyche-coordinator` the conviction
path depends on. Every finding below that is marked *reproduced* has a test in
this repository that fails on the honest expectation and passes on the current
behaviour; run them with `cargo test -p psyche-solana-tooling` and
`cargo test -p psyche-coordinator`.

## Summary

| # | Finding | Severity | Bounty class | Reproduced |
|---|---|---|---|---|
| 1 | Epoch-end accounting silently drops a conviction | Critical | B | yes |
| 2 | Bounty recipient is unvalidated when the verdict is omitted | Critical | B | yes |
| 3 | Votes from different rounds are pooled into one quorum | High | — | yes |
| 4 | Daemon and chain disagree on who is a verifier | High | A (enables) | yes |
| 15 | An unvalidated warmup witness index panics every later witness | High | — | yes |
| 5 | `run_finalize_slash` trusts a stale client index | High | — | no |
| 6 | `reset_for_epoch` destroys an appeal and an unfinished settlement | Medium-High | — | no |
| 7 | The appeal bench may contain the verifiers it is judging | Medium | — | no |
| 8 | A losing verifier escapes forfeit by leaving the epoch | Medium | — | no |
| 9 | No evidence consensus; the last voter overwrites the record | Medium | — | no |
| 10 | A challenged verdict has no deadline | Medium | — | no |
| 11 | Quorum can be one | Low | — | no |
| 12 | The voter cap can sit below quorum at scale | Low | — | no |
| 13 | Borsh accounts are sized with `std::mem::size_of` | Low | — | no |
| 14 | Slashing points and collateral units are coupled by convention | Info | — | no |
| 16 | The disclosure channel SECURITY.md names does not exist | Process | — | n/a |

Sections below run in numeric order; the table is sorted by severity, so the
numbers are identifiers rather than a reading order.

Findings 1 and 2 are Class B under `docs/REDTEAM_BOUNTY.md` — "recover bonded
collateral after conviction ... or finalise a withdraw that should have
forfeited" — and neither needs privileged keys.

The single most important structural observation is this: **every conviction
path in the treasurer ends in the same two lines of coordinator code.**
`run_submit_audit_verdict`, `run_submit_appeal_verdict`, `run_finalize_slash`,
`run_slash_losing_verifier` and the admin `run_slash` all CPI into
`slash_client`, which calls `Coordinator::eject` — and `eject` only sets a state
flag. No money moves there. The money moves once per epoch, in a single merge
walk in `instance_state.rs`. That walk had not been reviewed, and it is where
finding 1 lives. The committee machinery above it can be perfectly correct and
still not take a single token.

---

## 1. Epoch-end accounting silently drops a conviction (Critical, Class B)

**File:** `architectures/decentralized/solana-coordinator/programs/solana-coordinator/src/instance_state.rs:133-169`
**Test:** `memnet_treasurer_slash_accounting_order::exiting_out_of_join_order_voids_the_slash`

At `TickResult::EpochEnd` the program walks the permanent client records and
charges the slashing rate to everyone who left as `Ejected`:

```rust
let finished_clients = &self.coordinator.epoch_state.clients;      // 133
let exited_clients = &self.coordinator.epoch_state.exited_clients; // 134
let mut exited_client_index = 0;                                   // 138
for client in self.clients_state.clients.iter_mut() {              // 140
    ...
    if exited_client_index < exited_clients.len()
        && client.id == exited_clients[exited_client_index].id     // 156
    { ...; exited_client_index += 1; }                             // 167
}
```

This is a forward-only merge walk: one cursor, advanced only on a match. It is
correct only if `exited_clients` is in the same relative order as
`clients_state.clients`. It is not.

- `clients_state.clients` is append-only in **join order** (`instance_state.rs:366`,
  and a rejoining client reuses its slot at `:337`).
- `exited_clients` is filled by `move_clients_to_exited`
  (`shared/coordinator/src/coordinator.rs:1193-1197`) in **exit order**: a fresh
  batch is appended at the end of *every round* (`coordinator.rs:1043`), at
  warmup (`:1011`), at the members gate (`:954`) and at cooldown (`:1101`).

So whenever two clients exit in different rounds and the earlier-joined one
exits second, the cursor is already past the earlier-joined client when its
entry finally appears. It never goes back. That client is never charged.

**Failure scenario.** Clients A, B, C join in that order, all bonded 500, with
`slashing_rate_per_client = 200`.

1. C is convicted and ejected in round *r*. The round closes; `exited = [C]`.
2. A is convicted and ejected in round *r+k*. The round closes; `exited = [C, A]`.
3. At epoch end the walk visits A first: `exited[0]` is C, no match, cursor stays
   at 0. It visits B: no match. It visits C: match, C is charged 200, cursor → 1.
   The loop ends with `exited[1] = A` never examined.
4. `A.slashed` is still 0. `participant_bond_finalize_withdraw` computes
   `forfeited_amount = min(0 - 0, bond) = 0` and pays A its full 500 back.

The test asserts exactly this: C forfeits 200, A forfeits nothing, and A then
withdraws 500 out of the vault after a conviction that is recorded on chain.

**Why an attacker does not need luck.** Two identities are enough. Join the
attacker's main node first and a throwaway second. Drop the throwaway early in
the epoch so it lands in `exited_clients` first. From then on the main node can
be convicted at any point later in the epoch and will never forfeit. The cost is
one extra client slot.

It also fires with no attacker at all: a node that joined late and went offline
early is a completely ordinary event, and it voids the conviction of anyone who
joined before it.

**Fix.** Do not walk the two lists in lockstep. Match by identity — index
`clients_state.clients` by `NodeIdentity`, or iterate `exited_clients` in the
outer loop and look the permanent record up. The same applies to the
`finished_clients` cursor at `:141-153`; it happens to be safe today only
because `epoch_state.clients` is built from `get_active_clients_ids()`, which
preserves join order, and `retain` is stable. That is an invariant nobody
stated and nothing tests.

## 2. Bounty recipient is unvalidated when the verdict is omitted (Critical, Class B)

**File:** `architectures/decentralized/solana-treasurer/programs/solana-treasurer/src/logic/participant_bond_finalize_withdraw.rs:167-183`
**Test:** `memnet_treasurer_bounty_recipient::the_slashed_client_can_name_itself_as_the_bounty_recipient`

The bounty is split two ways. With voters, every recipient is unpacked and
checked against both the recorded voter and the collateral mint (`:194-201`).
Without voters, it is not checked at all:

```rust
if voters.is_empty() {
    let reporter = context.remaining_accounts.first()
        .ok_or(error!(ProgramError::MissingReporter))?;
    transfer(..., to: reporter.to_account_info(), ...)?;   // no owner check
}
```

`remaining_accounts` is supplied by the signer — who is the participant being
slashed. SPL's `transfer` will reject a non-token account and a mint mismatch,
so those two are covered by the token program. **The owner is not checked by
anyone.**

The reachability is the part that matters and it is broader than it looks.
`audit_verdict` is an `Option` account (`:57-65`). Anchor evaluates none of an
optional account's constraints — including its `seeds` — when the client
declines to pass it. So reaching the unchecked branch does not require that no
verdict exists; it only requires that the withdrawing participant **omit** the
verdict account. A target convicted by a full verifier quorum can leave the
`audit_verdict` slot empty and take the whole bounty itself, denying the
verifiers the payment the design promises them.

**Failure scenario.** Bond 500, `slashing_rate_per_client = 200`,
`slash_bounty_bps = 5000`. The participant is convicted and forfeits 200. At
`participant_bond_finalize_withdraw` it passes its own ATA as
`remaining_accounts[0]`. 100 tokens — the bounty half of its own forfeiture —
are transferred straight back to it. Net loss 100 instead of 200; the run's
`slash_bounty_bps` is silently halved for anyone who reads the code.

The test is `memnet_treasurer_bounty` with one argument changed.

**Fix.** Two parts, and both are needed:

1. In the empty branch, apply the same three checks the voter branch applies,
   and additionally reject `reporter.owner == user.key()`.
2. Stop letting the account be omitted. Derive the `AuditVerdict` PDA and
   require it whenever one exists — an `Option` that the beneficiary chooses is
   not a constraint. If the optionality is needed for runs that never audit,
   gate it on something the participant does not control.

## 3. Votes from different rounds are pooled into one quorum (High)

**File:** `.../logic/run_submit_audit_verdict.rs:103-122` and `:140-163`
**Test:** `memnet_treasurer_cross_round_quorum::votes_from_different_committees_are_pooled_into_one_quorum`

The committee is drawn per **round**: `from_coordinator_with_tie_breakers(coordinator, 0, ...)`
resolves offset 0 to `current_round()`, whose `random_seed` is replaced on every
`start_round_train` (`coordinator.rs:1167`). The verdict it writes into resets
per **epoch** (`:133-135`), and an epoch is many rounds by construction —
`Coordinator::check_config` rejects `max_round_train_time >= epoch_time`
(`coordinator.rs:1256`).

Nothing records which round a vote came from. `verdict.voters` accumulates
across rounds and is compared against `quorum`, computed from whichever round
happens to be current when the last vote lands.

**Failure scenario.** The security argument for the audit committee is sampling:
to convict, an attacker must hold ⅔ of the seats in a single draw. With
`verification_percent = 10` on a 100-node run that is 6 of 9 seats, which a 10%
sybil holder reaches with probability ≈10⁻⁶ in any given round. Under the
current code the same attacker instead waits: each round redraws, roughly 0.9 of
their nodes is seated per round, and after ~7 rounds they have accumulated 6
*distinct* verifier votes against the same honest target. Rounds are minutes.
The requirement collapses from "⅔ of one committee" to "⅔ of the union over the
epoch", which is not a security assumption at all.

The test convicts an honest node with two verifiers that were never seated on
the same committee, and asserts both that the two votes landed in different
rounds and that the second voter was absent from the first round's verifier set.

**Fix.** Record the round height on the verdict at the first vote and reject any
vote from a different round, resetting per round rather than per epoch. If a
quorum genuinely cannot be gathered inside one round, that is an argument for a
longer round or a smaller quorum — not for pooling draws.

## 4. Daemon and chain disagree on who is a verifier (High)

**Files:** `shared/coordinator/src/audit_selection.rs:59` versus
`.../logic/run_submit_audit_verdict.rs:104` and `run_submit_appeal_verdict.rs:96`
**Test:** `psyche_coordinator::committee_selection::tests::reserving_tie_breakers_moves_the_verifier_set`

Two constructors are live at once and they do not agree:

- The verifier daemon picks its work with `select_audits_for_current_round`
  (`solana-tooling/src/daemon.rs:69`), which calls `from_coordinator`. That
  passes `round.tie_breaker_tasks` — hardcoded `0` at every call site of
  `start_round_train` (`coordinator.rs:500`, `:1009`, `:1084`).
- The treasurer gates the vote with `from_coordinator_with_tie_breakers`,
  passing the run's `tie_breaker_committee_size`.

Reserving tie-breakers changes the verifier count
(`total × pct/100` versus `(total − tb) × pct/100`) **and** shifts the verifier
position window from `[0, v)` to `[tb, tb + v)`. The two views name different
nodes.

**Failure scenario.** A run turns appeals on, which is what
`tie_breaker_committee_size > 0` means. The daemon tells node X to replay and
report; X submits and the chain answers `VerifierNotAssigned`. Meanwhile the
nodes the chain *would* accept were never told to audit anything. Quorum is
rarely or never reached, so nothing is convicted — which makes Class A
("undetected fraud past the band") trivially achievable for as long as the
appeals court is enabled.

`docs/DISPUTE_DESIGN.md` states the appeals design is backward compatible
because both knobs default to 0. That is true, and it is also why this has not
been seen: the feature has never been switched on against a live daemon.

**Fix.** One source of truth. Have `select_audits_for_current_round` take the
tie-breaker count from the same place the treasurer does, or write
`tie_breaker_committee_size` into `round.tie_breaker_tasks` at
`start_round_train` so `from_coordinator` is correct by construction. The second
is preferable: it removes the second constructor rather than keeping two in
sync.

## 5. `run_finalize_slash` trusts a stale client index (High)

**File:** `.../logic/run_finalize_slash.rs:73` and `:83-93`

`run_finalize_slash` reads `target_index` off the verdict — recorded when the
audit vote was cast — and then requires that `epoch_state.clients[target_index]`
still be the target:

```rust
target_index = verdict.target_index;                       // 73
...
let target_client = ...clients.iter().nth(target_index)?;  // 87
if *target_client.id.signer() != verdict.target.to_bytes() { TargetMismatch }
```

But `epoch_state.clients` is compacted at the end of every round
(`coordinator.rs:1043`), and every client above a departing one shifts down. The
instruction takes no fresh index and there is no other way to finalise. Once the
index drifts the verdict is stuck in `SlashPending` permanently.

**Failure scenario.** A quorum convicts a target in round *r*; because
`challenge_window_seconds > 0` the verdict goes to `SlashPending`. The challenge
window is by design longer than a round — it has to be, or nobody could react —
so at least one round boundary passes before it can be finalised. Any client
below the target that drops offline in that window shifts the target's index.
`run_finalize_slash` then fails with `TargetMismatch` forever, and the
conviction never lands.

The only recovery is a fresh quorum in a *later epoch*, which triggers
`reset_for_epoch` and restarts the whole audit — see finding 6 for what that
costs.

Note that this is the *only* place a stored index is trusted across time.
`run_submit_audit_verdict`, `run_submit_appeal_verdict` and
`run_slash_losing_verifier` all resolve the target freshly, and are sound in
this respect.

**Fix.** Take `target_index` as an instruction parameter, exactly as
`run_submit_appeal_verdict` does (`:106-110`), and validate it against
`verdict.target`. The stored index should be treated as a log entry, not as an
input.

## 6. `reset_for_epoch` destroys an appeal and an unfinished settlement (Medium-High)

**Files:** `.../logic/run_submit_audit_verdict.rs:133-135`,
`.../state/audit_verdict.rs:67-83`

There is one `AuditVerdict` PDA per `(run, target)`, reused forever. Any
verifier submitting a verdict in a new epoch triggers:

```rust
} else if verdict.epoch != current_epoch {
    verdict.reset_for_epoch(current_epoch);
}
```

`reset_for_epoch` clears `status`, `challenger`, `voters`, `appeal_voters`,
`overturn_count`, `uphold_count` and `settled_count`.

**Failure scenario A — the settlement is erased.** A verdict is `Overturned`;
the losing verifiers are supposed to forfeit, one per `run_slash_losing_verifier`
transaction, indexed by `settled_count`. The crank is permissionless and
sequential, so it takes as many transactions as there were voters. Before it
finishes, any one of those same losing verifiers submits a fresh verdict against
the same target in the next epoch. `reset_for_epoch` wipes `voters` and
`settled_count`, and the crank now fails with `VerdictNotOverturned`. Every
verifier the crank had not yet reached escapes their forfeit, for the price of
one transaction, paid by one of the people escaping.

**Failure scenario B — the appeal is erased.** A target opens a challenge; the
tie-breaker bench is mid-vote. The epoch rolls. One new audit verdict resets the
account, and the challenge, the challenger and every appeal vote cast so far are
gone.

**Fix.** Do not reuse the account across disputes. Either seed the PDA with the
epoch as well as the target, or refuse `reset_for_epoch` unless the previous
verdict reached a terminal state *and* `settled_count == voters.len()`.

## 7. The appeal bench may contain the verifiers it is judging (Medium)

**File:** `.../logic/run_submit_appeal_verdict.rs:95-100`

Within one round, TieBreaker and Verifier positions are disjoint by construction
(`committee_selection.rs:187-195`), so the question as asked in the issue — can
the tie-breaker set overlap the verifier set — is **sound in-round**.

Across rounds it is not, and nothing pins the round. The appeal is judged under
whatever draw is current when the appeal votes land, which is a different draw
from the one that produced `verdict.voters`. The code never checks
`appeal_voters ∩ voters = ∅`.

**Failure scenario.** Verifier V votes to slash in round *r*. The target
challenges. In round *r+k* V is drawn as a TieBreaker and votes on the appeal of
its own verdict. V is not neutral: if the appeal is overturned,
`run_slash_losing_verifier` forfeits V's bond. V will vote Uphold every time.
`docs/DISPUTE_DESIGN.md` argues the bench is safe because it is "a disjoint,
separately sized set" and because a tie-breaker "earns either way" so the bounty
does not bias the vote. Neither argument survives a tie-breaker who is also a
defendant.

**Fix.** Reject an appeal vote from any key present in `verdict.voters`. It is
one `contains` call and it makes the disjointness claim true rather than
approximately true.

## 8. A losing verifier escapes forfeit by leaving the epoch (Medium)

**File:** `.../logic/run_slash_losing_verifier.rs:72-108`

The crank walks `voters` by index. That walk itself is sound — `settled_count`
must equal `voter_position` (`:66`), it cannot exceed `voters.len()` (`:63`), so
it cannot be replayed, skipped or run off the end. But:

```rust
verdict.settled_count += 1;                                // 93, unconditional
...
if let (Some(index), true) = (loser_index, slashable) {    // 108
    slash_client(...)
}
```

`settled_count` advances whether or not the loser was actually slashed, and
`slashable` is only true for `Healthy` or `Dropped` (`:83-84`). A voter that is
absent from `epoch_state.clients` — which happens at the first round boundary
after they stop participating — is skipped permanently. There is no retry.

**Failure scenario.** A verifier sees an overturn coming and stops
participating. One round later it is out of `epoch_state.clients`, the crank
reaches its position, logs `slashable=false`, increments past it, and it keeps
its bond. The deterrent that the whole appeals design rests on — "if you convict
wrongly, you pay" — is optional for anyone paying attention.

**Fix.** Do not advance `settled_count` on a miss. Record the loser as
outstanding and settle against the bond directly (the treasurer holds it; it
does not need the coordinator's client list to know who owes what).

## 9. No evidence consensus; the last voter overwrites the record (Medium)

**File:** `.../logic/run_submit_audit_verdict.rs:149-153`

Every vote overwrites `committed_hash`, `replayed_hash`, `target_index`,
`batch_start` and `batch_end`. Voters are counted, never compared. The quorum is
a count of signatures, not of agreement about what happened.

Being precise about the impact, because it is narrower than it first looks: the
slash CPI ignores the hashes entirely — `slash_client` logs them and calls
`slash(index)` → `eject(index)`. So a fabricated hash does **not** by itself
cause a wrong slash; the quorum is still required. What it corrupts is the only
on-chain evidence an appeal bench or an observer has, and the
`batch_start`/`batch_end` range that `run_slash_losing_verifier` later carries.
A verifier who votes last can therefore make a correct conviction look like it
rests on evidence that does not check out, which is a clean way to manufacture a
successful appeal.

**Fix.** First voter writes the evidence; subsequent voters must match it or be
rejected. Disagreement is information — it means at least one verifier is wrong
— and it should be visible, not overwritten.

## 10. A challenged verdict has no deadline (Medium)

**Files:** `.../logic/run_open_challenge.rs:55-68`, `run_finalize_slash.rs:65`

`run_finalize_slash` only accepts `SlashPending`. Once `run_open_challenge` sets
`Challenged`, the only exits are an overturn quorum or an uphold quorum among
the tie-breakers. There is no timeout, so a bench that never reaches quorum —
apathy, offline nodes, or `tie_breaker_committee_size` set larger than the live
client count — leaves the conviction unresolved indefinitely.

Combined with the fact that opening a challenge costs nothing beyond an already
posted bond, challenging is strictly dominant for a guilty target: it cannot
make their position worse and it may stall the slash forever.

`docs/DISPUTE_DESIGN.md` explicitly prices the missing challenge bond as a known
optional refinement, so the *cost* half of this is a documented deliberate
choice and not a finding. The *deadlock* half is not covered there and is: a
free action that can permanently block a conviction is worse than a free action
that merely delays one.

**Fix.** Give `Challenged` a deadline. If the bench has not reached quorum by
then, fall through to the verdict the verifiers already reached.

## 11. Quorum can be one (Low)

**File:** `.../logic/run_submit_audit_verdict.rs:121`

`(2 * verifier_nodes).div_ceil(3).max(1)` gives quorum 1 when there is a single
verifier seat, and `verifier_nodes = (total − tb) × pct / 100` rounds *down*, so
small runs land there routinely. On the shape of run currently deployed — a
handful of clients, `verification_percent` in the low tens — the committee is 0
or 1 seats. At 0 nobody can audit at all; at 1 a single verifier unilaterally
ejects any client it names.

Nothing validates that a run's configuration produces a committee capable of
meaning anything. The answer to the issue's question "can anyone lose a bond
without a quorum agreeing" is: not through a bug, but yes through a
configuration the program accepts without comment.

**Fix.** Reject a slash when `verifier_nodes` is below a floor, and surface the
computed committee size at `run_update` time.

## 12. The voter cap can sit below quorum at scale (Low)

**Files:** `.../state/audit_verdict.rs:3`, `.../logic/run_submit_audit_verdict.rs:143`

`MAX_VERDICT_VOTERS` is 64 and votes are refused at that count. Quorum is
`⌈2n/3⌉`, so above 96 verifier seats quorum exceeds the cap and can never be
reached. At `verification_percent = 10` that is about 960 clients — inside the
target scale, not beyond it.

The account sizing itself is correct: `space_with_discriminator()` matches the
Borsh layout field for field (4316 bytes), and both vectors are pre-allocated,
so no reallocation is involved.

**Fix.** Derive the cap from the maximum committee the coordinator can produce,
or refuse a configuration whose quorum exceeds it.

## 13. Borsh accounts are sized with `std::mem::size_of` (Low)

**Files:** `.../state/run.rs:32`, `.../state/participant.rs:20`

```rust
pub fn space_with_discriminator() -> usize { 8 + std::mem::size_of::<Run>() }
```

`size_of` reports the Rust layout, including alignment padding; Borsh serialises
packed. Today both over-allocate (`Run` 232 versus 229 needed), so nothing is
broken. But it is not a correct way to size a Borsh account, and it is silently
wrong in a way that only shows up once a field is added. `AuditVerdict` gets
this right by counting fields explicitly; the other two should match it.

## 14. Slashing points and collateral units are coupled by convention (Info)

`participant_bond_finalize_withdraw:113-118` subtracts `client.slashed` —
accumulated in units of `epoch_slashing_rate_per_client` — directly from
`participant.bond_amount`, which is in collateral base units. The two are only
the same thing because the operator sets them to be. `memnet_treasurer_bounty`
encodes the assumption (`BOND = 500`, `SLASHING_RATE = 200`, expecting
`BOND - SLASHING_RATE` back) but nothing on chain enforces it. A mint with a
different decimal count than the operator assumed changes what a conviction
costs, in either direction.

Worth a comment in `RunUpdateParams` at minimum.

## 15. An unvalidated warmup witness index panics every later witness (High)

**File:** `shared/coordinator/src/coordinator.rs:486-490`
**Test:** `psyche-coordinator --test warmup_witness_index` (three cases)

The round-witness path validates the caller's proof before storing it:
`verify_witness_for_client` → `verify_client` → `clients.get(index)`
(`committee_selection.rs:222-224`), a bounds-checked lookup. The warmup path
deliberately skips that check, and says so:

```rust
// Everyone can send a witness in the warmup phase so we don't need to check for the committee
let round = self.current_round().unwrap();
for witness in round.witnesses.iter() {
    if self.epoch_state.clients[witness.proof.index as usize].id == *from {   // 488
```

The skipped check would have bounded `proof.index`; the duplicate loop then
reads the unvalidated value back out of storage and indexes with it.
`FixedVec`'s `Index` impl is `self.get(index).expect("Index out of bounds")`
(`shared/core/src/fixed_vec.rs:185-187`), so the read panics, and a panic aborts
the transaction.

**Failure scenario.** Any joined client calls `warmup_witness` with
`proof.index = u64::MAX`. Nothing rejects it and it is stored. From that moment
every other client's `warmup_witness` transaction in that round panics at line
488 before it can do anything. Warmup advances to training only when
`round.witnesses.len() == witness_nodes` or on the timeout in `tick_warmup`
(`coordinator.rs:1008`), so the run is pushed onto the timeout path with a
single witness recorded, every epoch, at the cost of one transaction per warmup
by one participant.

The caller must be a registered client (`instance_state.rs:244` calls
`find_signer`), so this is not open to the world — but it is open to every
participant, including one that joined only to disrupt.

**Fix.** Bounds check `proof.index` against `epoch_state.clients.len()` on the
way in. Storing an index nobody validated and dereferencing it later is the
pattern to remove, not the specific value. Consider making `FixedVec`'s `Index`
impl unavailable in on-chain code so that a `get` with an explicit `None` arm is
the only way to read.

## 16. The disclosure channel SECURITY.md names does not exist (Process)

`SECURITY.md` and `docs/REDTEAM_BOUNTY.md` both instruct reporters to use
GitHub private vulnerability reporting and explicitly say not to open a public
issue for anything exploitable. Private vulnerability reporting is **disabled**
on both `wienerlabs/leviathan` and `wienerlabs/leviathan-net`
(`GET /repos/{owner}/{repo}/private-vulnerability-reporting` → `{"enabled": false}`).

An outside reporter has already hit this and said so in #15, having by then
posted partial exploit details publicly because there was nowhere else to put
them. That is a predictable outcome of publishing a policy that points at a
switch nobody turned on, and it will keep happening.

Enable it on both repositories before the bounty program is advertised
anywhere. It is a repository setting, not a code change.

---

## Read and found sound

Listed so the external audit does not spend hours re-deriving these.

**`run_submit_audit_verdict.rs`**
- Bond gate: `verifier_participant.bond_amount < run.bond_minimum_amount` is
  checked before anything else (`:83-87`), and `verifier_participant` is a PDA
  seeded by the signer, so it cannot be substituted.
- Double voting: `voters.contains(verifier_key)` (`:140-142`). Correct.
- Committee membership is checked against a freshly computed selection, not a
  caller-supplied proof (`:103-108`). Correct, and the right pattern: the
  coordinator's warmup witness path takes the opposite approach and pays for it
  in finding 15.
- Target spoofing: `params.target_index` is resolved against the live client
  list and the resulting signer compared to `params.target` (`:110-118`). An
  attacker cannot name one key and slash another.
- The `AuditVerdict` PDA is seeded by `params.target`, so the account and the
  claimed target cannot disagree.
- Quorum arithmetic `⌈2n/3⌉` is right for a two-thirds rule; the `.max(1)`
  boundary is finding 11, not an arithmetic error.

**`run_open_challenge.rs`**
- The verdict PDA is seeded by `challenger.key()`, so only the target can
  challenge its own conviction. That is the intended restriction and it is
  enforced structurally rather than by a comparison that could be forgotten.
- The window is closed correctly (`now >= pending_since_unix + window` rejects),
  and it is the exact complement of the check in `run_finalize_slash`, so there
  is no gap or overlap between "can still challenge" and "can now finalise".
- Status must be `SlashPending`, so a challenge cannot reopen a resolved verdict.

**`run_submit_appeal_verdict.rs`**
- Tie-breaker membership, bond minimum, duplicate voting and the voter cap are
  all checked, mirroring the audit path.
- `verdict.target != params.target` is rejected (`:123-125`) *in addition to*
  the PDA seeding — belt and braces, and correct.
- The slash CPI uses `params.target_index`, freshly validated in this same
  instruction (`:102-110`), not the stored one. Correct, and the reason this
  instruction does not suffer finding 5.
- State machine: `Voting → SlashPending → Challenged → {Upheld, Overturned}`
  moves in one direction only; every transition is guarded by an equality check
  on the current status. No backwards move exists other than
  `reset_for_epoch` (finding 6).

**`run_slash_losing_verifier.rs`**
- The index walk cannot be replayed, skipped or run past the end (`:63-68`).
  Sound, as the issue hoped. The problem is what happens on a miss (finding 8),
  not the walk.
- The loser is resolved by public key against the live client list, not by a
  stored index. Correct.

**`participant_bond_finalize_withdraw.rs`**
- The voter branch validates every recipient three ways: unpacks it as a
  `TokenAccount`, checks `owner == voter`, checks `mint == collateral_mint`
  (`:194-201`). This is the branch the tests cover and it is correct.
- Forfeit arithmetic: `saturating_sub` on the points difference, `min` against
  the bond, `min` again for the payout (`:113-124`). It cannot underflow and it
  cannot pay out more than the bond holds. `overflow-checks = true` is set for
  the release profile (`solana-treasurer/Cargo.toml:6`), so the remaining
  unchecked arithmetic aborts rather than wraps.
- `share = bounty / voters.len()` truncates, leaving a remainder in the vault.
  Not a leak.
- The appeal-verdict branch requires `Overturned` status, a matching run, and
  that the withdrawing user is one of the recorded original voters (`:152-162`).
  Correct.

**`participant_bond_deposit.rs` / `participant_bond_request_withdraw.rs`**
- Deposit rejects zero, moves tokens before crediting, and requires
  `user_collateral.owner == user` with no delegate.
- Request checks `collateral_amount > bond_amount - pending`, so pending
  withdrawals cannot be double-counted.

**`participant_claim.rs`**
- Bond gate, unclaimed-points check, and 1 collateral per point. The
  `participant_earned_points - claimed_earned_points` subtraction at `:91-92` is
  unchecked, but `clients_state.clients` is append-only (`instance_state.rs:366`)
  and entries are never removed or reset, so `earned` is monotonic and the
  subtraction cannot go negative. Sound, for a reason worth writing down.

**`run_slash.rs`, `run_set_slash_bounty.rs`, `run_set_challenge_config.rs`, `run_bond_config_update.rs`, `run_update.rs`**
- All authority-gated on `run.main_authority == authority.key()`, and the run
  account is cross-checked against both coordinator accounts where it matters.

**`CommitteeSelection`**
- `compute_shuffled_index` is a swap-or-not shuffle over `[0, total_nodes)`, so
  it is a permutation: one index maps to one position, and the position ranges
  for TieBreaker / Verifier / Trainer are disjoint. **A client cannot land in
  two committees in the same draw.** The overlap that does exist is across
  draws (finding 7).
- `CommitteeSelection::new` validates `total_nodes >= tie_breaker_nodes`,
  `verification_percent <= 100` and the witness count.

**Other programs** — `solana-distributor` and `solana-mining-pool` were read at
lower depth, since neither is on the conviction path. Merkle verification,
vesting, claim accounting, freeze guards and authority gating all appear
correct. The `.unwrap()` on the `u128 → u64` conversion in
`lender_claim.rs` is reachable only at balances near `u64::MAX` and aborts
rather than truncating; worth changing for tidiness, not urgency.

---

## The five questions in the issue

**Can anyone lose a bond without a quorum agreeing?**
Yes, in three ways. Through configuration, because a one-seat committee has
quorum one (11). Through accumulation, because votes from different draws are
pooled and no single committee ever has to agree (3). And the admin `run_slash`
path bypasses the committee entirely by design.

**Can a target be convicted twice for the same epoch?**
No. `VerdictAlreadyResolved` blocks a second resolution within an epoch, and
`eject` refuses a client that is already `Ejected`. Across epochs a target can
be convicted again, which is intended. The related defect is the opposite one:
a target convicted *once* may not be charged at all (1).

**Can an attacker force a slash of an honest node cheaply, and what does it cost
them?**
Yes, and far more cheaply than designed. Finding 3 reduces the cost from "⅔ of
one randomly drawn committee" to "⅔ of the union of the committees drawn over an
epoch". For a 10% sybil holder at `verification_percent = 10` on 100 nodes that
is roughly seven rounds of waiting instead of a ~10⁻⁶ chance per round. The
out-of-pocket cost is the minimum bond on each sybil identity, and those bonds
are never at risk, because the honest target has no way to appeal successfully
against a bench that may contain the accusers (7).

**Can the appeals path be used to escape a correct conviction?**
Yes — not by winning the appeal, but by never finishing it. A challenge is free
and `Challenged` has no deadline (10), so a guilty target can park a correct
conviction indefinitely. Separately, and outside the appeals path, a correct
conviction that reached `SlashPending` becomes permanently unfinalisable as soon
as the target's index drifts (5), and a conviction that fully lands may still
charge nothing (1).

**Is there any path where forfeited funds leave the vault to an address that is
not a recorded voter?**
Yes. Finding 2: when the verdict account is omitted, the bounty goes to an
arbitrary token account named by the person being slashed, with no owner check.
Every other payout path is checked.

---

## Recommended order of work

1. Finding 1, then finding 2. Both are Class B, neither needs privileged access,
   and until they are fixed the bond mechanism does not reliably take money from
   anyone.
2. Finding 4 before appeals are enabled on any run, and finding 3 before the
   committee is relied on for anything.
3. Findings 5, 6, 8 together — they are all "a conviction that was reached does
   not complete".
4. Finding 15, which is a one-line bounds check against a participant that can
   disrupt every warmup.
5. The rest, and finding 16, which costs nothing and should not wait for a
   release.
