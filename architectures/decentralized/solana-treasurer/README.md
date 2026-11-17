# Psyche Solana Treasurer

This smart contract provides an Incentive layer on top of the Psyche's coordinator program.

## How it works

The `Treasurer.Run` PDA can be created by an authority using the `run_create` IX.

Creating the `Treasurer.Run` PDA will automatically create a training `Coordinator.CoordinatorInstance` and `Coordinator.CoordinatorAccount` owned by the `Treasurer.Run` smart contract's PDA, it will use the `Coordinator.init_coordinator` IX.

The underlying `Coordinator.CoordinatorAccount` can then be configured indirectly, through using the `Treasurer.run_update` for all permissioned `Coordinator` IXs.

The underlying `CoordinatorAccount` permissionless IXs such as `tick` and `witness` can be used by simply using the `Coordinator` IXs directly, without using the `Treasurer` at all. Joining depends on the run's join authority: a run admitted by an outside key is joined with the `Coordinator`'s `join_run` and that key's authorization, while a run that is its own join authority (the `Treasurer.Run` PDA) is gated on the bond and joined only through `participant_join_run`.

A set of reward tokens can then be deposited inside of the `Treasurer.Run`'s ATA for fair distribution during the training of the underlying coordinator run.

Once a client has earned compute points in the underlying `Coordinator`'s run, that same client can then claim to have participated in the run by creating a `Treasurer.Participant` PDA.

Once that client has earned enough points and once the reward token has been deposited into the `Treasury.Run`'s ATA, the user can the directly withdraw the reward tokens to its wallet using the `participant_claim` IX.

The incentive reward rate can be configured in the `Coordinator.CoordinatorAccount` itself (epoch rewards rates).

## Solana Instructions

To achieve this the smart contract provides the following capabilities:

- `run_create`, Create a normal `Run` owned by the `Treasurer`
- `run_update`, Configure the underlying `Run`'s Psyche coordinator
- `participant_create`, Must be called before a user can claim reward tokens
- `participant_claim`, Once a user earned points on the `Run`'s coordinator, the user can withdraw the reward tokens proportional share of the `Run`'s treasury

## Bonds

Leviathan adds bonded participation on top of the rewards above. A conviction reaches a bond in two steps: the coordinator charges slashed points at the epoch boundary, and settlement forfeits them from the bond when the participant finalizes a withdrawal. Every bond check reads what is actually left, from chain state:

- the **unforfeited bond** is the posted bond less the convictions it has not settled yet, which is what a slash could still take; claims and challenges are held to it;
- the **committed bond** also leaves out a withdrawal already requested, which is the bond that stays behind new work; joins and verdict or appeal votes are held to it.

The bond instructions:

- `run_bond_config_update` sets the run's bond minimum and withdraw delay (a non-zero minimum requires a non-zero delay)
- `participant_bond_deposit` posts collateral as bond
- `participant_join_run` admits the participant to the next epoch of a run that is its own join authority, against its committed bond; the join authorization is active only inside this instruction (wienerlabs/leviathan-net#34)
- `participant_bond_request_withdraw` starts a withdrawal; the delay is fixed when it is requested
- `participant_bond_finalize_withdraw` settles any charged convictions out of the bond and returns the rest.
  - **When it is held:**
    - while the participant trains in an epoch the coordinator has not charged yet;
    - while its own conviction is still inside its challenge window or appeal, because a bond that left then would owe nothing;
    - while any verdict vote the participant cast can still cost it (wienerlabs/leviathan-net#39).
  - **Its own conviction:**
    - One that ejected it is charged first, if `run_charge_conviction` has not charged it already.
    - One that has run out its windows without reaching the ledger, and can no longer reach it because the participant is not in an epoch that will be charged, is charged here at the run's slashing rate and recorded as upheld. So is a `Due` conviction (wienerlabs/leviathan-net#36).
    - Either way the bounty is recorded on the verdict for its recipients and not paid here (wienerlabs/leviathan-net#64).
  - **Slashes no verdict accounts for**, such as a `run_slash` or an equivocation, pay their bounty to a reporter account the withdrawal names, which cannot be the participant's own (wienerlabs/leviathan-net#72).
- `run_charge_conviction` charges a convicted target's bond for a conviction that ejected it.
  - **When:** once the epoch it was ejected in has been charged by the coordinator.
  - **How much:** that epoch's charge, as settled points, out of what the bond still holds.
  - It records what the forfeit owes the conviction's voters, or the bench if it reached its quorum to uphold.
  - A target that never created a participant account has no bond and is charged nothing.
  - Anyone may send it. It touches no token account and waits on nobody, so a large committee, a closed token account or an absent target cannot hold it up.
  - A verdict that stands is taken over for a new dispute only once it is charged and every recipient credited, so the bounty cannot follow a later dispute (wienerlabs/leviathan-net#64).
- `run_pay_verdict_bounty` credits one recipient's bond with its share of what a verdict's losing side forfeited, once that is final:
  - an overturn's bench, out of the losing verifiers' forfeits;
  - a conviction's voters, or the bench that upheld it, out of the target's.

  It credits one seat per transaction. Anyone may send it, and each seat is paid once (wienerlabs/leviathan-net#39, #64).
- `participant_release_verdict_vote` stops a verdict vote holding its voter's bond once the verdict says it can no longer cost it. That is when the conviction stands, or when the verdict never reached quorum and either its round is over or the voter withdraws the vote itself. A withdrawn vote comes off the verdict's count. Anyone may send it, and each vote is released once. A verdict's account is taken over for a new dispute only once every vote on the old one is released or charged (wienerlabs/leviathan-net#39)
- `run_slash`, `run_submit_audit_verdict`, `run_open_challenge`, `run_submit_appeal_verdict`, `run_finalize_slash` and `run_slash_losing_verifier` convict and adjudicate, see `docs/DISPUTE_DESIGN.md`. A conviction ejects its target, and the coordinator charges ejections only when the epoch ends, so it ejects only where that charge is certain (`RoundTrain`, `RoundWitness`, `Cooldown`). A ruling reached anywhere else is recorded as `Due` and lands later, or at the target's withdrawal. A conviction (`run_slash`, or a vote of `run_submit_audit_verdict`) names the payload the target committed to: the epoch it was trained in (when that epoch started), its step, nonce and results digest, for the batch the conviction records. The treasurer rebuilds the commitment hash from those and the coordinator's run id, and the same transaction has to carry an ed25519 precompile instruction in which the target's p2p key, read from the coordinator roster, signs exactly that hash. A payload from outside the current epoch is refused too, by the epoch it names as well as its step, since a run restarted from an earlier step trains the step again (wienerlabs/leviathan-net#67). The verdict records the evidence, so anyone can rebuild the hash from the account (wienerlabs/leviathan-net#43). A run with a challenge window must also give appeals a deadline. The bench is drawn afresh each round, so `run_submit_appeal_verdict` refuses a seat to a verifier who voted on the verdict and to the target itself. `run_set_challenge_config` carries the bench size into the coordinator, which reserves those tie-breaker seats in every round it starts from then on, so the clients, the audit lottery and the treasurer draw one committee from the round; it is refused while a round's seed is committed (`RoundSeedCommitted`), and is sent again a slot later. `run_slash_losing_verifier` charges each verifier whose verdict an appeal overturned. It takes the epoch's slashing rate straight from the bond, wherever the verifier is, closes the verifier's vote, and records what the forfeit owes the bench. Once every loser is charged, `run_pay_verdict_bounty` credits each tie-breaker's bond with its share, one seat at a time, and anyone may send it. Switching appeals off finalises an appeal still open instead of holding it (wienerlabs/leviathan-net#39). `run_slash_equivocation` convicts a target that signed two different payloads for one batch, on the two signatures alone: anyone may send it, with an ed25519 precompile instruction for each commitment, and it lands only as an ejection the epoch will charge (wienerlabs/leviathan-net#46)
