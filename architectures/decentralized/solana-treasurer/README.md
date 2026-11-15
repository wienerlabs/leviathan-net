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
- `participant_bond_finalize_withdraw` settles any charged convictions out of the bond, pays any slash bounty, and returns the rest. It is held while the participant trains in an epoch the coordinator has not charged yet, and while its own conviction is still inside its challenge window or appeal, because a bond that left then would owe nothing. A conviction that has run out its windows without reaching the ledger, and can no longer reach it because the participant is not in an epoch that will be charged, is charged here against the bond at the run's slashing rate and recorded as upheld (wienerlabs/leviathan-net#36)
- `run_slash`, `run_submit_audit_verdict`, `run_open_challenge`, `run_submit_appeal_verdict`, `run_finalize_slash` and `run_slash_losing_verifier` convict and adjudicate, see `docs/DISPUTE_DESIGN.md`. A conviction ejects its target, and the coordinator charges ejections only when the epoch ends, so ejections, verdict votes and appeal votes all happen while an epoch is still to be charged; a run with a challenge window must also give appeals a deadline
