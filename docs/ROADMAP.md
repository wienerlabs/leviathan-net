# Finishing Leviathan

Roadmap to done, as of 28 October 2026. A typeset copy is in [ROADMAP.pdf](ROADMAP.pdf).

Leviathan is finished when a permissionless network on Solana mainnet trains an open model, checks every contribution, pays the people who trained it, and lets them choose the next one. This page lists what is done, what is left, and the order the rest has to happen in.

## 1. What done means

All five of these are true on mainnet. Until then the work is not finished, however much code has shipped.

- **The flagship run finishes.** An open-join, bonded run trains the 1B model to its announced token target, and the weights are published.
- **Every contribution is checked in public.** The audit lottery, the replay verifier, the jury and the appeals court run on mainnet, an outside audit has closed its findings, and convictions show on the dashboard.
- **$LEVI is live.** TGE directly on Solana rails after the legal opinion: Squads custody, the Raydium pool, vesting on chain. Pay for work, no yield.
- **The model earns.** Inference v0 serves the model, validates its outputs and pays revenue into the treasury.
- **The loop turns again.** A futarchy market picks the next model and the treasury funds its run.

> The loop turns again. But this time, it belongs to the people who train it.

## 2. Where it stands

The protocol and its security layer are built and proven on devnet. What is left is a real model on real data, the outside gates, and mainnet.

**35 of 75 items are done.** Counts are items, not effort.

| Phase | Status | Done | What it covers |
|---|---|---|---|
| 0, Proof | Done | 5 of 5 | Sim on real transformer gradients, decision log, whitepaper, the site |
| 1, Devnet core | Done | 10 of 10 | Bond custody, audit lottery, slash, verifier core, live conviction |
| 2, Genesis readiness | In progress | 15 of 34 | Jury, appeals, a live cheater slashed, rented GPUs. Next: model, data, rehearsals |
| 3, Mainnet | Prepared | 5 of 21 | Runbooks and briefs are written. Waits on the audit, counsel and tokenomics |
| 4, The second loop | Not started | 0 of 5 | Inference revenue and futarchy close the loop |

October so far:

- **9 Oct.** Two RTX 3090s rented on Nosana trained one model on the flagship devnet run, each node signing its own witness transactions.
- **10 Oct.** The verifier daemon convicted a live cheater from what it broadcast: distance 5.9968 against a 0.05 band, 0 of 500 bond returned.
- **14 Oct.** The internal security review was published: three passes, 3 critical and 6 high-severity bugs found, every code finding fixed.
- **16 Oct.** Trainers and the verifier now read the same batches across shards, and every run config is checked before it goes on chain.

## 3. The critical path

Twelve steps, in order. The audit and counsel run on outside calendars, so they are booked at step 3 and run alongside steps 4 to 8. Steps marked as milestones are gates.

1. **Decide how outsiders join.** An admission service on a join authority of its own, a pre-granted batch, or an open run. Nothing can be invited until this is decided. *Waits on the owner.* ([FIRST_RUN, section 0](https://github.com/wienerlabs/leviathan-net/blob/main/docs/FIRST_RUN.md))
2. **Pick the model and lock the data mix.** English and Turkish need a multilingual tokenizer, which means 4 bytes per token and a re-tokenized corpus: FineWeb-Edu or DCLM as the web base, FineWeb-2 for Turkish, mixed through WeightedHttp, pinned to commits and mirrored. *Waits on the owner, then build.* ([FIRST_RUN, section 2](https://github.com/wienerlabs/leviathan-net/blob/main/docs/FIRST_RUN.md))
3. **Book the outside audit and counsel.** The audit prep package and the legal briefing are written. Both have long lead times and gate everything with money at stake. *Outside calendar: waits on budget and outside parties.* ([leviathan#4](https://github.com/wienerlabs/leviathan/issues/4), [leviathan#5](https://github.com/wienerlabs/leviathan/issues/5))
4. **Run the FineWeb rehearsal on devnet.** The 19M Llama 2 init on the first FineWeb shard, with the verifier in dry run. Real text through the whole loop for the first time. *Waits on rented GPUs.* ([fineweb-rehearsal.toml](https://github.com/wienerlabs/leviathan-net/blob/main/config/leviathan/fineweb-rehearsal.toml))
5. **Calibrate the band for the chosen model.** Measure honest drift on the 3090, 4090, A100 and H100. Set too low, honest nodes get slashed; set too high, a cheater gets more room. *Waits on hardware.* ([net#7](https://github.com/wienerlabs/leviathan-net/issues/7))
6. **Close the engineering left in Phase 2.** CI that compiles the verifier, the collusion estimate, re-join after an epoch, the relay at relay.leviathan.run with a live scale test, and the loss curve on the dashboard. *Waits on build.* ([leviathan#17](https://github.com/wienerlabs/leviathan/issues/17), [leviathan#18](https://github.com/wienerlabs/leviathan/issues/18), [net#23](https://github.com/wienerlabs/leviathan-net/issues/23))
7. **Milestone: 350M rehearsal and the retro.** Internal nodes train 350M, then the telemetry retro turns today's no-go into a go for 1B, or names what still blocks it. *Waits on GPUs.* ([net#10](https://github.com/wienerlabs/leviathan-net/issues/10), [leviathan#3](https://github.com/wienerlabs/leviathan/issues/3))
8. **Milestone: first public run.** The 1B flagship, open to outsiders through the admission service, bondless as stage one and on devnet while the audit runs (owner decision 2), with a token target and a live loss curve. Until bonds switch on, a caught cheater forfeits its rewards and nothing else. *Waits on steps 1 to 7.* ([net#11](https://github.com/wienerlabs/leviathan-net/issues/11))
9. **Audit, counsel, tokenomics, name and license close.** The audit closes its critical and high findings, counsel's memo lands, and the token numbers, the license and the name are locked. *Outside calendar: waits on outside parties and the owner.* ([leviathan#6](https://github.com/wienerlabs/leviathan/issues/6), [leviathan#10](https://github.com/wienerlabs/leviathan/issues/10))
10. **Milestone: deploy to mainnet.** Fresh program keys made offline, Squads as upgrade and run authority, round seeds from VRF or a recent blockhash, a dedicated RPC, monitoring with on-call. *Waits on step 9.* ([leviathan#7](https://github.com/wienerlabs/leviathan/issues/7))
11. **Milestone: mainnet Genesis Run with bonds on, and the TGE.** A written go memo from engineering, ops and counsel, the red team live with a funded endowment, then unpause. For the first 24 hours, nothing changes except an emergency pause. *Waits on step 10.* ([leviathan#8](https://github.com/wienerlabs/leviathan/issues/8), [leviathan#2](https://github.com/wienerlabs/leviathan/issues/2))
12. **Milestone: close the second loop.** Inference v0 pays the treasury, a Wienerpad futarchy market picks the next model, and the treasury funds its run. *Waits on step 11.*

## 4. Every item

Sourced from the issues in the three repositories, the task ladder and the launch documents. Status is one of: done; shown (done in practice, the issue still to close); in progress; open. Milestones are marked.

### Phase 0, Proof, and Phase 1, Devnet core: 15 of 15 done

- Prior-art sweep of Psyche, training frameworks and verification
- Decision log locked, whitepaper v0
- Sim on real transformer gradients: honest, sign-flip, ALIE, audit; the bond law held at 9.8 rounds against a theoretical 10
- Economy calibration: break-even bond against audit rate on market GPU cost
- The site, leviathan.run, with the manifesto and the live dashboard
- Fork bootstrap of the Psyche substrate as leviathan-net
- Code map of the substrate, with the bond attach point decided
- Devnet deploy under our own program IDs
- Bond custody in the treasurer, with a withdraw delay as the challenge window
- Audit lottery drawn from the round seed
- Dispute and slash, carried into epoch-end settlement
- Aggregation that holds under sparse, sign-compressed transport
- Verifier core: replay and relative distance against a band
- Training swarm smoke on live devnet
- Live devnet conviction, bond forfeited into the run vault

### Phase 2, Genesis readiness: 15 of 34 done

| Status | Item | Track | Ref |
|---|---|---|---|
| Done | **Bonded jury vote replaces the single-key slash.** Two-thirds quorum; the daemon casts a bonded verdict instead of slashing alone. | Protocol | [net#4](https://github.com/wienerlabs/leviathan-net/issues/4) |
| Done | **Appeals court for wrongful convictions.** The accused can convene a larger bonded jury; tie-breakers earn a symmetric bounty from whoever loses. | Protocol | [net#25](https://github.com/wienerlabs/leviathan-net/pull/25) |
| Done | **Bond enforced at join time.** No posted bond, no admission to a bonded run. | Protocol | [net#3](https://github.com/wienerlabs/leviathan-net/issues/3) |
| Done | **Robust aggregation in the client apply path.** Centered clip and outlier excision before the sign step, opt-in. | Protocol | [net#6](https://github.com/wienerlabs/leviathan-net/issues/6) |
| Done | **A live cheater convicted from what it broadcast.** Distance 5.9968 against a 0.05 band, slash on chain, 0 of 500 bond returned. | Protocol | [net#29](https://github.com/wienerlabs/leviathan-net/pull/29) |
| Done | **Band calibration harness.** RTX 3060 Ti measured at 0.029 against the 0.05 default. | Protocol | [net#24](https://github.com/wienerlabs/leviathan-net/pull/24) |
| Done | **Rented GPUs trained one model on the flagship devnet run.** Two RTX 3090s on Nosana, each signing its own witness transactions. | Training and data | 9 Oct |
| Done | **Trainers and the verifier read the same batches.** Sorted shards, one sequence count, the run's shuffle; a multi-shard run can no longer slash an honest node. | Training and data | [90c27535](https://github.com/wienerlabs/leviathan-net/commit/90c27535) |
| Done | **FineWeb rehearsal config and a guard on every run config.** Model and data pinned to commits; a config string that would be cut short on chain now fails the test. | Training and data | [6a398074](https://github.com/wienerlabs/leviathan-net/commit/6a398074) |
| Done | **One-line node installer with bond funding.** Generate a wallet, fund the bond, join. | Network | [net#8](https://github.com/wienerlabs/leviathan-net/issues/8) |
| Done | **Dedicated devnet RPC.** Protocol operations off the shared public endpoint. | Network | [net#1](https://github.com/wienerlabs/leviathan-net/issues/1) |
| Done | **Self-hosted relay deployed, mesh proven end to end.** Clients can target our own iroh relay when they cannot hole-punch. | Network | [net#21](https://github.com/wienerlabs/leviathan-net/pull/21) |
| Done | **Monitoring design: six kill-switch alerts and an ops runbook.** Deployed with on-call in Phase 3. | Network | [leviathan#9](https://github.com/wienerlabs/leviathan/issues/9) |
| Done | **Internal security review before the outside audit.** Three passes over the programs, the verifier and the p2p layer; 3 critical and 6 high found and fixed. | Audit and legal | [leviathan#15](https://github.com/wienerlabs/leviathan/issues/15) |
| Done | **Red-team bounty design and a private disclosure channel.** Economic-security breaks are in scope; funding the endowment is a Phase 3 item. | Audit and legal | [REDTEAM_BOUNTY](https://github.com/wienerlabs/leviathan/blob/main/docs/REDTEAM_BOUNTY.md) |
| Shown | **Verifier daemon convicts in the swarm.** Shown live on 10 October; check the acceptance sentence and close the issue. | Protocol | [net#5](https://github.com/wienerlabs/leviathan-net/issues/5) |
| Shown | **Devnet redeploy with the grown Run layout.** Redeployed in August; confirm the real challenge-window bullet and close. | Protocol | [net#2](https://github.com/wienerlabs/leviathan-net/issues/2) |
| In progress | **Dashboard on live data, with the loss curve.** Renders the indexer telemetry today. Left: the refresh path, the Helius devnet RPC, and the loss curve, which needs the indexer to read WitnessMetadata. | Product | [net#26](https://github.com/wienerlabs/leviathan-net/issues/26), leviathan-web#1 |
| In progress | **Docs for what shipped since July.** Appeals, the live cheater run, rented GPUs, real data. | Product | leviathan-web#2 |
| Open | **Join policy for the first public run.** Admission service, pre-granted batch or open run, on a join authority used for this run alone. | Protocol | [FIRST_RUN, section 0](https://github.com/wienerlabs/leviathan-net/blob/main/docs/FIRST_RUN.md) |
| Open | **Model family and tokenizer for English and Turkish.** Sets VRAM, which cards can join, and 4 bytes per token. | Training and data | [FIRST_RUN, section 2](https://github.com/wienerlabs/leviathan-net/blob/main/docs/FIRST_RUN.md) |
| Open | **Data mix tokenized, pinned and mirrored.** datatrove shards of the web base and FineWeb-2 Turkish, mixed through WeightedHttp, with a mirror beside the Hub. | Training and data | data |
| Open | **FineWeb rehearsal on devnet.** 19M model, first FineWeb shard, verifier in dry run. | Training and data | [config README](https://github.com/wienerlabs/leviathan-net/blob/main/config/leviathan/README.md) |
| Open | **Band calibration across hardware classes.** 3090, 4090, A100 and H100, for the chosen model. | Protocol | [net#7](https://github.com/wienerlabs/leviathan-net/issues/7) |
| Open | **relay.leviathan.run and a live scale test.** Point the DNS at the self-hosted relay, then connect the Nosana swarm to it; that closes net#9. | Network | [net#22](https://github.com/wienerlabs/leviathan-net/issues/22), [net#23](https://github.com/wienerlabs/leviathan-net/issues/23) |
| Open | **Nodes re-join cleanly across epochs.** Needed before any run with churn. | Network | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |
| Open | **CI compiles the libtorch-bound crates.** Today the verifier, daemon, network and client can break with every gate green. | Protocol | [leviathan#17](https://github.com/wienerlabs/leviathan/issues/17) |
| Open | **Collusion capital accounts for committee sampling.** The published figure reads about an order of magnitude low. | Economics | [leviathan#18](https://github.com/wienerlabs/leviathan/issues/18) |
| Open | **Token target, batch schedule and round length.** Round length measured on a 3060, the target stated in tokens and then not changed. | Training and data | [FIRST_RUN, section 4](https://github.com/wienerlabs/leviathan-net/blob/main/docs/FIRST_RUN.md) |
| Open | **Committee size and the bond stage of the first public run.** Audit rate and bond are one decision; the band is settled. | Economics | [FIRST_RUN, section 3](https://github.com/wienerlabs/leviathan-net/blob/main/docs/FIRST_RUN.md) |
| Open | **GPU capacity for the Genesis runs.** Sourced from Nosana, with a budget for 350M and 1B. | Network | [leviathan#14](https://github.com/wienerlabs/leviathan/issues/14) |
| Open, milestone | **350M rehearsal with internal nodes.** The first run at a size that means something. | Training and data | [net#10](https://github.com/wienerlabs/leviathan-net/issues/10) |
| Open, milestone | **Telemetry retro: go or no-go for 1B.** The decision is no-go until the Phase 2 gates pass. | Training and data | [leviathan#3](https://github.com/wienerlabs/leviathan/issues/3) |
| Open, milestone | **First public run: the 1B flagship, bondless stage one.** Public invitation, a finish line in tokens, a loss curve anyone can watch. | Training and data | [net#11](https://github.com/wienerlabs/leviathan-net/issues/11) |

### Phase 3, Mainnet: 5 of 21 done

| Status | Item | Track | Ref |
|---|---|---|---|
| Done | **Legal briefing packet for counsel.** Bond and token questions framed for the memo. | Audit and legal | [LEGAL_BRIEFING](https://github.com/wienerlabs/leviathan/blob/main/docs/LEGAL_BRIEFING.md) |
| Done | **Audit prep package.** Scope, threat model and the internal findings for the outside auditor. | Audit and legal | [AUDIT_PREP](https://github.com/wienerlabs/leviathan/blob/main/docs/AUDIT_PREP.md) |
| Done | **Tokenomics and TGE design for $LEVI.** Squads custody, no launchpad, no yield; final numbers wait on counsel. | Economics | [TOKENOMICS](https://github.com/wienerlabs/leviathan/blob/main/docs/TOKENOMICS.md) |
| Done | **Mainnet deploy runbook.** Prerequisites, program order, authority handover, emergency path. | Network | [MAINNET_DEPLOY](https://github.com/wienerlabs/leviathan/blob/main/docs/launch/MAINNET_DEPLOY.md) |
| Done | **Genesis launch checklist and disclosures.** Launch-day sequence and the kill criteria. | Product | [GENESIS_LAUNCH](https://github.com/wienerlabs/leviathan/blob/main/docs/launch/GENESIS_LAUNCH.md) |
| Open | **Outside security audit, critical and high closed.** The gate for bonds, mainnet and the TGE. | Audit and legal | [leviathan#4](https://github.com/wienerlabs/leviathan/issues/4) |
| Open | **Legal opinion on the bond and the token.** Classification, collateral and payout structure. | Audit and legal | [leviathan#5](https://github.com/wienerlabs/leviathan/issues/5) |
| Open | **Token numbers locked after counsel.** Allocation, vesting, emission and the real bond parameters. | Economics | [leviathan#6](https://github.com/wienerlabs/leviathan/issues/6) |
| Open | **Round seeds that cannot be ground.** VRF or a recent blockhash in place of sha256 of the timestamp and slot. | Protocol | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |
| Open | **Fresh mainnet program keys under Squads.** Made offline; Squads holds the upgrade authority and the run authority. | Protocol | [MAINNET_DEPLOY](https://github.com/wienerlabs/leviathan/blob/main/docs/launch/MAINNET_DEPLOY.md) |
| Open | **Collateral mint chosen.** A stable token or the network token, per the legal structure. | Economics | [MAINNET_DEPLOY](https://github.com/wienerlabs/leviathan/blob/main/docs/launch/MAINNET_DEPLOY.md) |
| Open | **Dedicated mainnet RPC and websocket.** No public free endpoints on the critical path. | Network | [MAINNET_DEPLOY](https://github.com/wienerlabs/leviathan/blob/main/docs/launch/MAINNET_DEPLOY.md) |
| Open | **Monitoring deployed with an on-call rotation.** The six alerts live, paging a person. | Network | [OPS_RUNBOOK](https://github.com/wienerlabs/leviathan/blob/main/docs/ops/OPS_RUNBOOK.md) |
| Open | **Red team live.** Public brief, a funded endowment, a non-zero slash bounty. | Audit and legal | [leviathan#2](https://github.com/wienerlabs/leviathan/issues/2) |
| Open | **Name and trademark lock.** Trademark search with counsel, domains, handles. | Product | [leviathan#10](https://github.com/wienerlabs/leviathan/issues/10) |
| Open | **A license for the main repository.** wienerlabs/leviathan is public with no license file; the protocol fork is Apache 2.0. | Product | [leviathan](https://github.com/wienerlabs/leviathan) |
| Open, milestone | **Mainnet deployment.** Authorizer, coordinator, treasurer; addresses recorded at deploy time. | Network | [leviathan#7](https://github.com/wienerlabs/leviathan/issues/7) |
| Open, milestone | **Mainnet Genesis Run with bonds on.** After a signed go memo; money at stake. | Training and data | [leviathan#8](https://github.com/wienerlabs/leviathan/issues/8) |
| Open, milestone | **$LEVI TGE on Solana rails.** The distributor for airdrop and vesting, the pool opened, the addresses shown on the site. | Economics | [leviathan#6](https://github.com/wienerlabs/leviathan/issues/6) |
| Open | **Inference v0.** vLLM workers, TOPLOC validation and a settlement program that pays the treasury. | Protocol | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |
| Open | **Checkpoint lineage on Arweave.** Every published checkpoint traceable, through the zk-lokomotive rail. | Training and data | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |

### Phase 4, The second loop: 0 of 5 done

| Status | Item | Track | Ref |
|---|---|---|---|
| Open, milestone | **A futarchy market picks the next model.** Wienerpad prices the candidates on expected inference revenue. | Economics | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |
| Open | **The treasury funds the next run.** Inference revenue in, the next Genesis Run out. | Economics | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |
| Open | **DanteGPU supply integration.** A second source of GPUs beside Nosana. | Network | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |
| Open | **7B and larger run planning.** Sized from what the 1B run measured. | Training and data | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |
| Open | **TEE attestation lane for datacenter suppliers.** Confidential computing as a second way to prove honest work. | Protocol | [TASKS](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md) |

## 5. Owner decisions

In the order they block the path. Each one names the options and, where there is one, a recommendation.

1. **How outsiders join the first public run.** An admission service (about 150 lines, the join authority key online, every admission passes a policy), a pre-granted batch, or an open run. *Recommended:* the admission service, metering delegates too, since one grant can admit up to 65 identities. An open run only after the audit.
2. **Where stage one runs.** FIRST_RUN allows a bondless stage one before the audit; the mainnet deploy runbook makes the audit a prerequisite for deploying at all. Both hold only if stage one runs on devnet. *Recommended:* run stage one in public on devnet while the audit runs, and deploy mainnet once it closes. No money is at stake until the code has been audited.
3. **Model family and tokenizer.** English and Turkish together need a multilingual tokenizer, which doubles the bytes per token and fixes the VRAM floor for every node. *Next:* two or three candidates with parameter count, VRAM on a 3060 and Turkish token efficiency, side by side.
4. **Audit firm, counsel and their budget.** Both gate bonds, mainnet and the TGE, and both run on someone else's calendar. *Recommended:* book both now, against the prep package and the legal briefing that already exist.
5. **GPU budget for the 350M and 1B runs.** Rented on Nosana; the 3090 class is proven on the flagship devnet run.
6. **Token target, round length and committee size.** Set after the model is chosen. The target is stated in tokens and then left alone.
7. **A license for the main repository.** It is public with no license file, which leaves every right reserved by default.
8. **The name, as a trademark.** Leviathan ships as the working name; locking it needs a search with counsel.
9. **When the TGE happens.** Already settled in July: after a credible public run and the legal opinion, directly on Solana rails, no launchpad sale, no yield.

## 6. Stop rules

Survival over speed. The alerts are already written in [docs/ops](https://github.com/wienerlabs/leviathan/tree/main/docs/ops); these are the rules they enforce.

**Pause the run when**

- fraud slips past the audit (`UncaughtFraudSuspect`)
- the treasury pays out without matching claims (`TreasuryDrain`)
- the mesh splits beyond the SLA while epochs still pay (`MeshPartition`)
- honest nodes get slashed (`HonestFalsePositive`)
- the bond falls below the break-even penalty (`EconomicInsecurity`)
- the indexer falls behind and the dashboard goes stale (`IndexerLag`)
- a multisig or upgrade key is compromised

**Never before**

- bonds on, before the outside audit closes its critical and high findings
- an open run, before the audit
- a verifier that can slash, before its band is calibrated for the model and the hardware; dry run until then
- the TGE, before a credible public run and the legal opinion
- any parameter change in the first 24 hours after launch, except an emergency pause

## 7. Risks to watch

- CI does not compile the verifier, the daemon, the network or the client, so a broken verifier passes every gate ([leviathan#17](https://github.com/wienerlabs/leviathan/issues/17)).
- The collusion capital figure ignores committee sampling and reads about an order of magnitude low ([leviathan#18](https://github.com/wienerlabs/leviathan/issues/18)).
- Round seeds are sha256 of the timestamp and slot, which can be ground, until VRF or a recent blockhash replaces them.
- Config strings written on chain are cut short without an error (192 bytes for a URL, 64 for a file template). The run-config test catches it, but a multi-file data set needs WeightedHttp or a short mirror domain.
- Training data on one public host. At swarm scale the Hugging Face rate limits call for a mirror.
- GPU supply and price for the 1B run.
- How the bond and the token are classified. The design pays for work and offers no yield.

## Sources

Built from the repositories, issues and launch documents as they stood on 28 October 2026. Counts are items, not effort. Work counts as done when it is merged or shown on chain; an issue that only needs its paperwork is marked as shown.

- [wienerlabs/leviathan, the task ladder](https://github.com/wienerlabs/leviathan/blob/main/docs/TASKS.md)
- [The Genesis launch checklist](https://github.com/wienerlabs/leviathan/blob/main/docs/launch/GENESIS_LAUNCH.md) and [the mainnet deploy runbook](https://github.com/wienerlabs/leviathan/blob/main/docs/launch/MAINNET_DEPLOY.md)
- [The first public run](https://github.com/wienerlabs/leviathan-net/blob/main/docs/FIRST_RUN.md) in wienerlabs/leviathan-net
- [Protocol issues](https://github.com/wienerlabs/leviathan-net/issues) and [launch issues](https://github.com/wienerlabs/leviathan/issues)
- [leviathan.run](https://leviathan.run)
