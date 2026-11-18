//! Verdict votes hold their voters' bonds until they can no longer cost them
//! (wienerlabs/leviathan-net#39), and a verdict owes its winning side a bounty
//! out of the losing side's forfeit (wienerlabs/leviathan-net#64). These send
//! the instructions that free a vote, charge a losing verifier or a convicted
//! target, or credit a recipient its share, reading from chain whom each one
//! names.

use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use anyhow::Context;
use anyhow::Result;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_treasurer::accounts::ParticipantReleaseVerdictVoteAccounts;
use psyche_solana_treasurer::accounts::RunChargeConvictionAccounts;
use psyche_solana_treasurer::accounts::RunChargeDueConvictionAccounts;
use psyche_solana_treasurer::accounts::RunFinalizeSlashAccounts;
use psyche_solana_treasurer::accounts::RunPayVerdictBountyAccounts;
use psyche_solana_treasurer::accounts::RunSlashLosingVerifierAccounts;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::instruction::ParticipantReleaseVerdictVote;
use psyche_solana_treasurer::instruction::RunChargeConviction;
use psyche_solana_treasurer::instruction::RunChargeDueConviction;
use psyche_solana_treasurer::instruction::RunFinalizeSlash;
use psyche_solana_treasurer::instruction::RunPayVerdictBounty;
use psyche_solana_treasurer::instruction::RunSlashLosingVerifier;
use psyche_solana_treasurer::logic::ParticipantReleaseVerdictVoteParams;
use psyche_solana_treasurer::logic::RunChargeConvictionParams;
use psyche_solana_treasurer::logic::RunChargeDueConvictionParams;
use psyche_solana_treasurer::logic::RunFinalizeSlashParams;
use psyche_solana_treasurer::logic::RunPayVerdictBountyParams;
use psyche_solana_treasurer::logic::RunSlashLosingVerifierParams;
use psyche_solana_treasurer::state::appeal_window_open;
use psyche_solana_treasurer::state::conviction_landing;
use psyche_solana_treasurer::state::on_charging_roster;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::Landing;
use psyche_solana_treasurer::state::ConvictionCharge;
use psyche_solana_treasurer::state::VerdictStatus;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_toolbox_endpoint::ToolboxEndpoint;

use crate::get_accounts::get_run;

/// Releases share a transaction, each naming its own voter's participant
/// account next to the shared run, coordinator and verdict. Each one loads and
/// rewrites the whole verdict, so a few fit the default compute budget.
const RELEASES_PER_TRANSACTION: usize = 4;

/// `authority` releases `voter`'s vote on `target`'s verdict. Anyone may; the
/// voter itself may also withdraw a vote whose round is still under way.
pub fn participant_release_verdict_vote_instruction(
    authority: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
    voter: &Pubkey,
) -> Instruction {
    let accounts = ParticipantReleaseVerdictVoteAccounts {
        authority: *authority,
        run: *run,
        coordinator_account: *coordinator_account,
        audit_verdict: find_audit_verdict(run, target),
        participant: find_participant(run, voter),
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantReleaseVerdictVote {
            params: ParticipantReleaseVerdictVoteParams {
                target: *target,
                voter: *voter,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

pub async fn process_treasurer_participant_release_verdict_vote(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    authority: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
    voter: &Pubkey,
) -> Result<()> {
    let instruction = participant_release_verdict_vote_instruction(
        &authority.pubkey(),
        run,
        coordinator_account,
        target,
        voter,
    );
    endpoint
        .process_instruction_with_signers(payer, instruction, &[authority])
        .await?;
    Ok(())
}

/// `target`'s verdict account in `run`, if a dispute ever opened against it.
pub async fn get_audit_verdict(
    endpoint: &mut ToolboxEndpoint,
    run: &Pubkey,
    target: &Pubkey,
) -> Result<Option<AuditVerdict>> {
    let address = find_audit_verdict(run, target);
    let Some(account) = endpoint.get_account(&address).await? else {
        return Ok(None);
    };
    AuditVerdict::from_pda_account(&account.owner, &account.data).map_err(
        |error| anyhow::anyhow!("Unable to decode the audit verdict: {error}"),
    )
}

/// What [`settle_previous_dispute`] did to the last dispute against a target.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DisputeSettlement {
    /// A conviction whose windows had closed, or that was due, finalised.
    pub conviction_finalized: bool,
    /// A conviction that ejected the target charged against its bond, now
    /// that the ledger has charged the ejection, or a due one charged against
    /// it because the target is on no roster an epoch will still charge.
    pub conviction_charged: bool,
    /// Votes released from a standing conviction, or withdrawn from a dispute
    /// whose round is over.
    pub votes_freed: usize,
    /// Losing verifiers of an overturn charged by the crank.
    pub losers_charged: usize,
    /// Recipients credited their share of the losing side's forfeit: the
    /// bench of an overturn, or the voters, or the upholding bench, of a
    /// conviction that stands.
    pub seats_credited: usize,
}

/// The verdicts in `run` that are not settled: a dispute whose votes still
/// hold bonds, a conviction not yet finalised or charged, or a verdict that
/// still owes a recipient its share. Sorted by target.
pub async fn unsettled_disputes(
    endpoint: &mut ToolboxEndpoint,
    run: &Pubkey,
) -> Result<Vec<AuditVerdict>> {
    // A verdict account starts with its discriminator, then its bump, then
    // the run it belongs to.
    let discriminator: &[u8] =
        <AuditVerdict as anchor_lang::Discriminator>::DISCRIMINATOR;
    let addresses = endpoint
        .search_addresses(
            &psyche_solana_treasurer::ID,
            Some(AuditVerdict::space_with_discriminator()),
            &[(0, discriminator), (8 + 1, run.as_ref())],
        )
        .await?;
    let mut verdicts = vec![];
    for address in addresses {
        let Some(account) = endpoint.get_account(&address).await? else {
            continue;
        };
        let Some(verdict) =
            AuditVerdict::from_pda_account(&account.owner, &account.data)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "Unable to decode the audit verdict {address}: {error}"
                    )
                })?
        else {
            continue;
        };
        if verdict.run == *run && !verdict.is_settled() {
            verdicts.push(verdict);
        }
    }
    verdicts.sort_by_key(|verdict| verdict.target);
    Ok(verdicts)
}

/// What [`settle_open_disputes`] did across a run.
#[derive(Debug, Default)]
pub struct DisputeSweep {
    /// The verdicts that were not settled when the sweep began.
    pub open: usize,
    /// Convictions whose windows had closed, or that were due, finalised.
    pub convictions_finalized: usize,
    /// Convictions charged against their target's bond: one that ejected it,
    /// or a due one whose target is on no roster an epoch will still charge.
    pub convictions_charged: usize,
    /// Votes released or withdrawn.
    pub votes_freed: usize,
    /// Losing verifiers of an overturn charged.
    pub losers_charged: usize,
    /// Recipients credited their share of a bounty.
    pub seats_credited: usize,
    /// Targets whose dispute a step failed on, with why. The rest of the
    /// sweep goes on without them.
    pub failures: Vec<(Pubkey, anyhow::Error)>,
}

impl DisputeSweep {
    /// Whether the sweep did anything or met a failure worth reporting.
    pub fn is_quiet(&self) -> bool {
        self.convictions_finalized
            + self.convictions_charged
            + self.votes_freed
            + self.losers_charged
            + self.seats_credited
            == 0
            && self.failures.is_empty()
    }

    fn add(&mut self, settled: DisputeSettlement) {
        self.convictions_finalized += usize::from(settled.conviction_finalized);
        self.convictions_charged += usize::from(settled.conviction_charged);
        self.votes_freed += settled.votes_freed;
        self.losers_charged += settled.losers_charged;
        self.seats_credited += settled.seats_credited;
    }
}

/// Settles, as far as nobody's consent is needed, every dispute in `run` that
/// is not settled: [`settle_previous_dispute`] for each target in turn.
///
/// A dispute used to be settled only by a verifier about to open the next one
/// against the same target. A conviction's bounty is now credited only once
/// the target's bond is charged and each recipient credited by seat, so a
/// target nobody disputes again would leave its convicting voters unpaid. The
/// verifier daemon sweeps every pass instead (wienerlabs/leviathan-net#64).
pub async fn settle_open_disputes(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
) -> Result<DisputeSweep> {
    let open = unsettled_disputes(endpoint, run).await?;
    let mut sweep = DisputeSweep {
        open: open.len(),
        ..DisputeSweep::default()
    };
    for verdict in open {
        match settle_previous_dispute(
            endpoint,
            payer,
            run,
            coordinator_account,
            &verdict.target,
        )
        .await
        {
            Ok(settled) => sweep.add(settled),
            Err(err) => sweep.failures.push((verdict.target, err)),
        }
    }
    Ok(sweep)
}

/// How many times a step that another sender may be racing is read afresh and
/// tried again before its failure is reported.
const SETTLEMENT_ATTEMPTS: usize = 3;

/// Settles everything about `target`'s last dispute that nobody's consent is
/// needed for:
///
/// - it finalises a conviction whose windows have closed, or a due one that
///   can land now;
/// - it charges the target's bond for a conviction that ejected it, once the
///   epoch it was ejected in has been charged, or for a due one whose target
///   is on no roster an epoch will still charge, which it could otherwise
///   never land on (wienerlabs/leviathan-net#74);
/// - it frees the votes of a conviction that stands, or of a dispute whose
///   round is over;
/// - it charges an overturn's losing verifiers;
/// - it credits every recipient its share of the losing side's forfeit.
///
/// A verifier opening a new dispute against `target` sends this first: the
/// account is taken over only once the last dispute holds nobody's bond and
/// owes nobody a bounty. Other verifiers may be settling the same dispute at
/// the same time, so each step reads the verdict afresh, and a step someone
/// else already took is not an error.
///
/// Every step is tried, whatever an earlier one met: a vote that can be
/// released, or a seat that can be credited, does not wait on a finalisation
/// that keeps failing. The first failure is returned once all have run.
pub async fn settle_previous_dispute(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<DisputeSettlement> {
    let mut settlement = DisputeSettlement::default();
    if get_audit_verdict(endpoint, run, target).await?.is_none() {
        return Ok(settlement);
    }
    let mut first_error = None;
    let mut note = |error: anyhow::Error| {
        first_error.get_or_insert(error);
    };
    match finalize_closed_conviction(
        endpoint,
        payer,
        run,
        coordinator_account,
        target,
    )
    .await
    {
        Ok(finalized) => settlement.conviction_finalized = finalized,
        Err(error) => note(error),
    }
    match charge_ejected_conviction(
        endpoint,
        payer,
        run,
        coordinator_account,
        target,
    )
    .await
    {
        Ok(charged) => settlement.conviction_charged = charged,
        Err(error) => note(error),
    }
    match charge_due_conviction(
        endpoint,
        payer,
        run,
        coordinator_account,
        target,
    )
    .await
    {
        Ok(charged) => settlement.conviction_charged |= charged,
        Err(error) => note(error),
    }
    match free_settled_votes(endpoint, payer, run, coordinator_account, target)
        .await
    {
        Ok(freed) => settlement.votes_freed = freed,
        Err(error) => note(error),
    }
    match charge_losers(endpoint, payer, run, coordinator_account, target).await
    {
        Ok(charged) => settlement.losers_charged = charged,
        Err(error) => note(error),
    }
    match credit_seats(endpoint, payer, run, target).await {
        Ok(credited) => settlement.seats_credited = credited,
        Err(error) => note(error),
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(settlement),
    }
}

/// The coordinator state a settlement reads, taken in place: the account is
/// too large to copy onto an async task's stack.
struct CoordinatorView {
    round_over: bool,
    /// Where the target sits on the roster.
    target_index: Option<u64>,
    /// Whether a conviction finalised now would eject the target into an
    /// epoch that will charge it, as `conviction_landing` decides on chain.
    /// A due conviction lands only then: off the roster, outside a charging
    /// state, or with the target already ejected by another conviction, the
    /// program refuses to finalise it again.
    lands_now: bool,
    /// Whether the ledger has charged the ejection the verdict's conviction
    /// landed, if it landed one.
    ejection_charged: bool,
    /// Whether the target is on the roster of an epoch the ledger will still
    /// charge, as `on_charging_roster` decides on chain. A due conviction is
    /// charged against the bond directly only when it is not.
    on_charging_roster: bool,
}

async fn coordinator_view(
    endpoint: &mut ToolboxEndpoint,
    coordinator_account: &Pubkey,
    verdict: &AuditVerdict,
) -> Result<CoordinatorView> {
    let data = endpoint
        .get_account_data(coordinator_account)
        .await?
        .context("the coordinator account does not exist")?;
    let coordinator = &coordinator_account_from_bytes(&data)
        .map_err(|err| {
            anyhow::anyhow!(
                "Unable to decode coordinator_account data: {err:?}"
            )
        })?
        .state
        .coordinator;
    let signer = verdict.target.to_bytes();
    let target_index = coordinator
        .epoch_state
        .clients
        .iter()
        .position(|client| *client.id.signer() == signer)
        .map(|index| index as u64);
    let lands_now = target_index.is_some_and(|index| {
        matches!(
            conviction_landing(coordinator, &verdict.target, index),
            Ok(Landing::Eject(_))
        )
    });
    let view = CoordinatorView {
        round_over: verdict.round_over(coordinator),
        target_index,
        lands_now,
        ejection_charged: verdict.ejection_charged(coordinator),
        on_charging_roster: on_charging_roster(coordinator, &verdict.target),
    };
    Ok(view)
}

/// Finalises `target`'s conviction if its challenge window, or its appeal,
/// has closed, or if it is due and can land now. Returns whether it did.
async fn finalize_closed_conviction(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<bool> {
    let Some(verdict) = get_audit_verdict(endpoint, run, target).await? else {
        return Ok(false);
    };
    let run_state = get_run(endpoint, run)
        .await?
        .context("the run does not exist")?;
    let now = endpoint.get_sysvar_clock().await?.unix_timestamp;
    let view =
        coordinator_view(endpoint, coordinator_account, &verdict).await?;
    let lands_now = view.lands_now;
    let finalizable = match verdict.status {
        VerdictStatus::SlashPending => {
            now >= verdict.pending_since_unix
                + run_state.challenge_window_seconds
        },
        VerdictStatus::Challenged => !appeal_window_open(
            run_state.appeal_window_seconds,
            verdict.challenged_since_unix,
            now,
        ),
        VerdictStatus::Due => lands_now,
        VerdictStatus::Voting
        | VerdictStatus::Upheld
        | VerdictStatus::Overturned => false,
    };
    if !finalizable {
        return Ok(false);
    }
    // Off the roster the index is not read: the conviction is recorded due.
    let target_index = view.target_index.unwrap_or_default();
    let accounts = RunFinalizeSlashAccounts {
        payer: payer.pubkey(),
        run: *run,
        coordinator_instance: run_state.coordinator_instance,
        coordinator_account: *coordinator_account,
        audit_verdict: find_audit_verdict(run, target),
        coordinator_program: psyche_solana_coordinator::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunFinalizeSlash {
            params: RunFinalizeSlashParams {
                target: *target,
                target_index,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    match endpoint.process_instruction(payer, instruction).await {
        Ok(_) => Ok(true),
        // Someone else finalised it first.
        Err(_)
            if get_audit_verdict(endpoint, run, target)
                .await?
                .is_some_and(|now| now.status != verdict.status) =>
        {
            Ok(false)
        },
        Err(err) => Err(err),
    }
}

/// Charges `target`'s bond for the conviction that ejected it, once the epoch
/// it was ejected in has been charged. Returns whether it did.
async fn charge_ejected_conviction(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<bool> {
    let Some(verdict) = get_audit_verdict(endpoint, run, target).await? else {
        return Ok(false);
    };
    if !verdict.awaits_charge() {
        return Ok(false);
    }
    let view =
        coordinator_view(endpoint, coordinator_account, &verdict).await?;
    if !view.ejection_charged {
        return Ok(false);
    }
    let instruction = run_charge_conviction_instruction(
        &payer.pubkey(),
        run,
        coordinator_account,
        target,
    );
    match endpoint.process_instruction(payer, instruction).await {
        Ok(_) => Ok(true),
        // Someone else charged it first, or the target's withdrawal did.
        Err(_)
            if get_audit_verdict(endpoint, run, target)
                .await?
                .is_some_and(|now| now.charge != ConvictionCharge::Ejected) =>
        {
            Ok(false)
        },
        Err(err) => Err(err),
    }
}

/// Charges `target`'s bond for the conviction its verdict records as due, when
/// the target is on no roster an epoch will still charge: it could otherwise
/// land only if the target came back, or be charged only at the target's own
/// withdrawal (wienerlabs/leviathan-net#74). Returns whether it did.
async fn charge_due_conviction(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<bool> {
    let Some(verdict) = get_audit_verdict(endpoint, run, target).await? else {
        return Ok(false);
    };
    if verdict.status != VerdictStatus::Due {
        return Ok(false);
    }
    let view =
        coordinator_view(endpoint, coordinator_account, &verdict).await?;
    if view.on_charging_roster {
        return Ok(false);
    }
    let instruction = run_charge_due_conviction_instruction(
        &payer.pubkey(),
        run,
        coordinator_account,
        target,
    );
    match endpoint.process_instruction(payer, instruction).await {
        Ok(_) => Ok(true),
        // Someone else charged it first, or the target's withdrawal did.
        Err(_)
            if get_audit_verdict(endpoint, run, target)
                .await?
                .is_some_and(|now| now.status != VerdictStatus::Due) =>
        {
            Ok(false)
        },
        Err(err) => Err(err),
    }
}

/// Frees every vote on `target`'s verdict that a release would free without
/// its voter's consent.
///
/// A batch that fails is sent again one release at a time, so a release that
/// is refused, say because another sender just made it, does not hold back the
/// rest. The verdict is read afresh until nothing is left to free; a release
/// still refused after that is reported.
async fn free_settled_votes(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<usize> {
    let mut freed = 0;
    let mut last_error = None;
    for _ in 0..SETTLEMENT_ATTEMPTS {
        let Some(verdict) = get_audit_verdict(endpoint, run, target).await?
        else {
            return Ok(freed);
        };
        let view =
            coordinator_view(endpoint, coordinator_account, &verdict).await?;
        let voters = verdict.releasable_votes(None, view.round_over);
        if voters.is_empty() {
            return Ok(freed);
        }
        let release = |voter: &Pubkey| {
            participant_release_verdict_vote_instruction(
                &payer.pubkey(),
                run,
                coordinator_account,
                target,
                voter,
            )
        };
        for batch in voters.chunks(RELEASES_PER_TRANSACTION) {
            let instructions: Vec<Instruction> =
                batch.iter().map(release).collect();
            if endpoint
                .process_instructions_with_signers(payer, &instructions, &[])
                .await
                .is_ok()
            {
                freed += batch.len();
                continue;
            }
            for voter in batch {
                match endpoint.process_instruction(payer, release(voter)).await
                {
                    Ok(_) => freed += 1,
                    Err(err) => last_error = Some(err),
                }
            }
        }
    }
    let Some(verdict) = get_audit_verdict(endpoint, run, target).await? else {
        return Ok(freed);
    };
    let view =
        coordinator_view(endpoint, coordinator_account, &verdict).await?;
    match (
        verdict.releasable_votes(None, view.round_over).is_empty(),
        last_error,
    ) {
        (false, Some(err)) => Err(err),
        _ => Ok(freed),
    }
}

/// Charges every losing verifier the crank has not reached on `target`'s
/// overturned verdict. A position someone else charged first is skipped.
async fn charge_losers(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<usize> {
    let mut charged = 0;
    let mut misses = 0;
    loop {
        let Some(verdict) = get_audit_verdict(endpoint, run, target).await?
        else {
            return Ok(charged);
        };
        if verdict.status != VerdictStatus::Overturned
            || verdict.all_losers_settled()
        {
            return Ok(charged);
        }
        let position = verdict.settled_count;
        match process_treasurer_run_slash_losing_verifier(
            endpoint,
            payer,
            run,
            coordinator_account,
            target,
            position,
        )
        .await
        {
            Ok(()) => charged += 1,
            Err(err) => {
                misses += 1;
                let advanced = get_audit_verdict(endpoint, run, target)
                    .await?
                    .is_some_and(|now| now.settled_count > position);
                if !advanced && misses >= SETTLEMENT_ATTEMPTS {
                    return Err(err);
                }
            },
        }
    }
}

/// Credits every recipient on `target`'s verdict that has not been credited
/// its share, once what the verdict owes is final. A seat that fails does not
/// hold back the others; the first failure is returned once all were tried.
async fn credit_seats(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    target: &Pubkey,
) -> Result<usize> {
    let Some(verdict) = get_audit_verdict(endpoint, run, target).await? else {
        return Ok(0);
    };
    if verdict.bounty_final().is_err() || verdict.bounty_share() == 0 {
        return Ok(0);
    }
    let mut credited = 0;
    let mut first_error = None;
    for seat in 0..verdict.bounty_recipients().len() {
        if verdict.bounty_seat_paid(seat) {
            continue;
        }
        match process_treasurer_run_pay_verdict_bounty(
            endpoint,
            payer,
            run,
            target,
            seat as u16,
        )
        .await
        {
            Ok(()) => credited += 1,
            Err(err) => {
                // Someone else credited it first.
                let paid = get_audit_verdict(endpoint, run, target)
                    .await?
                    .is_some_and(|now| now.bounty_seat_paid(seat));
                if !paid {
                    first_error.get_or_insert(err);
                }
            },
        }
    }
    match first_error {
        Some(err) => Err(err),
        None => Ok(credited),
    }
}

/// Charges the losing verifier at `voter_position` of `target`'s overturned
/// verdict, adding what its forfeit owes the bench to the verdict. Whom to
/// charge is read from the verdict.
pub async fn process_treasurer_run_slash_losing_verifier(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
    voter_position: u16,
) -> Result<()> {
    let verdict = get_audit_verdict(endpoint, run, target)
        .await?
        .context("no dispute was ever opened against the target")?;
    // A position past the last voter names nobody; the program refuses it.
    let loser = verdict
        .voters
        .get(voter_position as usize)
        .copied()
        .unwrap_or_default();
    let accounts = RunSlashLosingVerifierAccounts {
        payer: payer.pubkey(),
        run: *run,
        coordinator_account: *coordinator_account,
        audit_verdict: find_audit_verdict(run, target),
        loser_participant: find_participant(run, &loser),
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSlashLosingVerifier {
            params: RunSlashLosingVerifierParams {
                target: *target,
                voter_position,
                loser,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}

/// The instruction charging `target`'s bond for the conviction that ejected
/// it.
pub fn run_charge_conviction_instruction(
    payer: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Instruction {
    let accounts = RunChargeConvictionAccounts {
        payer: *payer,
        run: *run,
        coordinator_account: *coordinator_account,
        audit_verdict: find_audit_verdict(run, target),
        target_participant: find_participant(run, target),
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunChargeConviction {
            params: RunChargeConvictionParams { target: *target },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// Charges `target`'s bond for the conviction that ejected it.
pub async fn process_treasurer_run_charge_conviction(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<()> {
    let instruction = run_charge_conviction_instruction(
        &payer.pubkey(),
        run,
        coordinator_account,
        target,
    );
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}

/// The instruction charging `target`'s bond for the conviction its verdict
/// records as due.
pub fn run_charge_due_conviction_instruction(
    payer: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Instruction {
    let accounts = RunChargeDueConvictionAccounts {
        payer: *payer,
        run: *run,
        coordinator_account: *coordinator_account,
        audit_verdict: find_audit_verdict(run, target),
        target_participant: find_participant(run, target),
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunChargeDueConviction {
            params: RunChargeDueConvictionParams { target: *target },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// Charges `target`'s bond for the conviction its verdict records as due.
pub async fn process_treasurer_run_charge_due_conviction(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    target: &Pubkey,
) -> Result<()> {
    let instruction = run_charge_due_conviction_instruction(
        &payer.pubkey(),
        run,
        coordinator_account,
        target,
    );
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}

/// The instruction crediting `recipient`, at `seat` of `target`'s verdict's
/// bounty recipients, its share of the losing side's forfeit.
pub fn run_pay_verdict_bounty_instruction(
    payer: &Pubkey,
    run: &Pubkey,
    target: &Pubkey,
    seat: u16,
    recipient: &Pubkey,
) -> Instruction {
    let accounts = RunPayVerdictBountyAccounts {
        payer: *payer,
        run: *run,
        audit_verdict: find_audit_verdict(run, target),
        recipient_participant: find_participant(run, recipient),
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunPayVerdictBounty {
            params: RunPayVerdictBountyParams {
                target: *target,
                seat,
                recipient: *recipient,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// Credits the recipient at `seat` of `target`'s verdict its share of the
/// losing side's forfeit. Whom to credit is read from the verdict.
pub async fn process_treasurer_run_pay_verdict_bounty(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    target: &Pubkey,
    seat: u16,
) -> Result<()> {
    let verdict = get_audit_verdict(endpoint, run, target)
        .await?
        .context("no dispute was ever opened against the target")?;
    let recipient = verdict
        .bounty_recipients()
        .get(seat as usize)
        .copied()
        .context("no recipient sits at that seat")?;
    let instruction = run_pay_verdict_bounty_instruction(
        &payer.pubkey(),
        run,
        target,
        seat,
        &recipient,
    );
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}
