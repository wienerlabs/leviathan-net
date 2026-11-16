//! Verdict votes hold their voters' bonds until they can no longer cost them
//! (wienerlabs/leviathan-net#39). These send the instructions that free a
//! vote, charge a losing one, or credit the bench its share of the losers'
//! forfeits, reading from chain whom each one names.

use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use anyhow::Context;
use anyhow::Result;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_treasurer::accounts::ParticipantReleaseVerdictVoteAccounts;
use psyche_solana_treasurer::accounts::RunFinalizeSlashAccounts;
use psyche_solana_treasurer::accounts::RunPayBenchBountyAccounts;
use psyche_solana_treasurer::accounts::RunSlashLosingVerifierAccounts;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::instruction::ParticipantReleaseVerdictVote;
use psyche_solana_treasurer::instruction::RunFinalizeSlash;
use psyche_solana_treasurer::instruction::RunPayBenchBounty;
use psyche_solana_treasurer::instruction::RunSlashLosingVerifier;
use psyche_solana_treasurer::logic::ParticipantReleaseVerdictVoteParams;
use psyche_solana_treasurer::logic::RunFinalizeSlashParams;
use psyche_solana_treasurer::logic::RunPayBenchBountyParams;
use psyche_solana_treasurer::logic::RunSlashLosingVerifierParams;
use psyche_solana_treasurer::state::appeal_window_open;
use psyche_solana_treasurer::state::AuditVerdict;
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
    /// Votes released from a standing conviction, or withdrawn from a dispute
    /// whose round is over.
    pub votes_freed: usize,
    /// Losing verifiers of an overturn charged by the crank.
    pub losers_charged: usize,
    /// Tie-breakers credited their share of the losers' forfeits.
    pub bench_paid: usize,
}

/// How many times a step that another sender may be racing is read afresh and
/// tried again before its failure is reported.
const SETTLEMENT_ATTEMPTS: usize = 3;

/// Settles everything about `target`'s last dispute that nobody's consent is
/// needed for:
///
/// - it finalises a conviction whose windows have closed, or a due one that
///   can land now;
/// - it frees the votes of a conviction that stands, or of a dispute whose
///   round is over;
/// - it charges an overturn's losing verifiers, and credits the bench.
///
/// A verifier opening a new dispute against `target` sends this first: the
/// account is taken over only once the last dispute holds nobody's bond and
/// owes the bench nothing. Other verifiers may be settling the same dispute at
/// the same time, so each step reads the verdict afresh, and a step someone
/// else already took is not an error.
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
    settlement.conviction_finalized = finalize_closed_conviction(
        endpoint,
        payer,
        run,
        coordinator_account,
        target,
    )
    .await?;
    settlement.votes_freed =
        free_settled_votes(endpoint, payer, run, coordinator_account, target)
            .await?;
    settlement.losers_charged =
        charge_losers(endpoint, payer, run, coordinator_account, target)
            .await?;
    settlement.bench_paid = credit_bench(endpoint, payer, run, target).await?;
    Ok(settlement)
}

/// The coordinator state a settlement reads, taken in place: the account is
/// too large to copy onto an async task's stack.
struct CoordinatorView {
    round_over: bool,
    /// Where the target sits on the roster, and whether an ejection now is
    /// certain to be charged.
    target_index: Option<u64>,
    ejections_charged: bool,
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
    let view = CoordinatorView {
        round_over: verdict.round_over(coordinator),
        target_index,
        ejections_charged: coordinator.ejections_charged(),
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
    let lands_now = view.target_index.is_some() && view.ejections_charged;
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

/// Credits every tie-breaker on `target`'s overturned verdict that has not
/// been credited its share, once every loser is charged.
async fn credit_bench(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    target: &Pubkey,
) -> Result<usize> {
    let Some(verdict) = get_audit_verdict(endpoint, run, target).await? else {
        return Ok(0);
    };
    if verdict.status != VerdictStatus::Overturned
        || !verdict.all_losers_settled()
        || verdict.bench_share() == 0
    {
        return Ok(0);
    }
    let mut credited = 0;
    for seat in 0..verdict.appeal_voters.len() {
        if verdict.bench_seat_paid(seat) {
            continue;
        }
        match process_treasurer_run_pay_bench_bounty(
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
                let paid = get_audit_verdict(endpoint, run, target)
                    .await?
                    .is_some_and(|now| now.bench_seat_paid(seat));
                if !paid {
                    return Err(err);
                }
            },
        }
    }
    Ok(credited)
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

/// The instruction crediting `tie_breaker`, at `seat` of `target`'s
/// overturned verdict, its share of the losers' forfeits.
pub fn run_pay_bench_bounty_instruction(
    payer: &Pubkey,
    run: &Pubkey,
    target: &Pubkey,
    seat: u16,
    tie_breaker: &Pubkey,
) -> Instruction {
    let accounts = RunPayBenchBountyAccounts {
        payer: *payer,
        run: *run,
        audit_verdict: find_audit_verdict(run, target),
        tie_breaker_participant: find_participant(run, tie_breaker),
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunPayBenchBounty {
            params: RunPayBenchBountyParams {
                target: *target,
                seat,
                tie_breaker: *tie_breaker,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// Credits the tie-breaker at `seat` of `target`'s overturned verdict its
/// share of the losers' forfeits. Whom to credit is read from the verdict.
pub async fn process_treasurer_run_pay_bench_bounty(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    target: &Pubkey,
    seat: u16,
) -> Result<()> {
    let verdict = get_audit_verdict(endpoint, run, target)
        .await?
        .context("no dispute was ever opened against the target")?;
    let tie_breaker = verdict
        .appeal_voters
        .get(seat as usize)
        .copied()
        .context("no tie-breaker sits at that seat")?;
    let instruction = run_pay_bench_bounty_instruction(
        &payer.pubkey(),
        run,
        target,
        seat,
        &tie_breaker,
    );
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}
