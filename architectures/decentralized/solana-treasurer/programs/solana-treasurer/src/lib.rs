pub mod logic;
pub mod state;

use anchor_lang::prelude::*;
use logic::*;

// The cluster a build targets is chosen here, not by patching this file at
// deploy time. A mainnet build carries a different program id, so a binary
// meant for devnet can never be pointed at mainnet by mistake.
#[cfg(not(feature = "mainnet"))]
declare_id!("Fq1Mv8osXqHxiiXjm4yhvQGE5wgx9QMueK8n2qwbqovV");
#[cfg(feature = "mainnet")]
declare_id!("A6Z8jZeKi81zUaozR7X7SGXtY8EyXm1YyTeFMuFgXEkW");

pub fn find_run(index: u64) -> Pubkey {
    Pubkey::find_program_address(
        &[state::Run::SEEDS_PREFIX, index.to_le_bytes().as_ref()],
        &crate::ID,
    )
    .0
}

pub fn find_participant(run: &Pubkey, user: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(
        &[
            state::Participant::SEEDS_PREFIX,
            run.as_ref(),
            user.as_ref(),
        ],
        &crate::ID,
    )
    .0
}

pub fn find_slash_report(
    run: &Pubkey,
    target: &Pubkey,
    charge_epoch_start: u64,
) -> Pubkey {
    Pubkey::find_program_address(
        &[
            state::SlashReport::SEEDS_PREFIX,
            run.as_ref(),
            target.as_ref(),
            charge_epoch_start.to_le_bytes().as_ref(),
        ],
        &crate::ID,
    )
    .0
}

pub fn find_equivocation_commitment(
    run: &Pubkey,
    reporter: &Pubkey,
    commitment: &[u8; 32],
) -> Pubkey {
    Pubkey::find_program_address(
        &[
            state::EquivocationCommitment::SEEDS_PREFIX,
            run.as_ref(),
            reporter.as_ref(),
            commitment.as_ref(),
        ],
        &crate::ID,
    )
    .0
}

pub fn find_equivocation_bounty(run: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(
        &[state::EquivocationBounty::SEEDS_PREFIX, run.as_ref()],
        &crate::ID,
    )
    .0
}

pub fn find_audit_verdict(run: &Pubkey, target: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(
        &[
            state::AuditVerdict::SEEDS_PREFIX,
            run.as_ref(),
            target.as_ref(),
        ],
        &crate::ID,
    )
    .0
}

#[program]
pub mod psyche_solana_treasurer {
    use super::*;

    pub fn run_create(
        context: Context<RunCreateAccounts>,
        params: RunCreateParams,
    ) -> Result<()> {
        run_create_processor(context, params)
    }

    pub fn run_update(
        context: Context<RunUpdateAccounts>,
        params: RunUpdateParams,
    ) -> Result<()> {
        run_update_processor(context, params)
    }

    pub fn participant_create(
        context: Context<ParticipantCreateAccounts>,
        params: ParticipantCreateParams,
    ) -> Result<()> {
        participant_create_processor(context, params)
    }

    pub fn participant_claim(
        context: Context<ParticipantClaimAccounts>,
        params: ParticipantClaimParams,
    ) -> Result<()> {
        participant_claim_processor(context, params)
    }

    pub fn run_bond_config_update(
        context: Context<RunBondConfigUpdateAccounts>,
        params: RunBondConfigUpdateParams,
    ) -> Result<()> {
        run_bond_config_update_processor(context, params)
    }

    pub fn participant_bond_deposit(
        context: Context<ParticipantBondDepositAccounts>,
        params: ParticipantBondDepositParams,
    ) -> Result<()> {
        participant_bond_deposit_processor(context, params)
    }

    pub fn participant_bond_request_withdraw(
        context: Context<ParticipantBondRequestWithdrawAccounts>,
        params: ParticipantBondRequestWithdrawParams,
    ) -> Result<()> {
        participant_bond_request_withdraw_processor(context, params)
    }

    pub fn participant_bond_finalize_withdraw(
        context: Context<ParticipantBondFinalizeWithdrawAccounts>,
        params: ParticipantBondFinalizeWithdrawParams,
    ) -> Result<()> {
        participant_bond_finalize_withdraw_processor(context, params)
    }

    pub fn run_slash(
        context: Context<RunSlashAccounts>,
        params: RunSlashParams,
    ) -> Result<()> {
        run_slash_processor(context, params)
    }

    pub fn run_set_slash_bounty(
        context: Context<RunSetSlashBountyAccounts>,
        params: RunSetSlashBountyParams,
    ) -> Result<()> {
        run_set_slash_bounty_processor(context, params)
    }

    pub fn participant_join_run(
        context: Context<ParticipantJoinRunAccounts>,
        params: ParticipantJoinRunParams,
    ) -> Result<()> {
        participant_join_run_processor(context, params)
    }

    pub fn run_submit_audit_verdict(
        context: Context<RunSubmitAuditVerdictAccounts>,
        params: RunSubmitAuditVerdictParams,
    ) -> Result<()> {
        run_submit_audit_verdict_processor(context, params)
    }

    pub fn run_set_challenge_config(
        context: Context<RunSetChallengeConfigAccounts>,
        params: RunSetChallengeConfigParams,
    ) -> Result<()> {
        run_set_challenge_config_processor(context, params)
    }

    pub fn run_open_challenge(
        context: Context<RunOpenChallengeAccounts>,
        params: RunOpenChallengeParams,
    ) -> Result<()> {
        run_open_challenge_processor(context, params)
    }

    pub fn run_submit_appeal_verdict(
        context: Context<RunSubmitAppealVerdictAccounts>,
        params: RunSubmitAppealVerdictParams,
    ) -> Result<()> {
        run_submit_appeal_verdict_processor(context, params)
    }

    pub fn run_finalize_slash(
        context: Context<RunFinalizeSlashAccounts>,
        params: RunFinalizeSlashParams,
    ) -> Result<()> {
        run_finalize_slash_processor(context, params)
    }

    pub fn run_slash_losing_verifier(
        context: Context<RunSlashLosingVerifierAccounts>,
        params: RunSlashLosingVerifierParams,
    ) -> Result<()> {
        run_slash_losing_verifier_processor(context, params)
    }

    pub fn run_pay_verdict_bounty(
        context: Context<RunPayVerdictBountyAccounts>,
        params: RunPayVerdictBountyParams,
    ) -> Result<()> {
        run_pay_verdict_bounty_processor(context, params)
    }

    pub fn run_charge_conviction(
        context: Context<RunChargeConvictionAccounts>,
        params: RunChargeConvictionParams,
    ) -> Result<()> {
        run_charge_conviction_processor(context, params)
    }

    pub fn run_charge_due_conviction(
        context: Context<RunChargeDueConvictionAccounts>,
        params: RunChargeDueConvictionParams,
    ) -> Result<()> {
        run_charge_due_conviction_processor(context, params)
    }

    pub fn run_commit_equivocation(
        context: Context<RunCommitEquivocationAccounts>,
        params: RunCommitEquivocationParams,
    ) -> Result<()> {
        run_commit_equivocation_processor(context, params)
    }

    pub fn run_slash_equivocation(
        context: Context<RunSlashEquivocationAccounts>,
        params: RunSlashEquivocationParams,
    ) -> Result<()> {
        run_slash_equivocation_processor(context, params)
    }

    pub fn run_close_equivocation_commitment(
        context: Context<RunCloseEquivocationCommitmentAccounts>,
        params: RunCloseEquivocationCommitmentParams,
    ) -> Result<()> {
        run_close_equivocation_commitment_processor(context, params)
    }

    pub fn run_set_equivocation_bounty(
        context: Context<RunSetEquivocationBountyAccounts>,
        params: RunSetEquivocationBountyParams,
    ) -> Result<()> {
        run_set_equivocation_bounty_processor(context, params)
    }

    pub fn run_charge_slash_report(
        context: Context<RunChargeSlashReportAccounts>,
        params: RunChargeSlashReportParams,
    ) -> Result<()> {
        run_charge_slash_report_processor(context, params)
    }

    pub fn run_pay_slash_report_bounty(
        context: Context<RunPaySlashReportBountyAccounts>,
        params: RunPaySlashReportBountyParams,
    ) -> Result<()> {
        run_pay_slash_report_bounty_processor(context, params)
    }

    pub fn participant_release_verdict_vote(
        context: Context<ParticipantReleaseVerdictVoteAccounts>,
        params: ParticipantReleaseVerdictVoteParams,
    ) -> Result<()> {
        participant_release_verdict_vote_processor(context, params)
    }
}

#[error_code]
pub enum ProgramError {
    #[msg("Invalid parameter")]
    InvalidParameter,

    #[msg("run_id must be 32 bytes or less")]
    RunIdInvalidLength,

    #[msg("Bond balance is insufficient for this request")]
    InsufficientBond,

    #[msg("The bond withdraw delay has not elapsed yet")]
    WithdrawDelayNotElapsed,

    /// No longer raised: a slash's reporter is recorded when it is made
    /// (wienerlabs/leviathan-net#72). Kept so later codes do not move.
    #[msg("A slash bounty is configured but no reporter account was provided")]
    MissingReporter,

    #[msg("Bond is below the run minimum required to claim rewards")]
    BondBelowMinimum,

    #[msg("A run that requires a bond must also set a positive bond withdraw delay")]
    BondWindowRequired,

    #[msg("The verifier is not a participant in the current epoch")]
    VerifierNotInEpoch,

    #[msg("The signer is not an assigned verifier for this round")]
    VerifierNotAssigned,

    #[msg("The target index does not match the provided target key")]
    TargetMismatch,

    #[msg("This verifier already submitted a verdict for this target this epoch")]
    DuplicateVerdict,

    #[msg("The verdict has already resolved into a slash this epoch")]
    VerdictAlreadyResolved,

    #[msg("The verdict voter set is full")]
    VerdictVotersFull,

    /// No longer raised: recipients are credited by seat, never paid through a
    /// token account (wienerlabs/leviathan-net#64). Kept so that every later
    /// error keeps its code.
    #[msg("A bounty recipient does not match the corresponding verdict voter")]
    BountyRecipientMismatch,

    #[msg("The verdict is not in a slash-pending state")]
    VerdictNotPending,

    #[msg("The challenge window has already closed")]
    ChallengeWindowClosed,

    #[msg("The challenge window is still open")]
    ChallengeWindowOpen,

    #[msg("The verdict is not under challenge")]
    VerdictNotChallenged,

    #[msg("The signer is not an assigned tie-breaker for this round")]
    NotTieBreaker,

    #[msg("This tie-breaker already voted on this appeal")]
    DuplicateAppealVerdict,

    #[msg("The appeal voter set is full")]
    AppealVotersFull,

    #[msg("The verdict was not overturned")]
    VerdictNotOverturned,

    #[msg("All losing verifiers have already been settled")]
    AllLosersSettled,

    #[msg("The verifier committee is too small for a quorum to mean anything")]
    VerifierCommitteeTooSmall,

    #[msg("The quorum for this run exceeds the number of voters a verdict can hold")]
    QuorumExceedsVoterCapacity,

    #[msg("The previous dispute for this target has not been settled yet")]
    PreviousVerdictUnsettled,

    // No longer returned: votes for different evidence are counted apart
    // rather than refused. Kept so the codes after it do not move.
    #[msg("This verdict already records different evidence for this target")]
    EvidenceMismatch,

    #[msg("A tie-breaker that voted on the original verdict cannot judge its appeal")]
    AppellateVotedOnVerdict,

    #[msg("The appeal window has not elapsed yet")]
    AppealWindowOpen,

    /// No longer raised: a slash's reporter is recorded when it is made
    /// (wienerlabs/leviathan-net#72). Kept so later codes do not move.
    #[msg("The bounty recipient must be a token account owned by someone other than the slashed participant")]
    InvalidBountyRecipient,

    #[msg("An audit verdict exists for this participant and must be provided")]
    AuditVerdictRequired,

    #[msg("This run is not gated by its own bond; join it through its join authority")]
    RunNotBondGated,

    #[msg("A conviction against this participant is still pending; withdraw once it resolves")]
    ConvictionPending,

    #[msg("This participant trains in an epoch that has not been charged yet; withdraw after it ends")]
    EpochNotSettled,

    #[msg("The target is not in an epoch that would charge its conviction yet")]
    TargetNotInChargingEpoch,

    #[msg("The committee in force was not drawn over the current roster; vote once a round draws one")]
    CommitteeNotDrawnForRoster,

    #[msg("A run with a challenge window must give its appeals a deadline")]
    AppealWindowRequired,

    #[msg("The transaction does not prove the target signed the commitment it is convicted on")]
    CommitmentNotSigned,

    #[msg(
        "The commitment a conviction cites was not made in the current epoch"
    )]
    CommitmentOutsideEpoch,

    #[msg("This participant voted on a verdict that can still cost it; release each vote once it can no longer, then withdraw")]
    VerdictVotesOpen,

    #[msg("This vote can still cost its voter: its verdict may yet be overturned, or was overturned and waits for run_slash_losing_verifier")]
    VoteStillAtStake,

    #[msg("This vote no longer holds its voter's bond")]
    VoteAlreadyReleased,

    #[msg("This key did not vote on this verdict")]
    NotAVoter,

    #[msg("The participant's count of open verdict votes disagrees with the verdicts")]
    OpenVerdictVotesInconsistent,

    #[msg("The bench is paid once every losing verifier has been charged")]
    LosersNotAllSettled,

    #[msg("This recipient's share has already been paid")]
    BountySeatPaid,

    #[msg("The losing side's forfeit owes the recipients nothing")]
    NothingOwedToSeats,

    #[msg("The two commitments are not two different commitments to the same step and batch")]
    EquivocationNotProven,

    #[msg("The target cannot be ejected into an epoch that will charge it now: it is off the roster, already ejected or withdrawn, or no ejection here would be charged")]
    EquivocationNotLandable,

    #[msg("The verdict holds no conviction that ejected its target and waits to be charged")]
    ConvictionNotChargeable,

    #[msg("The epoch the conviction ejected its target in has not been charged yet")]
    EjectionNotCharged,

    #[msg("The conviction's recipients are paid once the target's bond has paid for it")]
    ConvictionNotCharged,

    #[msg("The dispute is still open, so it owes nobody a bounty yet")]
    VerdictOwesNoBounty,

    #[msg("The participant a verdict convicts cannot judge its own appeal")]
    AppellateIsTarget,

    #[msg("The participant a verdict would convict cannot vote on it")]
    VerifierIsTarget,

    #[msg("The participant a slash convicts cannot be its reporter")]
    ReporterIsTarget,

    #[msg("The bond is held until every slash that ejected this participant has been charged against it")]
    SlashReportsUncharged,

    #[msg("The participant's count of uncharged slash reports disagrees with the reports")]
    SlashReportsInconsistent,

    #[msg("This slash names no reporter, so its bounty stays in the vault")]
    SlashReportOwesNoBounty,

    #[msg("The reporter committed to no such proof of equivocation")]
    EquivocationNotCommitted,

    #[msg("A proof of equivocation is revealed in a later slot than the reporter committed to it")]
    EquivocationCommittedTooRecently,

    #[msg("An equivocation's reporter may be given less than the whole of the target's forfeit")]
    EquivocationBountyTooHigh,

    #[msg("The verdict holds no conviction recorded as due")]
    ConvictionNotDue,

    #[msg("The target is on the roster of an epoch that will be charged; its conviction lands there through run_finalize_slash")]
    TargetInChargingEpoch,
}
