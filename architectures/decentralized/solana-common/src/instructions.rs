use anchor_client::anchor_lang::InstructionData;
use anchor_client::anchor_lang::ToAccountMetas;
use anchor_client::anchor_lang::system_program;
use anchor_client::solana_sdk::instruction::Instruction;
use anchor_client::solana_sdk::pubkey::Pubkey;
use anchor_spl::associated_token;
use anchor_spl::token;

pub fn coordinator_init_coordinator(
    payer: &Pubkey,
    run_id: &str,
    client_version: &str,
    coordinator_account: &Pubkey,
    main_authority: &Pubkey,
    join_authority: &Pubkey,
) -> Instruction {
    let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(run_id);
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::InitCoordinatorAccounts {
            payer: *payer,
            coordinator_instance,
            coordinator_account: *coordinator_account,
            system_program: system_program::ID,
        },
        psyche_solana_coordinator::instruction::InitCoordinator {
            params: psyche_solana_coordinator::logic::InitCoordinatorParams {
                main_authority: *main_authority,
                join_authority: *join_authority,
                run_id: run_id.to_string(),
                client_version: client_version.to_string(),
            },
        },
    )
}

pub fn coordinator_close_run(
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    main_authority: &Pubkey,
) -> Instruction {
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::FreeCoordinatorAccounts {
            authority: *main_authority,
            spill: *main_authority,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
        },
        psyche_solana_coordinator::instruction::FreeCoordinator {
            params: psyche_solana_coordinator::logic::FreeCoordinatorParams {},
        },
    )
}

pub fn coordinator_update(
    run_id: &str,
    coordinator_account: &Pubkey,
    main_authority: &Pubkey,
    metadata: Option<psyche_solana_coordinator::RunMetadata>,
    config: Option<psyche_coordinator::CoordinatorConfig>,
    model: Option<psyche_coordinator::model::Model>,
    progress: Option<psyche_coordinator::CoordinatorProgress>,
) -> Instruction {
    let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(run_id);
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::OwnerCoordinatorAccounts {
            authority: *main_authority,
            coordinator_instance,
            coordinator_account: *coordinator_account,
        },
        psyche_solana_coordinator::instruction::Update {
            metadata,
            config,
            model,
            progress,
        },
    )
}

pub fn coordinator_set_paused(
    run_id: &str,
    coordinator_account: &Pubkey,
    main_authority: &Pubkey,
    paused: bool,
) -> Instruction {
    let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(run_id);
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::OwnerCoordinatorAccounts {
            authority: *main_authority,
            coordinator_instance,
            coordinator_account: *coordinator_account,
        },
        psyche_solana_coordinator::instruction::SetPaused { paused },
    )
}

/// The instructions that pause a run, or resume it, given whether the next
/// round's seed is committed. While it is, the coordinator refuses a pause,
/// which could cancel a round whose seed is about to be known
/// (wienerlabs/leviathan-net#62), so `tick` goes first and starts that round,
/// and the pause lets it finish. Only when nothing is committed does the
/// pause go alone: a tick there could commit, and the pause behind it would
/// be refused. `set_paused` is the coordinator's `set_paused`, or the
/// treasurer's `run_update` that pauses through it.
///
/// Refused all the same, the pause is sent again a slot later, after `tick`
/// alone: in the committed slot itself the round cannot start yet, and to a
/// slot the sysvar no longer holds the tick in front renews the commitment,
/// which the refused pause reverts with it.
pub fn set_paused_instructions(
    seed_committed: bool,
    tick: Instruction,
    set_paused: Instruction,
) -> Vec<Instruction> {
    if seed_committed {
        vec![tick, set_paused]
    } else {
        vec![set_paused]
    }
}

pub fn coordinator_join_run(
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    authorization: &Pubkey,
    client_id: psyche_core::NodeIdentity,
) -> Instruction {
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::JoinRunAccounts {
            user: anchor_client::solana_sdk::pubkey::Pubkey::new_from_array(*client_id.signer()),
            authorization: *authorization,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
        },
        psyche_solana_coordinator::instruction::JoinRun {
            params: psyche_solana_coordinator::logic::JoinRunParams { client_id },
        },
    )
}

/// Joins the next epoch of a run whose join authority is the treasurer `run`
/// itself. The treasurer checks the bond the node has committed and opens the
/// node's join authorization only for this join, so a bond-gated run is joined
/// this way every epoch (wienerlabs/leviathan-net#34).
pub fn treasurer_participant_join_run(
    run: &Pubkey,
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    payer: &Pubkey,
    client_id: psyche_core::NodeIdentity,
) -> Instruction {
    let user = Pubkey::new_from_array(*client_id.signer());
    let participant = psyche_solana_treasurer::find_participant(run, &user);
    let authorization = psyche_solana_authorizer::find_authorization(
        run,
        &user,
        psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE,
    );
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::ParticipantJoinRunAccounts {
            payer: *payer,
            user,
            run: *run,
            participant,
            authorization,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
            authorizer_program: psyche_solana_authorizer::ID,
            coordinator_program: psyche_solana_coordinator::ID,
            system_program: system_program::ID,
            own_verdict: psyche_solana_treasurer::find_audit_verdict(run, &user),
        },
        psyche_solana_treasurer::instruction::ParticipantJoinRun {
            params: psyche_solana_treasurer::logic::ParticipantJoinRunParams { client_id },
        },
    )
}

pub fn coordinator_tick(
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    user: &Pubkey,
) -> Instruction {
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::PermissionlessCoordinatorAccounts {
            user: *user,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
            slot_hashes: anchor_client::solana_sdk::sysvar::slot_hashes::ID,
        },
        psyche_solana_coordinator::instruction::Tick {},
    )
}

pub fn coordinator_witness(
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    user: &Pubkey,
    witness: psyche_coordinator::Witness,
    metadata: psyche_coordinator::WitnessMetadata,
) -> Instruction {
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::PermissionlessCoordinatorAccounts {
            user: *user,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
            slot_hashes: anchor_client::solana_sdk::sysvar::slot_hashes::ID,
        },
        psyche_solana_coordinator::instruction::Witness {
            proof: witness.proof,
            participant_bloom: witness.participant_bloom,
            broadcast_bloom: witness.broadcast_bloom,
            broadcast_merkle: witness.broadcast_merkle,
            metadata,
        },
    )
}

pub fn coordinator_warmup_witness(
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    user: &Pubkey,
    witness: psyche_coordinator::Witness,
) -> Instruction {
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::PermissionlessCoordinatorAccounts {
            user: *user,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
            slot_hashes: anchor_client::solana_sdk::sysvar::slot_hashes::ID,
        },
        psyche_solana_coordinator::instruction::WarmupWitness {
            proof: witness.proof,
            participant_bloom: witness.participant_bloom,
            broadcast_bloom: witness.broadcast_bloom,
            broadcast_merkle: witness.broadcast_merkle,
        },
    )
}

pub fn coordinator_health_check(
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    user: &Pubkey,
    client_id: psyche_core::NodeIdentity,
    check: psyche_coordinator::CommitteeProof,
) -> Instruction {
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::PermissionlessCoordinatorAccounts {
            user: *user,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
            slot_hashes: anchor_client::solana_sdk::sysvar::slot_hashes::ID,
        },
        psyche_solana_coordinator::instruction::HealthCheck {
            id: client_id,
            committee: check.committee,
            position: check.position,
            index: check.index,
        },
    )
}

pub fn coordinator_checkpoint(
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    user: &Pubkey,
    repo: psyche_coordinator::model::Checkpoint,
) -> Instruction {
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::PermissionlessCoordinatorAccounts {
            user: *user,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
            slot_hashes: anchor_client::solana_sdk::sysvar::slot_hashes::ID,
        },
        psyche_solana_coordinator::instruction::Checkpoint { repo },
    )
}

pub fn coordinator_update_client_version(
    run_id: &str,
    coordinator_account: &Pubkey,
    main_authority: &Pubkey,
    new_version: &str,
) -> Instruction {
    let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(run_id);
    anchor_instruction(
        psyche_solana_coordinator::ID,
        psyche_solana_coordinator::accounts::OwnerCoordinatorAccounts {
            authority: *main_authority,
            coordinator_instance,
            coordinator_account: *coordinator_account,
        },
        psyche_solana_coordinator::instruction::UpdateClientVersion {
            new_version: new_version.to_string(),
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn treasurer_run_create(
    payer: &Pubkey,
    run_id: &str,
    client_version: &str,
    treasurer_index: u64,
    collateral_mint: &Pubkey,
    coordinator_account: &Pubkey,
    main_authority: &Pubkey,
    join_authority: &Pubkey,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    let run_collateral = associated_token::get_associated_token_address(&run, collateral_mint);
    let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(run_id);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::RunCreateAccounts {
            payer: *payer,
            run,
            run_collateral,
            collateral_mint: *collateral_mint,
            coordinator_instance,
            coordinator_account: *coordinator_account,
            coordinator_program: psyche_solana_coordinator::ID,
            associated_token_program: associated_token::ID,
            token_program: token::ID,
            system_program: system_program::ID,
        },
        psyche_solana_treasurer::instruction::RunCreate {
            params: psyche_solana_treasurer::logic::RunCreateParams {
                index: treasurer_index,
                main_authority: *main_authority,
                join_authority: *join_authority,
                run_id: run_id.to_string(),
                client_version: client_version.to_string(),
            },
        },
    )
}

pub fn treasurer_run_update(
    run_id: &str,
    treasurer_index: u64,
    coordinator_account: &Pubkey,
    main_authority: &Pubkey,
    params: psyche_solana_treasurer::logic::RunUpdateParams,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(run_id);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::RunUpdateAccounts {
            authority: *main_authority,
            run,
            coordinator_instance,
            coordinator_account: *coordinator_account,
            coordinator_program: psyche_solana_coordinator::ID,
        },
        psyche_solana_treasurer::instruction::RunUpdate { params },
    )
}

/// Sets the bond a treasurer run requires of its participants, and how long a
/// withdrawal waits, signed by the run's main authority. A non-zero minimum
/// needs a non-zero delay, which the program enforces.
pub fn treasurer_run_bond_config_update(
    treasurer_index: u64,
    main_authority: &Pubkey,
    params: psyche_solana_treasurer::logic::RunBondConfigUpdateParams,
) -> Instruction {
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::RunBondConfigUpdateAccounts {
            main_authority: *main_authority,
            run: psyche_solana_treasurer::find_run(treasurer_index),
        },
        psyche_solana_treasurer::instruction::RunBondConfigUpdate { params },
    )
}

pub fn treasurer_participant_create(
    payer: &Pubkey,
    treasurer_index: u64,
    user: &Pubkey,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    let participant = psyche_solana_treasurer::find_participant(&run, user);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::ParticipantCreateAccounts {
            payer: *payer,
            run,
            participant,
            user: *user,
            system_program: system_program::ID,
        },
        psyche_solana_treasurer::instruction::ParticipantCreate {
            params: psyche_solana_treasurer::logic::ParticipantCreateParams {},
        },
    )
}

pub fn treasurer_participant_claim(
    treasurer_index: u64,
    collateral_mint: &Pubkey,
    coordinator_account: &Pubkey,
    user: &Pubkey,
    claim_earned_points: u64,
) -> Instruction {
    let user_collateral = associated_token::get_associated_token_address(user, collateral_mint);
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    let run_collateral = associated_token::get_associated_token_address(&run, collateral_mint);
    let participant = psyche_solana_treasurer::find_participant(&run, user);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::ParticipantClaimAccounts {
            user: *user,
            user_collateral,
            run,
            run_collateral,
            participant,
            coordinator_account: *coordinator_account,
            token_program: token::ID,
            own_verdict: psyche_solana_treasurer::find_audit_verdict(&run, user),
        },
        psyche_solana_treasurer::instruction::ParticipantClaim {
            params: psyche_solana_treasurer::logic::ParticipantClaimParams {
                claim_earned_points,
            },
        },
    )
}

pub fn treasurer_participant_bond_deposit(
    treasurer_index: u64,
    collateral_mint: &Pubkey,
    user: &Pubkey,
    collateral_amount: u64,
) -> Instruction {
    let user_collateral = associated_token::get_associated_token_address(user, collateral_mint);
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    let run_collateral = associated_token::get_associated_token_address(&run, collateral_mint);
    let participant = psyche_solana_treasurer::find_participant(&run, user);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::ParticipantBondDepositAccounts {
            user: *user,
            user_collateral,
            run,
            run_collateral,
            participant,
            token_program: token::ID,
        },
        psyche_solana_treasurer::instruction::ParticipantBondDeposit {
            params: psyche_solana_treasurer::logic::ParticipantBondDepositParams {
                collateral_amount,
            },
        },
    )
}

pub fn treasurer_participant_bond_request_withdraw(
    treasurer_index: u64,
    user: &Pubkey,
    collateral_amount: u64,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    let participant = psyche_solana_treasurer::find_participant(&run, user);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::ParticipantBondRequestWithdrawAccounts {
            user: *user,
            run,
            participant,
        },
        psyche_solana_treasurer::instruction::ParticipantBondRequestWithdraw {
            params: psyche_solana_treasurer::logic::ParticipantBondRequestWithdrawParams {
                collateral_amount,
            },
        },
    )
}

pub fn treasurer_participant_bond_finalize_withdraw(
    treasurer_index: u64,
    collateral_mint: &Pubkey,
    coordinator_account: &Pubkey,
    user: &Pubkey,
) -> Instruction {
    let user_collateral = associated_token::get_associated_token_address(user, collateral_mint);
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    let run_collateral = associated_token::get_associated_token_address(&run, collateral_mint);
    let participant = psyche_solana_treasurer::find_participant(&run, user);
    // Always derived and always passed. The program pins this to its own PDA and
    // decides from the account's own contents whether a verdict exists, because
    // an optional account is one the beneficiary can leave out - and leaving it
    // out used to drop the withdrawal into the branch that paid an unchecked
    // recipient (wienerlabs/leviathan#15, finding 2).
    let audit_verdict = psyche_solana_treasurer::find_audit_verdict(&run, user);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::ParticipantBondFinalizeWithdrawAccounts {
            user: *user,
            user_collateral,
            run,
            run_collateral,
            coordinator_account: *coordinator_account,
            participant,
            audit_verdict,
            token_program: token::ID,
        },
        psyche_solana_treasurer::instruction::ParticipantBondFinalizeWithdraw {
            params: psyche_solana_treasurer::logic::ParticipantBondFinalizeWithdrawParams {},
        },
    )
}

/// `authority` releases `voter`'s vote on `target`'s verdict in run
/// `treasurer_index`, once the verdict says the vote can no longer cost it.
/// Anyone may send it; when `authority` is the voter itself, a vote whose
/// round is still under way is withdrawn as well
/// (wienerlabs/leviathan-net#39).
pub fn treasurer_participant_release_verdict_vote(
    treasurer_index: u64,
    coordinator_account: &Pubkey,
    authority: &Pubkey,
    target: &Pubkey,
    voter: &Pubkey,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::ParticipantReleaseVerdictVoteAccounts {
            authority: *authority,
            run,
            coordinator_account: *coordinator_account,
            audit_verdict: psyche_solana_treasurer::find_audit_verdict(&run, target),
            participant: psyche_solana_treasurer::find_participant(&run, voter),
        },
        psyche_solana_treasurer::instruction::ParticipantReleaseVerdictVote {
            params: psyche_solana_treasurer::logic::ParticipantReleaseVerdictVoteParams {
                target: *target,
                voter: *voter,
            },
        },
    )
}

/// Charges the losing verifier `loser`, at `voter_position` of `target`'s
/// overturned verdict in run `treasurer_index`, and records what its forfeit
/// owes the bench. Anyone may send it (wienerlabs/leviathan-net#39).
pub fn treasurer_run_slash_losing_verifier(
    treasurer_index: u64,
    coordinator_account: &Pubkey,
    payer: &Pubkey,
    target: &Pubkey,
    voter_position: u16,
    loser: &Pubkey,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::RunSlashLosingVerifierAccounts {
            payer: *payer,
            run,
            coordinator_account: *coordinator_account,
            audit_verdict: psyche_solana_treasurer::find_audit_verdict(&run, target),
            loser_participant: psyche_solana_treasurer::find_participant(&run, loser),
        },
        psyche_solana_treasurer::instruction::RunSlashLosingVerifier {
            params: psyche_solana_treasurer::logic::RunSlashLosingVerifierParams {
                target: *target,
                voter_position,
                loser: *loser,
            },
        },
    )
}

/// Credits `recipient`, at `seat` of `target`'s verdict's bounty recipients in
/// run `treasurer_index`, its share of the losing side's forfeit, to its bond:
/// a tie-breaker of an overturn once every loser is charged
/// (wienerlabs/leviathan-net#39), or a voter, or upholding tie-breaker, of a
/// conviction once the target is charged (wienerlabs/leviathan-net#64). Anyone
/// may send it.
pub fn treasurer_run_pay_verdict_bounty(
    treasurer_index: u64,
    payer: &Pubkey,
    target: &Pubkey,
    seat: u16,
    recipient: &Pubkey,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::RunPayVerdictBountyAccounts {
            payer: *payer,
            run,
            audit_verdict: psyche_solana_treasurer::find_audit_verdict(&run, target),
            recipient_participant: psyche_solana_treasurer::find_participant(&run, recipient),
        },
        psyche_solana_treasurer::instruction::RunPayVerdictBounty {
            params: psyche_solana_treasurer::logic::RunPayVerdictBountyParams {
                target: *target,
                seat,
                recipient: *recipient,
            },
        },
    )
}

/// Charges `target`'s bond in run `treasurer_index` for the conviction that
/// ejected it, once the epoch it was ejected in has been charged, and records
/// what that owes the conviction's recipients. Anyone may send it
/// (wienerlabs/leviathan-net#64).
pub fn treasurer_run_charge_conviction(
    treasurer_index: u64,
    coordinator_account: &Pubkey,
    payer: &Pubkey,
    target: &Pubkey,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::RunChargeConvictionAccounts {
            payer: *payer,
            run,
            coordinator_account: *coordinator_account,
            audit_verdict: psyche_solana_treasurer::find_audit_verdict(&run, target),
            target_participant: psyche_solana_treasurer::find_participant(&run, target),
        },
        psyche_solana_treasurer::instruction::RunChargeConviction {
            params: psyche_solana_treasurer::logic::RunChargeConvictionParams { target: *target },
        },
    )
}

/// Finalises `target`'s conviction in run `treasurer_index` once its windows
/// have closed, or once it is due and can land, with the target at
/// `target_index` on the roster now. Anyone may send it.
pub fn treasurer_run_finalize_slash(
    treasurer_index: u64,
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    payer: &Pubkey,
    target: &Pubkey,
    target_index: u64,
) -> Instruction {
    let run = psyche_solana_treasurer::find_run(treasurer_index);
    anchor_instruction(
        psyche_solana_treasurer::ID,
        psyche_solana_treasurer::accounts::RunFinalizeSlashAccounts {
            payer: *payer,
            run,
            coordinator_instance: *coordinator_instance,
            coordinator_account: *coordinator_account,
            audit_verdict: psyche_solana_treasurer::find_audit_verdict(&run, target),
            coordinator_program: psyche_solana_coordinator::ID,
        },
        psyche_solana_treasurer::instruction::RunFinalizeSlash {
            params: psyche_solana_treasurer::logic::RunFinalizeSlashParams {
                target: *target,
                target_index,
            },
        },
    )
}

pub fn authorizer_authorization_create(
    payer: &Pubkey,
    grantor: &Pubkey,
    grantee: &Pubkey,
    scope: &[u8],
) -> Instruction {
    let authorization = psyche_solana_authorizer::find_authorization(grantor, grantee, scope);
    anchor_instruction(
        psyche_solana_authorizer::ID,
        psyche_solana_authorizer::accounts::AuthorizationCreateAccounts {
            payer: *payer,
            grantor: *grantor,
            authorization,
            system_program: system_program::ID,
        },
        psyche_solana_authorizer::instruction::AuthorizationCreate {
            params: psyche_solana_authorizer::logic::AuthorizationCreateParams {
                grantee: *grantee,
                scope: scope.to_vec(),
            },
        },
    )
}

pub fn authorizer_authorization_grantor_update(
    grantor: &Pubkey,
    grantee: &Pubkey,
    scope: &[u8],
    active: bool,
) -> Instruction {
    let authorization = psyche_solana_authorizer::find_authorization(grantor, grantee, scope);
    anchor_instruction(
        psyche_solana_authorizer::ID,
        psyche_solana_authorizer::accounts::AuthorizationGrantorUpdateAccounts {
            grantor: *grantor,
            authorization,
        },
        psyche_solana_authorizer::instruction::AuthorizationGrantorUpdate {
            params: psyche_solana_authorizer::logic::AuthorizationGrantorUpdateParams { active },
        },
    )
}

pub fn authorizer_authorization_grantee_update(
    payer: &Pubkey,
    grantor: &Pubkey,
    grantee: &Pubkey,
    scope: &[u8],
    delegates_clear: bool,
    delegates_added: Vec<Pubkey>,
) -> Instruction {
    let authorization = psyche_solana_authorizer::find_authorization(grantor, grantee, scope);
    anchor_instruction(
        psyche_solana_authorizer::ID,
        psyche_solana_authorizer::accounts::AuthorizationGranteeUpdateAccounts {
            payer: *payer,
            grantee: *grantee,
            authorization,
            system_program: system_program::ID,
        },
        psyche_solana_authorizer::instruction::AuthorizationGranteeUpdate {
            params: psyche_solana_authorizer::logic::AuthorizationGranteeUpdateParams {
                delegates_clear,
                delegates_added,
            },
        },
    )
}

fn anchor_instruction<Accounts: ToAccountMetas, Args: InstructionData>(
    program_id: Pubkey,
    accounts: Accounts,
    args: Args,
) -> Instruction {
    Instruction {
        program_id,
        accounts: accounts.to_account_metas(None),
        data: args.data(),
    }
}

#[cfg(test)]
mod tests {
    use anchor_client::solana_sdk::signature::Keypair;
    use anchor_client::solana_sdk::signer::Signer;

    use super::*;

    fn instructions_for(seed_committed: bool) -> (Vec<Instruction>, Instruction, Instruction) {
        let run_id = "leviathan pause";
        let authority = Keypair::new().pubkey();
        let coordinator_instance = psyche_solana_coordinator::find_coordinator_instance(run_id);
        let coordinator_account = Keypair::new().pubkey();
        let tick = coordinator_tick(&coordinator_instance, &coordinator_account, &authority);
        let pause = coordinator_set_paused(run_id, &coordinator_account, &authority, true);
        (
            set_paused_instructions(seed_committed, tick.clone(), pause.clone()),
            tick,
            pause,
        )
    }

    /// With a round's seed committed, the pause is preceded by the tick that
    /// starts that round, which reads the `SlotHashes` sysvar.
    #[test]
    fn a_pause_behind_a_committed_seed_ticks_first() {
        let (sent, tick, pause) = instructions_for(true);
        assert_eq!(sent, vec![tick.clone(), pause]);
        assert!(tick
            .accounts
            .iter()
            .any(|meta| meta.pubkey == anchor_client::solana_sdk::sysvar::slot_hashes::ID));
    }

    /// With nothing committed the pause goes alone: a tick in front of it
    /// could commit, and the pause would then be refused.
    #[test]
    fn a_pause_with_nothing_committed_goes_alone() {
        let (sent, _, pause) = instructions_for(false);
        assert_eq!(sent, vec![pause]);
    }
}
