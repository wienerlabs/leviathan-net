use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use anchor_spl::associated_token;
use anchor_spl::token;
use anyhow::anyhow;
use anyhow::Result;
use psyche_core::NodeIdentity;
use crate::get_accounts::get_run;
use crate::signed_commitment::SignedCommitment;
use psyche_solana_authorizer::find_authorization;
use psyche_solana_coordinator::find_coordinator_instance;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_treasurer::accounts::ParticipantJoinRunAccounts;
use psyche_solana_treasurer::accounts::ParticipantBondDepositAccounts;
use psyche_solana_treasurer::accounts::ParticipantBondFinalizeWithdrawAccounts;
use psyche_solana_treasurer::accounts::ParticipantBondRequestWithdrawAccounts;
use psyche_solana_treasurer::accounts::ParticipantClaimAccounts;
use psyche_solana_treasurer::accounts::ParticipantCreateAccounts;
use psyche_solana_treasurer::accounts::RunBondConfigUpdateAccounts;
use psyche_solana_treasurer::accounts::RunCreateAccounts;
use psyche_solana_treasurer::accounts::RunSetSlashBountyAccounts;
use psyche_solana_treasurer::accounts::RunSlashAccounts;
use psyche_solana_treasurer::accounts::RunUpdateAccounts;
use psyche_solana_treasurer::find_participant;
use psyche_solana_treasurer::find_run;
use psyche_solana_treasurer::instruction::ParticipantJoinRun;
use psyche_solana_treasurer::instruction::ParticipantBondDeposit;
use psyche_solana_treasurer::instruction::ParticipantBondFinalizeWithdraw;
use psyche_solana_treasurer::instruction::ParticipantBondRequestWithdraw;
use psyche_solana_treasurer::instruction::ParticipantClaim;
use psyche_solana_treasurer::instruction::ParticipantCreate;
use psyche_solana_treasurer::instruction::RunBondConfigUpdate;
use psyche_solana_treasurer::instruction::RunCreate;
use psyche_solana_treasurer::instruction::RunSetSlashBounty;
use psyche_solana_treasurer::instruction::RunSlash;
use psyche_solana_treasurer::instruction::RunUpdate;
use psyche_solana_treasurer::logic::ParticipantBondDepositParams;
use psyche_solana_treasurer::logic::ParticipantBondFinalizeWithdrawParams;
use psyche_solana_treasurer::logic::ParticipantBondRequestWithdrawParams;
use psyche_solana_treasurer::logic::ParticipantClaimParams;
use psyche_solana_treasurer::logic::ParticipantCreateParams;
use psyche_solana_treasurer::logic::ParticipantJoinRunParams;
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunSetSlashBountyParams;
use psyche_solana_treasurer::accounts::RunSubmitAuditVerdictAccounts;
use psyche_solana_treasurer::accounts::RunSetChallengeConfigAccounts;
use psyche_solana_treasurer::accounts::RunOpenChallengeAccounts;
use psyche_solana_treasurer::accounts::RunSubmitAppealVerdictAccounts;
use psyche_solana_treasurer::accounts::RunFinalizeSlashAccounts;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::instruction::RunSubmitAuditVerdict;
use psyche_solana_treasurer::instruction::RunSetChallengeConfig;
use psyche_solana_treasurer::instruction::RunOpenChallenge;
use psyche_solana_treasurer::instruction::RunSubmitAppealVerdict;
use psyche_solana_treasurer::instruction::RunFinalizeSlash;
use psyche_solana_treasurer::logic::RunSlashParams;
use psyche_solana_treasurer::logic::RunSubmitAuditVerdictParams;
use psyche_solana_treasurer::logic::RunSetChallengeConfigParams;
use psyche_solana_treasurer::logic::RunOpenChallengeParams;
use psyche_solana_treasurer::logic::RunSubmitAppealVerdictParams;
use psyche_solana_treasurer::logic::RunFinalizeSlashParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use psyche_solana_treasurer::state::CommitmentEvidence;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::system_program;
use solana_sdk::sysvar;
use solana_toolbox_endpoint::ToolboxEndpoint;

pub async fn process_treasurer_run_create(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    collateral_mint: &Pubkey,
    coordinator_account: &Pubkey,
    params: RunCreateParams,
) -> Result<(Pubkey, Pubkey)> {
    let run = find_run(params.index);
    let run_collateral = ToolboxEndpoint::find_spl_associated_token_account(
        &run,
        collateral_mint,
    );
    let coordinator_instance = find_coordinator_instance(&params.run_id);
    let accounts = RunCreateAccounts {
        payer: payer.pubkey(),
        collateral_mint: *collateral_mint,
        run,
        run_collateral,
        coordinator_instance,
        coordinator_account: *coordinator_account,
        coordinator_program: psyche_solana_coordinator::ID,
        associated_token_program: associated_token::ID,
        token_program: token::ID,
        system_program: system_program::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunCreate { params }.data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint.process_instruction(payer, instruction).await?;
    Ok((run, coordinator_instance))
}

pub async fn process_treasurer_run_update(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    authority: &Keypair,
    run: &Pubkey,
    coordinator_instance: &Pubkey,
    coordinator_account: &Pubkey,
    params: RunUpdateParams,
) -> Result<()> {
    let accounts = RunUpdateAccounts {
        authority: authority.pubkey(),
        run: *run,
        coordinator_instance: *coordinator_instance,
        coordinator_account: *coordinator_account,
        coordinator_program: psyche_solana_coordinator::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunUpdate { params }.data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[authority])
        .await?;
    Ok(())
}

pub async fn process_treasurer_participant_create(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    run: &Pubkey,
) -> Result<()> {
    let participant = find_participant(run, &user.pubkey());
    let accounts = ParticipantCreateAccounts {
        payer: payer.pubkey(),
        user: user.pubkey(),
        run: *run,
        participant,
        system_program: system_program::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantCreate {
            params: ParticipantCreateParams {},
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[user])
        .await?;
    Ok(())
}

pub async fn process_treasurer_run_bond_config_update(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    main_authority: &Keypair,
    run: &Pubkey,
    params: RunBondConfigUpdateParams,
) -> Result<()> {
    let accounts = RunBondConfigUpdateAccounts {
        main_authority: main_authority.pubkey(),
        run: *run,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunBondConfigUpdate { params }.data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[main_authority])
        .await?;
    Ok(())
}

/// The run authority's slash instruction, convicting the client at `index`
/// on `evidence`. It is refused unless the same transaction proves the
/// client's p2p key signed the commitment; `process_treasurer_run_slash` sends
/// the two together.
#[allow(clippy::too_many_arguments)]
pub fn run_slash_instruction(
    authority: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    index: u64,
    batch_start: u64,
    batch_end: u64,
    evidence: CommitmentEvidence,
    replayed_hash: [u8; 32],
) -> Instruction {
    let coordinator_instance = find_coordinator_instance(run_id);
    let accounts = RunSlashAccounts {
        authority: *authority,
        run: *run,
        coordinator_instance,
        coordinator_account: *coordinator_account,
        coordinator_program: psyche_solana_coordinator::ID,
        instructions: sysvar::instructions::ID,
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSlash {
            params: RunSlashParams {
                index,
                batch_start,
                batch_end,
                evidence,
                replayed_hash,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// The run authority convicts the client at `index` on `commitment`, a payload
/// it signed, together with the precompile instruction that proves it.
#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_run_slash(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    authority: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    index: u64,
    commitment: &SignedCommitment,
    replayed_hash: [u8; 32],
) -> Result<()> {
    let instruction = run_slash_instruction(
        &authority.pubkey(),
        run,
        coordinator_account,
        run_id,
        index,
        commitment.batch_start,
        commitment.batch_end,
        commitment.evidence,
        replayed_hash,
    );
    endpoint
        .process_instructions_with_signers(
            payer,
            &[commitment.proof_instruction(), instruction],
            &[authority],
        )
        .await?;
    Ok(())
}

/// A verifier's vote to convict `target` on `evidence`. It is refused unless
/// the same transaction proves the target's p2p key signed the commitment;
/// `process_treasurer_run_submit_audit_verdict` sends the two together.
#[allow(clippy::too_many_arguments)]
pub fn run_submit_audit_verdict_instruction(
    payer: &Pubkey,
    verifier: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target: &Pubkey,
    target_index: u64,
    batch_start: u64,
    batch_end: u64,
    evidence: CommitmentEvidence,
    replayed_hash: [u8; 32],
) -> Instruction {
    let coordinator_instance = find_coordinator_instance(run_id);
    let verifier_participant = find_participant(run, verifier);
    let audit_verdict = find_audit_verdict(run, target);
    let accounts = RunSubmitAuditVerdictAccounts {
        payer: *payer,
        verifier: *verifier,
        verifier_participant,
        run: *run,
        coordinator_instance,
        coordinator_account: *coordinator_account,
        audit_verdict,
        coordinator_program: psyche_solana_coordinator::ID,
        system_program: system_program::ID,
        instructions: sysvar::instructions::ID,
    };
    Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSubmitAuditVerdict {
            params: RunSubmitAuditVerdictParams {
                target: *target,
                target_index,
                batch_start,
                batch_end,
                evidence,
                replayed_hash,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    }
}

/// `verifier` votes to convict `target` on `commitment`, a payload the target
/// signed, together with the precompile instruction that proves it.
#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_run_submit_audit_verdict(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    verifier: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target: &Pubkey,
    target_index: u64,
    commitment: &SignedCommitment,
    replayed_hash: [u8; 32],
) -> Result<()> {
    let instruction = run_submit_audit_verdict_instruction(
        &payer.pubkey(),
        &verifier.pubkey(),
        run,
        coordinator_account,
        run_id,
        target,
        target_index,
        commitment.batch_start,
        commitment.batch_end,
        commitment.evidence,
        replayed_hash,
    );
    endpoint
        .process_instructions_with_signers(
            payer,
            &[commitment.proof_instruction(), instruction],
            &[verifier],
        )
        .await?;
    Ok(())
}

pub async fn process_treasurer_run_set_slash_bounty(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    main_authority: &Keypair,
    run: &Pubkey,
    slash_bounty_bps: u16,
) -> Result<()> {
    let accounts = RunSetSlashBountyAccounts {
        main_authority: main_authority.pubkey(),
        run: *run,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSetSlashBounty {
            params: RunSetSlashBountyParams { slash_bounty_bps },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[main_authority])
        .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_participant_bond_finalize_withdraw_with_reporter(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    user_collateral: &Pubkey,
    collateral_mint: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    reporter_collateral: &Pubkey,
) -> Result<()> {
    let run_collateral = ToolboxEndpoint::find_spl_associated_token_account(
        run,
        collateral_mint,
    );
    let participant = find_participant(run, &user.pubkey());
    let accounts = ParticipantBondFinalizeWithdrawAccounts {
        user: user.pubkey(),
        user_collateral: *user_collateral,
        run: *run,
        run_collateral,
        coordinator_account: *coordinator_account,
        participant,
        audit_verdict: find_audit_verdict(run, &user.pubkey()),
        token_program: token::ID,
    };
    let mut metas = accounts.to_account_metas(None);
    metas.push(solana_sdk::instruction::AccountMeta::new(
        *reporter_collateral,
        false,
    ));
    let instruction = Instruction {
        accounts: metas,
        data: ParticipantBondFinalizeWithdraw {
            params: ParticipantBondFinalizeWithdrawParams {},
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[user])
        .await?;
    Ok(())
}

pub async fn process_treasurer_participant_bond_deposit(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    user_collateral: &Pubkey,
    collateral_mint: &Pubkey,
    run: &Pubkey,
    collateral_amount: u64,
) -> Result<()> {
    let run_collateral = ToolboxEndpoint::find_spl_associated_token_account(
        run,
        collateral_mint,
    );
    let participant = find_participant(run, &user.pubkey());
    let accounts = ParticipantBondDepositAccounts {
        user: user.pubkey(),
        user_collateral: *user_collateral,
        run: *run,
        run_collateral,
        participant,
        token_program: token::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantBondDeposit {
            params: ParticipantBondDepositParams { collateral_amount },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[user])
        .await?;
    Ok(())
}

pub async fn process_treasurer_participant_bond_request_withdraw(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    run: &Pubkey,
    collateral_amount: u64,
) -> Result<()> {
    let participant = find_participant(run, &user.pubkey());
    let accounts = ParticipantBondRequestWithdrawAccounts {
        user: user.pubkey(),
        run: *run,
        participant,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantBondRequestWithdraw {
            params: ParticipantBondRequestWithdrawParams { collateral_amount },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[user])
        .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_participant_bond_finalize_withdraw(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    user_collateral: &Pubkey,
    collateral_mint: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
) -> Result<()> {
    let run_collateral = ToolboxEndpoint::find_spl_associated_token_account(
        run,
        collateral_mint,
    );
    let participant = find_participant(run, &user.pubkey());
    let accounts = ParticipantBondFinalizeWithdrawAccounts {
        user: user.pubkey(),
        user_collateral: *user_collateral,
        run: *run,
        run_collateral,
        coordinator_account: *coordinator_account,
        participant,
        audit_verdict: find_audit_verdict(run, &user.pubkey()),
        token_program: token::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantBondFinalizeWithdraw {
            params: ParticipantBondFinalizeWithdrawParams {},
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[user])
        .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_participant_claim(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    user_collateral: &Pubkey,
    collateral_mint: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    claim_earned_points: u64,
) -> Result<()> {
    let run_collateral = ToolboxEndpoint::find_spl_associated_token_account(
        run,
        collateral_mint,
    );
    let participant = find_participant(run, &user.pubkey());
    let accounts = ParticipantClaimAccounts {
        user: user.pubkey(),
        user_collateral: *user_collateral,
        run: *run,
        run_collateral,
        coordinator_account: *coordinator_account,
        participant,
        token_program: token::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantClaim {
            params: ParticipantClaimParams {
                claim_earned_points,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[user])
        .await?;
    Ok(())
}

/// Joins the next epoch of a bond-gated run through the treasurer, which checks
/// the committed bond and opens the join authorization only for this join.
/// The run's coordinator accounts are read from the run itself.
pub async fn process_treasurer_participant_join_run(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    run: &Pubkey,
    client_id: NodeIdentity,
) -> Result<Pubkey> {
    let run_account = get_run(endpoint, run)
        .await?
        .ok_or_else(|| anyhow!("run {run} not found"))?;
    let participant = find_participant(run, &user.pubkey());
    let authorization =
        find_authorization(run, &user.pubkey(), JOIN_RUN_AUTHORIZATION_SCOPE);
    let accounts = ParticipantJoinRunAccounts {
        payer: payer.pubkey(),
        user: user.pubkey(),
        run: *run,
        participant,
        authorization,
        coordinator_instance: run_account.coordinator_instance,
        coordinator_account: run_account.coordinator_account,
        authorizer_program: psyche_solana_authorizer::ID,
        coordinator_program: psyche_solana_coordinator::ID,
        system_program: system_program::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantJoinRun {
            params: ParticipantJoinRunParams { client_id },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[user])
        .await?;
    Ok(authorization)
}

pub async fn process_treasurer_run_set_challenge_config(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    main_authority: &Keypair,
    run: &Pubkey,
    challenge_window_seconds: i64,
    tie_breaker_committee_size: u16,
    appeal_window_seconds: i64,
) -> Result<()> {
    let accounts = RunSetChallengeConfigAccounts {
        main_authority: main_authority.pubkey(),
        run: *run,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSetChallengeConfig {
            params: RunSetChallengeConfigParams {
                challenge_window_seconds,
                tie_breaker_committee_size,
                appeal_window_seconds,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[main_authority])
        .await?;
    Ok(())
}

pub async fn process_treasurer_run_open_challenge(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    challenger: &Keypair,
    run: &Pubkey,
) -> Result<()> {
    let challenger_participant = find_participant(run, &challenger.pubkey());
    let audit_verdict = find_audit_verdict(run, &challenger.pubkey());
    let coordinator_account = get_run(endpoint, run)
        .await?
        .ok_or_else(|| anyhow!("run {run} not found"))?
        .coordinator_account;
    let accounts = RunOpenChallengeAccounts {
        challenger: challenger.pubkey(),
        challenger_participant,
        run: *run,
        coordinator_account,
        audit_verdict,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunOpenChallenge {
            params: RunOpenChallengeParams {},
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[challenger])
        .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_run_submit_appeal_verdict(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    appellate: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target: &Pubkey,
    target_index: u64,
    overturn: bool,
) -> Result<()> {
    let coordinator_instance = find_coordinator_instance(run_id);
    let appellate_participant = find_participant(run, &appellate.pubkey());
    let audit_verdict = find_audit_verdict(run, target);
    let accounts = RunSubmitAppealVerdictAccounts {
        payer: payer.pubkey(),
        appellate: appellate.pubkey(),
        appellate_participant,
        run: *run,
        coordinator_instance,
        coordinator_account: *coordinator_account,
        audit_verdict,
        coordinator_program: psyche_solana_coordinator::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSubmitAppealVerdict {
            params: RunSubmitAppealVerdictParams {
                target: *target,
                target_index,
                overturn,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[appellate])
        .await?;
    Ok(())
}

pub async fn process_treasurer_run_finalize_slash(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    target: &Pubkey,
    target_index: u64,
) -> Result<()> {
    let coordinator_instance = find_coordinator_instance(run_id);
    let audit_verdict = find_audit_verdict(run, target);
    let accounts = RunFinalizeSlashAccounts {
        payer: payer.pubkey(),
        run: *run,
        coordinator_instance,
        coordinator_account: *coordinator_account,
        audit_verdict,
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
    endpoint.process_instruction(payer, instruction).await?;
    Ok(())
}
