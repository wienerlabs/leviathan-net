use anchor_lang::InstructionData;
use anchor_lang::ToAccountMetas;
use anchor_spl::associated_token;
use anchor_spl::token;
use anyhow::Result;
use psyche_solana_authorizer::find_authorization;
use psyche_solana_coordinator::find_coordinator_instance;
use psyche_solana_coordinator::logic::JOIN_RUN_AUTHORIZATION_SCOPE;
use psyche_solana_treasurer::accounts::ParticipantAuthorizeJoinAccounts;
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
use psyche_solana_treasurer::instruction::ParticipantAuthorizeJoin;
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
use psyche_solana_treasurer::logic::RunBondConfigUpdateParams;
use psyche_solana_treasurer::logic::RunCreateParams;
use psyche_solana_treasurer::logic::RunSetSlashBountyParams;
use psyche_solana_treasurer::accounts::RunSubmitAuditVerdictAccounts;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::instruction::RunSubmitAuditVerdict;
use psyche_solana_treasurer::logic::RunSlashParams;
use psyche_solana_treasurer::logic::RunSubmitAuditVerdictParams;
use psyche_solana_treasurer::logic::RunUpdateParams;
use solana_sdk::instruction::Instruction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::system_program;
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

#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_run_slash(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    authority: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    index: u64,
) -> Result<()> {
    process_treasurer_run_slash_with_hashes(
        endpoint,
        payer,
        authority,
        run,
        coordinator_account,
        run_id,
        index,
        0,
        0,
        [0x11; 32],
        [0x22; 32],
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn process_treasurer_run_slash_with_hashes(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    authority: &Keypair,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    run_id: &str,
    index: u64,
    batch_start: u64,
    batch_end: u64,
    committed_hash: [u8; 32],
    replayed_hash: [u8; 32],
) -> Result<()> {
    let coordinator_instance = find_coordinator_instance(run_id);
    let accounts = RunSlashAccounts {
        authority: authority.pubkey(),
        run: *run,
        coordinator_instance,
        coordinator_account: *coordinator_account,
        coordinator_program: psyche_solana_coordinator::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSlash {
            params: RunSlashParams {
                index,
                batch_start,
                batch_end,
                committed_hash,
                replayed_hash,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[authority])
        .await?;
    Ok(())
}

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
    batch_start: u64,
    batch_end: u64,
    committed_hash: [u8; 32],
    replayed_hash: [u8; 32],
) -> Result<()> {
    let coordinator_instance = find_coordinator_instance(run_id);
    let verifier_participant = find_participant(run, &verifier.pubkey());
    let audit_verdict = find_audit_verdict(run, target);
    let accounts = RunSubmitAuditVerdictAccounts {
        payer: payer.pubkey(),
        verifier: verifier.pubkey(),
        verifier_participant,
        run: *run,
        coordinator_instance,
        coordinator_account: *coordinator_account,
        audit_verdict,
        coordinator_program: psyche_solana_coordinator::ID,
        system_program: system_program::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: RunSubmitAuditVerdict {
            params: RunSubmitAuditVerdictParams {
                target: *target,
                target_index,
                batch_start,
                batch_end,
                committed_hash,
                replayed_hash,
            },
        }
        .data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[verifier])
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
        audit_verdict: None,
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
        audit_verdict: None,
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
pub async fn process_treasurer_participant_bond_finalize_withdraw_with_voters(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Keypair,
    user_collateral: &Pubkey,
    collateral_mint: &Pubkey,
    run: &Pubkey,
    coordinator_account: &Pubkey,
    voter_collaterals: &[Pubkey],
) -> Result<()> {
    let run_collateral = ToolboxEndpoint::find_spl_associated_token_account(
        run,
        collateral_mint,
    );
    let participant = find_participant(run, &user.pubkey());
    let audit_verdict = find_audit_verdict(run, &user.pubkey());
    let accounts = ParticipantBondFinalizeWithdrawAccounts {
        user: user.pubkey(),
        user_collateral: *user_collateral,
        run: *run,
        run_collateral,
        coordinator_account: *coordinator_account,
        participant,
        audit_verdict: Some(audit_verdict),
        token_program: token::ID,
    };
    let mut metas = accounts.to_account_metas(None);
    for voter_collateral in voter_collaterals {
        metas.push(solana_sdk::instruction::AccountMeta::new(
            *voter_collateral,
            false,
        ));
    }
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

pub async fn process_treasurer_participant_authorize_join(
    endpoint: &mut ToolboxEndpoint,
    payer: &Keypair,
    user: &Pubkey,
    run: &Pubkey,
) -> Result<Pubkey> {
    let participant = find_participant(run, user);
    let authorization = find_authorization(run, user, JOIN_RUN_AUTHORIZATION_SCOPE);
    let accounts = ParticipantAuthorizeJoinAccounts {
        payer: payer.pubkey(),
        user: *user,
        run: *run,
        participant,
        authorization,
        authorizer_program: psyche_solana_authorizer::ID,
        system_program: system_program::ID,
    };
    let instruction = Instruction {
        accounts: accounts.to_account_metas(None),
        data: ParticipantAuthorizeJoin {}.data(),
        program_id: psyche_solana_treasurer::ID,
    };
    endpoint
        .process_instruction_with_signers(payer, instruction, &[])
        .await?;
    Ok(authorization)
}
