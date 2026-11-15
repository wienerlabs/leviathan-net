use psyche_solana_treasurer::ProgramError;
use solana_sdk::instruction::InstructionError;
use solana_sdk::transaction::TransactionError;

/// The custom error a refused instruction returned, so a suite checks a refusal
/// for its reason and not merely for having failed.
pub fn refusal_code<T: std::fmt::Debug>(result: anyhow::Result<T>) -> u32 {
    let error = result.expect_err("the instruction was expected to be refused");
    match error.downcast_ref::<TransactionError>() {
        Some(TransactionError::InstructionError(
            _,
            InstructionError::Custom(code),
        )) => *code,
        _ => panic!("expected a program error, got: {error:#}"),
    }
}

/// The code the treasurer program returns for `error`.
pub fn treasurer_refusal(error: ProgramError) -> u32 {
    u32::from(error)
}
