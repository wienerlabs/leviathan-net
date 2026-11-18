use anchor_lang::prelude::*;
use psyche_coordinator::Coordinator;
use psyche_solana_coordinator::CoordinatorAccount;

use crate::logic::bounty_amount;
use crate::state::coordinator_slashed_points;
use crate::state::on_charging_roster;
use crate::state::AuditVerdict;
use crate::state::Participant;
use crate::state::Run;
use crate::state::VerdictStatus;
use crate::ProgramError;

#[derive(Accounts)]
#[instruction(params: RunChargeDueConvictionParams)]
pub struct RunChargeDueConvictionAccounts<'info> {
    #[account()]
    pub payer: Signer<'info>,

    #[account(
        mut,
        constraint = run.coordinator_account == coordinator_account.key(),
    )]
    pub run: Box<Account<'info, Run>>,

    // An account of another version is refused, not loaded: loading one
    // of another size panics (wienerlabs/leviathan-net#62).
    #[account(
        constraint = CoordinatorAccount::is_current(&coordinator_account)?,
    )]
    pub coordinator_account: AccountLoader<'info, CoordinatorAccount>,

    #[account(
        mut,
        seeds = [
            AuditVerdict::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump = audit_verdict.bump,
    )]
    pub audit_verdict: Box<Account<'info, AuditVerdict>>,

    /// CHECK: pinned to the target's participant PDA by the seeds below, and
    /// read only after this checks that it exists and belongs to this program.
    ///
    /// Not required to exist, as in `run_charge_conviction`: a run whose join
    /// authority is not the treasurer admits clients that never created a
    /// participant account. With no account at this address there is no bond,
    /// so the conviction is charged nothing, and still settled.
    #[account(
        mut,
        seeds = [
            Participant::SEEDS_PREFIX,
            run.key().as_ref(),
            params.target.as_ref(),
        ],
        bump,
    )]
    pub target_participant: UncheckedAccount<'info>,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RunChargeDueConvictionParams {
    pub target: Pubkey,
}

/// Charges the target's bond, `participant`, for the conviction `verdict`
/// records as due, at `rate`, and records the verdict as upheld and charged,
/// owing its recipients their bounty. Returns the forfeit, which the caller
/// takes off the run's bonded total.
///
/// Only while the target is on no roster an epoch will still charge: there the
/// conviction can still land as an ejection, through `run_finalize_slash`, and
/// is charged by the ledger. The forfeit comes out of what the ledger does not
/// already claim, as an overturned verifier's does, and is not counted as a
/// settled point, since the ledger never saw it. A target with no participant
/// account has no bond, and is charged nothing.
pub fn charge_due_conviction(
    verdict: &mut AuditVerdict,
    participant: Option<&mut Participant>,
    run: &Run,
    coordinator: &Coordinator,
    slashed_points: u64,
    rate: u64,
) -> Result<u64> {
    if verdict.status != VerdictStatus::Due {
        return err!(ProgramError::ConvictionNotDue);
    }
    if on_charging_roster(coordinator, &verdict.target) {
        return err!(ProgramError::TargetInChargingEpoch);
    }
    let forfeit = participant.map_or(0, |participant| {
        participant.forfeit_unclaimed_bond(rate, slashed_points)
    });
    verdict.settle_unlanded(bounty_amount(run, forfeit));
    Ok(forfeit)
}

/// Charges the bond of a target whose conviction is due, when the target is on
/// no roster an epoch will still charge, and records what that owes the
/// conviction's voters, or the bench that upheld it.
///
/// A conviction recorded as `Due` (wienerlabs/leviathan-net#36) used to be
/// settled only through its target: by `run_finalize_slash` once it was back
/// on the roster of a charging epoch, or by its own withdrawal. A target that
/// never came back and never withdrew left the verdict due for good, its
/// voters' bounty never recorded, and its bond held by its own conviction.
/// Now anyone settles it here, at the slashing rate in force, as the
/// withdrawal does (wienerlabs/leviathan-net#74). Each recipient is then
/// credited through `run_pay_verdict_bounty`, and the verdict is no longer
/// due, so a target that comes back later is not charged for it again.
pub fn run_charge_due_conviction_processor(
    context: Context<RunChargeDueConvictionAccounts>,
    params: RunChargeDueConvictionParams,
) -> Result<()> {
    let info = context.accounts.target_participant.to_account_info();
    let mut participant = {
        let data = info.try_borrow_data()?;
        Participant::from_pda_account(info.owner, &data)?
    };
    let forfeit = {
        let account = context.accounts.coordinator_account.load()?;
        charge_due_conviction(
            &mut context.accounts.audit_verdict,
            participant.as_mut(),
            &context.accounts.run,
            &account.state.coordinator,
            coordinator_slashed_points(&account, &params.target),
            account
                .state
                .clients_state
                .current_epoch_rates
                .slashing_rate_per_client,
        )?
    };
    if let Some(participant) = participant.as_ref() {
        let mut data = info.try_borrow_mut_data()?;
        participant.try_serialize(&mut &mut data[..])?;
    }
    let run = &mut context.accounts.run;
    run.total_bonded_amount -= forfeit;
    msg!(
        "charge_due_conviction: target={} forfeit {} owes {}",
        params.target,
        forfeit,
        context.accounts.audit_verdict.bounty_owed
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use psyche_coordinator::Client;
    use psyche_coordinator::ClientState;
    use psyche_coordinator::RunState;
    use psyche_core::FixedVec;
    use psyche_core::NodeIdentity;

    use super::*;
    use crate::state::CommitmentEvidence;
    use crate::state::ConvictionCharge;

    const RATE: u64 = 200;
    const BOUNTY_BPS: u16 = 5_000;

    fn target() -> Pubkey {
        Pubkey::new_from_array([9; 32])
    }

    fn run() -> Run {
        Run {
            bump: 0,
            index: 0,
            main_authority: Pubkey::default(),
            join_authority: Pubkey::default(),
            coordinator_account: Pubkey::default(),
            coordinator_instance: Pubkey::default(),
            collateral_mint: Pubkey::default(),
            total_claimed_collateral_amount: 0,
            total_claimed_earned_points: 0,
            total_bonded_amount: 0,
            bond_minimum_amount: 0,
            bond_withdraw_delay_seconds: 0,
            slash_bounty_bps: BOUNTY_BPS,
            challenge_window_seconds: 0,
            tie_breaker_committee_size: 0,
            appeal_window_seconds: 0,
        }
    }

    fn participant(bond: u64, settled_points: u64) -> Participant {
        Participant {
            bump: 0,
            claimed_collateral_amount: 0,
            claimed_earned_points: 0,
            bond_amount: bond,
            bond_withdraw_pending_amount: bond,
            bond_withdraw_requested_at: 0,
            bond_settled_slashed_points: settled_points,
            bond_withdraw_delay_snapshot: 0,
            open_verdict_votes: 0,
            uncharged_slash_reports: 0,
        }
    }

    fn verdict(status: VerdictStatus) -> AuditVerdict {
        AuditVerdict {
            bump: 0,
            run: Pubkey::new_unique(),
            target: target(),
            epoch: 0,
            round_height: 0,
            status,
            verdict_count: 2,
            committed_hash: [0; 32],
            replayed_hash: [0; 32],
            target_index: 0,
            batch_start: 0,
            batch_end: 0,
            evidence: CommitmentEvidence::default(),
            pending_since_unix: 0,
            challenged_since_unix: 0,
            challenger: Pubkey::default(),
            overturn_count: 0,
            uphold_count: 0,
            settled_count: 0,
            released_voters: 0,
            bounty_owed: 0,
            bounty_paid: 0,
            charge: ConvictionCharge::Unlanded,
            charge_epoch_start: 0,
            charge_rate: 0,
            appeal_upheld: false,
            voters: vec![Pubkey::new_unique(), Pubkey::new_unique()],
            appeal_voters: vec![],
            vote_commitments: vec![[0; 32]; 2],
        }
    }

    /// A coordinator in `run_state` whose roster is someone else, then the
    /// target in `state`.
    fn coordinator_with(
        run_state: RunState,
        state: ClientState,
    ) -> Box<Coordinator> {
        let mut coordinator = Box::new(Coordinator::zeroed());
        coordinator.run_state = run_state;
        coordinator.epoch_state.clients = FixedVec::from_iter([
            Client {
                id: NodeIdentity::from_single_key([1; 32]),
                state: ClientState::Healthy,
                exited_height: 0,
            },
            Client {
                id: NodeIdentity::from_single_key(target().to_bytes()),
                state,
                exited_height: 0,
            },
        ]);
        coordinator
    }

    /// Between epochs the last one has been charged, though its roster, the
    /// target on it, is still on the account.
    fn between_epochs() -> Box<Coordinator> {
        coordinator_with(RunState::WaitingForMembers, ClientState::Healthy)
    }

    #[test]
    fn a_due_conviction_off_every_charging_roster_is_charged_at_the_rate() {
        let mut due = verdict(VerdictStatus::Due);
        let mut p = participant(500, 0);
        let forfeit = charge_due_conviction(
            &mut due,
            Some(&mut p),
            &run(),
            &between_epochs(),
            0,
            RATE,
        )
        .unwrap();
        assert_eq!(forfeit, RATE);
        assert_eq!(p.bond_amount, 500 - RATE);
        // The pending withdrawal never asks for more than the bond holds.
        assert_eq!(p.bond_withdraw_pending_amount, 500 - RATE);
        // The ledger never saw it, so it is not a settled point.
        assert_eq!(p.bond_settled_slashed_points, 0);
        assert_eq!(due.status, VerdictStatus::Upheld);
        assert_eq!(due.charge, ConvictionCharge::Charged);
        assert_eq!(due.bounty_owed, RATE * BOUNTY_BPS as u64 / 10_000);
        assert_eq!(due.bounty_final(), Ok(()));
    }

    #[test]
    fn what_the_ledger_already_claims_is_left_to_it() {
        // The ledger charged 400 the bond has not paid yet: only 100 is free.
        let mut due = verdict(VerdictStatus::Due);
        let mut p = participant(500, 0);
        let forfeit = charge_due_conviction(
            &mut due,
            Some(&mut p),
            &run(),
            &between_epochs(),
            400,
            RATE,
        )
        .unwrap();
        assert_eq!(forfeit, 100);
        assert_eq!(p.bond_amount, 400);
        assert_eq!(p.unsettled_slashed_points(400), 400);
        assert_eq!(due.bounty_owed, 100 * BOUNTY_BPS as u64 / 10_000);
    }

    #[test]
    fn a_target_on_a_charging_roster_is_left_to_the_ledger() {
        let mut cases = vec![];
        for run_state in [
            RunState::Warmup,
            RunState::RoundTrain,
            RunState::RoundWitness,
            RunState::Cooldown,
        ] {
            // Healthy, a conviction lands there as an ejection; withdrawn or
            // ejected already, the epoch's charge is still to come.
            for state in [
                ClientState::Healthy,
                ClientState::Dropped,
                ClientState::Withdrawn,
                ClientState::Ejected,
            ] {
                cases.push(coordinator_with(run_state, state));
            }
        }
        // Moved off the roster at the round's end, still this epoch's.
        let mut left =
            coordinator_with(RunState::Cooldown, ClientState::Healthy);
        let mut exited = left.epoch_state.clients[1];
        exited.state = ClientState::Ejected;
        left.epoch_state.exited_clients = FixedVec::from_iter([exited]);
        left.epoch_state.clients =
            FixedVec::from_iter([left.epoch_state.clients[0]]);
        cases.push(left);

        for coordinator in cases {
            let mut due = verdict(VerdictStatus::Due);
            let mut p = participant(500, 0);
            assert_eq!(
                charge_due_conviction(
                    &mut due,
                    Some(&mut p),
                    &run(),
                    &coordinator,
                    0,
                    RATE,
                )
                .unwrap_err(),
                error!(ProgramError::TargetInChargingEpoch),
                "{}",
                coordinator.run_state
            );
            assert_eq!(due.status, VerdictStatus::Due);
            assert_eq!(due.charge, ConvictionCharge::Unlanded);
            assert_eq!(p.bond_amount, 500);
        }
    }

    #[test]
    fn a_target_off_the_roster_of_an_epoch_under_way_is_charged() {
        let mut coordinator =
            coordinator_with(RunState::RoundTrain, ClientState::Healthy);
        coordinator.epoch_state.clients =
            FixedVec::from_iter([coordinator.epoch_state.clients[0]]);
        let mut due = verdict(VerdictStatus::Due);
        let mut p = participant(500, 0);
        assert_eq!(
            charge_due_conviction(
                &mut due,
                Some(&mut p),
                &run(),
                &coordinator,
                0,
                RATE
            )
            .unwrap(),
            RATE
        );
        assert_eq!(due.status, VerdictStatus::Upheld);
    }

    #[test]
    fn only_a_due_conviction_is_charged_this_way_and_only_once() {
        for status in [
            VerdictStatus::Voting,
            VerdictStatus::SlashPending,
            VerdictStatus::Challenged,
            VerdictStatus::Upheld,
            VerdictStatus::Overturned,
        ] {
            let mut other = verdict(status);
            let mut p = participant(500, 0);
            assert_eq!(
                charge_due_conviction(
                    &mut other,
                    Some(&mut p),
                    &run(),
                    &between_epochs(),
                    0,
                    RATE,
                )
                .unwrap_err(),
                error!(ProgramError::ConvictionNotDue),
                "{status:?}"
            );
            assert_eq!(p.bond_amount, 500);
        }

        let mut due = verdict(VerdictStatus::Due);
        let mut p = participant(500, 0);
        charge_due_conviction(
            &mut due,
            Some(&mut p),
            &run(),
            &between_epochs(),
            0,
            RATE,
        )
        .unwrap();
        assert_eq!(
            charge_due_conviction(
                &mut due,
                Some(&mut p),
                &run(),
                &between_epochs(),
                0,
                RATE,
            )
            .unwrap_err(),
            error!(ProgramError::ConvictionNotDue),
        );
        assert_eq!(p.bond_amount, 500 - RATE);
    }

    #[test]
    fn a_target_with_no_account_is_settled_with_nothing() {
        let mut due = verdict(VerdictStatus::Due);
        let forfeit = charge_due_conviction(
            &mut due,
            None,
            &run(),
            &between_epochs(),
            0,
            RATE,
        )
        .unwrap();
        assert_eq!(forfeit, 0);
        assert_eq!(due.status, VerdictStatus::Upheld);
        assert_eq!(due.charge, ConvictionCharge::Charged);
        assert_eq!(due.bounty_owed, 0);
    }
}
