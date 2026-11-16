//! Reading a run off chain.
//!
//! Everything starts from the run id. The coordinator instance is the PDA the
//! coordinator program derives from it, and the instance names the coordinator
//! account. A treasurer-managed run is one whose instance's main authority is
//! a treasurer `Run` pointing back at that instance: `run_create` sets it up
//! that way.

use std::collections::HashMap;
use std::time::Duration;

use anchor_lang::AccountDeserialize;
use anchor_spl::token::TokenAccount;
use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use psyche_solana_coordinator::bytes_from_string;
use psyche_solana_coordinator::coordinator_account_from_bytes;
use psyche_solana_coordinator::CoordinatorInstance;
use psyche_solana_coordinator::CoordinatorInstanceState;
use psyche_solana_treasurer::find_audit_verdict;
use psyche_solana_treasurer::state::AuditVerdict;
use psyche_solana_treasurer::state::Run;
use solana_sdk::account::Account;
use solana_sdk::clock::Clock;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::sysvar;
use solana_toolbox_endpoint::ToolboxEndpoint;

use crate::metrics::IndexerMetrics;
use crate::observe::observe_run;
use crate::observe::RunObservation;
use crate::observe::SnapshotClock;
use crate::observe::TreasurerAccounts;
use crate::verdicts::VerdictLedger;

/// The most accounts one `getMultipleAccounts` request may name.
const MAX_ACCOUNTS_PER_REQUEST: usize = 100;

/// Where a run lives on chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunAddresses {
    pub coordinator_instance: Pubkey,
    pub coordinator_account: Pubkey,
    /// The treasurer run holding the run's bonds and rewards, and its vault.
    pub treasurer: Option<TreasurerAddresses>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreasurerAddresses {
    pub run: Pubkey,
    pub vault: Pubkey,
}

/// A treasurer run as it stood at the snapshot's slot, and the verdict
/// accounts of the run's registered clients that exist.
pub struct TreasurerSnapshot {
    pub run: Run,
    pub vault_balance: u64,
    pub verdicts: Vec<AuditVerdict>,
}

/// A run's accounts read together at one slot.
pub struct RunSnapshot {
    pub addresses: RunAddresses,
    pub clock: SnapshotClock,
    /// The coordinator account's bytes, validated when read. Kept as bytes and
    /// borrowed through [`RunSnapshot::state`]: the state is a few hundred
    /// kilobytes, and a copy of it on an async task's stack overflows it
    /// (wienerlabs/leviathan-net#35).
    coordinator_data: Vec<u8>,
    pub treasurer: Option<TreasurerSnapshot>,
}

impl RunSnapshot {
    pub fn state(&self) -> &CoordinatorInstanceState {
        &coordinator_account_from_bytes(&self.coordinator_data)
            .expect("the coordinator account was validated when it was read")
            .state
    }

    pub fn observe(&self, ledger: &mut VerdictLedger) -> RunObservation {
        let treasurer = self.treasurer.as_ref().map(|treasurer| TreasurerAccounts {
            run: &treasurer.run,
            vault_balance: treasurer.vault_balance,
            verdicts: &treasurer.verdicts,
        });
        observe_run(self.state(), self.clock, treasurer, ledger)
    }
}

fn decode_instance(account: Account) -> Result<CoordinatorInstance> {
    if account.owner != psyche_solana_coordinator::ID {
        bail!("the coordinator instance is not owned by the coordinator program");
    }
    CoordinatorInstance::try_deserialize(&mut account.data.as_slice())
        .map_err(|err| anyhow!("cannot decode the coordinator instance: {err}"))
}

/// The treasurer run behind `instance`, if `authority` is one: owned by the
/// treasurer program, and pointing back at the instance.
fn decode_treasurer(
    instance: &Pubkey,
    authority: &Pubkey,
    account: Option<Account>,
) -> Result<Option<(Run, TreasurerAddresses)>> {
    let Some(account) = account.filter(|account| account.owner == psyche_solana_treasurer::ID)
    else {
        return Ok(None);
    };
    let run = Run::try_deserialize(&mut account.data.as_slice())
        .map_err(|err| anyhow!("cannot decode the treasurer run {authority}: {err}"))?;
    if run.coordinator_instance != *instance {
        return Ok(None);
    }
    let vault = ToolboxEndpoint::find_spl_associated_token_account(authority, &run.collateral_mint);
    Ok(Some((
        run,
        TreasurerAddresses {
            run: *authority,
            vault,
        },
    )))
}

/// The treasurer run at `address`, read again in the snapshot, checked the way
/// `resolve_run` checked it: owned by the treasurer, and still the run of
/// `instance`.
fn decode_run(account: Account, address: &Pubkey, instance: &Pubkey) -> Result<Run> {
    if account.owner != psyche_solana_treasurer::ID {
        bail!("the treasurer run {address} is no longer owned by the treasurer program");
    }
    let run = Run::try_deserialize(&mut account.data.as_slice())
        .map_err(|err| anyhow!("cannot decode the treasurer run {address}: {err}"))?;
    if run.coordinator_instance != *instance {
        bail!("the treasurer run {address} no longer manages this run");
    }
    Ok(run)
}

fn decode_vault(account: Account, treasurer: &TreasurerAddresses, run: &Run) -> Result<u64> {
    let vault = TokenAccount::try_deserialize(&mut account.data.as_slice())
        .map_err(|err| anyhow!("cannot decode the vault {}: {err}", treasurer.vault))?;
    if vault.mint != run.collateral_mint || vault.owner != treasurer.run {
        bail!(
            "the vault {} is not the run's collateral account",
            treasurer.vault
        );
    }
    Ok(vault.amount)
}

/// The coordinator instance PDA of `run_id`, the address
/// `find_coordinator_instance` derives. A run id too long to be a PDA seed
/// names no run, and is refused here instead of panicking the way
/// `find_program_address` does on it.
pub fn coordinator_instance_of(run_id: &str) -> Result<Pubkey> {
    Pubkey::try_find_program_address(
        &[CoordinatorInstance::SEEDS_PREFIX, bytes_from_string(run_id)],
        &psyche_solana_coordinator::ID,
    )
    .map(|(address, _)| address)
    .ok_or_else(|| {
        anyhow!(
            "no run can have the id {run_id:?}: at {} bytes it is longer than a PDA seed allows",
            run_id.len()
        )
    })
}

/// Where `run_id` lives: its coordinator instance, its coordinator account,
/// and its treasurer run if one manages it.
pub async fn resolve_run(endpoint: &mut ToolboxEndpoint, run_id: &str) -> Result<RunAddresses> {
    let coordinator_instance = coordinator_instance_of(run_id)?;
    let instance = endpoint
        .get_account(&coordinator_instance)
        .await?
        .ok_or_else(|| {
            anyhow!(
                "no run {run_id}: its coordinator instance {coordinator_instance} does not exist"
            )
        })?;
    let instance = decode_instance(instance)?;
    let authority = endpoint.get_account(&instance.main_authority).await?;
    let treasurer = decode_treasurer(&coordinator_instance, &instance.main_authority, authority)?
        .map(|(_, addresses)| addresses);
    Ok(RunAddresses {
        coordinator_instance,
        coordinator_account: instance.coordinator_account,
        treasurer,
    })
}

/// The verdict accounts of `targets` in `treasurer_run` that exist, read in
/// as few requests as the RPC allows: a hundred to a request, so at most three
/// for the coordinator's 256 clients. Every client ever registered is read, not
/// only this epoch's roster, because a dispute outlives the target's seat: its
/// appeal and the losing-verifier crank can still move it.
async fn read_verdicts(
    endpoint: &mut ToolboxEndpoint,
    treasurer_run: &Pubkey,
    targets: &[Pubkey],
) -> Result<Vec<AuditVerdict>> {
    let addresses: Vec<Pubkey> = targets
        .iter()
        .map(|target| find_audit_verdict(treasurer_run, target))
        .collect();
    let mut verdicts = Vec::new();
    for chunk in addresses.chunks(MAX_ACCOUNTS_PER_REQUEST) {
        for (address, account) in chunk.iter().zip(endpoint.get_accounts(chunk).await?) {
            let Some(account) = account else {
                continue;
            };
            if let Some(verdict) = AuditVerdict::from_pda_account(&account.owner, &account.data)
                .map_err(|err| anyhow!("cannot decode the verdict {address}: {err}"))?
            {
                verdicts.push(verdict);
            }
        }
    }
    Ok(verdicts)
}

fn decode_clock(account: Option<Account>) -> Result<Clock> {
    let account = account.ok_or_else(|| anyhow!("the RPC returned no Clock sysvar"))?;
    solana_sdk::account::from_account::<Clock, _>(&account)
        .ok_or_else(|| anyhow!("cannot decode the Clock sysvar"))
}

/// `run_id`'s accounts, read at one slot.
///
/// The Clock sysvar, the coordinator account, the treasurer run and its vault
/// are named in one request, so they are one slot's state: the vault and the
/// bonds it backs are never read across a deposit or a withdrawal. The
/// verdict accounts follow, from the client list that slot holds. `tip` is
/// read last for the newest slot the RPC knows of; pass an endpoint at the
/// `processed` commitment, or `None` to ask `endpoint` again.
pub async fn read_run(
    endpoint: &mut ToolboxEndpoint,
    tip: Option<&mut ToolboxEndpoint>,
    run_id: &str,
) -> Result<RunSnapshot> {
    let addresses = resolve_run(endpoint, run_id).await?;
    let mut names = vec![sysvar::clock::ID, addresses.coordinator_account];
    if let Some(treasurer) = &addresses.treasurer {
        names.extend([treasurer.run, treasurer.vault]);
    }
    let mut accounts = endpoint.get_accounts(&names).await?.into_iter();
    let clock = decode_clock(accounts.next().flatten())?;
    let coordinator = accounts.next().flatten().ok_or_else(|| {
        anyhow!(
            "the coordinator account {} does not exist",
            addresses.coordinator_account
        )
    })?;
    if coordinator.owner != psyche_solana_coordinator::ID {
        bail!("the coordinator account is not owned by the coordinator program");
    }
    let coordinator_data = coordinator.data;
    let state = &coordinator_account_from_bytes(&coordinator_data)
        .map_err(|err| anyhow!("cannot decode the coordinator account: {err}"))?
        .state;

    let treasurer = match &addresses.treasurer {
        None => None,
        Some(treasurer_addresses) => {
            let run = decode_run(
                accounts
                    .next()
                    .flatten()
                    .ok_or_else(|| anyhow!("the treasurer run vanished"))?,
                &treasurer_addresses.run,
                &addresses.coordinator_instance,
            )?;
            let vault_balance = decode_vault(
                accounts.next().flatten().ok_or_else(|| {
                    anyhow!("the vault {} does not exist", treasurer_addresses.vault)
                })?,
                treasurer_addresses,
                &run,
            )?;
            let targets: Vec<Pubkey> = state
                .clients_state
                .clients
                .iter()
                .map(|client| Pubkey::new_from_array(*client.id.signer()))
                .collect();
            let verdicts = read_verdicts(endpoint, &treasurer_addresses.run, &targets).await?;
            Some(TreasurerSnapshot {
                run,
                vault_balance,
                verdicts,
            })
        }
    };

    let tip_slot = match tip {
        Some(tip) => tip.get_sysvar_clock().await?.slot,
        None => endpoint.get_sysvar_clock().await?.slot,
    };
    Ok(RunSnapshot {
        addresses,
        clock: SnapshotClock {
            slot: clock.slot,
            unix_timestamp: clock.unix_timestamp,
            tip_slot,
        },
        coordinator_data,
        treasurer,
    })
}

/// What the exporter remembers of one run between polls.
#[derive(Debug, Default)]
pub struct RunWatch {
    ledger: VerdictLedger,
    treasurer_seen: bool,
}

impl RunWatch {
    /// What `snapshot` says, unless it contradicts what earlier polls saw.
    ///
    /// A treasurer run never stops managing its run, so a snapshot without
    /// one, after one was seen, is a bad read rather than a run that lost its
    /// treasury. Taken at its word, it would remove every treasury series, and
    /// every alert on them, the drain included, would resolve in silence.
    pub fn observe(&mut self, snapshot: &RunSnapshot) -> Result<RunObservation> {
        self.admit(snapshot.treasurer.is_some())?;
        Ok(snapshot.observe(&mut self.ledger))
    }

    fn admit(&mut self, has_treasurer: bool) -> Result<()> {
        match (self.treasurer_seen, has_treasurer) {
            (true, false) => {
                bail!("the run's treasurer run was seen before and is missing from this read")
            }
            (_, true) => self.treasurer_seen = true,
            (false, false) => {}
        }
        Ok(())
    }
}

/// Polls `run_ids` every `interval` and exports what each poll reads, for as
/// long as the process runs. A poll that fails is counted and logged, and the
/// next one tries again.
pub async fn export_forever(
    endpoint: &mut ToolboxEndpoint,
    tip: &mut ToolboxEndpoint,
    run_ids: &[String],
    metrics: &IndexerMetrics,
    interval: Duration,
) -> ! {
    let mut watches: HashMap<&str, RunWatch> = HashMap::new();
    loop {
        for run_id in run_ids {
            let watch = watches.entry(run_id.as_str()).or_default();
            let observation = read_run(endpoint, Some(&mut *tip), run_id)
                .await
                .and_then(|snapshot| watch.observe(&snapshot));
            match observation {
                Ok(observation) => metrics.observe(run_id, &observation),
                Err(err) => {
                    eprintln!("[leviathan-indexer] cannot read run {run_id}: {err:#}");
                    metrics.poll_failed(run_id);
                }
            }
        }
        tokio::time::sleep(interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::RunWatch;

    #[test]
    fn a_treasurer_seen_once_cannot_silently_disappear() {
        let mut watch = RunWatch::default();
        // A coordinator-only run stays one.
        watch.admit(false).unwrap();
        watch.admit(false).unwrap();
        // Once a treasurer is seen, a read without it is refused, every time,
        // and a read with it is taken again.
        watch.admit(true).unwrap();
        assert!(watch.admit(false).is_err());
        assert!(watch.admit(false).is_err());
        watch.admit(true).unwrap();
    }
}
