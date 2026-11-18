use std::sync::{Arc, Mutex, PoisonError};

use leviathan_multitude::device::{DeviceState, Network, Thermal};
use leviathan_multitude::protocol::{address, signing_key};

use crate::session::{Activity, Report, Session, SessionError};

#[derive(Debug, thiserror::Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum MultitudeError {
    #[error("{0}")]
    Relay(String),
    #[error("{0}")]
    Training(String),
    #[error("{0}")]
    Key(String),
    #[error("{0}")]
    Storage(String),
}

impl From<SessionError> for MultitudeError {
    fn from(err: SessionError) -> Self {
        let message = err.to_string();
        match err {
            SessionError::Relay(_) | SessionError::Corpus => Self::Relay(message),
            SessionError::Training(_) => Self::Training(message),
            SessionError::Storage(_) => Self::Storage(message),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum ThermalLevel {
    Nominal,
    Fair,
    Serious,
    Critical,
}

impl From<Thermal> for ThermalLevel {
    fn from(thermal: Thermal) -> Self {
        match thermal {
            Thermal::Nominal => Self::Nominal,
            Thermal::Fair => Self::Fair,
            Thermal::Serious => Self::Serious,
            Thermal::Critical => Self::Critical,
        }
    }
}

impl From<ThermalLevel> for Thermal {
    fn from(level: ThermalLevel) -> Self {
        match level {
            ThermalLevel::Nominal => Self::Nominal,
            ThermalLevel::Fair => Self::Fair,
            ThermalLevel::Serious => Self::Serious,
            ThermalLevel::Critical => Self::Critical,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum NetworkKind {
    Unmetered,
    Metered,
    Offline,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct DeviceSnapshot {
    pub charging: bool,
    pub battery_percent: u8,
    pub network: NetworkKind,
    pub thermal: ThermalLevel,
    pub user_active: bool,
    pub free_memory_mb: u32,
}

impl From<DeviceSnapshot> for DeviceState {
    fn from(snapshot: DeviceSnapshot) -> Self {
        Self {
            charging: snapshot.charging,
            battery_percent: snapshot.battery_percent,
            network: match snapshot.network {
                NetworkKind::Unmetered => Network::Unmetered,
                NetworkKind::Metered => Network::Metered,
                NetworkKind::Offline => Network::Offline,
            },
            thermal: snapshot.thermal.into(),
            user_active: snapshot.user_active,
            free_memory_mb: snapshot.free_memory_mb,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum PhoneActivity {
    Trained,
    Waiting,
    Paused,
    Convicted,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct TickReport {
    pub activity: PhoneActivity,
    pub message: String,
    pub round: Option<u32>,
    pub global_loss: Option<f32>,
    pub phones_last_round: Option<u32>,
    pub rounds_counted: u32,
    pub audits_answered: u32,
}

impl From<Report> for TickReport {
    fn from(report: Report) -> Self {
        let (activity, message) = match report.activity {
            Activity::Trained {
                round,
                steps,
                loss,
                upload_bytes,
                seconds,
            } => (
                PhoneActivity::Trained,
                format!(
                    "Trained round {round}: {steps} steps in {seconds:.1}s, local loss {loss:.3}, sent {} KB",
                    upload_bytes.div_ceil(1024)
                ),
            ),
            Activity::Waiting(why) => (PhoneActivity::Waiting, capitalized(&why)),
            Activity::Paused(why) => (PhoneActivity::Paused, format!("Paused: {why}")),
            Activity::Convicted(why) => (PhoneActivity::Convicted, format!("Convicted: {why}")),
        };
        Self {
            activity,
            message,
            round: report.round.as_ref().map(|r| r.round),
            global_loss: report.round.as_ref().map(|r| r.global_loss),
            phones_last_round: report.round.as_ref().map(|r| r.phones_last_round),
            rounds_counted: report.rounds_counted,
            audits_answered: report.audits_answered,
        }
    }
}

fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

fn secret_bytes(secret: Vec<u8>) -> Result<[u8; 32], MultitudeError> {
    secret.try_into().map_err(|bytes: Vec<u8>| {
        MultitudeError::Key(format!("a key is 32 bytes, got {}", bytes.len()))
    })
}

#[derive(uniffi::Object)]
pub struct MultitudeNode {
    session: Mutex<Session>,
    address: String,
}

#[uniffi::export]
impl MultitudeNode {
    #[uniffi::constructor]
    pub fn new(
        relay_url: String,
        secret: Vec<u8>,
        state_path: Option<String>,
    ) -> Result<Arc<Self>, MultitudeError> {
        let session = Session::join(
            &relay_url,
            secret_bytes(secret)?,
            state_path.map(Into::into),
        )?;
        Ok(Arc::new(Self {
            address: session.address(),
            session: Mutex::new(session),
        }))
    }

    pub fn address(&self) -> String {
        self.address.clone()
    }

    pub fn tick(&self, device: DeviceSnapshot) -> Result<TickReport, MultitudeError> {
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(session.tick(&device.into())?.into())
    }
}

#[uniffi::export]
pub fn generate_secret() -> Vec<u8> {
    let mut secret = [0u8; 32];
    getrandom::fill(&mut secret).expect("the platform has a random source");
    secret.to_vec()
}

#[uniffi::export]
pub fn address_of(secret: Vec<u8>) -> Result<String, MultitudeError> {
    let key = signing_key(&secret_bytes(secret)?);
    Ok(address(&key.verifying_key().to_bytes()))
}

#[uniffi::export]
pub fn thermal_from_android(status: i32) -> ThermalLevel {
    Thermal::from_android(status).into()
}

#[uniffi::export]
pub fn thermal_from_ios(state: i64) -> ThermalLevel {
    Thermal::from_ios(state).into()
}
