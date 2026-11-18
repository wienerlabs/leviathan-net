uniffi::setup_scaffolding!();

mod ffi;
mod relay;
mod session;

pub use ffi::{
    address_of, generate_secret, thermal_from_android, thermal_from_ios, DeviceSnapshot,
    MultitudeError, MultitudeNode, NetworkKind, PhoneActivity, ThermalLevel, TickReport,
};
pub use relay::{RelayClient, RelayError};
pub use session::{Activity, Report, Session, SessionError};
