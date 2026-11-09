//! When a phone is allowed to train, and how much.
//!
//! Training is heavy, sustained compute. Done carelessly it drains the
//! battery, heats the phone (heat while charging is what ages a battery) and
//! eats a metered data plan. The default policy trains only on a charger, on an
//! unmetered network, with the screen off and the phone cool.
//!
//! The app reads the platform signals and maps them here:
//! iOS `ProcessInfo.thermalState` with [`Thermal::from_ios`], Android
//! `PowerManager.getCurrentThermalStatus()` with [`Thermal::from_android`].

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Thermal {
    Nominal,
    Fair,
    Serious,
    Critical,
}

impl Thermal {
    /// `ProcessInfo.ThermalState` raw values: nominal 0, fair 1, serious 2,
    /// critical 3.
    pub fn from_ios(raw: i64) -> Self {
        match raw {
            i64::MIN..=0 => Self::Nominal,
            1 => Self::Fair,
            2 => Self::Serious,
            _ => Self::Critical,
        }
    }

    /// `PowerManager.THERMAL_STATUS_*`: none 0, light 1, moderate 2, severe 3,
    /// critical 4, emergency 5, shutdown 6. Moderate already means the system
    /// is throttling, so it maps to serious.
    pub fn from_android(status: i32) -> Self {
        match status {
            i32::MIN..=0 => Self::Nominal,
            1 => Self::Fair,
            2 => Self::Serious,
            _ => Self::Critical,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Network {
    Unmetered,
    Metered,
    Offline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceState {
    pub charging: bool,
    pub battery_percent: u8,
    pub network: Network,
    pub thermal: Thermal,
    pub user_active: bool,
    pub free_memory_mb: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevicePolicy {
    pub require_charging: bool,
    pub min_battery_percent: u8,
    pub require_unmetered: bool,
    pub max_thermal: Thermal,
    pub require_idle: bool,
    pub min_free_memory_mb: u32,
}

impl Default for DevicePolicy {
    fn default() -> Self {
        Self {
            require_charging: true,
            min_battery_percent: 50,
            require_unmetered: true,
            max_thermal: Thermal::Fair,
            require_idle: true,
            min_free_memory_mb: 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauseReason {
    Offline,
    MeteredNetwork,
    NotCharging,
    BatteryLow(u8),
    TooHot(Thermal),
    InUse,
    LowMemory(u32),
}

impl std::fmt::Display for PauseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Offline => write!(f, "offline"),
            Self::MeteredNetwork => write!(f, "on a metered network"),
            Self::NotCharging => write!(f, "not charging"),
            Self::BatteryLow(percent) => write!(f, "battery at {percent}%"),
            Self::TooHot(thermal) => write!(f, "thermal state {thermal:?}"),
            Self::InUse => write!(f, "in use"),
            Self::LowMemory(mb) => write!(f, "{mb} MB free"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Train,
    Pause(PauseReason),
}

pub fn decide(policy: &DevicePolicy, state: &DeviceState) -> Decision {
    use PauseReason::*;
    let pause = if state.network == Network::Offline {
        Some(Offline)
    } else if policy.require_unmetered && state.network == Network::Metered {
        Some(MeteredNetwork)
    } else if policy.require_charging && !state.charging {
        Some(NotCharging)
    } else if state.battery_percent < policy.min_battery_percent {
        Some(BatteryLow(state.battery_percent))
    } else if state.thermal > policy.max_thermal {
        Some(TooHot(state.thermal))
    } else if policy.require_idle && state.user_active {
        Some(InUse)
    } else if state.free_memory_mb < policy.min_free_memory_mb {
        Some(LowMemory(state.free_memory_mb))
    } else {
        None
    };
    pause.map_or(Decision::Train, Decision::Pause)
}

/// Inner steps that fit in a round at the measured throughput.
///
/// A quarter of the round is kept for compression, upload and the audit
/// lottery. A phone that cannot fit `min_steps` returns 0 and sits the round
/// out instead of holding everyone up: in the main network a node that misses
/// `max_round_train_time` is dropped, and here it simply does not submit.
pub fn plan_inner_steps(
    tokens_per_second: f32,
    tokens_per_step: u64,
    round_seconds: f32,
    min_steps: u16,
    max_steps: u16,
) -> u16 {
    if tokens_per_second.is_nan()
        || tokens_per_second <= 0.0
        || tokens_per_step == 0
        || round_seconds.is_nan()
        || round_seconds <= 0.0
    {
        return 0;
    }
    let steps = (round_seconds * 0.75 * tokens_per_second / tokens_per_step as f32).floor();
    if steps < min_steps as f32 {
        0
    } else {
        steps.min(max_steps as f32) as u16
    }
}

/// Exponential moving average of measured training throughput.
#[derive(Clone, Copy, Debug)]
pub struct ThroughputMeter {
    alpha: f32,
    tokens_per_second: Option<f32>,
}

impl ThroughputMeter {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            tokens_per_second: None,
        }
    }

    pub fn record(&mut self, tokens: u64, seconds: f32) {
        if seconds.is_nan() || seconds <= 0.0 {
            return;
        }
        let sample = tokens as f32 / seconds;
        self.tokens_per_second = Some(match self.tokens_per_second {
            None => sample,
            Some(previous) => previous + self.alpha * (sample - previous),
        });
    }

    pub fn tokens_per_second(&self) -> Option<f32> {
        self.tokens_per_second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> DeviceState {
        DeviceState {
            charging: true,
            battery_percent: 80,
            network: Network::Unmetered,
            thermal: Thermal::Nominal,
            user_active: false,
            free_memory_mb: 4096,
        }
    }

    #[test]
    fn trains_only_when_every_condition_holds() {
        let policy = DevicePolicy::default();
        assert_eq!(decide(&policy, &ready()), Decision::Train);
        let cases = [
            (
                DeviceState {
                    charging: false,
                    ..ready()
                },
                PauseReason::NotCharging,
            ),
            (
                DeviceState {
                    network: Network::Metered,
                    ..ready()
                },
                PauseReason::MeteredNetwork,
            ),
            (
                DeviceState {
                    network: Network::Offline,
                    ..ready()
                },
                PauseReason::Offline,
            ),
            (
                DeviceState {
                    battery_percent: 20,
                    ..ready()
                },
                PauseReason::BatteryLow(20),
            ),
            (
                DeviceState {
                    thermal: Thermal::Serious,
                    ..ready()
                },
                PauseReason::TooHot(Thermal::Serious),
            ),
            (
                DeviceState {
                    user_active: true,
                    ..ready()
                },
                PauseReason::InUse,
            ),
            (
                DeviceState {
                    free_memory_mb: 300,
                    ..ready()
                },
                PauseReason::LowMemory(300),
            ),
        ];
        for (state, reason) in cases {
            assert_eq!(decide(&policy, &state), Decision::Pause(reason));
        }
    }

    #[test]
    fn platform_thermal_states_map_conservatively() {
        assert_eq!(Thermal::from_ios(1), Thermal::Fair);
        assert_eq!(Thermal::from_ios(3), Thermal::Critical);
        assert_eq!(Thermal::from_android(1), Thermal::Fair);
        assert_eq!(Thermal::from_android(2), Thermal::Serious);
        assert_eq!(Thermal::from_android(6), Thermal::Critical);
    }

    #[test]
    fn slow_phones_plan_fewer_steps_or_sit_out() {
        // 512 tokens per step, a 60 s round, 45 s of it for training.
        assert_eq!(plan_inner_steps(600.0, 512, 60.0, 2, 12), 12);
        assert_eq!(plan_inner_steps(100.0, 512, 60.0, 2, 12), 8);
        assert_eq!(plan_inner_steps(20.0, 512, 60.0, 2, 12), 0);
        assert_eq!(plan_inner_steps(0.0, 512, 60.0, 2, 12), 0);
    }

    #[test]
    fn throughput_meter_smooths() {
        let mut meter = ThroughputMeter::new(0.5);
        meter.record(1000, 1.0);
        meter.record(2000, 1.0);
        assert_eq!(meter.tokens_per_second(), Some(1500.0));
    }
}
