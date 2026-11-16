use prometheus_client::encoding::EncodeLabelSet;

/// The run a series is about. Every Leviathan series carries it, so one
/// Prometheus can watch several runs and a rule compares a run only with
/// itself.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RunLabels {
    pub run_id: String,
}

impl RunLabels {
    pub fn new(run_id: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
        }
    }
}

/// A run and the status of a verdict in it.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RunStatusLabels {
    pub run_id: String,
    pub status: String,
}

impl RunStatusLabels {
    pub fn new(run_id: &str, status: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            status: status.to_string(),
        }
    }
}

/// A run and the state of a client in it.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RunStateLabels {
    pub run_id: String,
    pub state: String,
}

impl RunStateLabels {
    pub fn new(run_id: &str, state: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            state: state.to_string(),
        }
    }
}
