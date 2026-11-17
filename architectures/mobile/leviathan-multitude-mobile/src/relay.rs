use std::time::Duration;

use leviathan_multitude::wire::{RoundInfo, RoundResult, RunInfo};
use leviathan_multitude::{Contribution, ErrorReveal};
use serde::de::DeserializeOwned;
use ureq::Agent;

const TIMEOUT: Duration = Duration::from_secs(60);
const MAX_BODY: u64 = 1 << 30;

#[derive(Debug, thiserror::Error)]
pub enum RelayError {
    #[error("relay unreachable: {0}")]
    Transport(String),
    #[error("relay answered {status}: {message}")]
    Refused { status: u16, message: String },
    #[error("relay sent something unreadable: {0}")]
    Decode(String),
}

fn transport(err: ureq::Error) -> RelayError {
    RelayError::Transport(err.to_string())
}

fn missing(path: &str) -> RelayError {
    RelayError::Refused {
        status: 404,
        message: format!("{path} not found"),
    }
}

pub struct RelayClient {
    agent: Agent,
    base: String,
}

impl RelayClient {
    pub fn new(base: &str) -> Self {
        let agent: Agent = Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            agent,
            base: base.trim_end_matches('/').to_string(),
        }
    }

    fn fetch(&self, path: &str) -> Result<Option<Vec<u8>>, RelayError> {
        let mut response = self
            .agent
            .get(format!("{}{path}", self.base))
            .call()
            .map_err(transport)?;
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .with_config()
            .limit(MAX_BODY)
            .read_to_vec()
            .map_err(transport)?;
        match status {
            200..=299 => Ok(Some(body)),
            404 => Ok(None),
            _ => Err(RelayError::Refused {
                status,
                message: String::from_utf8_lossy(&body).into_owned(),
            }),
        }
    }

    fn get<T: DeserializeOwned>(&self, path: &str) -> Result<Option<T>, RelayError> {
        self.fetch(path)?
            .map(|bytes| {
                postcard::from_bytes(&bytes).map_err(|err| RelayError::Decode(err.to_string()))
            })
            .transpose()
    }

    fn require<T: DeserializeOwned>(&self, path: &str) -> Result<T, RelayError> {
        self.get(path)?.ok_or_else(|| missing(path))
    }

    fn send(&self, path: &str, body: &[u8]) -> Result<String, RelayError> {
        let mut response = self
            .agent
            .post(format!("{}{path}", self.base))
            .header("content-type", "application/octet-stream")
            .send(body)
            .map_err(transport)?;
        let status = response.status().as_u16();
        let text = response.body_mut().read_to_string().map_err(transport)?;
        if (200..300).contains(&status) {
            Ok(text)
        } else {
            Err(RelayError::Refused {
                status,
                message: text,
            })
        }
    }

    pub fn run(&self) -> Result<RunInfo, RelayError> {
        self.require("/v1/run")
    }

    pub fn corpus(&self) -> Result<Vec<u8>, RelayError> {
        self.fetch("/v1/corpus")?
            .ok_or_else(|| missing("/v1/corpus"))
    }

    pub fn round(&self) -> Result<RoundInfo, RelayError> {
        self.require("/v1/round")
    }

    pub fn params(&self) -> Result<(u32, Vec<f32>), RelayError> {
        self.require("/v1/params")
    }

    pub fn result(&self, round: u32) -> Result<Option<RoundResult>, RelayError> {
        self.get(&format!("/v1/rounds/{round}"))
    }

    pub fn contributions(&self, round: u32) -> Result<Option<Vec<Contribution>>, RelayError> {
        self.get(&format!("/v1/rounds/{round}/contributions"))
    }

    pub fn contribute(&self, contribution: &Contribution) -> Result<(), RelayError> {
        self.send("/v1/contribute", &contribution.to_bytes())
            .map(drop)
    }

    pub fn reveal(&self, reveal: &ErrorReveal) -> Result<String, RelayError> {
        let body =
            postcard::to_stdvec(reveal).map_err(|err| RelayError::Decode(err.to_string()))?;
        self.send("/v1/reveal", &body)
    }
}
