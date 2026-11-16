//! The Prometheus side of Leviathan's monitoring: the series the verifier
//! daemon exports, how a registry is rendered for a scrape, and the endpoint
//! that serves it.
//!
//! The kill-switch rules in `telemetry/leviathan-killswitch.rules.yml` watch
//! these series together with the ones `leviathan-indexer` reads off chain
//! (wienerlabs/leviathan-net#57).

pub mod exposition;
pub mod labels;
#[cfg(feature = "server")]
pub mod server;
pub mod verifier;
