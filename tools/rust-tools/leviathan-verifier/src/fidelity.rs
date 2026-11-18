//! Checks [`simulate`] against real clients (wienerlabs/leviathan-net#48).
//!
//! A client started with `--write-gradients-dir` dumps its own payload when it
//! trains it, and each peer's once it matches the peer's commitment. So one
//! client's directory holds every node's payload of every round it trained
//! in. From those dumps the run's rounds are rebuilt as the coordinator ran
//! them:
//! - each epoch's training rounds, from height 0;
//! - the two rounds that close an epoch before the next one starts;
//! - which node trained which share of each batch.
//!
//! The simulation then trains the same rounds from the same checkpoint and
//! data, and has to send, byte for byte, what each node sent.

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;
use psyche_core::BatchId;
use psyche_data_provider::TokenizedDataProvider;
use psyche_modeling::DistroResult;
use psyche_network::distro_results_to_bytes;
use psyche_network::SerializedDistroResult;
use psyche_network::PAYLOAD_DUMP_EXTENSION;

use crate::contribution_key;
use crate::error_feedback::read_batch;
use crate::error_feedback::simulate;
use crate::error_feedback::Round;
use crate::error_feedback::Share;
use crate::error_feedback::Simulation;
use crate::replay::parse_assignment_key;
use crate::replay::DumpKey;

/// A payload a client dumped: who sent it, and for which epoch, step and
/// batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientDump {
    pub signer: String,
    pub key: DumpKey,
    pub path: PathBuf,
}

/// The payload dump at `path`, if its name is one
/// (`psyche_network::payload_dump_name`).
pub fn client_dump(path: &Path) -> Option<ClientDump> {
    let key = contribution_key(path)?;
    let signer = path
        .file_name()?
        .to_str()?
        .strip_prefix("result-")?
        .strip_suffix(&format!("-{key}.{PAYLOAD_DUMP_EXTENSION}"))?;
    Some(ClientDump {
        signer: signer.to_string(),
        key: parse_assignment_key(&key)?,
        path: path.to_path_buf(),
    })
}

/// Every payload dump in `dir`.
pub fn read_client_dumps(dir: &Path) -> Result<Vec<ClientDump>> {
    let mut dumps = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        if let Some(dump) = client_dump(&entry?.path()) {
            dumps.push(dump);
        }
    }
    if dumps.is_empty() {
        return Err(anyhow!("{} holds no payload dump", dir.display()));
    }
    Ok(dumps)
}

/// A round rebuilt from dumps, before its data is read: its step, its height,
/// and, if it trained, which node trained which share, in the batch's order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRound {
    pub step: u32,
    pub height: u32,
    pub shares: Option<Vec<(usize, BatchId)>>,
}

/// The rounds the run that `dumps` came from went through, and the signer of
/// each node, by node index. Every node has to have a dump of every training
/// round, the training rounds of an epoch have to follow each other, and the
/// next epoch has to start after the two rounds that close the last one.
pub fn plan_rounds(dumps: &[ClientDump]) -> Result<(Vec<PlannedRound>, Vec<String>)> {
    let mut signers: Vec<String> = dumps.iter().map(|dump| dump.signer.clone()).collect();
    signers.sort();
    signers.dedup();
    let node_of = |signer: &str| signers.iter().position(|known| known == signer);

    let mut epochs: BTreeMap<u64, BTreeMap<u32, Vec<(usize, BatchId)>>> = BTreeMap::new();
    for dump in dumps {
        let node = node_of(&dump.signer).ok_or_else(|| anyhow!("unknown {}", dump.signer))?;
        epochs
            .entry(dump.key.epoch_start)
            .or_default()
            .entry(dump.key.step)
            .or_default()
            .push((node, dump.key.batch_id));
    }

    let mut rounds: Vec<PlannedRound> = Vec::new();
    for (epoch_start, steps) in epochs {
        let first = *steps
            .keys()
            .next()
            .ok_or_else(|| anyhow!("epoch {epoch_start} has no step"))?;
        if let Some(last) = rounds.last() {
            let closing = [last.step + 1, last.step + 2];
            if first != last.step + 3 {
                return Err(anyhow!(
                    "epoch {epoch_start} starts at step {first}, not after the two rounds that close the epoch before it at step {}",
                    last.step
                ));
            }
            let height = last.height;
            rounds.extend(closing.iter().zip(1..).map(|(step, up)| PlannedRound {
                step: *step,
                height: height + up,
                shares: None,
            }));
        }
        for (height, (step, mut shares)) in (0u32..).zip(steps) {
            if step != first + height {
                return Err(anyhow!(
                    "epoch {epoch_start} trains step {step} after step {}",
                    first + height - 1
                ));
            }
            shares.sort_by_key(|(_, batch)| batch.0.start);
            let mut nodes: Vec<usize> = shares.iter().map(|(node, _)| *node).collect();
            nodes.sort_unstable();
            if nodes != (0..signers.len()).collect::<Vec<_>>() {
                return Err(anyhow!(
                    "step {step} has dumps from nodes {nodes:?}, not one from each of {}",
                    signers.len()
                ));
            }
            rounds.push(PlannedRound {
                step,
                height,
                shares: Some(shares),
            });
        }
    }
    Ok((rounds, signers))
}

/// The rounds of `planned`, each share with its batch read from `provider`.
pub async fn read_planned<P: TokenizedDataProvider>(
    provider: &mut P,
    planned: &[PlannedRound],
) -> Result<Vec<Round>> {
    let mut rounds = Vec::with_capacity(planned.len());
    for round in planned {
        let shares = match &round.shares {
            Some(shares) => {
                let mut read = Vec::with_capacity(shares.len());
                for (node, batch) in shares {
                    read.push(Share {
                        node: *node,
                        batch: *batch,
                        data: read_batch(provider, *batch).await?,
                    });
                }
                Some(read)
            }
            None => None,
        };
        rounds.push(Round {
            step: round.step,
            height: round.height,
            shares,
        });
    }
    Ok(rounds)
}

/// The bytes a node broadcasts, and a client dumps, for `results`.
pub fn wire_bytes(results: &[DistroResult]) -> Result<Vec<u8>> {
    let serialized: Vec<SerializedDistroResult> = results
        .iter()
        .map(SerializedDistroResult::try_from)
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| anyhow!("serializing results: {error:?}"))?;
    distro_results_to_bytes(&serialized).map_err(|error| anyhow!("encoding results: {error}"))
}

/// What the simulation sent against what the clients dumped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FidelityReport {
    /// Payloads the simulation sent byte for byte.
    pub matched: usize,
    /// The step and signer of every payload it sent otherwise.
    pub mismatched: Vec<(u32, String)>,
}

/// Trains `rounds` under `simulation`, which has to train as many nodes as
/// `signers` names and no shadow, and compares every payload each node sent
/// with the one in `dumps`.
pub fn check_fidelity(
    simulation: &Simulation,
    rounds: Vec<Round>,
    dumps: &[ClientDump],
    signers: &[String],
) -> Result<FidelityReport> {
    if simulation.node_count != signers.len() || simulation.shadow.is_some() {
        return Err(anyhow!(
            "the check trains the {} nodes that sent, and no shadow",
            signers.len()
        ));
    }
    let dump_of: BTreeMap<(u32, &str), &Path> = dumps
        .iter()
        .map(|dump| ((dump.key.step, dump.signer.as_str()), dump.path.as_path()))
        .collect();
    let mut compared: Vec<Result<(u32, String, bool)>> = Vec::new();
    simulate(simulation, rounds, &mut |observed| {
        for (share, sent) in observed.shares.iter().zip(observed.sent) {
            let step = observed.record.step;
            let signer = &signers[share.node];
            compared.push((|| {
                let path = dump_of
                    .get(&(step, signer.as_str()))
                    .ok_or_else(|| anyhow!("no dump of {signer} at step {step}"))?;
                let dumped =
                    std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
                Ok((step, signer.clone(), wire_bytes(sent)? == dumped))
            })());
        }
    })?;
    let mut report = FidelityReport::default();
    for comparison in compared {
        match comparison? {
            (_, _, true) => report.matched += 1,
            (step, signer, false) => report.mismatched.push((step, signer)),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use psyche_core::ClosedInterval;

    fn batch(start: u64, end: u64) -> BatchId {
        BatchId(ClosedInterval::new(start, end))
    }

    fn dump(signer: &str, epoch_start: u64, step: u32, start: u64) -> ClientDump {
        ClientDump {
            signer: signer.to_string(),
            key: DumpKey {
                epoch_start,
                step,
                batch_id: batch(start, start + 3),
            },
            path: PathBuf::new(),
        }
    }

    #[test]
    fn a_dump_names_its_signer_epoch_step_and_batch() {
        let name = psyche_network::payload_dump_name("8GMYMJPd", 1794935060, 3, batch(20, 23));
        let parsed = client_dump(Path::new(&name)).unwrap();
        assert_eq!(parsed.signer, "8GMYMJPd");
        assert_eq!(
            parsed.key,
            DumpKey {
                epoch_start: 1794935060,
                step: 3,
                batch_id: batch(20, 23)
            }
        );
        assert!(client_dump(Path::new(&format!("{name}.commitment"))).is_none());
        assert!(client_dump(Path::new("result-8GMYMJPd.vec-postcard")).is_none());
    }

    #[test]
    fn rounds_are_rebuilt_epoch_by_epoch_with_the_rounds_that_close_them() {
        // Two epochs: steps 1-3 train, 4-5 close the first, 6-7 train.
        let dumps = vec![
            dump("b", 10, 1, 4),
            dump("a", 10, 1, 0),
            dump("a", 10, 2, 12),
            dump("b", 10, 2, 8),
            dump("a", 10, 3, 16),
            dump("b", 10, 3, 20),
            dump("b", 20, 6, 40),
            dump("a", 20, 6, 44),
            dump("a", 20, 7, 48),
            dump("b", 20, 7, 52),
        ];
        let (rounds, signers) = plan_rounds(&dumps).unwrap();
        assert_eq!(signers, vec!["a", "b"]);
        let shape: Vec<(u32, u32, Option<Vec<usize>>)> = rounds
            .iter()
            .map(|round| {
                (
                    round.step,
                    round.height,
                    round
                        .shares
                        .as_ref()
                        .map(|shares| shares.iter().map(|(node, _)| *node).collect()),
                )
            })
            .collect();
        assert_eq!(
            shape,
            vec![
                (1, 0, Some(vec![0, 1])),
                (2, 1, Some(vec![1, 0])),
                (3, 2, Some(vec![0, 1])),
                (4, 3, None),
                (5, 4, None),
                (6, 0, Some(vec![1, 0])),
                (7, 1, Some(vec![0, 1])),
            ]
        );
        assert_eq!(
            rounds[1].shares.as_ref().unwrap()[0],
            (1, batch(8, 11)),
            "shares go in the batch's order"
        );
    }

    #[test]
    fn a_run_the_dumps_cannot_account_for_is_refused() {
        let missing_node = vec![
            dump("a", 10, 1, 0),
            dump("b", 10, 1, 4),
            dump("a", 10, 2, 8),
        ];
        assert!(plan_rounds(&missing_node).is_err());
        let gap = vec![
            dump("a", 10, 1, 0),
            dump("b", 10, 1, 4),
            dump("a", 10, 3, 16),
            dump("b", 10, 3, 20),
        ];
        assert!(plan_rounds(&gap).is_err());
        let early_epoch = vec![
            dump("a", 10, 1, 0),
            dump("b", 10, 1, 4),
            dump("a", 20, 3, 16),
            dump("b", 20, 3, 20),
        ];
        assert!(plan_rounds(&early_epoch).is_err());
    }
}
