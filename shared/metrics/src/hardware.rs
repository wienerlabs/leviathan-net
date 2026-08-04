//! What a node will say about its own hardware.
//!
//! Nothing in the protocol records hardware. The coordinator stores a node's
//! identity, its state and the height it left at, and that is all, so a fleet
//! view has to be built out of band from what nodes report about themselves.
//! That makes everything here self-reported and unverified, and it is labelled
//! that way everywhere it surfaces.
//!
//! What is deliberately *not* collected is as much the point as what is. No
//! serial number, no GPU UUID, no PCI bus id, no hostname, no address. Those
//! pin a report to a machine, and pinning a report to a machine is what turns
//! a fleet summary into a target list: `Round.random_seed` is public, so anyone
//! can already compute which identities drew the verifier seats for a round.
//! Publishing per-node hardware next to that tells an attacker not just who
//! must be silenced to break an audit, but what they are running and where to
//! find it. The aggregate answers "what is this network made of" without
//! answering "which box do I attack".

use nvml_wrapper::Nvml;
use serde::Deserialize;
use serde::Serialize;
use sysinfo::System;

/// One accelerator model and how many of them a node has.
///
/// Counted by model rather than listed per device, so two identical cards are
/// one entry with `count: 2` and no way to tell them apart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuModel {
    pub name: String,
    pub count: u32,
    /// Total VRAM per device, in bytes.
    pub memory_bytes: u64,
    /// CUDA compute capability as `major.minor`, when NVML reports one.
    pub compute_capability: Option<String>,
}

/// A single node's self-report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareReport {
    pub gpus: Vec<GpuModel>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub cpu_model: Option<String>,
    pub cpu_cores: usize,
    pub system_memory_bytes: u64,
    pub os: Option<String>,
}

impl HardwareReport {
    pub fn total_gpus(&self) -> u32 {
        self.gpus.iter().map(|gpu| gpu.count).sum()
    }

    pub fn total_vram_bytes(&self) -> u64 {
        self.gpus
            .iter()
            .map(|gpu| gpu.memory_bytes * gpu.count as u64)
            .sum()
    }
}

/// Reads the local machine. A host with no NVIDIA driver reports zero GPUs
/// rather than failing, because CPU-only nodes are a real configuration and a
/// missing NVML is not an error worth stopping a training run over.
pub fn local_report() -> HardwareReport {
    let mut system = System::new_all();
    system.refresh_all();

    let (gpus, driver_version, cuda_version) = match Nvml::init() {
        Ok(nvml) => read_nvml(&nvml),
        Err(_) => (Vec::new(), None, None),
    };

    HardwareReport {
        gpus,
        driver_version,
        cuda_version,
        cpu_model: system
            .cpus()
            .first()
            .map(|cpu| cpu.brand().trim().to_string())
            .filter(|brand| !brand.is_empty()),
        cpu_cores: system.cpus().len(),
        system_memory_bytes: system.total_memory(),
        os: System::long_os_version(),
    }
}

fn read_nvml(nvml: &Nvml) -> (Vec<GpuModel>, Option<String>, Option<String>) {
    let driver_version = nvml.sys_driver_version().ok();
    let cuda_version = nvml.sys_cuda_driver_version().ok().map(|version| {
        format!("{}.{}", version / 1000, (version % 1000) / 10)
    });

    let mut models: Vec<GpuModel> = Vec::new();
    let device_count = nvml.device_count().unwrap_or(0);

    for index in 0..device_count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        let Ok(name) = device.name() else {
            continue;
        };
        let memory_bytes = device.memory_info().map(|info| info.total).unwrap_or(0);
        let compute_capability = device
            .cuda_compute_capability()
            .ok()
            .map(|cc| format!("{}.{}", cc.major, cc.minor));

        match models.iter_mut().find(|model| {
            model.name == name
                && model.memory_bytes == memory_bytes
                && model.compute_capability == compute_capability
        }) {
            Some(existing) => existing.count += 1,
            None => models.push(GpuModel {
                name,
                count: 1,
                memory_bytes,
                compute_capability,
            }),
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    (models, driver_version, cuda_version)
}

/// The fleet view the dashboard renders.
///
/// Built by folding many [`HardwareReport`]s together. It carries counts and
/// nothing that maps a card back to the node that reported it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetSummary {
    pub nodes_reporting: u32,
    pub total_gpus: u32,
    pub total_vram_bytes: u64,
    pub total_cpu_cores: u64,
    pub gpus: Vec<GpuModel>,
    pub driver_versions: Vec<VersionCount>,
    pub cuda_versions: Vec<VersionCount>,
    pub operating_systems: Vec<VersionCount>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionCount {
    pub value: String,
    pub nodes: u32,
}

fn tally(counts: &mut Vec<VersionCount>, value: Option<String>) {
    let Some(value) = value else { return };
    match counts.iter_mut().find(|entry| entry.value == value) {
        Some(existing) => existing.nodes += 1,
        None => counts.push(VersionCount { value, nodes: 1 }),
    }
}

pub fn summarize(reports: &[HardwareReport]) -> FleetSummary {
    let mut summary = FleetSummary {
        nodes_reporting: reports.len() as u32,
        ..Default::default()
    };

    for report in reports {
        summary.total_cpu_cores += report.cpu_cores as u64;
        tally(&mut summary.driver_versions, report.driver_version.clone());
        tally(&mut summary.cuda_versions, report.cuda_version.clone());
        tally(&mut summary.operating_systems, report.os.clone());

        for gpu in &report.gpus {
            match summary.gpus.iter_mut().find(|model| {
                model.name == gpu.name
                    && model.memory_bytes == gpu.memory_bytes
                    && model.compute_capability == gpu.compute_capability
            }) {
                Some(existing) => existing.count += gpu.count,
                None => summary.gpus.push(gpu.clone()),
            }
        }
    }

    summary.total_gpus = summary.gpus.iter().map(|gpu| gpu.count).sum();
    summary.total_vram_bytes = summary
        .gpus
        .iter()
        .map(|gpu| gpu.memory_bytes * gpu.count as u64)
        .sum();
    summary.gpus.sort_by(|a, b| {
        b.count.cmp(&a.count).then_with(|| a.name.cmp(&b.name))
    });
    summary.driver_versions.sort_by(|a, b| b.nodes.cmp(&a.nodes));
    summary.cuda_versions.sort_by(|a, b| b.nodes.cmp(&a.nodes));
    summary.operating_systems.sort_by(|a, b| b.nodes.cmp(&a.nodes));

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(gpus: Vec<GpuModel>, driver: &str, cores: usize) -> HardwareReport {
        HardwareReport {
            gpus,
            driver_version: Some(driver.to_string()),
            cuda_version: Some("12.4".to_string()),
            cpu_model: Some("test cpu".to_string()),
            cpu_cores: cores,
            system_memory_bytes: 64 * 1024 * 1024 * 1024,
            os: Some("Linux".to_string()),
        }
    }

    fn gpu(name: &str, count: u32, memory_gb: u64) -> GpuModel {
        GpuModel {
            name: name.to_string(),
            count,
            memory_bytes: memory_gb * 1024 * 1024 * 1024,
            compute_capability: Some("8.9".to_string()),
        }
    }

    #[test]
    fn identical_cards_fold_into_one_entry() {
        let summary = summarize(&[
            report(vec![gpu("NVIDIA RTX 4090", 2, 24)], "550.54", 32),
            report(vec![gpu("NVIDIA RTX 4090", 4, 24)], "550.54", 64),
        ]);

        assert_eq!(summary.nodes_reporting, 2);
        assert_eq!(summary.gpus.len(), 1);
        assert_eq!(summary.gpus[0].count, 6);
        assert_eq!(summary.total_gpus, 6);
        assert_eq!(summary.total_cpu_cores, 96);
        assert_eq!(summary.driver_versions.len(), 1);
        assert_eq!(summary.driver_versions[0].nodes, 2);
    }

    #[test]
    fn same_model_at_different_vram_stays_separate() {
        let summary = summarize(&[
            report(vec![gpu("NVIDIA A100", 1, 40)], "550.54", 16),
            report(vec![gpu("NVIDIA A100", 1, 80)], "535.10", 16),
        ]);

        assert_eq!(summary.gpus.len(), 2);
        assert_eq!(summary.total_gpus, 2);
        assert_eq!(summary.driver_versions.len(), 2);
    }

    #[test]
    fn vram_totals_multiply_by_count() {
        let summary = summarize(&[report(vec![gpu("NVIDIA H100", 8, 80)], "550.54", 128)]);
        assert_eq!(summary.total_vram_bytes, 8 * 80 * 1024 * 1024 * 1024);
    }

    /// The summary is the only shape that is ever published, so nothing in it
    /// may carry a per-node handle. This asserts the shape rather than trusting
    /// that a later edit will remember why.
    #[test]
    fn summary_carries_no_node_attribution() {
        let summary = summarize(&[report(vec![gpu("NVIDIA H100", 8, 80)], "550.54", 128)]);
        let json = serde_json::to_string(&summary).unwrap();
        for forbidden in ["uuid", "serial", "hostname", "pci", "address", "node_id"] {
            assert!(
                !json.contains(forbidden),
                "fleet summary leaked a {forbidden}: {json}"
            );
        }
    }

    #[test]
    fn a_host_without_nvml_still_reports() {
        let local = local_report();
        assert_eq!(local.total_gpus() as usize, local.gpus.iter().map(|g| g.count as usize).sum::<usize>());
    }
}
