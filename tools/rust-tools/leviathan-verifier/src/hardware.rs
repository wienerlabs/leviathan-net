//! The device and precision a measurement trains on, as the offline tools
//! name them on the command line.

use anyhow::anyhow;
use anyhow::Result;

/// A device named `cpu`, `mps`, `cuda`, `cuda:N`, or `auto` (CUDA when there
/// is one, the CPU otherwise).
pub fn parse_device(s: &str) -> Result<tch::Device> {
    match s.trim().to_lowercase().as_str() {
        "cpu" => Ok(tch::Device::Cpu),
        "mps" => Ok(tch::Device::Mps),
        "cuda" => Ok(tch::Device::Cuda(0)),
        "auto" => Ok(tch::Device::cuda_if_available()),
        other => other
            .strip_prefix("cuda:")
            .and_then(|n| n.parse::<usize>().ok())
            .map(tch::Device::Cuda)
            .ok_or_else(|| {
                anyhow!("unknown device '{other}': expected cpu, mps, cuda, cuda:N, or auto")
            }),
    }
}

/// A dtype named `fp32`, `bf16` or `fp16`.
pub fn parse_dtype(s: &str) -> Result<tch::Kind> {
    match s.trim().to_lowercase().as_str() {
        "fp32" | "f32" | "float" => Ok(tch::Kind::Float),
        "bf16" | "bfloat16" => Ok(tch::Kind::BFloat16),
        "fp16" | "f16" | "half" => Ok(tch::Kind::Half),
        other => Err(anyhow!(
            "unknown dtype '{other}': expected fp32, bf16, or fp16"
        )),
    }
}

/// How a report names `device`.
pub fn device_label(device: tch::Device) -> String {
    match device {
        tch::Device::Cpu => "cpu".to_string(),
        tch::Device::Cuda(i) => format!("cuda:{i}"),
        other => format!("{other:?}").to_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devices_parse() {
        assert!(matches!(parse_device("cpu").unwrap(), tch::Device::Cpu));
        assert!(matches!(parse_device("CPU").unwrap(), tch::Device::Cpu));
        assert!(matches!(
            parse_device("cuda").unwrap(),
            tch::Device::Cuda(0)
        ));
        assert!(matches!(
            parse_device("cuda:2").unwrap(),
            tch::Device::Cuda(2)
        ));
        assert!(matches!(parse_device("mps").unwrap(), tch::Device::Mps));
        assert!(parse_device("auto").is_ok());
        assert!(parse_device("gpu").is_err());
        assert!(parse_device("cuda:x").is_err());
    }

    #[test]
    fn dtypes_parse() {
        assert!(matches!(parse_dtype("fp32").unwrap(), tch::Kind::Float));
        assert!(matches!(parse_dtype("float").unwrap(), tch::Kind::Float));
        assert!(matches!(parse_dtype("bf16").unwrap(), tch::Kind::BFloat16));
        assert!(matches!(
            parse_dtype("BFloat16").unwrap(),
            tch::Kind::BFloat16
        ));
        assert!(matches!(parse_dtype("fp16").unwrap(), tch::Kind::Half));
        assert!(parse_dtype("int8").is_err());
    }

    #[test]
    fn device_labels_are_stable() {
        assert_eq!(device_label(tch::Device::Cpu), "cpu");
        assert_eq!(device_label(tch::Device::Cuda(1)), "cuda:1");
        assert_eq!(device_label(tch::Device::Mps), "mps");
    }
}
