use serde::{Deserialize, Serialize};
use tch::{Kind, TchError};

/// This wrapper type only exists because tch doesn't expose the enum values directly.
/// It simply provides a serde ser/de impl for Kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerializableKind(Kind);

impl Serialize for SerializableKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u8(kind_to_u8(&self.0))
    }
}

impl<'de> Deserialize<'de> for SerializableKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = u8::deserialize(deserializer)?;
        u8_to_kind(v)
            .map(SerializableKind)
            .map_err(serde::de::Error::custom)
    }
}

impl SerializableKind {
    pub fn into_inner(self) -> Kind {
        self.0
    }

    pub fn new(kind: Kind) -> Self {
        SerializableKind(kind)
    }
}

impl From<SerializableKind> for Kind {
    fn from(value: SerializableKind) -> Self {
        value.0
    }
}

impl From<Kind> for SerializableKind {
    fn from(value: Kind) -> Self {
        Self(value)
    }
}

impl From<&SerializableKind> for Kind {
    fn from(value: &SerializableKind) -> Self {
        value.0
    }
}

impl From<&Kind> for SerializableKind {
    fn from(value: &Kind) -> Self {
        Self(*value)
    }
}

pub(crate) fn kind_to_u8(kind: &Kind) -> u8 {
    match kind {
        Kind::Uint8 => 0,
        Kind::Int8 => 1,
        Kind::Int16 => 2,
        Kind::Int => 3,
        Kind::Int64 => 4,
        Kind::Half => 5,
        Kind::Float => 6,
        Kind::Double => 7,
        Kind::ComplexHalf => 8,
        Kind::ComplexFloat => 9,
        Kind::ComplexDouble => 10,
        Kind::Bool => 11,
        Kind::QInt8 => 12,
        Kind::QUInt8 => 13,
        Kind::QInt32 => 14,
        Kind::BFloat16 => 15,
        Kind::QUInt4x2 => 16,
        Kind::QUInt2x4 => 17,
        Kind::Bits1x8 => 18,
        Kind::Bits2x4 => 19,
        Kind::Bits4x2 => 20,
        Kind::Bits8 => 21,
        Kind::Bits16 => 22,
        Kind::Float8e5m2 => 23,
        Kind::Float8e4m3fn => 24,
        Kind::Float8e5m2fnuz => 25,
        Kind::Float8e4m3fnuz => 26,
        Kind::UInt16 => 27,
        Kind::UInt32 => 28,
        Kind::UInt64 => 29,
        Kind::UInt1 => 30,
        Kind::UInt2 => 31,
        Kind::UInt3 => 32,
        Kind::UInt4 => 33,
        Kind::UInt5 => 34,
        Kind::UInt6 => 35,
        Kind::UInt7 => 36,
    }
}

fn u8_to_kind(v: u8) -> Result<Kind, TchError> {
    match v {
        0 => Ok(Kind::Uint8),
        1 => Ok(Kind::Int8),
        2 => Ok(Kind::Int16),
        3 => Ok(Kind::Int),
        4 => Ok(Kind::Int64),
        5 => Ok(Kind::Half),
        6 => Ok(Kind::Float),
        7 => Ok(Kind::Double),
        8 => Ok(Kind::ComplexHalf),
        9 => Ok(Kind::ComplexFloat),
        10 => Ok(Kind::ComplexDouble),
        11 => Ok(Kind::Bool),
        12 => Ok(Kind::QInt8),
        13 => Ok(Kind::QUInt8),
        14 => Ok(Kind::QInt32),
        15 => Ok(Kind::BFloat16),
        16 => Ok(Kind::QUInt4x2),
        17 => Ok(Kind::QUInt2x4),
        18 => Ok(Kind::Bits1x8),
        19 => Ok(Kind::Bits2x4),
        20 => Ok(Kind::Bits4x2),
        21 => Ok(Kind::Bits8),
        22 => Ok(Kind::Bits16),
        23 => Ok(Kind::Float8e5m2),
        24 => Ok(Kind::Float8e4m3fn),
        25 => Ok(Kind::Float8e5m2fnuz),
        26 => Ok(Kind::Float8e4m3fnuz),
        27 => Ok(Kind::UInt16),
        28 => Ok(Kind::UInt32),
        29 => Ok(Kind::UInt64),
        30 => Ok(Kind::UInt1),
        31 => Ok(Kind::UInt2),
        32 => Ok(Kind::UInt3),
        33 => Ok(Kind::UInt4),
        34 => Ok(Kind::UInt5),
        35 => Ok(Kind::UInt6),
        36 => Ok(Kind::UInt7),
        _ => Err(TchError::UnknownKind(v as i32)),
    }
}
