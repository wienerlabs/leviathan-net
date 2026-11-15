use crate::ClosedInterval;

use serde::{Deserialize, Serialize};
use std::{fmt, ops::RangeInclusive, str::FromStr};

#[derive(PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BatchId(pub ClosedInterval<u64>);

impl fmt::Display for BatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

impl fmt::Debug for BatchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

/// Reads a batch back from how it is displayed: `B[12, 19]` for a range, or
/// `B15` for a batch of one sample, which is how `ClosedInterval` prints one.
///
/// Dump names and logs carry batches in exactly that form. The parser used to
/// accept only the bracketed form, and choked on the space `Display` puts after
/// the comma. A batch of one sample, or any range, could therefore never be read
/// back from a dump name, and the verifier skipped those payloads
/// (wienerlabs/leviathan-net#43).
impl FromStr for BatchId {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let s = s.strip_prefix('B').unwrap_or(s);
        let (start, end) = match s.strip_prefix('[') {
            Some(range) => {
                let range = range.strip_suffix(']').ok_or("Missing ']' in input")?;
                let (start, end) = range.split_once(',').ok_or("Missing ',' in input")?;
                let start =
                    u64::from_str(start.trim()).map_err(|_| "Failed to parse start value")?;
                let end = u64::from_str(end.trim()).map_err(|_| "Failed to parse end value")?;
                (start, end)
            }
            None => {
                let single = u64::from_str(s).map_err(|_| "Failed to parse batch value")?;
                (single, single)
            }
        };
        if end < start {
            return Err("The batch ends before it starts");
        }
        Ok(BatchId(ClosedInterval { start, end }))
    }
}

impl BatchId {
    pub fn iter(&self) -> RangeInclusive<u64> {
        self.0.start..=self.0.end
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        (self.0.end - self.0.start + 1) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_reads_back_from_how_it_is_displayed() {
        for batch in [
            BatchId(ClosedInterval::new(12, 19)),
            BatchId(ClosedInterval::new(15, 15)),
            BatchId(ClosedInterval::new(0, 0)),
            BatchId(ClosedInterval::new(0, u64::MAX)),
        ] {
            assert_eq!(BatchId::from_str(&batch.to_string()), Ok(batch), "{batch}");
        }
    }

    #[test]
    fn the_prefix_and_spacing_are_optional() {
        let range = BatchId(ClosedInterval::new(3, 4));
        assert_eq!(BatchId::from_str("[3,4]"), Ok(range));
        assert_eq!(BatchId::from_str(" B[ 3 , 4 ] "), Ok(range));
        assert_eq!(
            BatchId::from_str("7"),
            Ok(BatchId(ClosedInterval::new(7, 7)))
        );
    }

    #[test]
    fn what_is_not_a_batch_is_refused() {
        for text in [
            "", "B", "B[]", "B[1]", "B[1, 2", "B[1, 2]x", "B[x, 2]", "B[1, y]", "Bz", "B[5, 4]",
        ] {
            assert!(BatchId::from_str(text).is_err(), "{text:?} parsed");
        }
    }
}
