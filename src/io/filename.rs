use crate::dtype::DType;
use crate::error::{Result, TrxError};

/// Parsed components of a TRX data filename like `positions.3.float32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrxFilename {
    pub name: String,
    pub ncols: usize,
    pub dtype: DType,
}

impl TrxFilename {
    /// Parse a filename stem (no directory, no leading path).
    ///
    /// Accepts two formats:
    /// - `{name}.{ncols}.{dtype}` (e.g. `positions.3.float32`)
    /// - `{name}.{dtype}` (e.g. `offsets.uint32`) — ncols defaults to 1
    pub fn parse(stem: &str) -> Result<Self> {
        // Try 3-part format first: split from the right.
        let parts3: Vec<&str> = stem.rsplitn(3, '.').collect();

        if parts3.len() == 3 {
            // Could be {name}.{ncols}.{dtype} or {name}.{something}.{dtype}
            if let Ok(ncols) = parts3[1].parse::<usize>() {
                if let Ok(dtype) = DType::parse(parts3[0]) {
                    return Ok(TrxFilename {
                        name: parts3[2].to_string(),
                        ncols,
                        dtype,
                    });
                }
            }
        }

        // Try 2-part format: {name}.{dtype} (ncols = 1)
        let parts2: Vec<&str> = stem.rsplitn(2, '.').collect();
        if parts2.len() == 2 {
            if let Ok(dtype) = DType::parse(parts2[0]) {
                return Ok(TrxFilename {
                    name: parts2[1].to_string(),
                    ncols: 1,
                    dtype,
                });
            }
        }

        Err(TrxError::Format(format!(
            "cannot parse TRX filename '{stem}'"
        )))
    }

    /// Format back to `{name}.{ncols}.{dtype}`.
    pub fn to_filename(&self) -> String {
        format!("{}.{}.{}", self.name, self.ncols, self.dtype.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_positions() {
        let f = TrxFilename::parse("positions.3.float32").unwrap();
        assert_eq!(f.name, "positions");
        assert_eq!(f.ncols, 3);
        assert_eq!(f.dtype, DType::Float32);
    }

    #[test]
    fn parse_offsets() {
        let f = TrxFilename::parse("offsets.1.uint32").unwrap();
        assert_eq!(f.name, "offsets");
        assert_eq!(f.ncols, 1);
        assert_eq!(f.dtype, DType::UInt32);
    }

    #[test]
    fn round_trip() {
        let f = TrxFilename {
            name: "fa".into(),
            ncols: 1,
            dtype: DType::Float32,
        };
        assert_eq!(TrxFilename::parse(&f.to_filename()).unwrap(), f);
    }

    #[test]
    fn parse_two_part() {
        // 1D data elements don't include ncols — ncols defaults to 1
        let f = TrxFilename::parse("offsets.uint32").unwrap();
        assert_eq!(f.name, "offsets");
        assert_eq!(f.ncols, 1);
        assert_eq!(f.dtype, DType::UInt32);
    }

    #[test]
    fn parse_invalid() {
        assert!(TrxFilename::parse("noext").is_err());
        assert!(TrxFilename::parse("foo.bar").is_err());
    }

    #[test]
    fn name_with_dots() {
        let f = TrxFilename::parse("my.metric.1.float64").unwrap();
        assert_eq!(f.name, "my.metric");
        assert_eq!(f.ncols, 1);
        assert_eq!(f.dtype, DType::Float64);
    }
}
