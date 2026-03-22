use bytemuck::{Pod, Zeroable};
use half::f16;
use std::fmt;

use crate::error::{Result, TrxError};

/// Supported element data types in TRX files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_of(self) -> usize {
        match self {
            DType::Int8 | DType::UInt8 => 1,
            DType::Float16 | DType::Int16 | DType::UInt16 => 2,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 | DType::UInt64 => 8,
        }
    }

    /// Canonical string name as used in TRX filenames.
    pub fn name(self) -> &'static str {
        match self {
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
        }
    }

    /// Parse a dtype string (e.g. `"float32"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "float16" => Ok(DType::Float16),
            "float32" => Ok(DType::Float32),
            "float64" => Ok(DType::Float64),
            "int8" => Ok(DType::Int8),
            "int16" => Ok(DType::Int16),
            "int32" => Ok(DType::Int32),
            "int64" => Ok(DType::Int64),
            "uint8" => Ok(DType::UInt8),
            "uint16" => Ok(DType::UInt16),
            "uint32" => Ok(DType::UInt32),
            "uint64" => Ok(DType::UInt64),
            _ => Err(TrxError::DType(s.to_string())),
        }
    }

    /// Whether this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, DType::Float16 | DType::Float32 | DType::Float64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Trait for scalar types that can be stored as TRX position coordinates.
///
/// Implementors must be [`Pod`] + [`Zeroable`] (for zero-copy casts) and
/// carry their [`DType`] as an associated constant.
pub trait TrxScalar: Pod + Zeroable + Copy + 'static + fmt::Debug {
    const DTYPE: DType;

    /// Convert to f32 for operations that need floating-point comparisons.
    fn to_f32(self) -> f32;

    /// Convert to f64 for operations that need higher precision.
    fn to_f64(self) -> f64;
}

impl TrxScalar for f32 {
    const DTYPE: DType = DType::Float32;
    fn to_f32(self) -> f32 { self }
    fn to_f64(self) -> f64 { self as f64 }
}

impl TrxScalar for f64 {
    const DTYPE: DType = DType::Float64;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self }
}

impl TrxScalar for f16 {
    const DTYPE: DType = DType::Float16;
    fn to_f32(self) -> f32 { f16::to_f32(self) }
    fn to_f64(self) -> f64 { f16::to_f64(self) }
}

impl TrxScalar for i8 {
    const DTYPE: DType = DType::Int8;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}
impl TrxScalar for i16 {
    const DTYPE: DType = DType::Int16;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}
impl TrxScalar for i32 {
    const DTYPE: DType = DType::Int32;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}
impl TrxScalar for i64 {
    const DTYPE: DType = DType::Int64;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}
impl TrxScalar for u8 {
    const DTYPE: DType = DType::UInt8;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}
impl TrxScalar for u16 {
    const DTYPE: DType = DType::UInt16;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}
impl TrxScalar for u32 {
    const DTYPE: DType = DType::UInt32;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}
impl TrxScalar for u64 {
    const DTYPE: DType = DType::UInt64;
    fn to_f32(self) -> f32 { self as f32 }
    fn to_f64(self) -> f64 { self as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_round_trip() {
        for dt in [
            DType::Float16,
            DType::Float32,
            DType::Float64,
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
        ] {
            assert_eq!(DType::parse(dt.name()).unwrap(), dt);
        }
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::Float32.size_of(), 4);
        assert_eq!(DType::Float64.size_of(), 8);
        assert_eq!(DType::Float16.size_of(), 2);
        assert_eq!(DType::UInt8.size_of(), 1);
    }

    #[test]
    fn dtype_parse_invalid() {
        assert!(DType::parse("complex128").is_err());
    }
}
