use half::f16;
use std::path::Path;

use crate::dtype::DType;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::io::filename::TrxFilename;
use crate::trx_file::TrxFile;

/// References to positions data, dispatched by runtime dtype.
pub enum PositionsRef<'a> {
    F16(&'a [[f16; 3]]),
    F32(&'a [[f32; 3]]),
    F64(&'a [[f64; 3]]),
}

/// A type-erased TRX container that can hold any position dtype.
///
/// Use this when the position dtype is not known at compile time (e.g. CLI tools
/// that accept arbitrary `.trx` files).
pub enum AnyTrxFile {
    F16(TrxFile<f16>),
    F32(TrxFile<f32>),
    F64(TrxFile<f64>),
}

impl AnyTrxFile {
    /// Load a TRX file, detecting the positions dtype at runtime.
    pub fn load(path: &Path) -> Result<Self> {
        let dtype = detect_positions_dtype(path)?;
        match dtype {
            DType::Float16 => Ok(AnyTrxFile::F16(TrxFile::<f16>::load(path)?)),
            DType::Float32 => Ok(AnyTrxFile::F32(TrxFile::<f32>::load(path)?)),
            DType::Float64 => Ok(AnyTrxFile::F64(TrxFile::<f64>::load(path)?)),
            other => Err(TrxError::DType(format!(
                "positions dtype {other} is not a float type"
            ))),
        }
    }

    /// Get a reference to the positions, dispatched by dtype.
    pub fn positions_ref(&self) -> PositionsRef<'_> {
        match self {
            AnyTrxFile::F16(f) => PositionsRef::F16(f.positions()),
            AnyTrxFile::F32(f) => PositionsRef::F32(f.positions()),
            AnyTrxFile::F64(f) => PositionsRef::F64(f.positions()),
        }
    }

    /// The positions dtype.
    pub fn dtype(&self) -> DType {
        match self {
            AnyTrxFile::F16(_) => DType::Float16,
            AnyTrxFile::F32(_) => DType::Float32,
            AnyTrxFile::F64(_) => DType::Float64,
        }
    }

    /// The header.
    pub fn header(&self) -> &Header {
        match self {
            AnyTrxFile::F16(f) => &f.header,
            AnyTrxFile::F32(f) => &f.header,
            AnyTrxFile::F64(f) => &f.header,
        }
    }

    /// Number of streamlines.
    pub fn nb_streamlines(&self) -> usize {
        match self {
            AnyTrxFile::F16(f) => f.nb_streamlines(),
            AnyTrxFile::F32(f) => f.nb_streamlines(),
            AnyTrxFile::F64(f) => f.nb_streamlines(),
        }
    }

    /// Number of vertices.
    pub fn nb_vertices(&self) -> usize {
        match self {
            AnyTrxFile::F16(f) => f.nb_vertices(),
            AnyTrxFile::F32(f) => f.nb_vertices(),
            AnyTrxFile::F64(f) => f.nb_vertices(),
        }
    }

    /// Dispatch to a closure with a concrete `&TrxFile<P>`.
    pub fn with_typed<R>(
        &self,
        on_f16: impl FnOnce(&TrxFile<f16>) -> R,
        on_f32: impl FnOnce(&TrxFile<f32>) -> R,
        on_f64: impl FnOnce(&TrxFile<f64>) -> R,
    ) -> R {
        match self {
            AnyTrxFile::F16(f) => on_f16(f),
            AnyTrxFile::F32(f) => on_f32(f),
            AnyTrxFile::F64(f) => on_f64(f),
        }
    }
}

impl std::fmt::Debug for AnyTrxFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyTrxFile::F16(t) => t.fmt(f),
            AnyTrxFile::F32(t) => t.fmt(f),
            AnyTrxFile::F64(t) => t.fmt(f),
        }
    }
}

/// Detect the positions dtype from a TRX path (directory or zip).
pub fn detect_positions_dtype(path: &Path) -> Result<DType> {
    if path.is_dir() {
        detect_positions_dtype_dir(path)
    } else if path.is_file() {
        detect_positions_dtype_zip(path)
    } else {
        Err(TrxError::FileNotFound(path.to_path_buf()))
    }
}

fn detect_positions_dtype_dir(dir: &Path) -> Result<DType> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("positions.") {
            let parsed = TrxFilename::parse(&name_str)?;
            return Ok(parsed.dtype);
        }
    }
    Err(TrxError::Format(
        "no positions file found in directory".into(),
    ))
}

fn detect_positions_dtype_zip(path: &Path) -> Result<DType> {
    let file = std::fs::File::open(path)?;
    let archive = zip::ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let name = archive.name_for_index(i).unwrap_or("");
        let basename = name.rsplit('/').next().unwrap_or(name);
        if basename.starts_with("positions.") {
            let parsed = TrxFilename::parse(basename)?;
            return Ok(parsed.dtype);
        }
    }
    Err(TrxError::Format(
        "no positions file found in zip archive".into(),
    ))
}
