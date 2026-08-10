pub mod tck;
pub mod trk;
pub mod tt;
pub mod vtk;

use std::path::Path;

use crate::any_trx_file::AnyTrxFile;
use crate::dtype::DType;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::tractogram::Tractogram;
pub use vtk::{
    inspect_vtk_declared_space, vtk_import_warnings, VtkCoordinateMode, VtkCoordinateSpace,
};

/// Supported tractogram file formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// TRX format (directory or `.trx` zip archive).
    Trx,
    /// TrackVis `.trk` / `.trk.gz`.
    Trk,
    /// MRtrix `.tck` / `.tck.gz`.
    Tck,
    /// VTK legacy polydata `.vtk`.
    Vtk,
    /// DSI Studio Tiny Track `.tt.gz` (import only).
    TinyTrack,
}

/// Options that control tractogram conversion behaviour.
#[derive(Clone, Debug)]
pub struct ConversionOptions {
    /// Optional header override for formats that do not carry TRX-style metadata.
    pub header: Option<Header>,
    /// Positions dtype to use when writing TRX output.
    pub trx_positions_dtype: DType,
    /// How VTK coordinates should be interpreted when reading.
    pub vtk_coordinate_mode: VtkCoordinateMode,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            header: None,
            trx_positions_dtype: DType::Float32,
            vtk_coordinate_mode: VtkCoordinateMode::AssumeRas,
        }
    }
}

/// Detect the tractogram format from a file path or directory.
///
/// Returns `Err` if the path does not match a known format.
pub fn detect_format(path: &Path) -> Result<Format> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            TrxError::Argument(format!("cannot determine format for {}", path.display()))
        })?;

    if file_name.ends_with(".trx") || path.is_dir() {
        return Ok(Format::Trx);
    }
    if file_name.ends_with(".trk") || file_name.ends_with(".trk.gz") {
        return Ok(Format::Trk);
    }
    if file_name.ends_with(".tck") || file_name.ends_with(".tck.gz") {
        return Ok(Format::Tck);
    }
    if file_name.ends_with(".vtk") {
        return Ok(Format::Vtk);
    }
    if file_name.ends_with(".tt") || file_name.ends_with(".tt.gz") {
        return Ok(Format::TinyTrack);
    }

    Err(TrxError::Format(format!(
        "unsupported tractogram format for {}",
        path.display()
    )))
}

/// Read a tractogram from any supported format into the neutral in-memory representation.
///
/// Dispatches to the format-specific reader based on [`detect_format`].
pub fn read_tractogram(path: &Path, options: &ConversionOptions) -> Result<Tractogram> {
    match detect_format(path)? {
        Format::Trx => Ok(Tractogram::from(&AnyTrxFile::load(path)?)),
        Format::Trk => trk::read_trk(path, options.header.clone()),
        Format::Tck => tck::read_tck(path, options.header.clone()),
        Format::Vtk => vtk::read_vtk(path, options.header.clone(), options.vtk_coordinate_mode),
        Format::TinyTrack => tt::read_tt(path),
    }
}

/// Write a tractogram to any supported output format.
///
/// Dispatches to the format-specific writer based on [`detect_format`].
pub fn write_tractogram(
    path: &Path,
    tractogram: &Tractogram,
    options: &ConversionOptions,
) -> Result<()> {
    match detect_format(path)? {
        Format::Trx => {
            let mut tractogram = tractogram.clone();
            if let Some(header) = &options.header {
                tractogram.set_spatial_metadata(header.voxel_to_rasmm, header.dimensions);
            }
            match tractogram.to_trx(options.trx_positions_dtype)? {
                AnyTrxFile::F16(file) => file.save(path),
                AnyTrxFile::F32(file) => file.save(path),
                AnyTrxFile::F64(file) => file.save(path),
            }
        }
        Format::Trk => {
            let mut tractogram = tractogram.clone();
            if let Some(header) = &options.header {
                tractogram.set_spatial_metadata(header.voxel_to_rasmm, header.dimensions);
            }
            crate::legacy_io::write_trk(path, &tractogram, None).map_err(|err| {
                TrxError::Format(format!(
                    "failed to write TrackVis file {}: {err}",
                    path.display()
                ))
            })
        }
        Format::Tck => tck::write_tck(path, tractogram),
        Format::Vtk => vtk::write_vtk(path, tractogram),
        Format::TinyTrack => Err(TrxError::Format(
            "Tiny Track (.tt/.tt.gz) conversion is not implemented yet".into(),
        )),
    }
}

/// Convert between tractogram file formats in one step.
///
/// Reads from `input`, writes to `output`, applying the given [`ConversionOptions`].
pub fn convert(input: &Path, output: &Path, options: &ConversionOptions) -> Result<()> {
    if detect_format(input)? == Format::Trk && detect_format(output)? == Format::Trx {
        return trk::convert_trk_to_trx(input, output, options);
    }
    let tractogram = read_tractogram(input, options)?;
    write_tractogram(output, &tractogram, options)
}

#[cfg(test)]
mod tests {
    use super::{ConversionOptions, VtkCoordinateMode};

    #[test]
    fn conversion_options_default_to_ras_for_vtk_parity() {
        assert_eq!(
            ConversionOptions::default().vtk_coordinate_mode,
            VtkCoordinateMode::AssumeRas
        );
    }
}
