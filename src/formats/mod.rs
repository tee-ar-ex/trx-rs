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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Trx,
    Trk,
    Tck,
    Vtk,
    TinyTrack,
}

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
            vtk_coordinate_mode: VtkCoordinateMode::HeaderOrWarn,
        }
    }
}

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

pub fn read_tractogram(path: &Path, options: &ConversionOptions) -> Result<Tractogram> {
    match detect_format(path)? {
        Format::Trx => Ok(Tractogram::from(&AnyTrxFile::load(path)?)),
        Format::Trk => trk::read_trk(path, options.header.clone()),
        Format::Tck => tck::read_tck(path, options.header.clone()),
        Format::Vtk => vtk::read_vtk(path, options.header.clone(), options.vtk_coordinate_mode),
        Format::TinyTrack => tt::read_tt(path),
    }
}

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
        Format::Trk => Err(TrxError::Format(
            "TrackVis (.trk/.trk.gz) export is intentionally unsupported; convert to TRX instead"
                .into(),
        )),
        Format::Tck => tck::write_tck(path, tractogram),
        Format::Vtk => vtk::write_vtk(path, tractogram),
        Format::TinyTrack => Err(TrxError::Format(
            "Tiny Track (.tt/.tt.gz) conversion is not implemented yet".into(),
        )),
    }
}

pub fn convert(input: &Path, output: &Path, options: &ConversionOptions) -> Result<()> {
    if detect_format(input)? == Format::Trk && detect_format(output)? == Format::Trx {
        return trk::convert_trk_to_trx(input, output, options);
    }
    let tractogram = read_tractogram(input, options)?;
    write_tractogram(output, &tractogram, options)
}
