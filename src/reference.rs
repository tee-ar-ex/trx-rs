use std::path::Path;

use nifti::{NiftiObject, ReaderOptions};

use crate::any_trx_file::AnyTrxFile;
use crate::error::{Result, TrxError};
use crate::header::Header;

/// Load TRX-compatible spatial metadata from a reference file.
///
/// Supported reference types:
/// - `.trx` directories or archives
/// - `.nii` / `.nii.gz` images
pub fn header_from_reference(path: &Path) -> Result<Header> {
    if path.is_dir() {
        return Ok(AnyTrxFile::load(path)?.header().clone());
    }

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            TrxError::Argument(format!(
                "cannot determine reference format for {}",
                path.display()
            ))
        })?;

    if file_name.ends_with(".trx") {
        return Ok(AnyTrxFile::load(path)?.header().clone());
    }

    if file_name.ends_with(".nii") || file_name.ends_with(".nii.gz") {
        return header_from_nifti(path);
    }

    Err(TrxError::Format(format!(
        "unsupported reference format for {}",
        path.display()
    )))
}

fn header_from_nifti(path: &Path) -> Result<Header> {
    let obj = ReaderOptions::new().read_file(path).map_err(|err| {
        TrxError::Format(format!(
            "failed to read NIfTI reference {}: {err}",
            path.display()
        ))
    })?;
    let header = obj.header();
    let affine = header.affine::<f64>();

    Ok(Header {
        voxel_to_rasmm: [
            [
                affine[(0, 0)],
                affine[(0, 1)],
                affine[(0, 2)],
                affine[(0, 3)],
            ],
            [
                affine[(1, 0)],
                affine[(1, 1)],
                affine[(1, 2)],
                affine[(1, 3)],
            ],
            [
                affine[(2, 0)],
                affine[(2, 1)],
                affine[(2, 2)],
                affine[(2, 3)],
            ],
            [
                affine[(3, 0)],
                affine[(3, 1)],
                affine[(3, 2)],
                affine[(3, 3)],
            ],
        ],
        dimensions: [
            u64::from(header.dim[1]),
            u64::from(header.dim[2]),
            u64::from(header.dim[3]),
        ],
        nb_streamlines: 0,
        nb_vertices: 0,
        extra: Default::default(),
    })
}
