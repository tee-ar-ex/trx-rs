use std::fs;
use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;

use crate::dtype::TrxScalar;
use crate::error::Result;
use crate::trx_file::{DataPerGroup, TrxFile};

fn offsets_as_u32_bytes(offsets: &[u32]) -> Vec<u8> {
    crate::mmap_backing::vec_to_bytes(offsets.to_vec())
}

/// Load a TRX file from a `.trx` zip archive.
///
/// Extracts the archive to a temporary directory, then delegates to the
/// directory loader. The `TempDir` handle is stored in the returned `TrxFile`
/// so the temp files remain alive while mmaps reference them.
pub fn load_from_zip<P: TrxScalar>(path: &Path) -> Result<TrxFile<P>> {
    let file = fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    let tempdir = tempfile::TempDir::new()?;
    let temp_path = tempdir.path().to_path_buf();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let entry_path = temp_path.join(entry.name());

        if entry.is_dir() {
            fs::create_dir_all(&entry_path)?;
        } else {
            if let Some(parent) = entry_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = fs::File::create(&entry_path)?;
            std::io::copy(&mut entry, &mut out_file)?;
        }
    }

    crate::io::directory::load_from_directory(&temp_path, Some(tempdir))
}

/// Save a `TrxFile<P>` to a `.trx` zip archive with deflate compression.
pub fn save_to_zip<P: TrxScalar>(trx: &TrxFile<P>, path: &Path) -> Result<()> {
    save_to_zip_with(trx, path, zip::CompressionMethod::Deflated)
}

/// Save a `TrxFile<P>` to a `.trx` zip archive with the given compression method.
pub fn save_to_zip_with<P: TrxScalar>(
    trx: &TrxFile<P>,
    path: &Path,
    compression: zip::CompressionMethod,
) -> Result<()> {
    let file = fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default()
        .compression_method(compression)
        .large_file(true);

    // Header
    let header_json = trx.header().to_json()?;
    zip.start_file("header.json", options)?;
    zip.write_all(header_json.as_bytes())?;

    // Positions
    let pos_filename = format!("positions.3.{}", P::DTYPE.name());
    zip.start_file(&pos_filename, options)?;
    zip.write_all(trx.positions_bytes())?;

    // Offsets default to compact uint32 on disk.
    zip.start_file("offsets.uint32", options)?;
    let offsets_bytes = offsets_as_u32_bytes(trx.offsets());
    zip.write_all(&offsets_bytes)?;

    // DPS
    write_data_map(&mut zip, "dps", trx.dps_arrays(), options)?;

    // DPV
    write_data_map(&mut zip, "dpv", trx.dpv_arrays(), options)?;

    // Groups
    write_data_map(&mut zip, "groups", trx.group_arrays(), options)?;

    // DPG
    write_dpg_map(&mut zip, "dpg", trx.dpg_arrays(), options)?;

    zip.finish()?;
    Ok(())
}

fn write_data_map<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    prefix: &str,
    arrays: &std::collections::HashMap<String, crate::trx_file::DataArray>,
    options: SimpleFileOptions,
) -> Result<()> {
    for (name, arr) in arrays {
        let filename = crate::io::filename::TrxFilename {
            name: name.clone(),
            ncols: arr.ncols(),
            dtype: arr.dtype(),
        }
        .to_filename();
        let entry_name = format!("{prefix}/{filename}");
        zip.start_file(&entry_name, options)?;
        zip.write_all(arr.as_bytes())?;
    }
    Ok(())
}

fn write_dpg_map<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    prefix: &str,
    groups: &DataPerGroup,
    options: SimpleFileOptions,
) -> Result<()> {
    for (group, arrays) in groups {
        for (name, arr) in arrays {
            let filename = crate::io::filename::TrxFilename {
                name: name.clone(),
                ncols: arr.ncols(),
                dtype: arr.dtype(),
            }
            .to_filename();
            let entry_name = format!("{prefix}/{group}/{filename}");
            zip.start_file(&entry_name, options)?;
            zip.write_all(arr.as_bytes())?;
        }
    }
    Ok(())
}
