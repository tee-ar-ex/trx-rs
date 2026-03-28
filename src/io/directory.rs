use bytemuck::cast_slice;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::io::filename::TrxFilename;
use crate::mmap_backing::MmapBacking;
use crate::trx_file::{DataArray, TrxFile};

fn offsets_as_u32_bytes(offsets: &[u32]) -> Vec<u8> {
    crate::mmap_backing::vec_to_bytes(offsets.to_vec())
}

/// Memory-map a file as read-only.
fn mmap_file(path: &Path) -> Result<Mmap> {
    let file = fs::File::open(path)?;
    // SAFETY: We trust the file won't be modified externally while mapped.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

/// Load data arrays from a subdirectory (e.g. `dps/`, `dpv/`, `groups/`).
fn load_data_dir(dir: &Path) -> Result<HashMap<String, DataArray>> {
    let mut map = HashMap::new();
    if !dir.exists() {
        return Ok(map);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| TrxError::Format(format!("invalid filename: {}", path.display())))?;

        let parsed = TrxFilename::parse(file_name)?;
        let mmap = mmap_file(&path)?;

        map.insert(
            parsed.name.clone(),
            DataArray {
                backing: MmapBacking::ReadOnly(mmap),
                ncols: parsed.ncols,
                dtype: parsed.dtype,
            },
        );
    }

    Ok(map)
}

/// Find a file matching a given name prefix in a directory, regardless of
/// the ncols/dtype suffix.
fn find_file_with_prefix(dir: &Path, prefix: &str) -> Result<std::path::PathBuf> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(prefix) && name_str.chars().nth(prefix.len()) == Some('.') {
            return Ok(entry.path());
        }
    }
    Err(TrxError::FileNotFound(dir.join(prefix)))
}

/// Load a `TrxFile<P>` from an uncompressed directory.
pub fn load_from_directory<P: TrxScalar>(
    dir: &Path,
    tempdir: Option<tempfile::TempDir>,
) -> Result<TrxFile<P>> {
    if !dir.is_dir() {
        return Err(TrxError::FileNotFound(dir.to_path_buf()));
    }

    // Header
    let header = Header::from_file(&dir.join("header.json"))?;

    // Positions
    let pos_path = find_file_with_prefix(dir, "positions")?;
    let pos_fname = pos_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| TrxError::Format("invalid positions filename".into()))?;
    let pos_parsed = TrxFilename::parse(pos_fname)?;

    if pos_parsed.dtype != P::DTYPE {
        return Err(TrxError::DType(format!(
            "expected positions dtype {}, got {}",
            P::DTYPE,
            pos_parsed.dtype
        )));
    }
    if pos_parsed.ncols != 3 {
        return Err(TrxError::Format(format!(
            "positions must have 3 columns, got {}",
            pos_parsed.ncols
        )));
    }

    let positions_backing = MmapBacking::ReadOnly(mmap_file(&pos_path)?);

    // Offsets
    let off_path = find_file_with_prefix(dir, "offsets")?;
    let off_fname = off_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| TrxError::Format("invalid offsets filename".into()))?;
    let off_parsed = TrxFilename::parse(off_fname)?;

    let offsets_mmap = mmap_file(&off_path)?;
    let offsets_backing = convert_offsets_to_u32(
        &offsets_mmap,
        off_parsed.dtype,
        header.nb_streamlines as usize,
        header.nb_vertices as usize,
    )?;

    // DPS, DPV, groups
    let dps = load_data_dir(&dir.join("dps"))?;
    let dpv = load_data_dir(&dir.join("dpv"))?;
    let groups = load_data_dir(&dir.join("groups"))?;

    Ok(TrxFile::from_parts(
        header,
        positions_backing,
        offsets_backing,
        dps,
        dpv,
        groups,
        tempdir,
    ))
}

/// Convert offset bytes to u32, handling uint64→u32 narrowing and
/// ensuring the sentinel value (nb_vertices) is present.
fn convert_offsets_to_u32(
    mmap: &Mmap,
    dtype: DType,
    nb_streamlines: usize,
    nb_vertices: usize,
) -> Result<MmapBacking> {
    match dtype {
        DType::UInt64 => {
            let values: &[u64] = cast_slice(mmap.as_ref());
            // Check if sentinel is present
            if values.len() == nb_streamlines {
                // Missing sentinel — append nb_vertices
                let mut owned: Vec<u32> = values
                    .iter()
                    .copied()
                    .map(|value| {
                        u32::try_from(value).map_err(|_| {
                            TrxError::Format(format!("offset {value} exceeds uint32 range"))
                        })
                    })
                    .collect::<Result<_>>()?;
                owned.push(nb_vertices as u32);
                let bytes: Vec<u8> = crate::mmap_backing::vec_to_bytes(owned);
                Ok(MmapBacking::Owned(bytes))
            } else if values.len() == nb_streamlines + 1 {
                let owned: Vec<u32> = values
                    .iter()
                    .copied()
                    .map(|value| {
                        u32::try_from(value).map_err(|_| {
                            TrxError::Format(format!("offset {value} exceeds uint32 range"))
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(MmapBacking::Owned(crate::mmap_backing::vec_to_bytes(owned)))
            } else {
                Err(TrxError::Format(format!(
                    "unexpected offset count: {} (expected {} or {})",
                    values.len(),
                    nb_streamlines,
                    nb_streamlines + 1,
                )))
            }
        }
        DType::UInt32 => {
            let values: &[u32] = cast_slice(mmap.as_ref());
            let mut out: Vec<u32> = values.to_vec();
            if out.len() == nb_streamlines {
                out.push(nb_vertices as u32);
            }
            let bytes: Vec<u8> = crate::mmap_backing::vec_to_bytes(out);
            Ok(MmapBacking::Owned(bytes))
        }
        other => Err(TrxError::DType(format!(
            "offsets must be uint32 or uint64, got {other}"
        ))),
    }
}

/// Save a `TrxFile<P>` to an uncompressed directory.
pub fn save_to_directory<P: TrxScalar>(trx: &TrxFile<P>, dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)?;

    // Header
    trx.header.write_to(&dir.join("header.json"))?;

    // Positions
    let pos_filename = format!("positions.3.{}", P::DTYPE.name());
    fs::write(dir.join(&pos_filename), trx.positions_bytes())?;

    // Offsets default to compact uint32 on disk.
    let offsets_filename = "offsets.1.uint32";
    let offsets_bytes = offsets_as_u32_bytes(trx.offsets());
    fs::write(dir.join(offsets_filename), offsets_bytes)?;

    // DPS
    save_data_dir(&trx.dps, &dir.join("dps"))?;

    // DPV
    save_data_dir(&trx.dpv, &dir.join("dpv"))?;

    // Groups
    save_data_dir(&trx.groups, &dir.join("groups"))?;

    Ok(())
}

fn save_data_dir(arrays: &HashMap<String, DataArray>, dir: &Path) -> Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    for (name, arr) in arrays {
        let filename = TrxFilename {
            name: name.clone(),
            ncols: arr.ncols,
            dtype: arr.dtype,
        }
        .to_filename();
        fs::write(dir.join(&filename), arr.backing.as_bytes())?;
    }
    Ok(())
}
