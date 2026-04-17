use bytemuck::cast_slice;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::io::filename::TrxFilename;
use crate::mmap_backing::vec_to_bytes;
use crate::mmap_backing::MmapBacking;
use crate::trx_file::{DataArray, DataPerGroup, TrxFile, TrxParts};

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
            DataArray::from_backing(MmapBacking::ReadOnly(mmap), parsed.ncols, parsed.dtype),
        );
    }

    Ok(map)
}

fn load_dpg_dir(dir: &Path) -> Result<DataPerGroup> {
    let mut out = HashMap::new();
    if !dir.exists() {
        return Ok(out);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let group_name = entry.file_name().to_string_lossy().to_string();
        let data = load_data_dir(&path)?;
        if !data.is_empty() {
            out.insert(group_name, data);
        }
    }

    Ok(out)
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
    let dpg = load_dpg_dir(&dir.join("dpg"))?;

    Ok(TrxFile::from_parts(TrxParts {
        header,
        positions_backing,
        offsets_backing,
        dps,
        dpv,
        groups,
        dpg,
        tempdir,
    }))
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
    trx.header().write_to(&dir.join("header.json"))?;

    // Positions
    let pos_filename = format!("positions.3.{}", P::DTYPE.name());
    fs::write(dir.join(&pos_filename), trx.positions_bytes())?;

    // Offsets default to compact uint32 on disk.
    let offsets_filename = "offsets.uint32";
    let offsets_bytes = offsets_as_u32_bytes(trx.offsets());
    fs::write(dir.join(offsets_filename), offsets_bytes)?;

    // DPS
    save_data_dir(trx.dps_arrays(), &dir.join("dps"))?;

    // DPV
    save_data_dir(trx.dpv_arrays(), &dir.join("dpv"))?;

    // Groups
    save_data_dir(trx.group_arrays(), &dir.join("groups"))?;

    // DPG
    save_dpg_dir(trx.dpg_arrays(), &dir.join("dpg"))?;

    Ok(())
}

pub fn append_dps_to_directory(
    dir: &Path,
    dps: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    let header = Header::from_file(&dir.join("header.json"))?;
    validate_row_count("DPS", dps, header.nb_streamlines as usize)?;
    append_arrays_to_directory(&dir.join("dps"), dps, overwrite)
}

pub fn append_dpv_to_directory(
    dir: &Path,
    dpv: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    let header = Header::from_file(&dir.join("header.json"))?;
    validate_row_count("DPV", dpv, header.nb_vertices as usize)?;
    append_arrays_to_directory(&dir.join("dpv"), dpv, overwrite)
}

pub fn append_groups_to_directory(
    dir: &Path,
    groups: &HashMap<String, Vec<u32>>,
    overwrite: bool,
) -> Result<()> {
    let header = Header::from_file(&dir.join("header.json"))?;
    let groups_dir = dir.join("groups");
    fs::create_dir_all(&groups_dir)?;
    for (name, members) in groups {
        validate_group_members(name, members, header.nb_streamlines as usize)?;
        let target = groups_dir.join(format!("{name}.uint32"));
        if !overwrite {
            if let Some(existing) = find_named_array_file(&groups_dir, name)? {
                if existing.exists() {
                    continue;
                }
            }
        } else if let Some(existing) = find_named_array_file(&groups_dir, name)? {
            if existing != target && existing.exists() {
                fs::remove_file(existing)?;
            }
        }
        fs::write(target, vec_to_bytes(members.clone()))?;
    }
    Ok(())
}

pub fn append_dpg_to_directory(dir: &Path, dpg: &DataPerGroup, overwrite: bool) -> Result<()> {
    let groups_dir = dir.join("groups");
    let dpg_root = dir.join("dpg");
    for (group, entries) in dpg {
        if find_named_array_file(&groups_dir, group)?.is_none() {
            return Err(TrxError::Argument(format!(
                "cannot add DPG entries for missing group '{group}'"
            )));
        }
        let group_dir = dpg_root.join(group);
        fs::create_dir_all(&group_dir)?;
        for (name, arr) in entries {
            let target = group_dir.join(filename_for_array(name, arr));
            if !overwrite {
                if let Some(existing) = find_named_array_file(&group_dir, name)? {
                    if existing.exists() {
                        continue;
                    }
                }
            } else if let Some(existing) = find_named_array_file(&group_dir, name)? {
                if existing != target && existing.exists() {
                    fs::remove_file(existing)?;
                }
            }
            fs::write(target, arr.as_bytes())?;
        }
    }
    Ok(())
}

pub fn delete_dps_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    delete_named_arrays(&dir.join("dps"), names)
}

pub fn delete_dpv_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    delete_named_arrays(&dir.join("dpv"), names)
}

pub fn delete_groups_from_directory(dir: &Path, names: &[&str]) -> Result<()> {
    let groups_dir = dir.join("groups");
    for name in names {
        if let Some(path) = find_named_array_file(&groups_dir, name)? {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        let dpg_group = dir.join("dpg").join(name);
        if dpg_group.exists() {
            fs::remove_dir_all(dpg_group)?;
        }
    }
    Ok(())
}

pub fn delete_dpg_from_directory(dir: &Path, group: &str, names: Option<&[&str]>) -> Result<()> {
    let group_dir = dir.join("dpg").join(group);
    match names {
        None | Some([]) => {
            if group_dir.exists() {
                fs::remove_dir_all(group_dir)?;
            }
        }
        Some(names) => {
            for name in names {
                if let Some(path) = find_named_array_file(&group_dir, name)? {
                    if path.exists() {
                        fs::remove_file(path)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn save_data_dir(arrays: &HashMap<String, DataArray>, dir: &Path) -> Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    for (name, arr) in arrays {
        let filename = filename_for_array(name, arr);
        fs::write(dir.join(&filename), arr.as_bytes())?;
    }
    Ok(())
}

fn save_dpg_dir(arrays: &DataPerGroup, dir: &Path) -> Result<()> {
    if arrays.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    for (group, entries) in arrays {
        save_data_dir(entries, &dir.join(group))?;
    }
    Ok(())
}

fn append_arrays_to_directory(
    dir: &Path,
    arrays: &HashMap<String, DataArray>,
    overwrite: bool,
) -> Result<()> {
    fs::create_dir_all(dir)?;
    for (name, arr) in arrays {
        let target = dir.join(filename_for_array(name, arr));
        if !overwrite {
            if let Some(existing) = find_named_array_file(dir, name)? {
                if existing.exists() {
                    continue;
                }
            }
        } else if let Some(existing) = find_named_array_file(dir, name)? {
            if existing != target && existing.exists() {
                fs::remove_file(existing)?;
            }
        }
        fs::write(target, arr.as_bytes())?;
    }
    Ok(())
}

fn delete_named_arrays(dir: &Path, names: &[&str]) -> Result<()> {
    for name in names {
        if let Some(path) = find_named_array_file(dir, name)? {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

fn find_named_array_file(dir: &Path, name: &str) -> Result<Option<std::path::PathBuf>> {
    if !dir.exists() {
        return Ok(None);
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
        if parsed.name == name {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn validate_row_count(
    kind: &str,
    arrays: &HashMap<String, DataArray>,
    expected_rows: usize,
) -> Result<()> {
    for (name, arr) in arrays {
        if arr.nrows() != expected_rows {
            return Err(TrxError::Format(format!(
                "{kind} '{name}' has {} rows, expected {expected_rows}",
                arr.nrows()
            )));
        }
    }
    Ok(())
}

fn validate_group_members(name: &str, members: &[u32], nb_streamlines: usize) -> Result<()> {
    for &member in members {
        if member as usize >= nb_streamlines {
            return Err(TrxError::Format(format!(
                "group '{name}' contains streamline index {member}, but NB_STREAMLINES is {nb_streamlines}"
            )));
        }
    }
    Ok(())
}

fn filename_for_array(name: &str, arr: &DataArray) -> String {
    TrxFilename {
        name: name.to_string(),
        ncols: arr.ncols(),
        dtype: arr.dtype(),
    }
    .to_filename()
}
