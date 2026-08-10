use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use zip::write::SimpleFileOptions;

use super::archive_edit::{self, ArchiveOp};
use crate::dtype::TrxScalar;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::io::filename::TrxFilename;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::trx_file::{DataArray, DataPerGroup, TrxFile, TrxParts};

/// On-disk width for the `offsets.*` array. The TRX spec accepts both
/// `uint32` and `uint64`; we auto-pick at write time via [`pick_for`] so
/// every-day tractograms (≤ 4 G vertices) stay compact and only genuinely
/// huge files pay the doubled-width cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OffsetsDtype {
    U32,
    // Reserved for the day the in-memory offset representation widens past
    // u32. The reader already accepts `offsets.uint64`; the writer will
    // start emitting it once `pick_for` can return this variant.
    #[allow(dead_code)]
    U64,
}

impl OffsetsDtype {
    /// Pick the narrowest dtype that fits every offset in the slice.
    pub(crate) fn pick_for(offsets: &[u32]) -> Self {
        // The in-memory representation is `&[u32]`, so by definition every
        // value fits in `u32`. We still keep this function as the canonical
        // place to widen the rule if the in-memory type ever changes.
        let _ = offsets;
        OffsetsDtype::U32
    }

    /// Filename suffix written for this dtype (e.g. `"uint64"`).
    pub(crate) fn suffix(self) -> &'static str {
        match self {
            OffsetsDtype::U32 => "uint32",
            OffsetsDtype::U64 => "uint64",
        }
    }

    /// Serialise an in-memory `u32` offset slice to disk bytes at this width.
    pub(crate) fn encode(self, offsets: &[u32]) -> Vec<u8> {
        match self {
            OffsetsDtype::U32 => crate::mmap_backing::vec_to_bytes(offsets.to_vec()),
            OffsetsDtype::U64 => {
                let widened: Vec<u64> = offsets.iter().map(|&o| o as u64).collect();
                crate::mmap_backing::vec_to_bytes(widened)
            }
        }
    }
}

#[derive(Debug, Default)]
struct TrxArchiveIndex {
    dps: HashMap<String, String>,
    dpv: HashMap<String, String>,
    groups: HashMap<String, String>,
    dpg: HashMap<String, HashMap<String, String>>,
}

fn get_entry_backing(
    arc_mmap: &Arc<memmap2::Mmap>,
    entry: &mut zip::read::ZipFile,
    align_requirement: usize,
) -> Result<MmapBacking> {
    let size = entry.size() as usize;
    if size == 0 {
        return Ok(MmapBacking::Owned(Vec::new()));
    }
    if entry.compression() == zip::CompressionMethod::Stored {
        let data_start = entry.data_start() as usize;
        if data_start + size > arc_mmap.len() {
            return Err(TrxError::Format(
                "zip entry data exceeds archive boundaries".into(),
            ));
        }
        let ptr_val = arc_mmap.as_ptr() as usize + data_start;
        let req = align_requirement.max(1);
        if ptr_val.is_multiple_of(req) {
            Ok(MmapBacking::SharedSlice {
                mmap: Arc::clone(arc_mmap),
                offset: data_start,
                len: size,
            })
        } else {
            let slice = &arc_mmap[data_start..data_start + size];
            let num_u64s = size.div_ceil(8);
            let mut values = vec![0u64; num_u64s];
            let bytes_slice = bytemuck::cast_slice_mut(&mut values);
            bytes_slice[..size].copy_from_slice(slice);
            Ok(MmapBacking::OwnedU64(values, size))
        }
    } else {
        let num_u64s = size.div_ceil(8);
        let mut values = vec![0u64; num_u64s];
        let bytes_slice = bytemuck::cast_slice_mut(&mut values);
        // We use read_exact or read_to_end into a subslice
        Read::read_exact(entry, &mut bytes_slice[..size])?;
        Ok(MmapBacking::OwnedU64(values, size))
    }
}

fn load_zip_offsets(
    arc_mmap: &Arc<memmap2::Mmap>,
    entry: &mut zip::read::ZipFile,
    dtype: crate::dtype::DType,
    nb_streamlines: usize,
    nb_vertices: usize,
) -> Result<MmapBacking> {
    match dtype {
        crate::dtype::DType::UInt64 => {
            if entry.size() % 8 != 0 {
                return Err(TrxError::Format(format!(
                    "offsets entry size {} is not a multiple of 8 for uint64",
                    entry.size()
                )));
            }
            let mut values = vec![0u64; entry.size() as usize / 8];
            Read::read_exact(entry, bytemuck::cast_slice_mut(&mut values))?;
            if values.len() == nb_streamlines {
                let mut owned: Vec<u32> = values
                    .iter()
                    .copied()
                    .map(|v| {
                        u32::try_from(v).map_err(|_| {
                            TrxError::Format(format!("offset {v} exceeds uint32 range"))
                        })
                    })
                    .collect::<Result<_>>()?;
                owned.push(nb_vertices as u32);
                let len = owned.len() * 4;
                Ok(MmapBacking::OwnedU32(owned, len))
            } else if values.len() == nb_streamlines + 1 {
                let owned: Vec<u32> = values
                    .iter()
                    .copied()
                    .map(|v| {
                        u32::try_from(v).map_err(|_| {
                            TrxError::Format(format!("offset {v} exceeds uint32 range"))
                        })
                    })
                    .collect::<Result<_>>()?;
                let len = owned.len() * 4;
                Ok(MmapBacking::OwnedU32(owned, len))
            } else {
                Err(TrxError::Format(format!(
                    "unexpected offset count: {} (expected {} or {})",
                    values.len(),
                    nb_streamlines,
                    nb_streamlines + 1,
                )))
            }
        }
        crate::dtype::DType::UInt32 => {
            if entry.size() % 4 != 0 {
                return Err(TrxError::Format(format!(
                    "offsets entry size {} is not a multiple of 4 for uint32",
                    entry.size()
                )));
            }
            let num_u32 = entry.size() as usize / 4;
            if num_u32 == nb_streamlines {
                let mut values = vec![0u32; num_u32];
                Read::read_exact(entry, bytemuck::cast_slice_mut(&mut values))?;
                let mut out = values;
                out.push(nb_vertices as u32);
                let len = out.len() * 4;
                Ok(MmapBacking::OwnedU32(out, len))
            } else if num_u32 == nb_streamlines + 1 {
                get_entry_backing(arc_mmap, entry, std::mem::align_of::<u32>())
            } else {
                Err(TrxError::Format(format!(
                    "unexpected offset count: {} (expected {} or {})",
                    num_u32,
                    nb_streamlines,
                    nb_streamlines + 1,
                )))
            }
        }
        other => Err(TrxError::DType(format!(
            "offsets must be uint32 or uint64, got {other}"
        ))),
    }
}

/// Load a TRX file from a `.trx` zip archive.
///
/// Memory-maps the zip file directly from disk and parses entries in-memory
/// without creating temporary directories.
pub fn load_from_zip<P: TrxScalar>(path: &Path) -> Result<TrxFile<P>> {
    let file = fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let arc_mmap = Arc::new(mmap);
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(arc_mmap.as_ref()))?;

    let header: Header = {
        let mut header_entry = archive.by_name("header.json")?;
        let mut bytes = Vec::new();
        Read::read_to_end(&mut header_entry, &mut bytes)?;
        serde_json::from_slice(&bytes)?
    };

    let mut positions_backing = None;
    let mut offsets_backing = None;
    let mut dps = HashMap::new();
    let mut dpv = HashMap::new();
    let mut groups = HashMap::new();
    let mut dpg: DataPerGroup = HashMap::new();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if entry.is_dir() || name.ends_with('/') || name == "header.json" {
            continue;
        }

        if let Some(rest) = name.strip_prefix("dps/") {
            let parsed = TrxFilename::parse(rest)?;
            let backing = get_entry_backing(&arc_mmap, &mut entry, parsed.dtype.size_of())?;
            dps.insert(
                parsed.name,
                DataArray::from_backing(backing, parsed.ncols, parsed.dtype),
            );
        } else if let Some(rest) = name.strip_prefix("dpv/") {
            let parsed = TrxFilename::parse(rest)?;
            let backing = get_entry_backing(&arc_mmap, &mut entry, parsed.dtype.size_of())?;
            dpv.insert(
                parsed.name,
                DataArray::from_backing(backing, parsed.ncols, parsed.dtype),
            );
        } else if let Some(rest) = name.strip_prefix("groups/") {
            let parsed = TrxFilename::parse(rest)?;
            let backing = get_entry_backing(&arc_mmap, &mut entry, parsed.dtype.size_of())?;
            groups.insert(
                parsed.name,
                DataArray::from_backing(backing, parsed.ncols, parsed.dtype),
            );
        } else if let Some(rest) = name.strip_prefix("dpg/") {
            if let Some((group, file_name)) = rest.split_once('/') {
                let parsed = TrxFilename::parse(file_name)?;
                let backing = get_entry_backing(&arc_mmap, &mut entry, parsed.dtype.size_of())?;
                let group_map = dpg.entry(group.to_string()).or_default();
                group_map.insert(
                    parsed.name,
                    DataArray::from_backing(backing, parsed.ncols, parsed.dtype),
                );
            }
        } else if name.starts_with("positions.") {
            let pos_parsed = TrxFilename::parse(&name)?;
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
            positions_backing = Some(get_entry_backing(
                &arc_mmap,
                &mut entry,
                std::mem::align_of::<P>(),
            )?);
        } else if name.starts_with("offsets.") {
            let off_parsed = TrxFilename::parse(&name)?;
            let backing = load_zip_offsets(
                &arc_mmap,
                &mut entry,
                off_parsed.dtype,
                header.nb_streamlines as usize,
                header.nb_vertices as usize,
            )?;
            offsets_backing = Some(backing);
        }
    }

    let positions_backing = match positions_backing {
        Some(b) => b,
        None => {
            if header.nb_vertices == 0 {
                MmapBacking::Owned(Vec::new())
            } else {
                return Err(TrxError::FileNotFound(path.join("positions")));
            }
        }
    };

    let offsets_backing = match offsets_backing {
        Some(b) => b,
        None => {
            if header.nb_streamlines == 0 {
                MmapBacking::Owned(vec_to_bytes(vec![0u32]))
            } else {
                return Err(TrxError::FileNotFound(path.join("offsets")));
            }
        }
    };

    Ok(TrxFile::from_parts(TrxParts {
        header,
        positions_backing,
        offsets_backing,
        dps,
        dpv,
        groups,
        dpg,
    }))
}

/// Save a `TrxFile<P>` to a `.trx` zip archive.
///
/// All entries are written uncompressed (Stored). DEFLATE rarely pays off on
/// the float-dominated payload of a TRX file: compression ratios are typically
/// <15% and write time grows substantially. Callers who want to compress the
/// `groups/` entries (uint32 streamline-index lists, which do tend to have
/// runs) can use [`save_to_zip_with`].
pub fn save_to_zip<P: TrxScalar>(trx: &TrxFile<P>, path: &Path) -> Result<()> {
    save_to_zip_with(trx, path, zip::CompressionMethod::Stored)
}

/// Save a `TrxFile<P>` to a `.trx` zip archive, applying `groups_compression`
/// to `groups/` entries only. All other entries (header, positions, offsets,
/// dps, dpv, dpg) are always Stored.
pub fn save_to_zip_with<P: TrxScalar>(
    trx: &TrxFile<P>,
    path: &Path,
    groups_compression: zip::CompressionMethod,
) -> Result<()> {
    let offsets_dtype = OffsetsDtype::pick_for(trx.offsets());
    let file = fs::File::create(path)?;
    let buf_writer = BufWriter::new(file);
    let mut zip = zip::ZipWriter::new(buf_writer);
    let stored = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .with_alignment(8)
        .large_file(true);
    let groups_opts = SimpleFileOptions::default()
        .compression_method(groups_compression)
        .large_file(true);

    // Header
    let header_json = trx.header().to_json()?;
    zip.start_file("header.json", stored)?;
    zip.write_all(header_json.as_bytes())?;

    // Positions
    let pos_filename = format!("positions.3.{}", P::DTYPE.name());
    zip.start_file(&pos_filename, stored)?;
    zip.write_all(trx.positions_bytes())?;

    // Offsets — written at `offsets_dtype`'s width.
    let offsets_filename = format!("offsets.{}", offsets_dtype.suffix());
    zip.start_file(&offsets_filename, stored)?;
    let offsets_bytes = offsets_dtype.encode(trx.offsets());
    zip.write_all(&offsets_bytes)?;

    // DPS / DPV — float-heavy, Stored.
    write_data_map(&mut zip, "dps", trx.dps_arrays(), stored)?;
    write_data_map(&mut zip, "dpv", trx.dpv_arrays(), stored)?;

    // Groups — uint32 membership lists; honor caller's compression choice.
    write_data_map(&mut zip, "groups", trx.group_arrays(), groups_opts)?;

    // DPG — tiny per-group scalars, Stored.
    write_dpg_map(&mut zip, "dpg", trx.dpg_arrays(), stored)?;

    let mut buf_writer = zip.finish()?;
    buf_writer.flush()?;
    Ok(())
}

/// Append DPS arrays to a TRX zip archive, optionally overwriting existing entries.
pub fn append_dps_to_zip(
    path: &Path,
    dps: &HashMap<String, DataArray>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let header = read_header_from_zip(path)?;
    validate_row_count("DPS", dps, header.nb_streamlines as usize)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for (name, arr) in dps {
        let target = data_entry_path("dps", name, arr);
        plan_data_write(
            &index.dps,
            name,
            target,
            arr,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    archive_edit::apply_archive_ops(path, ops)
}

/// Append DPV arrays to a TRX zip archive, optionally overwriting existing entries.
pub fn append_dpv_to_zip(
    path: &Path,
    dpv: &HashMap<String, DataArray>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let header = read_header_from_zip(path)?;
    validate_row_count("DPV", dpv, header.nb_vertices as usize)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for (name, arr) in dpv {
        let target = data_entry_path("dpv", name, arr);
        plan_data_write(
            &index.dpv,
            name,
            target,
            arr,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    archive_edit::apply_archive_ops(path, ops)
}

/// Append group membership arrays to a TRX zip archive, optionally overwriting existing entries.
pub fn append_groups_to_zip(
    path: &Path,
    groups: &HashMap<String, Vec<u32>>,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let header = read_header_from_zip(path)?;
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for (name, members) in groups {
        validate_group_members(name, members, header.nb_streamlines as usize)?;
        let target = format!("groups/{name}.uint32");
        let bytes = vec_to_bytes(members.clone());
        plan_bytes_write(
            index.groups.get(name),
            target,
            bytes,
            overwrite,
            compression,
            &mut ops,
        )?;
    }
    archive_edit::apply_archive_ops(path, ops)
}

/// Append DPG (data-per-group) entries to a TRX zip archive, optionally overwriting existing entries.
pub fn append_dpg_to_zip(
    path: &Path,
    dpg: &DataPerGroup,
    compression: zip::CompressionMethod,
    overwrite: bool,
) -> Result<()> {
    let index = build_archive_index(path)?;
    let available_groups: BTreeSet<&str> = index.groups.keys().map(String::as_str).collect();
    let mut ops = Vec::new();
    for (group, arrays) in dpg {
        if !available_groups.contains(group.as_str()) {
            return Err(TrxError::Argument(format!(
                "cannot add DPG entries for missing group '{group}'"
            )));
        }
        let existing = index.dpg.get(group);
        for (name, arr) in arrays {
            let target = format!("dpg/{group}/{}", filename_for_array(name, arr));
            let existing_path = existing.and_then(|entries| entries.get(name));
            plan_bytes_write(
                existing_path,
                target,
                arr.as_bytes().to_vec(),
                overwrite,
                compression,
                &mut ops,
            )?;
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

/// Delete named DPS arrays from a TRX zip archive.
pub fn delete_dps_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for name in names {
        if let Some(entry_path) = index.dps.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

/// Delete named DPV arrays from a TRX zip archive.
pub fn delete_dpv_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for name in names {
        if let Some(entry_path) = index.dpv.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

/// Delete named groups (and their DPG entries) from a TRX zip archive.
pub fn delete_groups_from_zip(path: &Path, names: &[&str]) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    for name in names {
        if let Some(entry_path) = index.groups.get(*name) {
            ops.push(ArchiveOp::Delete {
                path: entry_path.clone(),
            });
        }
        ops.push(ArchiveOp::DeletePrefix {
            prefix: format!("dpg/{name}"),
        });
    }
    archive_edit::apply_archive_ops(path, ops)
}

/// Delete DPG entries for a specific group from a TRX zip archive.
///
/// When `names` is `None` or empty, the entire DPG prefix for the group is removed.
/// When `names` lists specific fields, only those entries are deleted.
pub fn delete_dpg_from_zip(path: &Path, group: &str, names: Option<&[&str]>) -> Result<()> {
    let index = build_archive_index(path)?;
    let mut ops = Vec::new();
    match names {
        None => ops.push(ArchiveOp::DeletePrefix {
            prefix: format!("dpg/{group}"),
        }),
        Some([]) => ops.push(ArchiveOp::DeletePrefix {
            prefix: format!("dpg/{group}"),
        }),
        Some(names) => {
            if let Some(entries) = index.dpg.get(group) {
                for name in names {
                    if let Some(entry_path) = entries.get(*name) {
                        ops.push(ArchiveOp::Delete {
                            path: entry_path.clone(),
                        });
                    }
                }
            }
        }
    }
    archive_edit::apply_archive_ops(path, ops)
}

fn read_header_from_zip(path: &Path) -> Result<Header> {
    let bytes = archive_edit::read_archive_entry(path, "header.json")?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn build_archive_index(path: &Path) -> Result<TrxArchiveIndex> {
    let entries = archive_edit::archive_entry_names(path)?;
    let mut index = TrxArchiveIndex::default();

    for entry in entries {
        if let Some(rest) = entry.strip_prefix("dps/") {
            index_entry(&mut index.dps, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("dpv/") {
            index_entry(&mut index.dpv, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("groups/") {
            index_entry(&mut index.groups, &entry, rest)?;
        } else if let Some(rest) = entry.strip_prefix("dpg/") {
            if let Some((group, file_name)) = rest.split_once('/') {
                let parsed = TrxFilename::parse(file_name)?;
                let group_entries = index.dpg.entry(group.to_string()).or_default();
                if group_entries.insert(parsed.name, entry.clone()).is_some() {
                    return Err(TrxError::Format(format!(
                        "duplicate DPG entry path for group '{group}'"
                    )));
                }
            }
        }
    }

    Ok(index)
}

fn index_entry(
    index: &mut HashMap<String, String>,
    full_path: &str,
    file_name: &str,
) -> Result<()> {
    if file_name.ends_with('/') {
        return Ok(());
    }
    let parsed = TrxFilename::parse(file_name)?;
    if index.insert(parsed.name, full_path.to_string()).is_some() {
        return Err(TrxError::Format(format!(
            "duplicate archive entry for '{full_path}'"
        )));
    }
    Ok(())
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

fn data_entry_path(prefix: &str, name: &str, arr: &DataArray) -> String {
    format!("{prefix}/{}", filename_for_array(name, arr))
}

fn filename_for_array(name: &str, arr: &DataArray) -> String {
    TrxFilename {
        name: name.to_string(),
        ncols: arr.ncols(),
        dtype: arr.dtype(),
    }
    .to_filename()
}

fn plan_data_write(
    existing: &HashMap<String, String>,
    logical_name: &str,
    target_path: String,
    arr: &DataArray,
    overwrite: bool,
    compression: zip::CompressionMethod,
    ops: &mut Vec<ArchiveOp>,
) -> Result<()> {
    plan_bytes_write(
        existing.get(logical_name),
        target_path,
        arr.as_bytes().to_vec(),
        overwrite,
        compression,
        ops,
    )
}

fn plan_bytes_write(
    existing_path: Option<&String>,
    target_path: String,
    bytes: Vec<u8>,
    overwrite: bool,
    compression: zip::CompressionMethod,
    ops: &mut Vec<ArchiveOp>,
) -> Result<()> {
    match existing_path {
        None => ops.push(ArchiveOp::Add {
            path: target_path,
            bytes,
            compression,
        }),
        Some(_) if !overwrite => {}
        Some(existing) if existing == &target_path => ops.push(ArchiveOp::Replace {
            path: target_path,
            bytes,
            compression,
        }),
        Some(existing) => {
            ops.push(ArchiveOp::Delete {
                path: existing.clone(),
            });
            ops.push(ArchiveOp::Add {
                path: target_path,
                bytes,
                compression,
            });
        }
    }
    Ok(())
}

fn write_data_map<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    prefix: &str,
    arrays: &HashMap<String, DataArray>,
    options: SimpleFileOptions,
) -> Result<()> {
    for (name, arr) in arrays {
        let filename = filename_for_array(name, arr);
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
            let filename = filename_for_array(name, arr);
            let entry_name = format!("{prefix}/{group}/{filename}");
            zip.start_file(&entry_name, options)?;
            zip.write_all(arr.as_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::stream::TrxStream;

    #[test]
    fn zip_round_trip_deflated_and_stored() {
        let mut stream = TrxStream::<f32>::new(Header::identity_affine(), [100, 100, 100]);
        stream.push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let trx = stream.finalize();

        let dir = tempfile::TempDir::new().unwrap();
        let zip_path = dir.path().join("test_deflated.trx");

        save_to_zip_with(&trx, &zip_path, zip::CompressionMethod::Deflated).unwrap();
        let loaded = load_from_zip::<f32>(&zip_path).unwrap();

        assert_eq!(loaded.nb_streamlines(), 1);
        assert_eq!(loaded.nb_vertices(), 2);
        assert_eq!(loaded.streamline(0), trx.streamline(0));
    }

    #[test]
    fn zip_direct_mapping_without_temp_dir() {
        let mut stream = TrxStream::<f32>::new(Header::identity_affine(), [100, 100, 100]);
        stream.push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let trx = stream.finalize();

        let dir = tempfile::TempDir::new().unwrap();
        let zip_path = dir.path().join("stored.trx");

        save_to_zip_with(&trx, &zip_path, zip::CompressionMethod::Stored).unwrap();
        let loaded = load_from_zip::<f32>(&zip_path).unwrap();

        assert!(loaded.is_file_backed());
        assert_eq!(loaded.nb_streamlines(), 1);
        assert_eq!(loaded.nb_vertices(), 2);
        assert_eq!(loaded.streamline(0), trx.streamline(0));
    }

    #[test]
    fn zip_unaligned_offsets_size_errors() {
        let dir = tempfile::TempDir::new().unwrap();
        let zip_path = dir.path().join("corrupt_offsets.trx");

        let file = std::fs::File::create(&zip_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);

        let header = Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [100, 100, 100],
            nb_streamlines: 0,
            nb_vertices: 0,
            extra: Default::default(),
        };
        let json = serde_json::to_string(&header).unwrap();
        zip.start_file("header.json", zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(json.as_bytes()).unwrap();

        // Write positions.bit32.ncols3.raw (0 vertices)
        zip.start_file("positions.bit32.ncols3.raw", zip::write::SimpleFileOptions::default()).unwrap();

        // Write unaligned offsets (e.g. 5 bytes instead of multiple of 4 or 8)
        zip.start_file("offsets.bit32.ncols1.raw", zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(&[0u8; 5]).unwrap();

        zip.finish().unwrap();

        let res = load_from_zip::<f32>(&zip_path);
        assert!(res.is_err());
    }
}
