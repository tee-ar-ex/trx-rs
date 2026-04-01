use std::collections::{BTreeSet, HashMap};
use std::fs::{self, OpenOptions};
use std::path::Path;

use half::f16;
use memmap2::MmapOptions;

use crate::any_trx_file::{AnyTrxFile, PositionsRef};
use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::mmap_backing::MmapBacking;
use crate::trx_file::{DataArray, DataArrayInfo, FromF32, TrxFile, TrxParts};

#[derive(Clone, Debug, Default)]
pub struct ConcatenateOptions {
    pub delete_dpv: bool,
    pub delete_dps: bool,
    pub delete_groups: bool,
    pub positions_dtype: Option<DType>,
}

#[derive(Clone, Copy, Debug)]
struct ArraySpec {
    ncols: usize,
    dtype: DType,
}

#[derive(Clone, Copy, Debug)]
struct GroupSpec {
    dtype: DType,
    len: usize,
}

/// Concatenate multiple TRX files with Python-like semantics.
///
/// Header spatial metadata are copied from the first input without validation.
/// `dpg` is intentionally dropped to match `trx-python` concatenate behavior.
pub fn concatenate_any_trx(
    inputs: &[&AnyTrxFile],
    options: &ConcatenateOptions,
) -> Result<AnyTrxFile> {
    if inputs.is_empty() {
        return Err(TrxError::Argument("no shards to merge".into()));
    }

    let first = inputs[0];
    let target_dtype = options.positions_dtype.unwrap_or_else(|| first.dtype());
    let total_streamlines = inputs.iter().map(|input| input.nb_streamlines()).sum();
    let total_vertices = inputs.iter().map(|input| input.nb_vertices()).sum();

    let dps_specs = retained_array_specs(inputs, ArrayKind::Dps, options.delete_dps)?;
    let dpv_specs = retained_array_specs(inputs, ArrayKind::Dpv, options.delete_dpv)?;
    let group_specs = retained_group_specs(inputs)?;

    match target_dtype {
        DType::Float16 => concatenate_into::<f16>(
            inputs,
            total_streamlines,
            total_vertices,
            dps_specs,
            dpv_specs,
            group_specs,
            options,
        )
        .map(AnyTrxFile::F16),
        DType::Float32 => concatenate_into::<f32>(
            inputs,
            total_streamlines,
            total_vertices,
            dps_specs,
            dpv_specs,
            group_specs,
            options,
        )
        .map(AnyTrxFile::F32),
        DType::Float64 => concatenate_into::<f64>(
            inputs,
            total_streamlines,
            total_vertices,
            dps_specs,
            dpv_specs,
            group_specs,
            options,
        )
        .map(AnyTrxFile::F64),
        other => Err(TrxError::DType(format!(
            "TRX positions must be float16, float32, or float64, got {other}"
        ))),
    }
}

/// Merge multiple typed TRX files into one using file-backed output.
pub fn merge_trx_shards<P: TrxScalar>(shards: &[&TrxFile<P>]) -> Result<TrxFile<P>> {
    if shards.is_empty() {
        return Err(TrxError::Argument("no shards to merge".into()));
    }

    let total_streamlines = shards.iter().map(|input| input.nb_streamlines()).sum();
    let total_vertices = shards.iter().map(|input| input.nb_vertices()).sum();
    let dps_specs = retained_array_specs_typed(shards, ArrayKind::Dps, false)?;
    let dpv_specs = retained_array_specs_typed(shards, ArrayKind::Dpv, false)?;
    let group_specs = retained_group_specs_typed(shards)?;

    concatenate_typed_from_trx(
        shards,
        total_streamlines,
        total_vertices,
        dps_specs,
        dpv_specs,
        group_specs,
        &ConcatenateOptions {
            positions_dtype: Some(P::DTYPE),
            ..Default::default()
        },
    )
}

fn concatenate_into<P>(
    inputs: &[&AnyTrxFile],
    total_streamlines: usize,
    total_vertices: usize,
    dps_specs: HashMap<String, ArraySpec>,
    dpv_specs: HashMap<String, ArraySpec>,
    group_specs: HashMap<String, GroupSpec>,
    options: &ConcatenateOptions,
) -> Result<TrxFile<P>>
where
    P: TrxScalar + FromF32,
{
    let mut header = inputs[0].header().clone();
    header.nb_streamlines = total_streamlines as u64;
    header.nb_vertices = total_vertices as u64;

    let tempdir = tempfile::TempDir::new()?;
    let tempdir_path = tempdir.path().to_path_buf();

    let mut positions_backing = create_mmap_backing(
        &tempdir_path.join(format!("positions.3.{}", P::DTYPE.name())),
        total_vertices * 3 * std::mem::size_of::<P>(),
    )?;
    let mut offsets_backing =
        create_mmap_backing(&tempdir_path.join("offsets.uint32"), (total_streamlines + 1) * 4)?;
    let mut dps = create_data_map(&tempdir_path.join("dps"), &dps_specs, total_streamlines)?;
    let mut dpv = create_data_map(&tempdir_path.join("dpv"), &dpv_specs, total_vertices)?;
    let mut groups = if options.delete_groups {
        HashMap::new()
    } else {
        create_group_map(&tempdir_path.join("groups"), &group_specs)?
    };

    fill_positions::<P>(inputs, &mut positions_backing)?;
    fill_offsets(inputs, &mut offsets_backing)?;
    copy_retained_arrays(inputs, &dps_specs, &mut dps, ArrayKind::Dps)?;
    copy_retained_arrays(inputs, &dpv_specs, &mut dpv, ArrayKind::Dpv)?;
    if !options.delete_groups {
        copy_groups(inputs, &group_specs, &mut groups)?;
    }

    Ok(TrxFile::from_parts(TrxParts {
        header,
        positions_backing,
        offsets_backing,
        dps,
        dpv,
        groups,
        dpg: HashMap::new(),
        tempdir: Some(tempdir),
    }))
}

fn concatenate_typed_from_trx<P>(
    inputs: &[&TrxFile<P>],
    total_streamlines: usize,
    total_vertices: usize,
    dps_specs: HashMap<String, ArraySpec>,
    dpv_specs: HashMap<String, ArraySpec>,
    group_specs: HashMap<String, GroupSpec>,
    options: &ConcatenateOptions,
) -> Result<TrxFile<P>>
where
    P: TrxScalar,
{
    let mut header = inputs[0].header().clone();
    header.nb_streamlines = total_streamlines as u64;
    header.nb_vertices = total_vertices as u64;

    let tempdir = tempfile::TempDir::new()?;
    let tempdir_path = tempdir.path().to_path_buf();

    let mut positions_backing = create_mmap_backing(
        &tempdir_path.join(format!("positions.3.{}", P::DTYPE.name())),
        total_vertices * 3 * std::mem::size_of::<P>(),
    )?;
    let mut offsets_backing =
        create_mmap_backing(&tempdir_path.join("offsets.uint32"), (total_streamlines + 1) * 4)?;
    let mut dps = create_data_map(&tempdir_path.join("dps"), &dps_specs, total_streamlines)?;
    let mut dpv = create_data_map(&tempdir_path.join("dpv"), &dpv_specs, total_vertices)?;
    let mut groups = if options.delete_groups {
        HashMap::new()
    } else {
        create_group_map(&tempdir_path.join("groups"), &group_specs)?
    };

    fill_positions_typed(inputs, &mut positions_backing)?;
    fill_offsets_typed(inputs, &mut offsets_backing)?;
    copy_retained_arrays_typed(inputs, &dps_specs, &mut dps, ArrayKind::Dps)?;
    copy_retained_arrays_typed(inputs, &dpv_specs, &mut dpv, ArrayKind::Dpv)?;
    if !options.delete_groups {
        copy_groups_typed(inputs, &group_specs, &mut groups)?;
    }

    Ok(TrxFile::from_parts(TrxParts {
        header,
        positions_backing,
        offsets_backing,
        dps,
        dpv,
        groups,
        dpg: HashMap::new(),
        tempdir: Some(tempdir),
    }))
}

#[derive(Clone, Copy)]
enum ArrayKind {
    Dps,
    Dpv,
}

fn retained_array_specs(
    inputs: &[&AnyTrxFile],
    kind: ArrayKind,
    delete_all: bool,
) -> Result<HashMap<String, ArraySpec>> {
    let union_keys = array_keys_union(inputs, kind);
    let reference = array_infos(inputs[0], kind);

    for input in inputs.iter().skip(1) {
        let current = array_infos(input, kind);
        for key in &union_keys {
            match (reference.get(key), current.get(key)) {
                (Some(lhs), Some(rhs)) => ensure_matching_array_info(kind, key, *lhs, *rhs)?,
                _ if delete_all => {}
                _ => {
                    return Err(TrxError::Argument(format!(
                        "{} key '{key}' does not exist in all TrxFile inputs",
                        array_kind_name(kind)
                    )))
                }
            }
        }
    }

    if delete_all {
        return Ok(HashMap::new());
    }

    Ok(reference
        .into_iter()
        .map(|(name, info)| {
            (
                name,
                ArraySpec {
                    ncols: info.ncols,
                    dtype: info.dtype,
                },
            )
        })
        .collect())
}

fn retained_group_specs(inputs: &[&AnyTrxFile]) -> Result<HashMap<String, GroupSpec>> {
    let mut specs: HashMap<String, GroupSpec> = HashMap::new();
    let mut lengths: HashMap<String, usize> = HashMap::new();

    for input in inputs {
        let groups = group_infos(input);
        for (name, info) in groups {
            match specs.get(&name) {
                Some(existing) if existing.dtype != info.dtype => {
                    return Err(TrxError::Argument(format!(
                        "group key '{name}' has different dtypes across inputs"
                    )))
                }
                Some(_) => {}
                None => {
                    specs.insert(
                        name.clone(),
                        GroupSpec {
                            dtype: info.dtype,
                            len: 0,
                        },
                    );
                }
            }
            *lengths.entry(name).or_insert(0usize) += info.nrows;
        }
    }

    for (name, len) in lengths {
        specs
            .get_mut(&name)
            .expect("lengths and specs must be in sync")
            .len = len;
    }

    Ok(specs)
}

fn retained_array_specs_typed<P: TrxScalar>(
    inputs: &[&TrxFile<P>],
    kind: ArrayKind,
    delete_all: bool,
) -> Result<HashMap<String, ArraySpec>> {
    let union_keys: BTreeSet<String> = inputs
        .iter()
        .flat_map(|input| match kind {
            ArrayKind::Dps => input.dps_arrays().keys(),
            ArrayKind::Dpv => input.dpv_arrays().keys(),
        })
        .cloned()
        .collect();
    let reference: HashMap<String, DataArrayInfo> = match kind {
        ArrayKind::Dps => inputs[0]
            .dps_arrays()
            .iter()
            .map(|(name, arr)| (name.clone(), arr.info()))
            .collect(),
        ArrayKind::Dpv => inputs[0]
            .dpv_arrays()
            .iter()
            .map(|(name, arr)| (name.clone(), arr.info()))
            .collect(),
    };

    for input in inputs.iter().skip(1) {
        let current: HashMap<String, DataArrayInfo> = match kind {
            ArrayKind::Dps => input
                .dps_arrays()
                .iter()
                .map(|(name, arr)| (name.clone(), arr.info()))
                .collect(),
            ArrayKind::Dpv => input
                .dpv_arrays()
                .iter()
                .map(|(name, arr)| (name.clone(), arr.info()))
                .collect(),
        };
        for key in &union_keys {
            match (reference.get(key), current.get(key)) {
                (Some(lhs), Some(rhs)) => ensure_matching_array_info(kind, key, *lhs, *rhs)?,
                _ if delete_all => {}
                _ => {
                    return Err(TrxError::Argument(format!(
                        "{} key '{key}' does not exist in all TrxFile inputs",
                        array_kind_name(kind)
                    )))
                }
            }
        }
    }

    if delete_all {
        return Ok(HashMap::new());
    }

    Ok(reference
        .into_iter()
        .map(|(name, info)| {
            (
                name,
                ArraySpec {
                    ncols: info.ncols,
                    dtype: info.dtype,
                },
            )
        })
        .collect())
}

fn retained_group_specs_typed<P: TrxScalar>(inputs: &[&TrxFile<P>]) -> Result<HashMap<String, GroupSpec>> {
    let mut specs: HashMap<String, GroupSpec> = HashMap::new();
    let mut lengths: HashMap<String, usize> = HashMap::new();

    for input in inputs {
        for (name, arr) in input.group_arrays() {
            match specs.get(name) {
                Some(existing) if existing.dtype != arr.dtype() => {
                    return Err(TrxError::Argument(format!(
                        "group key '{name}' has different dtypes across inputs"
                    )))
                }
                Some(_) => {}
                None => {
                    specs.insert(
                        name.clone(),
                        GroupSpec {
                            dtype: arr.dtype(),
                            len: 0,
                        },
                    );
                }
            }
            *lengths.entry(name.clone()).or_insert(0usize) += arr.nrows();
        }
    }

    for (name, len) in lengths {
        specs
            .get_mut(&name)
            .expect("lengths and specs must be in sync")
            .len = len;
    }
    Ok(specs)
}

fn array_keys_union(inputs: &[&AnyTrxFile], kind: ArrayKind) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for input in inputs {
        for key in array_infos(input, kind).into_keys() {
            keys.insert(key);
        }
    }
    keys
}

fn array_infos(file: &AnyTrxFile, kind: ArrayKind) -> HashMap<String, DataArrayInfo> {
    match kind {
        ArrayKind::Dps => file.dps_entries().into_iter().collect(),
        ArrayKind::Dpv => file.dpv_entries().into_iter().collect(),
    }
}

fn group_infos(file: &AnyTrxFile) -> HashMap<String, DataArrayInfo> {
    file.with_typed(
        |trx| {
            trx.group_arrays()
                .iter()
                .map(|(name, arr)| (name.clone(), arr.info()))
                .collect()
        },
        |trx| {
            trx.group_arrays()
                .iter()
                .map(|(name, arr)| (name.clone(), arr.info()))
                .collect()
        },
        |trx| {
            trx.group_arrays()
                .iter()
                .map(|(name, arr)| (name.clone(), arr.info()))
                .collect()
        },
    )
}

fn ensure_matching_array_info(
    kind: ArrayKind,
    key: &str,
    lhs: DataArrayInfo,
    rhs: DataArrayInfo,
) -> Result<()> {
    if lhs.dtype != rhs.dtype {
        return Err(TrxError::Argument(format!(
            "{} key '{key}' has different dtypes across inputs",
            array_kind_name(kind)
        )));
    }
    if lhs.ncols != rhs.ncols {
        return Err(TrxError::Argument(format!(
            "{} key '{key}' has different column counts across inputs",
            array_kind_name(kind)
        )));
    }
    Ok(())
}

fn array_kind_name(kind: ArrayKind) -> &'static str {
    match kind {
        ArrayKind::Dps => "dps",
        ArrayKind::Dpv => "dpv",
    }
}

fn create_mmap_backing(path: &Path, len: usize) -> Result<MmapBacking> {
    if len == 0 {
        return Ok(MmapBacking::Owned(Vec::new()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.set_len(len as u64)?;
    let mmap = unsafe { MmapOptions::new().len(len).map_mut(&file)? };
    Ok(MmapBacking::ReadWrite(mmap))
}

fn create_data_map(
    dir: &Path,
    specs: &HashMap<String, ArraySpec>,
    rows: usize,
) -> Result<HashMap<String, DataArray>> {
    let mut out = HashMap::new();
    if specs.is_empty() {
        return Ok(out);
    }
    fs::create_dir_all(dir)?;
    for (name, spec) in specs {
        let filename = crate::io::filename::TrxFilename {
            name: name.clone(),
            ncols: spec.ncols,
            dtype: spec.dtype,
        }
        .to_filename();
        let len = rows
            .checked_mul(spec.ncols)
            .and_then(|v| v.checked_mul(spec.dtype.size_of()))
            .ok_or_else(|| TrxError::Argument(format!("array '{name}' is too large")))?;
        out.insert(
            name.clone(),
            DataArray::from_backing(create_mmap_backing(&dir.join(filename), len)?, spec.ncols, spec.dtype),
        );
    }
    Ok(out)
}

fn create_group_map(dir: &Path, specs: &HashMap<String, GroupSpec>) -> Result<HashMap<String, DataArray>> {
    let mut out = HashMap::new();
    if specs.is_empty() {
        return Ok(out);
    }
    fs::create_dir_all(dir)?;
    for (name, spec) in specs {
        let filename = crate::io::filename::TrxFilename {
            name: name.clone(),
            ncols: 1,
            dtype: spec.dtype,
        }
        .to_filename();
        let len = spec
            .len
            .checked_mul(spec.dtype.size_of())
            .ok_or_else(|| TrxError::Argument(format!("group '{name}' is too large")))?;
        out.insert(
            name.clone(),
            DataArray::from_backing(create_mmap_backing(&dir.join(filename), len)?, 1, spec.dtype),
        );
    }
    Ok(out)
}

fn fill_positions<P>(inputs: &[&AnyTrxFile], backing: &mut MmapBacking) -> Result<()>
where
    P: TrxScalar + FromF32,
{
    let dst: &mut [[P; 3]] = backing.cast_slice_mut()?;
    let mut cursor = 0usize;
    for input in inputs {
        let count = input.nb_vertices();
        let target = &mut dst[cursor..cursor + count];
        match input.positions_ref() {
            PositionsRef::F16(src) => copy_positions(src, target),
            PositionsRef::F32(src) => copy_positions(src, target),
            PositionsRef::F64(src) => copy_positions(src, target),
        }
        cursor += count;
    }
    Ok(())
}

fn copy_positions<Src, Dst>(src: &[[Src; 3]], dst: &mut [[Dst; 3]])
where
    Src: TrxScalar,
    Dst: TrxScalar + FromF32,
{
    for (src_row, dst_row) in src.iter().zip(dst.iter_mut()) {
        dst_row[0] = Dst::from_f32(src_row[0].to_f32());
        dst_row[1] = Dst::from_f32(src_row[1].to_f32());
        dst_row[2] = Dst::from_f32(src_row[2].to_f32());
    }
}

fn fill_offsets(inputs: &[&AnyTrxFile], backing: &mut MmapBacking) -> Result<()> {
    let dst: &mut [u32] = backing.cast_slice_mut()?;
    let mut cursor = 0usize;
    let mut vertex_base = 0u32;
    for input in inputs {
        for offset in input.offsets_vec().into_iter().take(input.nb_streamlines()) {
            dst[cursor] = offset
                .checked_add(vertex_base)
                .ok_or_else(|| TrxError::Argument("offset overflow during concatenate".into()))?;
            cursor += 1;
        }
        vertex_base = vertex_base
            .checked_add(input.nb_vertices() as u32)
            .ok_or_else(|| TrxError::Argument("vertex count overflow during concatenate".into()))?;
    }
    dst[cursor] = vertex_base;
    Ok(())
}

fn fill_positions_typed<P>(inputs: &[&TrxFile<P>], backing: &mut MmapBacking) -> Result<()>
where
    P: TrxScalar,
{
    let dst: &mut [[P; 3]] = backing.cast_slice_mut()?;
    let mut cursor = 0usize;
    for input in inputs {
        let count = input.nb_vertices();
        dst[cursor..cursor + count].copy_from_slice(input.positions());
        cursor += count;
    }
    Ok(())
}

fn fill_offsets_typed<P: TrxScalar>(inputs: &[&TrxFile<P>], backing: &mut MmapBacking) -> Result<()> {
    let dst: &mut [u32] = backing.cast_slice_mut()?;
    let mut cursor = 0usize;
    let mut vertex_base = 0u32;
    for input in inputs {
        for &offset in input.offsets().iter().take(input.nb_streamlines()) {
            dst[cursor] = offset
                .checked_add(vertex_base)
                .ok_or_else(|| TrxError::Argument("offset overflow during concatenate".into()))?;
            cursor += 1;
        }
        vertex_base = vertex_base
            .checked_add(input.nb_vertices() as u32)
            .ok_or_else(|| TrxError::Argument("vertex count overflow during concatenate".into()))?;
    }
    dst[cursor] = vertex_base;
    Ok(())
}

fn copy_retained_arrays(
    inputs: &[&AnyTrxFile],
    specs: &HashMap<String, ArraySpec>,
    outputs: &mut HashMap<String, DataArray>,
    kind: ArrayKind,
) -> Result<()> {
    for (name, spec) in specs {
        let row_bytes = spec.ncols * spec.dtype.size_of();
        let dst = outputs
            .get_mut(name)
            .expect("output map should contain every retained array")
            .as_bytes_mut()?;
        let mut byte_cursor = 0usize;
        for input in inputs {
            let src = array_map(input, kind)
                .get(name)
                .expect("retained array must exist in every input");
            let bytes = src.as_bytes();
            let len = src
                .nrows()
                .checked_mul(row_bytes)
                .ok_or_else(|| TrxError::Argument(format!("array '{name}' is too large")))?;
            dst[byte_cursor..byte_cursor + len].copy_from_slice(bytes);
            byte_cursor += len;
        }
    }
    Ok(())
}

fn copy_retained_arrays_typed<P: TrxScalar>(
    inputs: &[&TrxFile<P>],
    specs: &HashMap<String, ArraySpec>,
    outputs: &mut HashMap<String, DataArray>,
    kind: ArrayKind,
) -> Result<()> {
    for (name, spec) in specs {
        let row_bytes = spec.ncols * spec.dtype.size_of();
        let dst = outputs
            .get_mut(name)
            .expect("output map should contain every retained array")
            .as_bytes_mut()?;
        let mut byte_cursor = 0usize;
        for input in inputs {
            let src = match kind {
                ArrayKind::Dps => input.dps_arrays().get(name),
                ArrayKind::Dpv => input.dpv_arrays().get(name),
            }
            .expect("retained array must exist in every input");
            let bytes = src.as_bytes();
            let len = src
                .nrows()
                .checked_mul(row_bytes)
                .ok_or_else(|| TrxError::Argument(format!("array '{name}' is too large")))?;
            dst[byte_cursor..byte_cursor + len].copy_from_slice(bytes);
            byte_cursor += len;
        }
    }
    Ok(())
}

fn copy_groups(
    inputs: &[&AnyTrxFile],
    specs: &HashMap<String, GroupSpec>,
    outputs: &mut HashMap<String, DataArray>,
) -> Result<()> {
    let mut positions: HashMap<String, usize> = specs.keys().map(|name| (name.clone(), 0)).collect();
    let mut streamline_base = 0u32;
    for input in inputs {
        let src_groups = group_map(input);
        for (name, arr) in src_groups {
            let cursor = positions
                .get_mut(name)
                .expect("output positions must exist for every group");
            let dst = outputs
                .get_mut(name)
                .expect("output arrays must exist for every group");
            let count = arr.nrows();
            copy_group_with_offset(arr, *cursor, streamline_base, dst)?;
            *cursor += count;
        }
        streamline_base = streamline_base
            .checked_add(input.nb_streamlines() as u32)
            .ok_or_else(|| TrxError::Argument("streamline count overflow during concatenate".into()))?;
    }
    Ok(())
}

fn copy_groups_typed<P: TrxScalar>(
    inputs: &[&TrxFile<P>],
    specs: &HashMap<String, GroupSpec>,
    outputs: &mut HashMap<String, DataArray>,
) -> Result<()> {
    let mut positions: HashMap<String, usize> = specs.keys().map(|name| (name.clone(), 0)).collect();
    let mut streamline_base = 0u32;
    for input in inputs {
        for (name, arr) in input.group_arrays() {
            let cursor = positions
                .get_mut(name)
                .expect("output positions must exist for every group");
            let dst = outputs
                .get_mut(name)
                .expect("output arrays must exist for every group");
            let count = arr.nrows();
            copy_group_with_offset(arr, *cursor, streamline_base, dst)?;
            *cursor += count;
        }
        streamline_base = streamline_base
            .checked_add(input.nb_streamlines() as u32)
            .ok_or_else(|| TrxError::Argument("streamline count overflow during concatenate".into()))?;
    }
    Ok(())
}

fn copy_group_with_offset(
    src: &DataArray,
    dst_start: usize,
    streamline_base: u32,
    dst: &mut DataArray,
) -> Result<()> {
    match src.dtype() {
        DType::Int8 => copy_group_typed::<i8>(src.cast_slice(), dst_start, streamline_base, dst),
        DType::Int16 => copy_group_typed::<i16>(src.cast_slice(), dst_start, streamline_base, dst),
        DType::Int32 => copy_group_typed::<i32>(src.cast_slice(), dst_start, streamline_base, dst),
        DType::Int64 => copy_group_typed::<i64>(src.cast_slice(), dst_start, streamline_base, dst),
        DType::UInt8 => copy_group_typed::<u8>(src.cast_slice(), dst_start, streamline_base, dst),
        DType::UInt16 => copy_group_typed::<u16>(src.cast_slice(), dst_start, streamline_base, dst),
        DType::UInt32 => copy_group_typed::<u32>(src.cast_slice(), dst_start, streamline_base, dst),
        DType::UInt64 => copy_group_typed::<u64>(src.cast_slice(), dst_start, streamline_base, dst),
        other => Err(TrxError::DType(format!(
            "group arrays must use integer dtype, got {other}"
        ))),
    }
}

fn copy_group_typed<T>(
    src: &[T],
    dst_start: usize,
    streamline_base: u32,
    dst: &mut DataArray,
) -> Result<()>
where
    T: Copy + TryFrom<u64> + Into<i128> + bytemuck::Pod,
    <T as TryFrom<u64>>::Error: std::fmt::Display,
{
    let dst_values: &mut [T] = dst.cast_slice_mut()?;
    let addend = i128::from(streamline_base);
    for (index, &value) in src.iter().enumerate() {
        let current = value.into();
        let shifted = current + addend;
        let shifted_u64 = u64::try_from(shifted).map_err(|_| {
            TrxError::Argument("group index underflow during concatenate".into())
        })?;
        dst_values[dst_start + index] = T::try_from(shifted_u64).map_err(|err| {
            TrxError::Argument(format!("group index overflow during concatenate: {err}"))
        })?;
    }
    Ok(())
}

fn array_map(file: &AnyTrxFile, kind: ArrayKind) -> &HashMap<String, DataArray> {
    match file {
        AnyTrxFile::F16(trx) => match kind {
            ArrayKind::Dps => trx.dps_arrays(),
            ArrayKind::Dpv => trx.dpv_arrays(),
        },
        AnyTrxFile::F32(trx) => match kind {
            ArrayKind::Dps => trx.dps_arrays(),
            ArrayKind::Dpv => trx.dpv_arrays(),
        },
        AnyTrxFile::F64(trx) => match kind {
            ArrayKind::Dps => trx.dps_arrays(),
            ArrayKind::Dpv => trx.dpv_arrays(),
        },
    }
}

fn group_map(file: &AnyTrxFile) -> &HashMap<String, DataArray> {
    match file {
        AnyTrxFile::F16(trx) => trx.group_arrays(),
        AnyTrxFile::F32(trx) => trx.group_arrays(),
        AnyTrxFile::F64(trx) => trx.group_arrays(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::mmap_backing::vec_to_bytes;

    fn sample_header() -> Header {
        Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [10, 20, 30],
            nb_streamlines: 0,
            nb_vertices: 0,
            extra: Default::default(),
        }
    }

    fn build_trx(
        positions: Vec<[f32; 3]>,
        offsets: Vec<u32>,
        dps: HashMap<String, DataArray>,
        dpv: HashMap<String, DataArray>,
        groups: HashMap<String, DataArray>,
        dpg: HashMap<String, HashMap<String, DataArray>>,
    ) -> TrxFile<f32> {
        let mut header = sample_header();
        header.nb_streamlines = offsets.len().saturating_sub(1) as u64;
        header.nb_vertices = positions.len() as u64;
        TrxFile::from_parts(TrxParts {
            header,
            positions_backing: MmapBacking::Owned(vec_to_bytes(positions)),
            offsets_backing: MmapBacking::Owned(vec_to_bytes(offsets)),
            dps,
            dpv,
            groups,
            dpg,
            tempdir: None,
        })
    }

    fn scalar_u32(value: u32) -> DataArray {
        DataArray::owned_bytes(vec_to_bytes(vec![value]), 1, DType::UInt32)
    }

    fn scalar_f32(value: f32) -> DataArray {
        DataArray::owned_bytes(vec_to_bytes(vec![value]), 1, DType::Float32)
    }

    fn vertex_f32(values: Vec<f32>) -> DataArray {
        DataArray::owned_bytes(vec_to_bytes(values), 1, DType::Float32)
    }

    #[test]
    fn concatenate_preserves_counts_and_group_union_and_drops_dpg() {
        let mut groups_a = HashMap::new();
        groups_a.insert(
            "left".into(),
            DataArray::owned_bytes(vec_to_bytes(vec![0u32]), 1, DType::UInt32),
        );
        let mut dpg_a = HashMap::new();
        dpg_a.insert(
            "left".into(),
            HashMap::from([("color".into(), DataArray::owned_bytes(vec![1, 2, 3], 3, DType::UInt8))]),
        );

        let a = build_trx(
            vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            vec![0, 2],
            HashMap::from([("weight".into(), scalar_f32(1.0))]),
            HashMap::from([("fa".into(), vertex_f32(vec![0.1, 0.2]))]),
            groups_a,
            dpg_a,
        );

        let mut groups_b = HashMap::new();
        groups_b.insert(
            "right".into(),
            DataArray::owned_bytes(vec_to_bytes(vec![0u32]), 1, DType::UInt32),
        );
        let b = build_trx(
            vec![[2.0, 2.0, 2.0]],
            vec![0, 1],
            HashMap::from([("weight".into(), scalar_f32(2.0))]),
            HashMap::from([("fa".into(), vertex_f32(vec![0.3]))]),
            groups_b,
            HashMap::new(),
        );

        let any_a = AnyTrxFile::F32(a);
        let any_b = AnyTrxFile::F32(b);
        let merged = concatenate_any_trx(&[&any_a, &any_b], &ConcatenateOptions::default()).unwrap();

        match merged {
            AnyTrxFile::F32(trx) => {
                assert_eq!(trx.nb_streamlines(), 2);
                assert_eq!(trx.nb_vertices(), 3);
                assert_eq!(trx.header().dimensions, [10, 20, 30]);
                assert!(trx.is_file_backed());
                assert_eq!(trx.group("left").unwrap(), &[0]);
                assert_eq!(trx.group("right").unwrap(), &[1]);
                assert!(trx.dpg_group_names().is_empty());
            }
            _ => panic!("expected float32 output"),
        }
    }

    #[test]
    fn concatenate_errors_on_missing_dps_without_delete_flag() {
        let a = build_trx(
            vec![[0.0, 0.0, 0.0]],
            vec![0, 1],
            HashMap::from([("weight".into(), scalar_f32(1.0))]),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let b = build_trx(
            vec![[1.0, 1.0, 1.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let any_a = AnyTrxFile::F32(a);
        let any_b = AnyTrxFile::F32(b);
        let err = concatenate_any_trx(&[&any_a, &any_b], &ConcatenateOptions::default()).unwrap_err();
        assert!(err.to_string().contains("dps key 'weight'"));
    }

    #[test]
    fn concatenate_delete_dps_drops_category() {
        let a = build_trx(
            vec![[0.0, 0.0, 0.0]],
            vec![0, 1],
            HashMap::from([("weight".into(), scalar_f32(1.0))]),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let b = build_trx(
            vec![[1.0, 1.0, 1.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let any_a = AnyTrxFile::F32(a);
        let any_b = AnyTrxFile::F32(b);
        let merged = concatenate_any_trx(
            &[&any_a, &any_b],
            &ConcatenateOptions {
                delete_dps: true,
                ..Default::default()
            },
        )
        .unwrap();
        match merged {
            AnyTrxFile::F32(trx) => assert!(trx.dps_names().is_empty()),
            _ => panic!("expected float32 output"),
        }
    }

    #[test]
    fn concatenate_errors_on_dpv_column_mismatch() {
        let a = build_trx(
            vec![[0.0, 0.0, 0.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::from([(
                "fa".into(),
                DataArray::owned_bytes(vec_to_bytes(vec![0.1f32]), 1, DType::Float32),
            )]),
            HashMap::new(),
            HashMap::new(),
        );
        let b = build_trx(
            vec![[1.0, 1.0, 1.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::from([(
                "fa".into(),
                DataArray::owned_bytes(vec_to_bytes(vec![0.2f32, 0.3f32]), 2, DType::Float32),
            )]),
            HashMap::new(),
            HashMap::new(),
        );
        let any_a = AnyTrxFile::F32(a);
        let any_b = AnyTrxFile::F32(b);
        let err = concatenate_any_trx(&[&any_a, &any_b], &ConcatenateOptions::default()).unwrap_err();
        assert!(err.to_string().contains("column counts"));
    }

    #[test]
    fn concatenate_errors_on_group_dtype_mismatch() {
        let a = build_trx(
            vec![[0.0, 0.0, 0.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::new(),
            HashMap::from([("bundle".into(), scalar_u32(0))]),
            HashMap::new(),
        );
        let b = build_trx(
            vec![[1.0, 1.0, 1.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::new(),
            HashMap::from([(
                "bundle".into(),
                DataArray::owned_bytes(vec_to_bytes(vec![0u16]), 1, DType::UInt16),
            )]),
            HashMap::new(),
        );
        let any_a = AnyTrxFile::F32(a);
        let any_b = AnyTrxFile::F32(b);
        let err = concatenate_any_trx(&[&any_a, &any_b], &ConcatenateOptions::default()).unwrap_err();
        assert!(err.to_string().contains("group key 'bundle'"));
    }

    #[test]
    fn concatenate_respects_positions_dtype_override() {
        let a = build_trx(
            vec![[0.0, 0.0, 0.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let b = build_trx(
            vec![[1.0, 1.0, 1.0]],
            vec![0, 1],
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let any_a = AnyTrxFile::F32(a);
        let any_b = AnyTrxFile::F32(b);
        let merged = concatenate_any_trx(
            &[&any_a, &any_b],
            &ConcatenateOptions {
                positions_dtype: Some(DType::Float16),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(merged, AnyTrxFile::F16(_)));
    }
}
