//! Copy metadata (DPS / DPV / groups, optionally DPG) from a donor TRX onto a
//! target TRX whose streamlines and vertices already match.
//!
//! Unlike [`concatenate_any_trx`](crate::concatenate_any_trx) and
//! [`subset_streamlines`](crate::subset_streamlines), this operation does not
//! touch positions, offsets, or the header — it only grafts named arrays from
//! the source onto the target's metadata `HashMap`s.

use std::collections::HashMap;

use crate::any_trx_file::AnyTrxFile;
use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::trx_file::{DataArray, DataPerGroup, TrxFile};

/// Options controlling [`copy_metadata_any_trx`].
///
/// When `dps`, `dpv`, and `groups` are all `None`, every array in each
/// category is copied. Once any of them is `Some`, only the kinds with an
/// explicit filter are copied; unfiltered kinds are skipped.
#[derive(Clone, Debug, Default)]
pub struct CopyMetadataOptions {
    /// Names of DPS arrays to copy, or `None` to fall back to the global
    /// "copy everything" rule described above.
    pub dps: Option<Vec<String>>,
    pub dpv: Option<Vec<String>>,
    pub groups: Option<Vec<String>>,
    /// If `true`, also copy data-per-group entries for the selected groups.
    pub copy_dpg: bool,
    /// If `true`, donor entries replace existing target entries with the same
    /// name. If `false`, name collisions are an error.
    pub overwrite_conflicting_metadata: bool,
    /// If `true`, donor arrays whose row count does not match the target's
    /// streamline / vertex count are skipped with a warning rather than
    /// aborting the operation.
    pub skip_mismatched: bool,
}

/// Copy metadata from `source` onto `target`, returning the modified target.
pub fn copy_metadata_any_trx(
    target: AnyTrxFile,
    source: &AnyTrxFile,
    opts: &CopyMetadataOptions,
) -> Result<AnyTrxFile> {
    match target {
        AnyTrxFile::F16(trx) => copy_metadata(trx, source, opts).map(AnyTrxFile::F16),
        AnyTrxFile::F32(trx) => copy_metadata(trx, source, opts).map(AnyTrxFile::F32),
        AnyTrxFile::F64(trx) => copy_metadata(trx, source, opts).map(AnyTrxFile::F64),
    }
}

/// Typed variant of [`copy_metadata_any_trx`].
pub fn copy_metadata<P: TrxScalar>(
    mut target: TrxFile<P>,
    source: &AnyTrxFile,
    opts: &CopyMetadataOptions,
) -> Result<TrxFile<P>> {
    let plan = SelectionPlan::from_options(opts);

    if (plan.copies(Kind::Dps) || plan.copies(Kind::Group))
        && target.nb_streamlines() != source.nb_streamlines()
    {
        return Err(TrxError::Argument(format!(
            "donor and target streamline counts differ ({} vs {}); cannot copy DPS or groups",
            source.nb_streamlines(),
            target.nb_streamlines()
        )));
    }
    if plan.copies(Kind::Dpv) && target.nb_vertices() != source.nb_vertices() {
        return Err(TrxError::Argument(format!(
            "donor and target vertex counts differ ({} vs {}); cannot copy DPV",
            source.nb_vertices(),
            target.nb_vertices()
        )));
    }

    let dps = collect_named(source, Kind::Dps, plan.filter(Kind::Dps))?;
    let dpv = collect_named(source, Kind::Dpv, plan.filter(Kind::Dpv))?;
    let groups = collect_named(source, Kind::Group, plan.filter(Kind::Group))?;

    let dps = filter_by_rows(dps, target.nb_streamlines(), Kind::Dps, opts.skip_mismatched)?;
    let dpv = filter_by_rows(dpv, target.nb_vertices(), Kind::Dpv, opts.skip_mismatched)?;
    validate_group_indices(&groups, target.nb_streamlines())?;

    if !opts.overwrite_conflicting_metadata {
        check_no_conflict(target.dps_arrays(), &dps, Kind::Dps)?;
        check_no_conflict(target.dpv_arrays(), &dpv, Kind::Dpv)?;
        check_no_conflict(target.group_arrays(), &groups, Kind::Group)?;
    }

    extend_map(target.dps_arrays_mut(), dps);
    extend_map(target.dpv_arrays_mut(), dpv);
    extend_map(target.group_arrays_mut(), groups);

    if opts.copy_dpg {
        let dpg = collect_dpg(source, plan.filter(Kind::Group));
        if !opts.overwrite_conflicting_metadata {
            check_no_dpg_conflict(target.dpg_arrays(), &dpg)?;
        }
        merge_dpg(target.dpg_arrays_mut(), dpg);
    }

    Ok(target)
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Kind {
    Dps,
    Dpv,
    Group,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Dps => "DPS",
            Kind::Dpv => "DPV",
            Kind::Group => "group",
        }
    }
}

/// What to copy, per kind: nothing, every entry, or a specific name list.
enum Selection<'a> {
    Skip,
    All,
    Named(&'a [String]),
}

struct SelectionPlan<'a> {
    dps: Selection<'a>,
    dpv: Selection<'a>,
    groups: Selection<'a>,
}

impl<'a> SelectionPlan<'a> {
    fn from_options(opts: &'a CopyMetadataOptions) -> Self {
        let any_filter = opts.dps.is_some() || opts.dpv.is_some() || opts.groups.is_some();
        Self {
            dps: Self::select(opts.dps.as_deref(), any_filter),
            dpv: Self::select(opts.dpv.as_deref(), any_filter),
            groups: Self::select(opts.groups.as_deref(), any_filter),
        }
    }

    fn select(filter: Option<&'a [String]>, any_filter: bool) -> Selection<'a> {
        match (filter, any_filter) {
            (Some(names), _) => Selection::Named(names),
            (None, false) => Selection::All,
            (None, true) => Selection::Skip,
        }
    }

    fn filter(&self, kind: Kind) -> &Selection<'_> {
        match kind {
            Kind::Dps => &self.dps,
            Kind::Dpv => &self.dpv,
            Kind::Group => &self.groups,
        }
    }

    fn copies(&self, kind: Kind) -> bool {
        !matches!(self.filter(kind), Selection::Skip)
    }
}

fn collect_named(
    source: &AnyTrxFile,
    kind: Kind,
    selection: &Selection<'_>,
) -> Result<Vec<(String, DataArray)>> {
    let pick = |arrays: &HashMap<String, DataArray>| -> Result<Vec<(String, DataArray)>> {
        match selection {
            Selection::Skip => Ok(Vec::new()),
            Selection::All => Ok(arrays
                .iter()
                .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                .collect()),
            Selection::Named(names) => names
                .iter()
                .map(|name| {
                    arrays
                        .get(name.as_str())
                        .map(|arr| (name.clone(), arr.clone_owned()))
                        .ok_or_else(|| {
                            TrxError::Argument(format!(
                                "donor has no {} named '{name}'",
                                kind.label()
                            ))
                        })
                })
                .collect(),
        }
    };

    source.with_typed(
        |s| pick(arrays_for(s, kind)),
        |s| pick(arrays_for(s, kind)),
        |s| pick(arrays_for(s, kind)),
    )
}

fn arrays_for<P: TrxScalar>(trx: &TrxFile<P>, kind: Kind) -> &HashMap<String, DataArray> {
    match kind {
        Kind::Dps => trx.dps_arrays(),
        Kind::Dpv => trx.dpv_arrays(),
        Kind::Group => trx.group_arrays(),
    }
}

fn filter_by_rows(
    entries: Vec<(String, DataArray)>,
    expected_rows: usize,
    kind: Kind,
    skip_mismatched: bool,
) -> Result<Vec<(String, DataArray)>> {
    entries
        .into_iter()
        .filter_map(|(name, arr)| {
            if arr.nrows() == expected_rows {
                return Some(Ok((name, arr)));
            }
            if skip_mismatched {
                eprintln!(
                    "warning: skipping {} '{name}' (rows {} != target {expected_rows})",
                    kind.label(),
                    arr.nrows()
                );
                return None;
            }
            Some(Err(TrxError::Argument(format!(
                "{} '{name}' has {} rows, target expects {expected_rows}",
                kind.label(),
                arr.nrows()
            ))))
        })
        .collect()
}

fn validate_group_indices(groups: &[(String, DataArray)], nb_streamlines: usize) -> Result<()> {
    for (name, arr) in groups {
        if let Some(max) = max_group_index(arr, name)? {
            if max >= nb_streamlines as u64 {
                return Err(TrxError::Argument(format!(
                    "group '{name}' references streamline index {max} but target has only \
                     {nb_streamlines} streamlines"
                )));
            }
        }
    }
    Ok(())
}

fn max_group_index(arr: &DataArray, name: &str) -> Result<Option<u64>> {
    match arr.dtype() {
        DType::Int8 => Ok(arr.cast_slice::<i8>().iter().map(|&v| v as i64).max().map(check_nonneg).transpose()?),
        DType::Int16 => Ok(arr.cast_slice::<i16>().iter().map(|&v| v as i64).max().map(check_nonneg).transpose()?),
        DType::Int32 => Ok(arr.cast_slice::<i32>().iter().map(|&v| v as i64).max().map(check_nonneg).transpose()?),
        DType::Int64 => Ok(arr.cast_slice::<i64>().iter().copied().max().map(check_nonneg).transpose()?),
        DType::UInt8 => Ok(arr.cast_slice::<u8>().iter().map(|&v| v as u64).max()),
        DType::UInt16 => Ok(arr.cast_slice::<u16>().iter().map(|&v| v as u64).max()),
        DType::UInt32 => Ok(arr.cast_slice::<u32>().iter().map(|&v| v as u64).max()),
        DType::UInt64 => Ok(arr.cast_slice::<u64>().iter().copied().max()),
        other => Err(TrxError::DType(format!(
            "group '{name}' uses non-integer dtype {other}"
        ))),
    }
}

fn check_nonneg(value: i64) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| TrxError::Argument(format!("group index {value} is negative")))
}

fn check_no_conflict(
    target: &HashMap<String, DataArray>,
    incoming: &[(String, DataArray)],
    kind: Kind,
) -> Result<()> {
    if let Some((name, _)) = incoming.iter().find(|(name, _)| target.contains_key(name)) {
        return Err(TrxError::Argument(format!(
            "{} '{name}' already exists in target; pass overwrite_conflicting_metadata to replace",
            kind.label()
        )));
    }
    Ok(())
}

fn extend_map(target: &mut HashMap<String, DataArray>, incoming: Vec<(String, DataArray)>) {
    target.extend(incoming);
}

fn collect_dpg(source: &AnyTrxFile, selection: &Selection<'_>) -> DataPerGroup {
    let pick = |dpg: &DataPerGroup| -> DataPerGroup {
        let want_group = |group: &str| -> bool {
            match selection {
                Selection::Skip => false,
                Selection::All => true,
                Selection::Named(names) => names.iter().any(|n| n == group),
            }
        };
        dpg.iter()
            .filter(|(group, _)| want_group(group.as_str()))
            .map(|(group, entries)| {
                let cloned = entries
                    .iter()
                    .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                    .collect();
                (group.clone(), cloned)
            })
            .collect()
    };

    source.with_typed(
        |s| pick(s.dpg_arrays()),
        |s| pick(s.dpg_arrays()),
        |s| pick(s.dpg_arrays()),
    )
}

fn check_no_dpg_conflict(target: &DataPerGroup, incoming: &DataPerGroup) -> Result<()> {
    for (group, entries) in incoming {
        let Some(existing) = target.get(group) else {
            continue;
        };
        if let Some(name) = entries.keys().find(|name| existing.contains_key(name.as_str())) {
            return Err(TrxError::Argument(format!(
                "DPG '{group}/{name}' already exists in target; pass \
                 overwrite_conflicting_metadata to replace"
            )));
        }
    }
    Ok(())
}

fn merge_dpg(target: &mut DataPerGroup, incoming: DataPerGroup) {
    for (group, entries) in incoming {
        target.entry(group).or_default().extend(entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::mmap_backing::{vec_to_bytes, MmapBacking};
    use crate::trx_file::TrxParts;

    fn header() -> Header {
        Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [10, 10, 10],
            nb_streamlines: 0,
            nb_vertices: 0,
            extra: Default::default(),
        }
    }

    fn build(positions: Vec<[f32; 3]>, offsets: Vec<u32>) -> TrxFile<f32> {
        let mut h = header();
        h.nb_streamlines = offsets.len().saturating_sub(1) as u64;
        h.nb_vertices = positions.len() as u64;
        TrxFile::from_parts(TrxParts {
            header: h,
            positions_backing: MmapBacking::Owned(vec_to_bytes(positions)),
            offsets_backing: MmapBacking::Owned(vec_to_bytes(offsets)),
            dps: HashMap::new(),
            dpv: HashMap::new(),
            groups: HashMap::new(),
            dpg: HashMap::new(),
            tempdir: None,
        })
    }

    fn scalar_f32(values: Vec<f32>) -> DataArray {
        DataArray::owned_bytes(vec_to_bytes(values), 1, DType::Float32)
    }

    fn group_u32(indices: Vec<u32>) -> DataArray {
        DataArray::owned_bytes(vec_to_bytes(indices), 1, DType::UInt32)
    }

    #[test]
    fn copies_all_metadata_when_no_filters() {
        let mut donor = build(vec![[0.0; 3], [1.0; 3], [2.0; 3]], vec![0, 2, 3]);
        donor.dps_arrays_mut().insert("weight".into(), scalar_f32(vec![0.5, 1.5]));
        donor.dpv_arrays_mut().insert("fa".into(), scalar_f32(vec![0.1, 0.2, 0.3]));
        donor.group_arrays_mut().insert("bundle".into(), group_u32(vec![0, 1]));

        let target = build(vec![[0.0; 3], [1.0; 3], [2.0; 3]], vec![0, 2, 3]);

        let merged = copy_metadata_any_trx(
            AnyTrxFile::F32(target),
            &AnyTrxFile::F32(donor),
            &CopyMetadataOptions::default(),
        )
        .unwrap();

        let AnyTrxFile::F32(out) = merged else {
            panic!("dtype changed");
        };
        assert_eq!(out.dps::<f32>("weight").unwrap().as_flat_slice(), &[0.5, 1.5]);
        assert_eq!(out.dpv::<f32>("fa").unwrap().as_flat_slice(), &[0.1, 0.2, 0.3]);
        assert_eq!(out.group("bundle").unwrap(), &[0, 1]);
    }

    #[test]
    fn streamline_count_mismatch_errors_when_dps_requested() {
        let mut donor = build(vec![[0.0; 3]], vec![0, 1]);
        donor.dps_arrays_mut().insert("w".into(), scalar_f32(vec![1.0]));
        let target = build(vec![[0.0; 3], [1.0; 3]], vec![0, 1, 2]);

        let err = copy_metadata_any_trx(
            AnyTrxFile::F32(target),
            &AnyTrxFile::F32(donor),
            &CopyMetadataOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("streamline counts differ"));
    }

    #[test]
    fn name_collision_errors_without_overwrite() {
        let mut donor = build(vec![[0.0; 3]], vec![0, 1]);
        donor.dps_arrays_mut().insert("w".into(), scalar_f32(vec![2.0]));
        let mut target = build(vec![[0.0; 3]], vec![0, 1]);
        target.dps_arrays_mut().insert("w".into(), scalar_f32(vec![1.0]));

        let err = copy_metadata_any_trx(
            AnyTrxFile::F32(target),
            &AnyTrxFile::F32(donor),
            &CopyMetadataOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("DPS 'w'"));
    }

    #[test]
    fn overwrite_replaces_existing_entry() {
        let mut donor = build(vec![[0.0; 3]], vec![0, 1]);
        donor.dps_arrays_mut().insert("w".into(), scalar_f32(vec![2.0]));
        let mut target = build(vec![[0.0; 3]], vec![0, 1]);
        target.dps_arrays_mut().insert("w".into(), scalar_f32(vec![1.0]));

        let merged = copy_metadata_any_trx(
            AnyTrxFile::F32(target),
            &AnyTrxFile::F32(donor),
            &CopyMetadataOptions {
                overwrite_conflicting_metadata: true,
                ..Default::default()
            },
        )
        .unwrap();
        let AnyTrxFile::F32(out) = merged else {
            panic!("dtype changed");
        };
        assert_eq!(out.dps::<f32>("w").unwrap().as_flat_slice(), &[2.0]);
    }

    #[test]
    fn selective_copy_only_named_dps() {
        let mut donor = build(vec![[0.0; 3]], vec![0, 1]);
        donor.dps_arrays_mut().insert("a".into(), scalar_f32(vec![1.0]));
        donor.dps_arrays_mut().insert("b".into(), scalar_f32(vec![2.0]));
        donor.dpv_arrays_mut().insert("fa".into(), scalar_f32(vec![0.5]));
        let target = build(vec![[0.0; 3]], vec![0, 1]);

        let merged = copy_metadata_any_trx(
            AnyTrxFile::F32(target),
            &AnyTrxFile::F32(donor),
            &CopyMetadataOptions {
                dps: Some(vec!["a".into()]),
                ..Default::default()
            },
        )
        .unwrap();
        let AnyTrxFile::F32(out) = merged else {
            panic!("dtype changed");
        };
        assert_eq!(out.dps_names(), vec!["a"]);
        assert!(out.dpv_names().is_empty(), "dpv should not have been copied");
    }

    #[test]
    fn group_index_out_of_bounds_errors() {
        let mut donor = build(vec![[0.0; 3]], vec![0, 1]);
        donor.group_arrays_mut().insert("bad".into(), group_u32(vec![5]));
        let target = build(vec![[0.0; 3]], vec![0, 1]);

        let err = copy_metadata_any_trx(
            AnyTrxFile::F32(target),
            &AnyTrxFile::F32(donor),
            &CopyMetadataOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("references streamline index 5"));
    }
}
