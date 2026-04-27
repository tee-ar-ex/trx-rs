use std::collections::HashSet;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use trx_rs::{
    concatenate_any_trx, convert, copy_metadata_any_trx, detect_format, header_from_reference,
    inspect_vtk_declared_space, read_tractogram, retain_representative_indices, subset_streamlines,
    write_tractogram, AnyTrxFile, ConcatenateOptions, ConversionOptions, CopyMetadataOptions,
    DType, DuplicateRemovalMode, DuplicateRemovalParams, Format, Header, Tractogram, TrxError,
    TrxScalar, VtkCoordinateMode,
};

#[derive(Parser, Debug)]
#[command(name = "trx")]
#[command(about = "TRX tractogram tools", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert between supported tractogram formats.
    Convert {
        input: PathBuf,
        output: PathBuf,
        #[arg(long = "positions-dtype", value_enum, default_value = "f32")]
        positions_dtype: PositionDtype,
        #[arg(long = "vtk-space", value_enum, default_value = "ras")]
        vtk_space: VtkSpaceArg,
    },
    /// Print a concise summary of a tractogram.
    Info {
        input: PathBuf,
        /// Compute and print min/mean/max streamline lengths in mm (TRX only).
        #[arg(long)]
        stats: bool,
    },
    /// Concatenate TRX files into one output TRX.
    Concatenate {
        #[arg(required = true, num_args = 2..)]
        inputs: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long = "delete-dpv")]
        delete_dpv: bool,
        #[arg(long = "delete-dps")]
        delete_dps: bool,
        #[arg(long = "delete-groups")]
        delete_groups: bool,
        #[arg(short = 'r', long = "reference")]
        reference: Option<PathBuf>,
        #[arg(long = "positions-dtype", value_enum)]
        positions_dtype: Option<PositionDtype>,
        #[arg(long = "input-group-name")]
        input_group_names: Vec<String>,
        #[arg(long = "vtk-space", value_enum, default_value = "ras")]
        vtk_space: VtkSpaceArg,
    },
    /// Rewrite a TRX file with a different positions dtype.
    ManipulateDtype {
        input: PathBuf,
        output: PathBuf,
        #[arg(long = "positions-dtype", value_enum, default_value = "f32")]
        positions_dtype: PositionDtype,
    },
    /// Compare two tractograms and report differences (exit 1 if different).
    Compare { input1: PathBuf, input2: PathBuf },
    /// Update the voxel-to-RAS affine in a TRX file from a reference NIfTI or TRX.
    UpdateAffine {
        input: PathBuf,
        reference: PathBuf,
        output: PathBuf,
    },
    /// Validate a TRX file; optionally remove invalid or duplicate streamlines.
    Validate {
        input: PathBuf,
        /// Output TRX (dry-run report only if omitted).
        output: Option<PathBuf>,
        /// Remove streamlines with any vertex outside the volume bounding box.
        #[arg(long = "remove-invalid")]
        remove_invalid: bool,
        /// Remove duplicate streamlines (exact byte match).
        #[arg(long = "remove-identical")]
        remove_identical: bool,
    },
    /// Copy DPS / DPV / groups (and optionally DPG) from one TRX onto another.
    ///
    /// The donor and target must describe the same streamlines (matching
    /// `nb_streamlines` and `nb_vertices` for the kinds being copied). When
    /// `--output` is omitted, the target is rewritten in place via a sibling
    /// temp file and atomic rename.
    CopyMetadata {
        /// Target TRX whose metadata will be augmented.
        target: PathBuf,
        /// Donor TRX whose metadata is read.
        #[arg(long = "from")]
        from: PathBuf,
        /// Output path. If omitted, TARGET is rewritten in place.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Copy only this DPS array (repeatable).
        #[arg(long = "dps")]
        dps: Vec<String>,
        /// Copy only this DPV array (repeatable).
        #[arg(long = "dpv")]
        dpv: Vec<String>,
        /// Copy only this group (repeatable).
        #[arg(long = "group")]
        groups: Vec<String>,
        /// Also copy data-per-group entries for the selected groups.
        #[arg(long = "copy-dpg")]
        copy_dpg: bool,
        /// If a metadata key already exists on the target, replace it with the
        /// donor's. Without this flag, name collisions abort the operation.
        #[arg(long = "overwrite-conflicting-metadata")]
        overwrite_conflicting_metadata: bool,
        /// Skip donor arrays whose row count does not match the target instead
        /// of aborting.
        #[arg(long = "skip-mismatched")]
        skip_mismatched: bool,
    },
    /// Extract a subset of streamlines from a TRX file.
    Subset {
        input: PathBuf,
        /// Output TRX for the merged subset. Required unless --group-export is set.
        output: Option<PathBuf>,
        /// Include streamlines from this group (repeatable; result is their union).
        #[arg(long = "group")]
        groups: Vec<String>,
        /// Keep only this many streamlines (applied after group filtering).
        #[arg(long = "num-streamlines")]
        num_streamlines: Option<usize>,
        /// How to select when --num-streamlines is smaller than the candidate set.
        #[arg(long = "selection-method", value_enum, default_value = "first")]
        selection_method: SelectionMethod,
        /// Random seed for reproducible random selection.
        #[arg(long)]
        seed: Option<u64>,
        /// Export each group to its own file. Use {groupname} as a placeholder,
        /// e.g. `bundles/{groupname}.tck`. Supports any output format.
        /// When combined with --group, only those named groups are exported.
        #[arg(long = "group-export")]
        group_export: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PositionDtype {
    F16,
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum VtkSpaceArg {
    HeaderOrWarn,
    Ras,
    Lps,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SelectionMethod {
    First,
    Random,
}

impl From<VtkSpaceArg> for VtkCoordinateMode {
    fn from(value: VtkSpaceArg) -> Self {
        match value {
            VtkSpaceArg::HeaderOrWarn => VtkCoordinateMode::HeaderOrWarn,
            VtkSpaceArg::Ras => VtkCoordinateMode::AssumeRas,
            VtkSpaceArg::Lps => VtkCoordinateMode::AssumeLps,
        }
    }
}

impl From<PositionDtype> for DType {
    fn from(d: PositionDtype) -> Self {
        match d {
            PositionDtype::F16 => DType::Float16,
            PositionDtype::F32 => DType::Float32,
            PositionDtype::F64 => DType::Float64,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => {
            if code != 0 {
                std::process::exit(code);
            }
        }
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}

fn run(cli: Cli) -> trx_rs::Result<i32> {
    match cli.command {
        Command::Convert {
            input,
            output,
            positions_dtype,
            vtk_space,
        } => run_convert(&input, &output, positions_dtype.into(), vtk_space.into()).map(ok),
        Command::Info { input, stats } => print_info(&input, stats).map(ok),
        Command::Concatenate {
            inputs,
            output,
            delete_dpv,
            delete_dps,
            delete_groups,
            reference,
            positions_dtype,
            input_group_names,
            vtk_space,
        } => concatenate_trx(
            &inputs,
            &output,
            reference.as_deref(),
            delete_dpv,
            delete_dps,
            delete_groups,
            positions_dtype.map(Into::into),
            &input_group_names,
            vtk_space.into(),
        )
        .map(ok),
        Command::ManipulateDtype {
            input,
            output,
            positions_dtype,
        } => rewrite_trx_dtype(&input, &output, positions_dtype.into()).map(ok),
        Command::Compare { input1, input2 } => {
            compare_tractograms(&input1, &input2).map(|identical| if identical { 0 } else { 1 })
        }
        Command::UpdateAffine {
            input,
            reference,
            output,
        } => update_affine(&input, &reference, &output).map(ok),
        Command::Validate {
            input,
            output,
            remove_invalid,
            remove_identical,
        } => validate_trx(&input, output.as_deref(), remove_invalid, remove_identical).map(ok),
        Command::CopyMetadata {
            target,
            from,
            output,
            dps,
            dpv,
            groups,
            copy_dpg,
            overwrite_conflicting_metadata,
            skip_mismatched,
        } => copy_metadata_trx(
            &target,
            &from,
            output.as_deref(),
            &dps,
            &dpv,
            &groups,
            copy_dpg,
            overwrite_conflicting_metadata,
            skip_mismatched,
        )
        .map(ok),
        Command::Subset {
            input,
            output,
            groups,
            num_streamlines,
            selection_method,
            seed,
            group_export,
        } => {
            let opts = SubsetOptions {
                groups,
                num_streamlines,
                selection_method,
                seed,
            };
            subset_trx(&input, output.as_deref(), &opts, group_export.as_deref()).map(ok)
        }
    }
}

fn ok(_: ()) -> i32 {
    0
}

// ── convert ──────────────────────────────────────────────────────────────────

fn run_convert(
    input: &Path,
    output: &Path,
    dtype: DType,
    vtk_coordinate_mode: VtkCoordinateMode,
) -> trx_rs::Result<()> {
    let input_format = detect_format(input)?;
    let output_format = detect_format(output)?;
    if input_format == Format::Trx && output_format == Format::Trx {
        return rewrite_trx_dtype(input, output, dtype);
    }
    convert(
        input,
        output,
        &ConversionOptions {
            header: None,
            trx_positions_dtype: dtype,
            vtk_coordinate_mode,
        },
    )
}

fn rewrite_trx_dtype(input: &Path, output: &Path, dtype: DType) -> trx_rs::Result<()> {
    let input_format = detect_format(input)?;
    let output_format = detect_format(output)?;
    if input_format != Format::Trx || output_format != Format::Trx {
        return Err(TrxError::Argument(
            "manipulate-dtype only supports TRX input and output".into(),
        ));
    }
    let file = AnyTrxFile::load(input)?;
    let rewritten = file.convert_positions_dtype(dtype)?;
    rewritten.save(output)
}

// ── concatenate ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn concatenate_trx(
    inputs: &[PathBuf],
    output: &Path,
    reference: Option<&Path>,
    delete_dpv: bool,
    delete_dps: bool,
    delete_groups: bool,
    positions_dtype: Option<DType>,
    input_group_names: &[String],
    vtk_coordinate_mode: VtkCoordinateMode,
) -> trx_rs::Result<()> {
    if detect_format(output)? != Format::Trx {
        return Err(TrxError::Argument(
            "concatenate output must be a TRX path".into(),
        ));
    }
    let header = reference.map(header_from_reference).transpose()?;
    let loaded: Vec<AnyTrxFile> = inputs
        .iter()
        .map(|path| load_concat_input(path, header.clone(), vtk_coordinate_mode))
        .collect::<trx_rs::Result<_>>()?;
    let refs: Vec<&AnyTrxFile> = loaded.iter().collect();
    let merged = concatenate_any_trx(
        &refs,
        &ConcatenateOptions {
            delete_dpv,
            delete_dps,
            delete_groups,
            positions_dtype,
            input_group_names: normalize_cli_group_names(input_group_names),
        },
    )?;
    merged.save(output)
}

fn normalize_cli_group_names(values: &[String]) -> Vec<Option<String>> {
    values
        .iter()
        .map(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect()
}

fn load_concat_input(
    path: &Path,
    header: Option<Header>,
    vtk_coordinate_mode: VtkCoordinateMode,
) -> trx_rs::Result<AnyTrxFile> {
    match detect_format(path)? {
        Format::Trx => AnyTrxFile::load(path),
        Format::Trk => {
            let tractogram = read_tractogram(path, &ConversionOptions::default())?;
            tractogram.to_trx(DType::Float32)
        }
        Format::Tck | Format::Vtk => {
            if header.is_none() {
                return Err(TrxError::Argument(format!(
                    "--reference is required for {}",
                    path.display()
                )));
            }
            let tractogram = read_tractogram(
                path,
                &ConversionOptions {
                    header,
                    vtk_coordinate_mode,
                    ..Default::default()
                },
            )?;
            tractogram.to_trx(DType::Float32)
        }
        Format::TinyTrack => {
            let tractogram = read_tractogram(path, &ConversionOptions::default())?;
            tractogram.to_trx(DType::Float32)
        }
    }
}

// ── copy-metadata ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn copy_metadata_trx(
    target: &Path,
    from: &Path,
    output: Option<&Path>,
    dps: &[String],
    dpv: &[String],
    groups: &[String],
    copy_dpg: bool,
    overwrite_conflicting_metadata: bool,
    skip_mismatched: bool,
) -> trx_rs::Result<()> {
    if detect_format(target)? != Format::Trx {
        return Err(TrxError::Argument(
            "copy-metadata target must be a TRX path".into(),
        ));
    }
    if detect_format(from)? != Format::Trx {
        return Err(TrxError::Argument(
            "copy-metadata --from must be a TRX path".into(),
        ));
    }
    if let Some(out) = output {
        if detect_format(out)? != Format::Trx {
            return Err(TrxError::Argument(
                "copy-metadata --output must be a TRX path".into(),
            ));
        }
    }

    let target_file = AnyTrxFile::load(target)?;
    let donor = AnyTrxFile::load(from)?;
    let opts = CopyMetadataOptions {
        dps: cli_filter(dps),
        dpv: cli_filter(dpv),
        groups: cli_filter(groups),
        copy_dpg,
        overwrite_conflicting_metadata,
        skip_mismatched,
    };
    let merged = copy_metadata_any_trx(target_file, &donor, &opts)?;

    match output {
        Some(out) => merged.save(out),
        None => save_in_place(merged, target),
    }
}

fn cli_filter(values: &[String]) -> Option<Vec<String>> {
    if values.is_empty() {
        None
    } else {
        Some(values.to_vec())
    }
}

/// Save `file` to `path`, going through a sibling temp path so a crash
/// mid-write cannot corrupt the original.
fn save_in_place(file: AnyTrxFile, path: &Path) -> trx_rs::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let basename = path
        .file_name()
        .ok_or_else(|| TrxError::Argument(format!("invalid target path '{}'", path.display())))?;
    let temp_path = parent.join(format!(
        ".{}.copy-metadata.tmp-{}",
        basename.to_string_lossy(),
        std::process::id()
    ));

    // Clear any leftover temp from a previous failed run.
    let _ = std::fs::remove_file(&temp_path);
    let _ = std::fs::remove_dir_all(&temp_path);

    if let Err(err) = file.save(&temp_path) {
        let _ = std::fs::remove_file(&temp_path);
        let _ = std::fs::remove_dir_all(&temp_path);
        return Err(err);
    }

    // Drop the source file's mmaps before replacing the on-disk path.
    drop(file);

    if path.exists() {
        if path.is_dir() {
            std::fs::remove_dir_all(path)?;
        } else {
            std::fs::remove_file(path)?;
        }
    }
    std::fs::rename(&temp_path, path)?;
    Ok(())
}

// ── info ──────────────────────────────────────────────────────────────────────

fn print_info(path: &Path, stats: bool) -> trx_rs::Result<()> {
    match detect_format(path)? {
        Format::Trx => {
            let file = AnyTrxFile::load(path)?;
            println!("format: trx");
            println!("positions_dtype: {}", file.dtype().name());
            println!("nb_streamlines: {}", file.nb_streamlines());
            println!("nb_vertices: {}", file.nb_vertices());
            println!("dimensions: {:?}", file.header().dimensions);
            println!("dps: {}", file.dps_entries().len());
            println!("dpv: {}", file.dpv_entries().len());
            println!("groups: {}", file.groups_owned().len());
            file.with_typed(print_trx_dpg_info, print_trx_dpg_info, print_trx_dpg_info);
            if stats {
                file.with_typed(print_length_stats, print_length_stats, print_length_stats);
            }
        }
        format => {
            let tractogram = read_tractogram(path, &ConversionOptions::default())?;
            println!("format: {}", format_name(format));
            println!("nb_streamlines: {}", tractogram.nb_streamlines());
            println!("nb_vertices: {}", tractogram.nb_vertices());
            println!("dimensions: {:?}", tractogram.header().dimensions);
            println!("groups: {}", tractogram.groups().len());
            println!("dpg_groups: {}", tractogram.dpg().len());
            if format == Format::Vtk {
                let declared = inspect_vtk_declared_space(path)?;
                let label = match declared {
                    Some(trx_rs::VtkCoordinateSpace::Ras) => "RAS",
                    Some(trx_rs::VtkCoordinateSpace::Lps) => "LPS",
                    None => "missing",
                };
                println!("vtk_declared_space: {label}");
            }
            if stats {
                eprintln!("warning: --stats is only supported for TRX files");
            }
        }
    }
    Ok(())
}

fn print_trx_dpg_info<P: TrxScalar>(trx: &trx_rs::TrxFile<P>) {
    println!("dpg_groups: {}", trx.dpg_group_names().len());
}

fn print_length_stats<P: TrxScalar>(trx: &trx_rs::TrxFile<P>) {
    let mut count = 0usize;
    let mut total = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for points in trx.streamlines() {
        let length: f64 = points
            .windows(2)
            .map(|pair| {
                let dx = (pair[1][0].to_f32() - pair[0][0].to_f32()) as f64;
                let dy = (pair[1][1].to_f32() - pair[0][1].to_f32()) as f64;
                let dz = (pair[1][2].to_f32() - pair[0][2].to_f32()) as f64;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .sum();
        count += 1;
        total += length;
        min = min.min(length);
        max = max.max(length);
    }

    if count == 0 {
        println!("min_length_mm: N/A");
        println!("mean_length_mm: N/A");
        println!("max_length_mm: N/A");
        return;
    }
    println!("min_length_mm: {min:.4}");
    println!("mean_length_mm: {:.4}", total / count as f64);
    println!("max_length_mm: {max:.4}");
}

fn format_name(format: Format) -> &'static str {
    match format {
        Format::Trx => "trx",
        Format::Trk => "trk",
        Format::Tck => "tck",
        Format::Vtk => "vtk",
        Format::TinyTrack => "tt",
    }
}

// ── compare ───────────────────────────────────────────────────────────────────

struct TractogramSummary {
    nb_streamlines: usize,
    nb_vertices: usize,
    dimensions: [u64; 3],
    dps: HashSet<String>,
    dpv: HashSet<String>,
    groups: HashSet<String>,
}

fn load_tractogram_summary(path: &Path) -> trx_rs::Result<TractogramSummary> {
    if detect_format(path)? == Format::Trx {
        let file = AnyTrxFile::load(path)?;
        Ok(TractogramSummary {
            nb_streamlines: file.nb_streamlines(),
            nb_vertices: file.nb_vertices(),
            dimensions: file.header().dimensions,
            dps: file.dps_entries().into_iter().map(|(k, _)| k).collect(),
            dpv: file.dpv_entries().into_iter().map(|(k, _)| k).collect(),
            groups: file.groups_owned().into_iter().map(|(k, _)| k).collect(),
        })
    } else {
        let tractogram = read_tractogram(path, &ConversionOptions::default())?;
        Ok(TractogramSummary {
            nb_streamlines: tractogram.nb_streamlines(),
            nb_vertices: tractogram.nb_vertices(),
            dimensions: tractogram.header().dimensions,
            dps: HashSet::new(),
            dpv: HashSet::new(),
            groups: tractogram.groups().keys().cloned().collect(),
        })
    }
}

fn report_eq<T: std::fmt::Debug + PartialEq>(label: &str, a: &T, b: &T) -> bool {
    let same = a == b;
    println!("[{}] {label}: {a:?} vs {b:?}", if same { "=" } else { "!" });
    same
}

fn report_set_diff(label: &str, a: &HashSet<String>, b: &HashSet<String>) -> bool {
    let only_a: Vec<_> = a.difference(b).collect();
    let only_b: Vec<_> = b.difference(a).collect();
    if only_a.is_empty() && only_b.is_empty() {
        println!("[=] {label}: {a:?}");
        true
    } else {
        println!("[!] {label}: only in first={only_a:?} only in second={only_b:?}");
        false
    }
}

fn compare_tractograms(path1: &Path, path2: &Path) -> trx_rs::Result<bool> {
    let s1 = load_tractogram_summary(path1)?;
    let s2 = load_tractogram_summary(path2)?;

    // Use `&` (not `&&`) so every field is printed even when earlier ones differ.
    let identical = report_eq("nb_streamlines", &s1.nb_streamlines, &s2.nb_streamlines)
        & report_eq("nb_vertices", &s1.nb_vertices, &s2.nb_vertices)
        & report_eq("dimensions", &s1.dimensions, &s2.dimensions)
        & report_set_diff("dps", &s1.dps, &s2.dps)
        & report_set_diff("dpv", &s1.dpv, &s2.dpv)
        & report_set_diff("groups", &s1.groups, &s2.groups);

    println!(
        "result: {}",
        if identical { "identical" } else { "different" }
    );
    Ok(identical)
}

// ── update-affine ─────────────────────────────────────────────────────────────

fn update_affine(input: &Path, reference: &Path, output: &Path) -> trx_rs::Result<()> {
    if detect_format(input)? != Format::Trx || detect_format(output)? != Format::Trx {
        return Err(TrxError::Argument(
            "update-affine requires TRX input and output".into(),
        ));
    }
    let ref_header = header_from_reference(reference)?;
    let file = AnyTrxFile::load(input)?;
    let old_header = file.header().clone();
    let new_header = Header {
        voxel_to_rasmm: ref_header.voxel_to_rasmm,
        dimensions: ref_header.dimensions,
        nb_streamlines: old_header.nb_streamlines,
        nb_vertices: old_header.nb_vertices,
        extra: old_header.extra,
    };
    file.with_updated_header(new_header).save(output)
}

// ── validate ──────────────────────────────────────────────────────────────────

fn validate_trx(
    input: &Path,
    output: Option<&Path>,
    remove_invalid: bool,
    remove_identical: bool,
) -> trx_rs::Result<()> {
    if detect_format(input)? != Format::Trx {
        return Err(TrxError::Argument("validate requires a TRX input".into()));
    }
    let file = AnyTrxFile::load(input)?;
    let original_count = file.nb_streamlines();

    let (invalid_removed, duplicate_removed, final_indices) = match &file {
        AnyTrxFile::F16(trx) => compute_valid_indices(trx, remove_invalid, remove_identical)?,
        AnyTrxFile::F32(trx) => compute_valid_indices(trx, remove_invalid, remove_identical)?,
        AnyTrxFile::F64(trx) => compute_valid_indices(trx, remove_invalid, remove_identical)?,
    };

    println!("original_streamlines: {original_count}");
    if remove_invalid {
        println!("removed_invalid: {invalid_removed}");
    }
    if remove_identical {
        println!("removed_duplicates: {duplicate_removed}");
    }
    println!("remaining_streamlines: {}", final_indices.len());

    if let Some(out) = output {
        if detect_format(out)? != Format::Trx {
            return Err(TrxError::Argument(
                "validate output must be a TRX path".into(),
            ));
        }
        let result = match &file {
            AnyTrxFile::F16(trx) => AnyTrxFile::F16(subset_streamlines(trx, &final_indices)?),
            AnyTrxFile::F32(trx) => AnyTrxFile::F32(subset_streamlines(trx, &final_indices)?),
            AnyTrxFile::F64(trx) => AnyTrxFile::F64(subset_streamlines(trx, &final_indices)?),
        };
        result.save(out)?;
    }

    Ok(())
}

fn compute_valid_indices<P: TrxScalar>(
    trx: &trx_rs::TrxFile<P>,
    remove_invalid: bool,
    remove_identical: bool,
) -> trx_rs::Result<(usize, usize, Vec<usize>)> {
    let bounds_indices: Vec<usize> = if remove_invalid {
        let rasmm_to_voxel = invert_affine(&trx.header().voxel_to_rasmm)
            .ok_or_else(|| TrxError::Format("voxel_to_rasmm affine is not invertible".into()))?;
        let dims = trx.header().dimensions;
        (0..trx.nb_streamlines())
            .filter(|&i| streamline_within_bounds(trx.streamline(i), &rasmm_to_voxel, dims))
            .collect()
    } else {
        (0..trx.nb_streamlines()).collect()
    };

    let invalid_removed = trx.nb_streamlines() - bounds_indices.len();

    if !remove_identical {
        return Ok((invalid_removed, 0, bounds_indices));
    }

    // Materialise the bounds-filtered subset so we can run dedup on it.
    let valid_subset = subset_streamlines(trx, &bounds_indices)?;
    let params = DuplicateRemovalParams {
        mode: DuplicateRemovalMode::Exact,
        ..Default::default()
    };
    let keep_in_subset = retain_representative_indices(&valid_subset, &params);
    let duplicate_removed = valid_subset.nb_streamlines() - keep_in_subset.len();

    // Map subset-local indices back to original-file indices.
    let final_indices = keep_in_subset
        .into_iter()
        .map(|i| bounds_indices[i])
        .collect();

    Ok((invalid_removed, duplicate_removed, final_indices))
}

fn streamline_within_bounds<P: TrxScalar>(
    points: &[[P; 3]],
    rasmm_to_voxel: &[[f64; 4]; 4],
    dims: [u64; 3],
) -> bool {
    points.iter().all(|point| {
        let x = point[0].to_f32() as f64;
        let y = point[1].to_f32() as f64;
        let z = point[2].to_f32() as f64;
        let vx = rasmm_to_voxel[0][0] * x
            + rasmm_to_voxel[0][1] * y
            + rasmm_to_voxel[0][2] * z
            + rasmm_to_voxel[0][3];
        let vy = rasmm_to_voxel[1][0] * x
            + rasmm_to_voxel[1][1] * y
            + rasmm_to_voxel[1][2] * z
            + rasmm_to_voxel[1][3];
        let vz = rasmm_to_voxel[2][0] * x
            + rasmm_to_voxel[2][1] * y
            + rasmm_to_voxel[2][2] * z
            + rasmm_to_voxel[2][3];
        vx >= 0.0
            && vy >= 0.0
            && vz >= 0.0
            && vx < dims[0] as f64
            && vy < dims[1] as f64
            && vz < dims[2] as f64
    })
}

fn invert_affine(a: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let r = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let t = [a[0][3], a[1][3], a[2][3]];

    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;

    let ri = [
        [
            (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * inv_det,
            (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * inv_det,
            (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * inv_det,
        ],
        [
            (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * inv_det,
            (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * inv_det,
            (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * inv_det,
        ],
        [
            (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * inv_det,
            (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * inv_det,
            (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * inv_det,
        ],
    ];

    let ti = [
        -(ri[0][0] * t[0] + ri[0][1] * t[1] + ri[0][2] * t[2]),
        -(ri[1][0] * t[0] + ri[1][1] * t[1] + ri[1][2] * t[2]),
        -(ri[2][0] * t[0] + ri[2][1] * t[1] + ri[2][2] * t[2]),
    ];

    Some([
        [ri[0][0], ri[0][1], ri[0][2], ti[0]],
        [ri[1][0], ri[1][1], ri[1][2], ti[1]],
        [ri[2][0], ri[2][1], ri[2][2], ti[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

// ── subset ────────────────────────────────────────────────────────────────────

struct SubsetOptions {
    groups: Vec<String>,
    num_streamlines: Option<usize>,
    selection_method: SelectionMethod,
    seed: Option<u64>,
}

fn subset_trx(
    input: &Path,
    output: Option<&Path>,
    opts: &SubsetOptions,
    group_export: Option<&str>,
) -> trx_rs::Result<()> {
    if output.is_none() && group_export.is_none() {
        return Err(TrxError::Argument(
            "either OUTPUT or --group-export must be specified".into(),
        ));
    }
    if detect_format(input)? != Format::Trx {
        return Err(TrxError::Argument("subset requires a TRX input".into()));
    }
    let file = AnyTrxFile::load(input)?;

    if let Some(out) = output {
        if detect_format(out)? != Format::Trx {
            return Err(TrxError::Argument(
                "subset output must be a TRX path".into(),
            ));
        }
        let result = match &file {
            AnyTrxFile::F16(trx) => AnyTrxFile::F16(subset_typed(trx, opts)?),
            AnyTrxFile::F32(trx) => AnyTrxFile::F32(subset_typed(trx, opts)?),
            AnyTrxFile::F64(trx) => AnyTrxFile::F64(subset_typed(trx, opts)?),
        };
        result.save(out)?;
    }

    if let Some(pattern) = group_export {
        export_groups_by_pattern(&file, &opts.groups, pattern)?;
    }

    Ok(())
}

fn subset_typed<P: TrxScalar>(
    trx: &trx_rs::TrxFile<P>,
    opts: &SubsetOptions,
) -> trx_rs::Result<trx_rs::TrxFile<P>> {
    let mut indices: Vec<usize> = if opts.groups.is_empty() {
        (0..trx.nb_streamlines()).collect()
    } else {
        let mut set = HashSet::new();
        for name in &opts.groups {
            let group_indices = trx
                .group(name)
                .map_err(|_| TrxError::Argument(format!("group '{name}' not found in TRX file")))?;
            for &idx in group_indices {
                set.insert(idx as usize);
            }
        }
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort_unstable();
        v
    };

    if let Some(n) = opts.num_streamlines {
        if n < indices.len() {
            match opts.selection_method {
                SelectionMethod::First => indices.truncate(n),
                SelectionMethod::Random => {
                    use rand::seq::SliceRandom;
                    use rand::SeedableRng;
                    let mut rng = match opts.seed {
                        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
                        None => rand::rngs::StdRng::from_entropy(),
                    };
                    indices.shuffle(&mut rng);
                    indices.truncate(n);
                    indices.sort_unstable();
                }
            }
        }
    }

    subset_streamlines(trx, &indices)
}

fn export_groups_by_pattern(
    file: &AnyTrxFile,
    filter_groups: &[String],
    pattern: &str,
) -> trx_rs::Result<()> {
    if !pattern.contains("{groupname}") {
        return Err(TrxError::Argument(
            "--group-export pattern must contain '{groupname}' placeholder".into(),
        ));
    }

    let all_groups = file.groups_owned();
    let filter_set: HashSet<&str> = filter_groups.iter().map(String::as_str).collect();
    let has_any_groups = !all_groups.is_empty();
    let groups_to_export: Vec<_> = all_groups
        .into_iter()
        .filter(|(name, _)| filter_set.is_empty() || filter_set.contains(name.as_str()))
        .collect();

    if groups_to_export.is_empty() {
        if !has_any_groups {
            eprintln!("Warning: --group-export produced no output — the input TRX file has no groups.");
        } else if !filter_set.is_empty() {
            eprintln!(
                "Warning: --group-export produced no output — none of the requested groups ({}) were found in the file.",
                filter_groups.join(", ")
            );
        } else {
            eprintln!("Warning: --group-export produced no output files.");
        }
        return Ok(());
    }

    for (group_name, streamline_indices) in groups_to_export {
        let out_path_str = pattern.replace("{groupname}", &group_name);
        let out_path = PathBuf::from(&out_path_str);

        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let indices: Vec<usize> = streamline_indices.iter().map(|&i| i as usize).collect();
        let group_file = match file {
            AnyTrxFile::F16(trx) => AnyTrxFile::F16(subset_streamlines(trx, &indices)?),
            AnyTrxFile::F32(trx) => AnyTrxFile::F32(subset_streamlines(trx, &indices)?),
            AnyTrxFile::F64(trx) => AnyTrxFile::F64(subset_streamlines(trx, &indices)?),
        };

        if detect_format(&out_path)? == Format::Trx {
            group_file.save(&out_path)?;
        } else {
            let tractogram = Tractogram::from(&group_file);
            write_tractogram(&out_path, &tractogram, &ConversionOptions::default())?;
        }

        println!(
            "exported {group_name}: {} streamlines → {out_path_str}",
            indices.len()
        );
    }

    Ok(())
}
