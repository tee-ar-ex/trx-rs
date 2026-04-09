use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use trx_rs::{
    concatenate_any_trx, convert, detect_format, header_from_reference, read_tractogram,
    AnyTrxFile, ConcatenateOptions, ConversionOptions, DType, Format, TrxError,
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
    },
    /// Print a concise summary of a tractogram.
    Info { input: PathBuf },
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
    },
    /// Rewrite a TRX file with a different positions dtype.
    ManipulateDtype {
        input: PathBuf,
        output: PathBuf,
        #[arg(long = "positions-dtype", value_enum, default_value = "f32")]
        positions_dtype: PositionDtype,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PositionDtype {
    F16,
    F32,
    F64,
}

impl PositionDtype {
    fn into_dtype(self) -> DType {
        match self {
            Self::F16 => DType::Float16,
            Self::F32 => DType::Float32,
            Self::F64 => DType::Float64,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> trx_rs::Result<()> {
    match cli.command {
        Command::Convert {
            input,
            output,
            positions_dtype,
        } => run_convert(&input, &output, positions_dtype.into_dtype()),
        Command::Info { input } => print_info(&input),
        Command::Concatenate {
            inputs,
            output,
            delete_dpv,
            delete_dps,
            delete_groups,
            reference,
            positions_dtype,
            input_group_names,
        } => concatenate_trx(
            &inputs,
            &output,
            reference.as_deref(),
            delete_dpv,
            delete_dps,
            delete_groups,
            positions_dtype.map(PositionDtype::into_dtype),
            &input_group_names,
        ),
        Command::ManipulateDtype {
            input,
            output,
            positions_dtype,
        } => rewrite_trx_dtype(&input, &output, positions_dtype.into_dtype()),
    }
}

fn run_convert(input: &Path, output: &Path, dtype: DType) -> trx_rs::Result<()> {
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
) -> trx_rs::Result<()> {
    if detect_format(output)? != Format::Trx {
        return Err(TrxError::Argument(
            "concatenate output must be a TRX path".into(),
        ));
    }

    let header = reference.map(header_from_reference).transpose()?;
    let loaded: Vec<AnyTrxFile> = inputs
        .iter()
        .map(|path| load_concat_input(path, header.clone()))
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

fn load_concat_input(path: &Path, header: Option<trx_rs::Header>) -> trx_rs::Result<AnyTrxFile> {
    match detect_format(path)? {
        Format::Trx => AnyTrxFile::load(path),
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

fn print_info(path: &Path) -> trx_rs::Result<()> {
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
            Ok(())
        }
        format => {
            let tractogram = read_tractogram(path, &ConversionOptions::default())?;
            println!("format: {}", format_name(format));
            println!("nb_streamlines: {}", tractogram.nb_streamlines());
            println!("nb_vertices: {}", tractogram.nb_vertices());
            println!("dimensions: {:?}", tractogram.header().dimensions);
            println!("groups: {}", tractogram.groups().len());
            println!("dpg_groups: {}", tractogram.dpg().len());
            Ok(())
        }
    }
}

fn print_trx_dpg_info<P: trx_rs::TrxScalar>(trx: &trx_rs::TrxFile<P>) {
    println!("dpg_groups: {}", trx.dpg_group_names().len());
}

fn format_name(format: Format) -> &'static str {
    match format {
        Format::Trx => "trx",
        Format::Tck => "tck",
        Format::Vtk => "vtk",
        Format::TinyTrack => "tt",
    }
}
