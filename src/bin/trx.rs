use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use trx_rs::{
    convert, detect_format, merge_trx_shards, read_tractogram, AnyTrxFile, ConversionOptions,
    DType, Format, TrxError,
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
        #[arg(long = "positions-dtype", value_enum)]
        positions_dtype: Option<PositionDtype>,
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
            positions_dtype,
        } => concatenate_trx(
            &inputs,
            &output,
            positions_dtype.map(PositionDtype::into_dtype),
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

fn concatenate_trx(
    inputs: &[PathBuf],
    output: &Path,
    positions_dtype: Option<DType>,
) -> trx_rs::Result<()> {
    for input in inputs {
        if detect_format(input)? != Format::Trx {
            return Err(TrxError::Argument(format!(
                "concatenate currently only supports TRX inputs: {}",
                input.display()
            )));
        }
    }
    if detect_format(output)? != Format::Trx {
        return Err(TrxError::Argument(
            "concatenate output must be a TRX path".into(),
        ));
    }

    let loaded: Vec<AnyTrxFile> = inputs
        .iter()
        .map(|path| AnyTrxFile::load(path))
        .collect::<trx_rs::Result<_>>()?;
    let target_dtype = positions_dtype.unwrap_or_else(|| loaded[0].dtype());
    let normalized: Vec<AnyTrxFile> = loaded
        .iter()
        .map(|file| file.convert_positions_dtype(target_dtype))
        .collect::<trx_rs::Result<_>>()?;

    match target_dtype {
        DType::Float16 => merge_normalized(&normalized, output, expect_f16),
        DType::Float32 => merge_normalized(&normalized, output, expect_f32),
        DType::Float64 => merge_normalized(&normalized, output, expect_f64),
        other => Err(TrxError::DType(format!(
            "unsupported positions dtype {other}"
        ))),
    }
}

fn merge_normalized<P: trx_rs::TrxScalar>(
    files: &[AnyTrxFile],
    output: &Path,
    expect: fn(&AnyTrxFile) -> trx_rs::Result<&trx_rs::TrxFile<P>>,
) -> trx_rs::Result<()> {
    let refs = files
        .iter()
        .map(expect)
        .collect::<trx_rs::Result<Vec<_>>>()?;
    merge_trx_shards(&refs)?.save(output)
}

fn expect_f16(file: &AnyTrxFile) -> trx_rs::Result<&trx_rs::TrxFile<half::f16>> {
    match file {
        AnyTrxFile::F16(trx) => Ok(trx),
        _ => Err(TrxError::DType("expected float16 TRX".into())),
    }
}

fn expect_f32(file: &AnyTrxFile) -> trx_rs::Result<&trx_rs::TrxFile<f32>> {
    match file {
        AnyTrxFile::F32(trx) => Ok(trx),
        _ => Err(TrxError::DType("expected float32 TRX".into())),
    }
}

fn expect_f64(file: &AnyTrxFile) -> trx_rs::Result<&trx_rs::TrxFile<f64>> {
    match file {
        AnyTrxFile::F64(trx) => Ok(trx),
        _ => Err(TrxError::DType("expected float64 TRX".into())),
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
