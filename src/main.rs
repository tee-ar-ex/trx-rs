use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments
    if args.len() < 3 {
        eprintln!("Usage: {} <input> <output> [--ref <nifti>]", args[0]);
        std::process::exit(1);
    }

    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let mut ref_nifti = None;

    let mut idx = 3;
    while idx < args.len() {
        if args[idx] == "--ref" {
            if idx + 1 < args.len() {
                ref_nifti = Some(PathBuf::from(&args[idx + 1]));
                idx += 2;
            } else {
                eprintln!("Missing path for --ref");
                std::process::exit(1);
            }
        } else {
            eprintln!("Unknown argument: {}", args[idx]);
            std::process::exit(1);
        }
    }

    let tractogram =
        trx_rs::formats::read_tractogram(&input, &trx_rs::formats::ConversionOptions::default())?;

    if output.extension().map_or(false, |ext| ext == "trx") {
        trx_rs::legacy_io::write_trx(&output, &tractogram, ref_nifti.as_deref())?;
    } else if output.extension().map_or(false, |ext| ext == "trk") {
        trx_rs::legacy_io::write_trk(&output, &tractogram, ref_nifti.as_deref())?;
    } else {
        eprintln!("Unsupported output format");
        std::process::exit(1);
    }

    Ok(())
}
