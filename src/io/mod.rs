pub mod directory;
pub mod filename;
pub mod zip;

use std::path::Path;

use crate::dtype::TrxScalar;
use crate::error::{Result, TrxError};
use crate::trx_file::TrxFile;

/// Load a TRX file, auto-detecting format.
///
/// - If `path` is a directory, loads from the directory.
/// - If `path` has a `.trx` extension (or is a file), loads from zip.
pub fn load<P: TrxScalar>(path: &Path) -> Result<TrxFile<P>> {
    if path.is_dir() {
        directory::load_from_directory(path, None)
    } else if path.is_file() {
        zip::load_from_zip(path)
    } else {
        Err(TrxError::FileNotFound(path.to_path_buf()))
    }
}
