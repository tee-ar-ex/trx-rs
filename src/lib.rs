//! # trx-rs
//!
//! Rust library for reading and writing TRX tractography files.
//!
//! TRX is a file format for storing brain tractography streamlines as
//! memory-mapped binary arrays, either as plain directories or ZIP archives.

pub mod any_trx_file;
pub mod dtype;
pub mod error;
pub mod header;
pub mod io;
pub mod mmap_backing;
pub mod ops;
pub mod stream;
pub mod trx_file;
pub mod typed_view;
pub mod vertex;

// Re-exports for convenience
pub use any_trx_file::{AnyTrxFile, PositionsRef};
pub use dtype::{DType, TrxScalar};
pub use error::{Result, TrxError};
pub use header::Header;
pub use mmap_backing::MmapBacking;
pub use stream::TrxStream;
pub use trx_file::TrxFile;
pub use typed_view::TypedView2D;
pub use vertex::Position3;
