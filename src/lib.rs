//! # trx-rs
//!
//! Rust library for reading and writing TRX tractography files.
//!
//! TRX is a file format for storing brain tractography streamlines as
//! memory-mapped binary arrays, either as plain directories or ZIP archives.

pub mod any_trx_file;
pub mod dtype;
pub mod error;
pub mod formats;
pub mod header;
pub mod io;
pub mod mmap_backing;
pub mod ops;
pub mod reference;
pub mod stream;
pub mod tractogram;
pub mod trx_file;
pub mod typed_view;
pub mod vertex;

// Re-exports for convenience
pub use any_trx_file::{AnyTrxFile, PositionsRef};
pub use dtype::{DType, TrxScalar};
pub use error::{Result, TrxError};
pub use formats::{
    convert, detect_format, inspect_vtk_declared_space, read_tractogram, vtk_import_warnings,
    write_tractogram, ConversionOptions, Format, VtkCoordinateMode, VtkCoordinateSpace,
};
pub use header::Header;
pub use mmap_backing::MmapBacking;
pub use ops::{
    build_streamline_aabbs, build_streamline_aabbs_from_slices, concatenate_any_trx, difference,
    difference_indices, intersection, intersection_indices, merge_trx_shards, query_aabb,
    query_aabb_cached, remove_duplicates, remove_duplicates_tractogram,
    retain_representative_indices, retain_tractogram_representative_indices, streamline_union,
    subset_streamlines, ConcatenateOptions, DuplicateRemovalMode, DuplicateRemovalParams,
    StreamlineAabb,
};
pub use reference::header_from_reference;
pub use stream::TrxStream;
pub use tractogram::Tractogram;
pub use trx_file::{DataArray, DataArrayInfo, TrxFile};
pub use typed_view::TypedView2D;
pub use vertex::Position3;
