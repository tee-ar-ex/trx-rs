use bytemuck::{cast_slice, Pod};
use std::collections::HashMap;
use std::path::Path;

use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::mmap_backing::MmapBacking;
use crate::typed_view::TypedView2D;

/// Named data array with column count and dtype metadata.
#[derive(Debug)]
pub struct DataArray {
    pub backing: MmapBacking,
    pub ncols: usize,
    pub dtype: DType,
}

/// Core TRX container, generic over the position scalar type `P`.
///
/// Owns all memory-mapped (or heap-allocated) backings. Typed views borrow
/// from `self` — no explicit lifetime annotations needed.
///
/// # Field ordering
/// `_tempdir` must be declared AFTER mmap fields so it is dropped last,
/// keeping the temp directory alive while mmaps reference files within it.
pub struct TrxFile<P: TrxScalar> {
    pub header: Header,

    /// Positions backing — `N × 3` elements of type `P`.
    positions_backing: MmapBacking,

    /// Offsets backing — `(nb_streamlines + 1)` u32 values.
    /// TRX offsets are normalized to uint32 internally.
    offsets_backing: MmapBacking,

    /// Data per streamline: name → DataArray with `nb_streamlines` rows.
    pub dps: HashMap<String, DataArray>,

    /// Data per vertex: name → DataArray with `nb_vertices` rows.
    pub dpv: HashMap<String, DataArray>,

    /// Groups: name → DataArray of uint32 streamline indices.
    pub groups: HashMap<String, DataArray>,

    /// Temp directory handle (for zip-extracted files). Kept alive until drop.
    _tempdir: Option<tempfile::TempDir>,

    _phantom: std::marker::PhantomData<P>,
}

impl<P: TrxScalar> TrxFile<P> {
    /// Create an empty TrxFile with the given header.
    pub fn empty(header: Header) -> Self {
        Self {
            header,
            positions_backing: MmapBacking::Owned(Vec::new()),
            offsets_backing: MmapBacking::Owned(Vec::new()),
            dps: HashMap::new(),
            dpv: HashMap::new(),
            groups: HashMap::new(),
            _tempdir: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Construct from pre-built components (used by the loader).
    pub(crate) fn from_parts(
        header: Header,
        positions_backing: MmapBacking,
        offsets_backing: MmapBacking,
        dps: HashMap<String, DataArray>,
        dpv: HashMap<String, DataArray>,
        groups: HashMap<String, DataArray>,
        tempdir: Option<tempfile::TempDir>,
    ) -> Self {
        Self {
            header,
            positions_backing,
            offsets_backing,
            dps,
            dpv,
            groups,
            _tempdir: tempdir,
            _phantom: std::marker::PhantomData,
        }
    }

    // ── Positions ───────────────────────────────────────────────────

    /// Positions as a flat slice of `[P; 3]` arrays.
    pub fn positions(&self) -> &[[P; 3]] {
        cast_slice(self.positions_backing.as_bytes())
    }

    /// Positions as a `TypedView2D` with 3 columns.
    pub fn positions_2d(&self) -> TypedView2D<'_, P> {
        let flat: &[P] = cast_slice(self.positions_backing.as_bytes());
        TypedView2D::new(flat, 3)
    }

    /// Raw position bytes for direct GPU buffer upload.
    pub fn positions_bytes(&self) -> &[u8] {
        self.positions_backing.as_bytes()
    }

    /// Number of vertices (points) across all streamlines.
    pub fn nb_vertices(&self) -> usize {
        self.positions().len()
    }

    // ── Offsets ─────────────────────────────────────────────────────

    /// Offsets as a slice of `u32`. Length is `nb_streamlines + 1`.
    /// The i-th streamline spans `positions[offsets[i]..offsets[i+1]]`.
    pub fn offsets(&self) -> &[u32] {
        self.offsets_backing.cast_slice()
    }

    /// Number of streamlines.
    pub fn nb_streamlines(&self) -> usize {
        let offsets = self.offsets();
        if offsets.is_empty() {
            0
        } else {
            offsets.len() - 1
        }
    }

    // ── Streamline access ───────────────────────────────────────────

    /// Get the i-th streamline as a slice of `[P; 3]` points.
    pub fn streamline(&self, i: usize) -> &[[P; 3]] {
        let offsets = self.offsets();
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        &self.positions()[start..end]
    }

    /// Iterate over all streamlines.
    pub fn streamlines(&self) -> StreamlineIter<'_, P> {
        StreamlineIter {
            positions: self.positions(),
            offsets: self.offsets(),
            index: 0,
        }
    }

    /// Length (number of points) of each streamline.
    pub fn streamline_lengths(&self) -> Vec<usize> {
        let offsets = self.offsets();
        offsets.windows(2).map(|w| (w[1] - w[0]) as usize).collect()
    }

    // ── DPS / DPV / Group access ────────────────────────────────────

    /// Get a DPS (data-per-streamline) array cast to type `T`.
    pub fn dps<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self
            .dps
            .get(name)
            .ok_or_else(|| TrxError::Argument(format!("no DPS named '{name}'")))?;
        let data: &[T] = cast_slice(arr.backing.as_bytes());
        Ok(TypedView2D::new(data, arr.ncols))
    }

    /// Get a DPV (data-per-vertex) array cast to type `T`.
    pub fn dpv<T: Pod>(&self, name: &str) -> Result<TypedView2D<'_, T>> {
        let arr = self
            .dpv
            .get(name)
            .ok_or_else(|| TrxError::Argument(format!("no DPV named '{name}'")))?;
        let data: &[T] = cast_slice(arr.backing.as_bytes());
        Ok(TypedView2D::new(data, arr.ncols))
    }

    /// Get group member indices (always u32).
    pub fn group(&self, name: &str) -> Result<&[u32]> {
        let arr = self
            .groups
            .get(name)
            .ok_or_else(|| TrxError::Argument(format!("no group named '{name}'")))?;
        Ok(arr.backing.cast_slice())
    }

    /// List DPS field names.
    pub fn dps_names(&self) -> Vec<&str> {
        self.dps.keys().map(|s| s.as_str()).collect()
    }

    /// List DPV field names.
    pub fn dpv_names(&self) -> Vec<&str> {
        self.dpv.keys().map(|s| s.as_str()).collect()
    }

    /// List group names.
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.keys().map(|s| s.as_str()).collect()
    }

    // ── Loading (convenience) ───────────────────────────────────────

    /// Load a TRX file from a directory or `.trx` zip archive.
    pub fn load(path: &Path) -> Result<Self> {
        crate::io::load::<P>(path)
    }

    // ── Saving ──────────────────────────────────────────────────────

    /// Save to a directory.
    pub fn save_to_directory(&self, path: &Path) -> Result<()> {
        crate::io::directory::save_to_directory(self, path)
    }

    /// Save to a `.trx` zip archive (deflate compression).
    pub fn save_to_zip(&self, path: &Path) -> Result<()> {
        crate::io::zip::save_to_zip(self, path)
    }

    /// Save to a `.trx` zip archive without compression (stored mode).
    pub fn save_to_zip_stored(&self, path: &Path) -> Result<()> {
        crate::io::zip::save_to_zip_with(self, path, zip::CompressionMethod::Stored)
    }

    /// Save — auto-detects format from extension (`.trx` = zip, otherwise directory).
    pub fn save(&self, path: &Path) -> Result<()> {
        if path.extension().and_then(|e| e.to_str()) == Some("trx") {
            self.save_to_zip(path)
        } else {
            self.save_to_directory(path)
        }
    }
}

impl<P: TrxScalar> std::fmt::Debug for TrxFile<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrxFile")
            .field("dtype", &P::DTYPE)
            .field("nb_streamlines", &self.nb_streamlines())
            .field("nb_vertices", &self.nb_vertices())
            .field("dps", &self.dps_names())
            .field("dpv", &self.dpv_names())
            .field("groups", &self.group_names())
            .finish()
    }
}

/// Iterator over streamlines, yielding `&[[P; 3]]` slices.
pub struct StreamlineIter<'a, P: TrxScalar> {
    positions: &'a [[P; 3]],
    offsets: &'a [u32],
    index: usize,
}

impl<'a, P: TrxScalar> Iterator for StreamlineIter<'a, P> {
    type Item = &'a [[P; 3]];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index + 1 >= self.offsets.len() {
            return None;
        }
        let start = self.offsets[self.index] as usize;
        let end = self.offsets[self.index + 1] as usize;
        self.index += 1;
        Some(&self.positions[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1 - self.index
        };
        (remaining, Some(remaining))
    }
}

impl<'a, P: TrxScalar> ExactSizeIterator for StreamlineIter<'a, P> {}
