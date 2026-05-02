//! Polyline → Catmull-Rom fitting.
//!
//! Reduces a dense streamline to a sparse set of control points whose
//! Catmull-Rom interpolation reproduces the input within a user-specified
//! Hausdorff tolerance. Useful for editor tools (which want few sculptable
//! control points) and for compressing dense atlas tractograms.
//!
//! The algorithm:
//!
//! 1. Start with the two endpoints as kept control points.
//! 2. For each segment between consecutive kept points, tessellate the
//!    Catmull-Rom curve, then walk the *interior* input vertices in that
//!    segment and find the one whose perpendicular distance to the
//!    tessellated polyline is largest.
//! 3. If any segment's worst vertex is farther than `epsilon_mm`, add the
//!    worst vertex from each violating segment as a new kept point and
//!    iterate. Adding many points per iteration converges in O(log n)
//!    passes for smooth curves while staying stable on degenerate inputs.
//! 4. Stop when no segment's worst error exceeds the tolerance.
//!
//! Distances are computed as squared distances throughout (no `sqrt` in the
//! hot loop) and the per-segment tessellation buffer is reused across
//! iterations so the working set stays compact.
//!
//! # Endpoint convention
//!
//! Both endpoints of every input streamline are always retained, matching
//! how `trxviz-draw` and most editor UIs treat streamline terminations
//! (cortical/sub-cortical anchors are anatomically meaningful).
//!
//! # Coordinate units
//!
//! `epsilon_mm` matches the units of `points`, which in TRX is RAS+ mm.

mod catmull_rom;
mod tractogram;

pub use catmull_rom::{
    fit_catmull_rom_indices, sample_catmull_rom, sample_catmull_rom_into,
    simplify_streamline, MIN_KEPT_POINTS,
};
pub use tractogram::{
    simplify_tractogram, FittedMarker, SimplifyOptions, SimplifyStats, FITTED_MARKER_KEY,
};
