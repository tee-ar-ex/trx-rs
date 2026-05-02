//! Apply Catmull-Rom fitting at the [`Tractogram`] level.
//!
//! Produces a new tractogram with simplified positions and a
//! `header.extra["catmull_rom_fitted"]` marker so downstream tools can
//! recognise the file as ready-to-edit (no second pass of fitting needed).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::{Result, TrxError};
use crate::mmap_backing::vec_to_bytes;
use crate::tractogram::Tractogram;
use crate::trx_file::DataArray;

use super::catmull_rom::simplify_streamline;

/// JSON key written under [`Tractogram::extra_mut`] to mark a tractogram as
/// already Catmull-Rom-fitted.
pub const FITTED_MARKER_KEY: &str = "catmull_rom_fitted";

/// Schema for the [`FITTED_MARKER_KEY`] payload. Stable across releases of
/// this crate; consumers should treat unknown fields as ignorable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedMarker {
    pub version: u32,
    pub epsilon_mm: f32,
    pub fitter: String,
}

impl FittedMarker {
    pub fn new(epsilon_mm: f32) -> Self {
        Self {
            version: 1,
            epsilon_mm,
            fitter: "trx-rs".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimplifyOptions {
    /// Hausdorff tolerance in the same units as the tractogram positions
    /// (RAS+ mm for TRX). Smaller values keep more control points.
    pub epsilon_mm: f32,
    /// If `Some`, only this group's streamlines are written to the output.
    /// `None` keeps all streamlines and preserves every group.
    pub group: Option<String>,
    /// Default `width` value (mm) written as a per-vertex DPV column. Lets
    /// downstream editors honour future width-driven dispersal without
    /// requiring the user to enter values up-front.
    pub default_width_mm: f32,
    /// Default `tension` value written as a per-vertex DPV column.
    pub default_tension: f32,
    /// Number of streamlines to process in parallel (per Rayon default if
    /// `None`). Currently the implementation is single-threaded; this knob
    /// is reserved for a future `parallel` feature gate.
    pub _parallel_chunks: Option<usize>,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        Self {
            epsilon_mm: 1.0,
            group: None,
            default_width_mm: 1.0,
            default_tension: 0.5,
            _parallel_chunks: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimplifyStats {
    pub input_streamlines: usize,
    pub output_streamlines: usize,
    pub input_vertices: usize,
    pub output_vertices: usize,
    pub groups_preserved: usize,
}

impl SimplifyStats {
    pub fn vertex_compression_ratio(&self) -> f32 {
        if self.output_vertices == 0 {
            0.0
        } else {
            self.input_vertices as f32 / self.output_vertices as f32
        }
    }
}

/// Build a new [`Tractogram`] from `input` with each streamline simplified
/// to a Catmull-Rom-fittable control polygon. Preserves the source header
/// (so spatial metadata round-trips) and writes the
/// [`FITTED_MARKER_KEY`] marker.
///
/// When `opts.group` is set, only streamlines belonging to that group are
/// kept; output groups are remapped to the new (compacted) streamline
/// indices.
pub fn simplify_tractogram(
    input: &Tractogram,
    opts: &SimplifyOptions,
) -> Result<(Tractogram, SimplifyStats)> {
    if !opts.epsilon_mm.is_finite() || opts.epsilon_mm <= 0.0 {
        return Err(TrxError::Argument(format!(
            "epsilon_mm must be a positive finite value, got {}",
            opts.epsilon_mm
        )));
    }

    // Decide which streamlines we're keeping. `keep_input_idx[i] = Some(out_idx)`
    // when input streamline `i` should be written; `None` means dropped.
    let nb_in = input.nb_streamlines();
    let keep_set: Option<std::collections::HashSet<u32>> = match &opts.group {
        Some(name) => {
            let members = input
                .group(name)
                .ok_or_else(|| TrxError::Argument(format!("group `{name}` not found")))?;
            Some(members.iter().copied().collect())
        }
        None => None,
    };

    let mut output = Tractogram::with_header(input.header().clone());
    let mut stats = SimplifyStats::default();
    let mut new_widths: Vec<f32> = Vec::new();
    let mut new_tensions: Vec<f32> = Vec::new();
    // Map old streamline index → new streamline index (only set for kept).
    let mut remap: Vec<Option<u32>> = vec![None; nb_in];

    for input_idx in 0..nb_in {
        if let Some(keep) = &keep_set {
            if !keep.contains(&(input_idx as u32)) {
                continue;
            }
        }
        let dense = input.streamline(input_idx);
        stats.input_streamlines += 1;
        stats.input_vertices += dense.len();

        let simplified = simplify_streamline(dense, opts.epsilon_mm);
        let out_idx = output.nb_streamlines() as u32;
        output.push_streamline(&simplified)?;
        let cp_count = simplified.len();
        stats.output_vertices += cp_count;
        new_widths.extend(std::iter::repeat(opts.default_width_mm).take(cp_count));
        new_tensions.extend(std::iter::repeat(opts.default_tension).take(cp_count));
        remap[input_idx] = Some(out_idx);
    }
    stats.output_streamlines = output.nb_streamlines();

    if !new_widths.is_empty() {
        output.insert_dpv("width", scalar_dpv(new_widths));
        output.insert_dpv("tension", scalar_dpv(new_tensions));
    }

    let mut group_remap: HashMap<String, Vec<u32>> = HashMap::new();
    for (name, members) in input.groups() {
        if let Some(filter) = &opts.group {
            if name != filter {
                continue;
            }
        }
        let mut remapped: Vec<u32> = members
            .iter()
            .filter_map(|&idx| {
                remap
                    .get(idx as usize)
                    .copied()
                    .flatten()
            })
            .collect();
        if remapped.is_empty() {
            continue;
        }
        remapped.sort_unstable();
        group_remap.insert(name.clone(), remapped);
    }
    stats.groups_preserved = group_remap.len();
    for (name, members) in group_remap {
        output.insert_group(name, members);
    }

    let marker = FittedMarker::new(opts.epsilon_mm);
    let json = serde_json::to_value(&marker)
        .map_err(|e| TrxError::Argument(format!("serialise fitted marker: {e}")))?;
    output.extra_mut().insert(FITTED_MARKER_KEY.to_string(), json);

    Ok((output, stats))
}

fn scalar_dpv(values: Vec<f32>) -> DataArray {
    DataArray::owned_bytes(vec_to_bytes(values), 1, DType::Float32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tractogram::Tractogram;

    fn dense_streamline(seed: f32, n: usize) -> Vec<[f32; 3]> {
        (0..=n)
            .map(|i| {
                let t = i as f32 / n as f32;
                let x = seed + 50.0 * t;
                let y = 10.0 * (t * std::f32::consts::PI).sin();
                let z = 5.0 * t;
                [x, y, z]
            })
            .collect()
    }

    fn build_input(streamline_count: usize, vertices: usize) -> Tractogram {
        let mut t = Tractogram::new();
        for s in 0..streamline_count {
            let pts = dense_streamline(s as f32 * 1.0, vertices);
            t.push_streamline(&pts).unwrap();
        }
        t.insert_group(
            "even",
            (0..streamline_count as u32).step_by(2).collect(),
        );
        t.insert_group(
            "odd",
            (1..streamline_count as u32).step_by(2).collect(),
        );
        t
    }

    #[test]
    fn simplify_compresses_and_marks() {
        let input = build_input(4, 100);
        let (output, stats) =
            simplify_tractogram(&input, &SimplifyOptions::default()).unwrap();

        assert_eq!(stats.input_streamlines, 4);
        assert_eq!(stats.output_streamlines, 4);
        assert!(stats.output_vertices < stats.input_vertices);
        assert!(stats.vertex_compression_ratio() > 4.0);

        // Marker present.
        let marker = output.extra().get(FITTED_MARKER_KEY).unwrap();
        let parsed: FittedMarker = serde_json::from_value(marker.clone()).unwrap();
        assert_eq!(parsed.version, 1);
        assert!((parsed.epsilon_mm - 1.0).abs() < 1e-6);

        // DPVs match new vertex count.
        let widths = output.dpv_arrays().get("width").unwrap();
        assert_eq!(widths.dtype(), DType::Float32);
        assert_eq!(widths.cast_slice::<f32>().len(), output.nb_vertices());

        // Both groups remap.
        assert_eq!(stats.groups_preserved, 2);
        assert_eq!(output.group("even").unwrap().len(), 2);
        assert_eq!(output.group("odd").unwrap().len(), 2);
    }

    #[test]
    fn group_filter_drops_others() {
        let input = build_input(4, 100);
        let opts = SimplifyOptions {
            group: Some("even".to_string()),
            ..SimplifyOptions::default()
        };
        let (output, stats) = simplify_tractogram(&input, &opts).unwrap();
        assert_eq!(stats.input_streamlines, 2);
        assert_eq!(stats.output_streamlines, 2);
        assert_eq!(output.groups().len(), 1);
        assert_eq!(output.group("even").unwrap(), &[0, 1]);
    }

    #[test]
    fn unknown_group_is_an_error() {
        let input = build_input(2, 50);
        let opts = SimplifyOptions {
            group: Some("nonexistent".to_string()),
            ..SimplifyOptions::default()
        };
        let err = simplify_tractogram(&input, &opts).unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn invalid_epsilon_rejected() {
        let input = build_input(1, 50);
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let opts = SimplifyOptions {
                epsilon_mm: bad,
                ..SimplifyOptions::default()
            };
            assert!(simplify_tractogram(&input, &opts).is_err());
        }
    }
}
