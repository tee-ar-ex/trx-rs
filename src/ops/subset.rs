use std::collections::HashMap;

use crate::dtype::TrxScalar;
use crate::error::Result;
use crate::header::Header;
use crate::mmap_backing::MmapBacking;
use crate::trx_file::{DataArray, TrxFile};

/// Extract a subset of streamlines by index, producing a new `TrxFile`.
///
/// All DPS, DPV, and group arrays are remapped accordingly.
pub fn subset_streamlines<P: TrxScalar>(trx: &TrxFile<P>, indices: &[usize]) -> Result<TrxFile<P>> {
    let old_offsets = trx.offsets();
    let old_positions = trx.positions();

    // Build new positions and offsets
    let mut new_positions: Vec<[P; 3]> = Vec::new();
    let mut new_offsets: Vec<u32> = vec![0];

    for &idx in indices {
        let start = old_offsets[idx] as usize;
        let end = old_offsets[idx + 1] as usize;
        new_positions.extend_from_slice(&old_positions[start..end]);
        new_offsets.push(new_positions.len() as u32);
    }

    let nb_streamlines = indices.len() as u64;
    let nb_vertices = new_positions.len() as u64;

    // Remap DPS (data per streamline)
    let new_dps = remap_dps(&trx.dps, indices);

    // Remap DPV (data per vertex)
    let new_dpv = remap_dpv(&trx.dpv, old_offsets, indices);

    // Remap groups
    let new_groups = remap_groups(&trx.groups, indices);

    let header = Header {
        voxel_to_rasmm: trx.header.voxel_to_rasmm,
        dimensions: trx.header.dimensions,
        nb_streamlines,
        nb_vertices,
        extra: trx.header.extra.clone(),
    };

    let pos_bytes = crate::mmap_backing::vec_to_bytes(new_positions);
    let off_bytes = crate::mmap_backing::vec_to_bytes(new_offsets);

    Ok(TrxFile::from_parts(
        header,
        MmapBacking::Owned(pos_bytes),
        MmapBacking::Owned(off_bytes),
        new_dps,
        new_dpv,
        new_groups,
        None,
    ))
}

/// Remap DPS arrays: select rows by streamline index.
fn remap_dps(dps: &HashMap<String, DataArray>, indices: &[usize]) -> HashMap<String, DataArray> {
    let mut out = HashMap::new();
    for (name, arr) in dps {
        let row_bytes = arr.ncols * arr.dtype.size_of();
        let src = arr.backing.as_bytes();
        let mut dst = Vec::with_capacity(indices.len() * row_bytes);
        for &idx in indices {
            let start = idx * row_bytes;
            dst.extend_from_slice(&src[start..start + row_bytes]);
        }
        out.insert(
            name.clone(),
            DataArray {
                backing: MmapBacking::Owned(dst),
                ncols: arr.ncols,
                dtype: arr.dtype,
            },
        );
    }
    out
}

/// Remap DPV arrays: select vertex ranges corresponding to selected streamlines.
fn remap_dpv(
    dpv: &HashMap<String, DataArray>,
    offsets: &[u32],
    indices: &[usize],
) -> HashMap<String, DataArray> {
    let mut out = HashMap::new();
    for (name, arr) in dpv {
        let elem_bytes = arr.ncols * arr.dtype.size_of();
        let src = arr.backing.as_bytes();
        let mut dst = Vec::new();
        for &idx in indices {
            let vstart = offsets[idx] as usize;
            let vend = offsets[idx + 1] as usize;
            let byte_start = vstart * elem_bytes;
            let byte_end = vend * elem_bytes;
            dst.extend_from_slice(&src[byte_start..byte_end]);
        }
        out.insert(
            name.clone(),
            DataArray {
                backing: MmapBacking::Owned(dst),
                ncols: arr.ncols,
                dtype: arr.dtype,
            },
        );
    }
    out
}

/// Remap groups: update streamline indices to reflect the new ordering.
fn remap_groups(
    groups: &HashMap<String, DataArray>,
    indices: &[usize],
) -> HashMap<String, DataArray> {
    // Build a reverse map: old_index → new_index
    let mut old_to_new: HashMap<usize, u32> = HashMap::new();
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx as u32);
    }

    let mut out = HashMap::new();
    for (name, arr) in groups {
        let old_members: &[u32] = arr.backing.cast_slice();
        let new_members: Vec<u32> = old_members
            .iter()
            .filter_map(|&m| old_to_new.get(&(m as usize)).copied())
            .collect();
        let bytes = crate::mmap_backing::vec_to_bytes(new_members);
        out.insert(
            name.clone(),
            DataArray {
                backing: MmapBacking::Owned(bytes),
                ncols: 1,
                dtype: crate::dtype::DType::UInt32,
            },
        );
    }
    out
}

/// Per-streamline axis-aligned bounding box: `[min_x, min_y, min_z, max_x, max_y, max_z]`.
///
/// Stored as f32, matching trx-cpp's comparison precision.
pub type StreamlineAabb = [f32; 6];

/// Compute per-streamline AABBs for all streamlines in a `TrxFile`.
///
/// Returns a `Vec` of length `nb_streamlines`, where each entry is
/// `[min_x, min_y, min_z, max_x, max_y, max_z]` in f32.
///
/// This is intended to be computed once and reused across multiple queries,
/// matching trx-cpp's `build_streamline_aabbs()` / AABB cache pattern.
pub fn build_streamline_aabbs<P: TrxScalar>(trx: &TrxFile<P>) -> Vec<StreamlineAabb> {
    let offsets = trx.offsets();
    let positions = trx.positions();
    let n = trx.nb_streamlines();
    let mut aabbs = Vec::with_capacity(n);

    for i in 0..n {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for point in positions.iter().take(end).skip(start) {
            let x = point[0].to_f32();
            let y = point[1].to_f32();
            let z = point[2].to_f32();
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if z < min_z {
                min_z = z;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
            if z > max_z {
                max_z = z;
            }
        }

        aabbs.push([min_x, min_y, min_z, max_x, max_y, max_z]);
    }

    aabbs
}

/// Query pre-computed AABBs against a query box, returning matching streamline indices.
///
/// This is the fast path: O(N) with 6 float comparisons per streamline,
/// matching trx-cpp's `query_aabb()` implementation.
pub fn query_aabb_cached(aabbs: &[StreamlineAabb], min: [f64; 3], max: [f64; 3]) -> Vec<usize> {
    let min_x = min[0] as f32;
    let min_y = min[1] as f32;
    let min_z = min[2] as f32;
    let max_x = max[0] as f32;
    let max_y = max[1] as f32;
    let max_z = max[2] as f32;

    let mut result = Vec::new();

    for (i, aabb) in aabbs.iter().enumerate() {
        if aabb[0] <= max_x
            && aabb[3] >= min_x
            && aabb[1] <= max_y
            && aabb[4] >= min_y
            && aabb[2] <= max_z
            && aabb[5] >= min_z
        {
            result.push(i);
        }
    }

    result
}

/// Find streamline indices whose AABB intersects the given query box.
///
/// Convenience wrapper that builds AABBs on the fly. For repeated queries on the
/// same data, use [`build_streamline_aabbs`] + [`query_aabb_cached`] instead.
///
/// `min` and `max` define the query AABB corners. Comparisons are done in f32,
/// matching trx-cpp's behavior.
pub fn query_aabb<P: TrxScalar>(trx: &TrxFile<P>, min: [f64; 3], max: [f64; 3]) -> Vec<usize> {
    let aabbs = build_streamline_aabbs(trx);
    query_aabb_cached(&aabbs, min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::TrxStream;

    fn make_test_trx() -> TrxFile<f32> {
        let mut stream = TrxStream::<f32>::new(Header::identity_affine(), [100, 100, 100]);
        stream.push_streamline(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        stream.push_streamline(&[[10.0, 10.0, 10.0], [11.0, 11.0, 11.0], [12.0, 12.0, 12.0]]);
        stream.push_streamline(&[[20.0, 20.0, 20.0]]);
        stream.finalize()
    }

    #[test]
    fn subset_basic() {
        let trx = make_test_trx();
        let sub = subset_streamlines(&trx, &[0, 2]).unwrap();
        assert_eq!(sub.nb_streamlines(), 2);
        assert_eq!(sub.nb_vertices(), 3);
        assert_eq!(sub.streamline(0), &[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(sub.streamline(1), &[[20.0, 20.0, 20.0]]);
    }

    #[test]
    fn query_aabb_basic() {
        let trx = make_test_trx();
        let hits = query_aabb(&trx, [9.0, 9.0, 9.0], [15.0, 15.0, 15.0]);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn query_aabb_cached_basic() {
        let trx = make_test_trx();
        let aabbs = build_streamline_aabbs(&trx);
        assert_eq!(aabbs.len(), 3);

        // Streamline 0: [0,0,0] to [1,1,1]
        assert_eq!(aabbs[0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        // Streamline 1: [10,10,10] to [12,12,12]
        assert_eq!(aabbs[1], [10.0, 10.0, 10.0, 12.0, 12.0, 12.0]);

        // Query should match same results as non-cached version
        let hits = query_aabb_cached(&aabbs, [9.0, 9.0, 9.0], [15.0, 15.0, 15.0]);
        assert_eq!(hits, vec![1]);

        // Query that hits all
        let hits = query_aabb_cached(&aabbs, [-1.0, -1.0, -1.0], [25.0, 25.0, 25.0]);
        assert_eq!(hits, vec![0, 1, 2]);
    }
}
