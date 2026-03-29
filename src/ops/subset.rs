use std::collections::HashMap;

use crate::dtype::TrxScalar;
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::mmap_backing::MmapBacking;
use crate::trx_file::{DataArray, DataPerGroup, TrxFile, TrxParts};

/// Extract a subset of streamlines by index, producing a new `TrxFile`.
///
/// All DPS, DPV, and group arrays are remapped accordingly.
pub fn subset_streamlines<P: TrxScalar>(trx: &TrxFile<P>, indices: &[usize]) -> Result<TrxFile<P>> {
    let (new_positions, new_offsets) =
        collect_positions_and_offsets(trx.positions(), trx.offsets(), indices)?;

    let nb_streamlines = indices.len() as u64;
    let nb_vertices = new_positions.len() as u64;

    // Remap DPS (data per streamline)
    let new_dps = remap_dps(trx.dps_arrays(), indices);

    // Remap DPV (data per vertex)
    let new_dpv = remap_dpv(trx.dpv_arrays(), trx.offsets(), indices);

    // Remap groups
    let new_groups = remap_groups(trx.group_arrays(), indices);
    let new_dpg = remap_dpg(trx.dpg_arrays(), &new_groups);

    let header = Header {
        voxel_to_rasmm: trx.header().voxel_to_rasmm,
        dimensions: trx.header().dimensions,
        nb_streamlines,
        nb_vertices,
        extra: trx.header().extra.clone(),
    };

    let pos_bytes = crate::mmap_backing::vec_to_bytes(new_positions);
    let off_bytes = crate::mmap_backing::vec_to_bytes(new_offsets);

    Ok(TrxFile::from_parts(TrxParts {
        header,
        positions_backing: MmapBacking::Owned(pos_bytes),
        offsets_backing: MmapBacking::Owned(off_bytes),
        dps: new_dps,
        dpv: new_dpv,
        groups: new_groups,
        dpg: new_dpg,
        tempdir: None,
    }))
}

/// Remap DPS arrays: select rows by streamline index.
fn remap_dps(dps: &HashMap<String, DataArray>, indices: &[usize]) -> HashMap<String, DataArray> {
    dps.iter()
        .map(|(name, arr)| {
            let row_bytes = arr.ncols() * arr.dtype().size_of();
            let dst = copy_row_bytes(arr.as_bytes(), row_bytes, indices);
            (
                name.clone(),
                DataArray::from_backing(MmapBacking::Owned(dst), arr.ncols(), arr.dtype()),
            )
        })
        .collect()
}

/// Remap DPV arrays: select vertex ranges corresponding to selected streamlines.
fn remap_dpv(
    dpv: &HashMap<String, DataArray>,
    offsets: &[u32],
    indices: &[usize],
) -> HashMap<String, DataArray> {
    dpv.iter()
        .map(|(name, arr)| {
            let row_bytes = arr.ncols() * arr.dtype().size_of();
            let dst = copy_vertex_ranges(arr.as_bytes(), row_bytes, offsets, indices);
            (
                name.clone(),
                DataArray::from_backing(MmapBacking::Owned(dst), arr.ncols(), arr.dtype()),
            )
        })
        .collect()
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
        let old_members: &[u32] = arr.cast_slice();
        let new_members: Vec<u32> = old_members
            .iter()
            .filter_map(|&m| old_to_new.get(&(m as usize)).copied())
            .collect();
        let bytes = crate::mmap_backing::vec_to_bytes(new_members);
        out.insert(
            name.clone(),
            DataArray::from_backing(MmapBacking::Owned(bytes), 1, crate::dtype::DType::UInt32),
        );
    }
    out
}

fn remap_dpg(dpg: &DataPerGroup, groups: &HashMap<String, DataArray>) -> DataPerGroup {
    let mut out = HashMap::new();
    for (group_name, entries) in dpg {
        if let Some(group_members) = groups.get(group_name) {
            if group_members.as_bytes().is_empty() {
                continue;
            }
            out.insert(
                group_name.clone(),
                entries
                    .iter()
                    .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                    .collect(),
            );
        }
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamlineAabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl StreamlineAabb {
    pub fn min(&self) -> [f32; 3] {
        self.min
    }

    pub fn max(&self) -> [f32; 3] {
        self.max
    }

    pub fn overlaps_box(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        self.min[0] <= max[0]
            && self.max[0] >= min[0]
            && self.min[1] <= max[1]
            && self.max[1] >= min[1]
            && self.min[2] <= max[2]
            && self.max[2] >= min[2]
    }
}

/// Compute per-streamline AABBs for all streamlines in a `TrxFile`.
///
/// Returns a `Vec` of length `nb_streamlines`, where each entry is
/// `[min_x, min_y, min_z, max_x, max_y, max_z]` in f32.
///
/// This is intended to be computed once and reused across multiple queries,
/// matching trx-cpp's `build_streamline_aabbs()` / AABB cache pattern.
pub fn build_streamline_aabbs<P: TrxScalar>(trx: &TrxFile<P>) -> Vec<StreamlineAabb> {
    build_streamline_aabbs_from_iter(
        trx.offsets(),
        trx.streamlines().map(|streamline| {
            streamline
                .iter()
                .map(|point| [point[0].to_f32(), point[1].to_f32(), point[2].to_f32()])
        }),
    )
}

pub fn build_streamline_aabbs_from_slices(
    positions: &[[f32; 3]],
    offsets: &[u32],
) -> Vec<StreamlineAabb> {
    build_streamline_aabbs_from_iter(
        offsets,
        offsets.windows(2).map(|window| {
            positions[window[0] as usize..window[1] as usize]
                .iter()
                .copied()
        }),
    )
}

/// Query pre-computed AABBs against a query box, returning matching streamline indices.
///
/// This is the fast path: O(N) with 6 float comparisons per streamline,
/// matching trx-cpp's `query_aabb()` implementation.
pub fn query_aabb_cached(aabbs: &[StreamlineAabb], min: [f64; 3], max: [f64; 3]) -> Vec<usize> {
    let min = [min[0] as f32, min[1] as f32, min[2] as f32];
    let max = [max[0] as f32, max[1] as f32, max[2] as f32];

    aabbs
        .iter()
        .enumerate()
        .filter_map(|(index, aabb)| aabb.overlaps_box(min, max).then_some(index))
        .collect()
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

fn collect_positions_and_offsets<P: TrxScalar>(
    positions: &[[P; 3]],
    offsets: &[u32],
    indices: &[usize],
) -> Result<(Vec<[P; 3]>, Vec<u32>)> {
    let mut new_positions = Vec::new();
    let mut new_offsets = vec![0];

    for &idx in indices {
        let window = offsets
            .get(idx..=idx + 1)
            .ok_or_else(|| TrxError::Argument(format!("streamline index {idx} out of bounds")))?;
        new_positions.extend_from_slice(&positions[window[0] as usize..window[1] as usize]);
        new_offsets.push(
            u32::try_from(new_positions.len()).map_err(|_| {
                TrxError::Argument("subset would exceed u32::MAX vertices".into())
            })?,
        );
    }

    Ok((new_positions, new_offsets))
}

fn copy_row_bytes(src: &[u8], row_bytes: usize, indices: &[usize]) -> Vec<u8> {
    let mut dst = Vec::with_capacity(indices.len() * row_bytes);
    for &idx in indices {
        let start = idx * row_bytes;
        dst.extend_from_slice(&src[start..start + row_bytes]);
    }
    dst
}

fn copy_vertex_ranges(src: &[u8], row_bytes: usize, offsets: &[u32], indices: &[usize]) -> Vec<u8> {
    let mut dst = Vec::new();
    for &idx in indices {
        let start = offsets[idx] as usize * row_bytes;
        let end = offsets[idx + 1] as usize * row_bytes;
        dst.extend_from_slice(&src[start..end]);
    }
    dst
}

fn build_streamline_aabbs_from_iter<I, J>(offsets: &[u32], streamlines: I) -> Vec<StreamlineAabb>
where
    I: Iterator<Item = J>,
    J: Iterator<Item = [f32; 3]>,
{
    let mut aabbs = Vec::with_capacity(offsets.len().saturating_sub(1));

    for streamline in streamlines {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for point in streamline {
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
        aabbs.push(StreamlineAabb { min, max });
    }

    aabbs
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
        assert_eq!(aabbs[0].min(), [0.0, 0.0, 0.0]);
        assert_eq!(aabbs[0].max(), [1.0, 1.0, 1.0]);

        // Streamline 1: [10,10,10] to [12,12,12]
        assert_eq!(aabbs[1].min(), [10.0, 10.0, 10.0]);
        assert_eq!(aabbs[1].max(), [12.0, 12.0, 12.0]);

        // Query should match same results as non-cached version
        let hits = query_aabb_cached(&aabbs, [9.0, 9.0, 9.0], [15.0, 15.0, 15.0]);
        assert_eq!(hits, vec![1]);

        // Query that hits all
        let hits = query_aabb_cached(&aabbs, [-1.0, -1.0, -1.0], [25.0, 25.0, 25.0]);
        assert_eq!(hits, vec![0, 1, 2]);
    }
}
