use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::dtype::TrxScalar;
use crate::error::Result;
use crate::tractogram::Tractogram;
use crate::trx_file::TrxFile;

use super::subset::subset_streamlines;

/// A hashable representation of a streamline for set operations.
/// Uses the raw bytes of the positions to compute identity.
#[derive(Clone)]
struct StreamlineKey(Vec<u8>);

impl PartialEq for StreamlineKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for StreamlineKey {}

impl Hash for StreamlineKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

fn streamline_key<P: TrxScalar>(points: &[[P; 3]]) -> StreamlineKey {
    StreamlineKey(bytemuck::cast_slice::<[P; 3], u8>(points).to_vec())
}

pub fn retain_representative_indices<P: TrxScalar>(
    trx: &TrxFile<P>,
    params: &DuplicateRemovalParams,
) -> Vec<usize> {
    retain_representative_indices_impl(
        trx.offsets().len().saturating_sub(1),
        |index| {
            trx.streamline(index)
                .iter()
                .map(|point| [point[0].to_f32(), point[1].to_f32(), point[2].to_f32()])
                .collect()
        },
        params,
    )
}

pub fn remove_duplicates<P: TrxScalar>(
    trx: &TrxFile<P>,
    params: &DuplicateRemovalParams,
) -> Result<TrxFile<P>> {
    let indices = retain_representative_indices(trx, params);
    subset_streamlines(trx, &indices)
}

pub fn retain_tractogram_representative_indices(
    tractogram: &Tractogram,
    params: &DuplicateRemovalParams,
) -> Vec<usize> {
    retain_representative_indices_impl(
        tractogram.nb_streamlines(),
        |index| tractogram.streamline(index).to_vec(),
        params,
    )
}

pub fn remove_duplicates_tractogram(
    tractogram: &Tractogram,
    params: &DuplicateRemovalParams,
) -> Result<Tractogram> {
    let indices = retain_tractogram_representative_indices(tractogram, params);
    tractogram.subset_streamlines(&indices)
}

fn retain_representative_indices_impl(
    streamline_count: usize,
    mut points_for_index: impl FnMut(usize) -> Vec<[f32; 3]>,
    params: &DuplicateRemovalParams,
) -> Vec<usize> {
    let mut retained = Vec::<usize>::new();
    let mut exact_seen = HashSet::<StreamlineKey>::new();

    if matches!(params.mode, DuplicateRemovalMode::Exact) {
        for streamline_index in 0..streamline_count {
            let points = canonicalize_streamline_points(&points_for_index(streamline_index));
            if exact_seen.insert(canonical_streamline_key(&points)) {
                retained.push(streamline_index);
            }
        }
        return retained;
    }

    let mut representatives = Vec::<RepresentativeDescriptor>::new();
    let mut endpoint_buckets = HashMap::<(VoxelKey, VoxelKey), Vec<usize>>::new();

    for streamline_index in 0..streamline_count {
        let points = canonicalize_streamline_points(&points_for_index(streamline_index));
        if !exact_seen.insert(canonical_streamline_key(&points)) {
            continue;
        }

        let descriptor = RepresentativeDescriptor::from_points(&points, params);
        let mut candidate_indices = HashSet::<usize>::new();
        for start_bucket in neighboring_voxels(descriptor.start_bucket, 1) {
            for end_bucket in neighboring_voxels(descriptor.end_bucket, 1) {
                if let Some(indices) = endpoint_buckets.get(&(start_bucket, end_bucket)) {
                    candidate_indices.extend(indices.iter().copied());
                }
            }
        }

        let is_duplicate = candidate_indices.into_iter().any(|candidate_index| {
            representatives_match(&representatives[candidate_index], &descriptor, params)
        });
        if is_duplicate {
            continue;
        }

        let representative_index = representatives.len();
        representatives.push(descriptor);
        endpoint_buckets
            .entry((
                representatives[representative_index].start_bucket,
                representatives[representative_index].end_bucket,
            ))
            .or_default()
            .push(representative_index);
        retained.push(streamline_index);
    }

    retained
}

impl RepresentativeDescriptor {
    fn from_points(points: &[[f32; 3]], params: &DuplicateRemovalParams) -> Self {
        let (bbox_min, bbox_max) = compute_bounding_box(points);
        let start = points.first().copied().unwrap_or([0.0, 0.0, 0.0]);
        let end = points.last().copied().unwrap_or(start);
        let endpoint_cell = params.endpoint_tolerance_mm.max(1e-3);
        let tolerance_mm = params.tolerance_mm.max(1e-3);

        Self {
            points: points.to_vec(),
            length_mm: streamline_length(points),
            bbox_min,
            bbox_max,
            start,
            end,
            start_bucket: quantize_point(start, endpoint_cell),
            end_bucket: quantize_point(end, endpoint_cell),
            voxels: rasterize_streamline_voxels(points, tolerance_mm),
            segment_hash: SegmentSpatialHash::build(points, tolerance_mm),
        }
    }
}

fn representatives_match(
    left: &RepresentativeDescriptor,
    right: &RepresentativeDescriptor,
    params: &DuplicateRemovalParams,
) -> bool {
    if !lengths_compatible(left.length_mm, right.length_mm, params) {
        return false;
    }
    if !expanded_aabb_overlap(left, right, params.tolerance_mm) {
        return false;
    }
    if euclidean_distance(left.start, right.start) > params.endpoint_tolerance_mm
        || euclidean_distance(left.end, right.end) > params.endpoint_tolerance_mm
    {
        return false;
    }
    if voxel_overlap_fraction(&left.voxels, &right.voxels) < params.min_shared_voxel_fraction {
        return false;
    }
    symmetric_streamline_match(left, right, params.tolerance_mm)
}

fn symmetric_streamline_match(
    left: &RepresentativeDescriptor,
    right: &RepresentativeDescriptor,
    tolerance_mm: f32,
) -> bool {
    points_match_streamline(&left.points, right, tolerance_mm)
        && points_match_streamline(&right.points, left, tolerance_mm)
}

fn points_match_streamline(
    points: &[[f32; 3]],
    representative: &RepresentativeDescriptor,
    tolerance_mm: f32,
) -> bool {
    if representative.segment_hash.segments.is_empty() {
        let tol2 = tolerance_mm * tolerance_mm;
        return points.iter().copied().all(|point| {
            representative
                .points
                .iter()
                .copied()
                .any(|other| squared_distance(point, other) <= tol2)
        });
    }

    points.iter().copied().all(|point| {
        representative
            .segment_hash
            .point_within_tolerance(point, tolerance_mm)
    })
}

fn canonical_streamline_key(points: &[[f32; 3]]) -> StreamlineKey {
    StreamlineKey(bytemuck::cast_slice::<[f32; 3], u8>(points).to_vec())
}

fn canonicalize_streamline_points(points: &[[f32; 3]]) -> Vec<[f32; 3]> {
    if points.len() <= 1 || canonical_orientation(points) {
        points.to_vec()
    } else {
        points.iter().rev().copied().collect()
    }
}

fn canonical_orientation(points: &[[f32; 3]]) -> bool {
    for index in 0..points.len() {
        let forward = points[index];
        let reverse = points[points.len() - 1 - index];
        match compare_point(forward, reverse) {
            Ordering::Less => return true,
            Ordering::Greater => return false,
            Ordering::Equal => continue,
        }
    }
    true
}

fn compare_point(left: [f32; 3], right: [f32; 3]) -> Ordering {
    for axis in 0..3 {
        match left[axis]
            .partial_cmp(&right[axis])
            .unwrap_or_else(|| left[axis].to_bits().cmp(&right[axis].to_bits()))
        {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

fn compute_bounding_box(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    if points.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (min, max)
    }
}

fn streamline_length(points: &[[f32; 3]]) -> f32 {
    points
        .windows(2)
        .map(|window| euclidean_distance(window[0], window[1]))
        .sum()
}

fn lengths_compatible(left: f32, right: f32, params: &DuplicateRemovalParams) -> bool {
    let shorter = left.min(right);
    let longer = left.max(right);
    longer - shorter
        <= shorter * 0.1 + params.endpoint_tolerance_mm * 2.0 + params.tolerance_mm * 2.0
}

fn expanded_aabb_overlap(
    left: &RepresentativeDescriptor,
    right: &RepresentativeDescriptor,
    tolerance_mm: f32,
) -> bool {
    left.bbox_min[0] - tolerance_mm <= right.bbox_max[0]
        && left.bbox_max[0] + tolerance_mm >= right.bbox_min[0]
        && left.bbox_min[1] - tolerance_mm <= right.bbox_max[1]
        && left.bbox_max[1] + tolerance_mm >= right.bbox_min[1]
        && left.bbox_min[2] - tolerance_mm <= right.bbox_max[2]
        && left.bbox_max[2] + tolerance_mm >= right.bbox_min[2]
}

fn rasterize_streamline_voxels(points: &[[f32; 3]], cell_size: f32) -> Vec<VoxelKey> {
    let mut voxels = HashSet::<VoxelKey>::new();
    for point in points.iter().copied() {
        voxels.insert(quantize_point(point, cell_size));
    }
    for window in points.windows(2) {
        let start = window[0];
        let end = window[1];
        let length = euclidean_distance(start, end);
        let steps = ((length / (cell_size * 0.5)).ceil() as usize).max(1);
        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let point = [
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t,
            ];
            voxels.insert(quantize_point(point, cell_size));
        }
    }
    let mut voxels = voxels.into_iter().collect::<Vec<_>>();
    voxels.sort_unstable();
    voxels
}

fn voxel_overlap_fraction(left: &[VoxelKey], right: &[VoxelKey]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return if left.is_empty() && right.is_empty() {
            1.0
        } else {
            0.0
        };
    }

    let mut intersection = 0usize;
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => left_index += 1,
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                intersection += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    intersection as f32 / left.len().min(right.len()) as f32
}

fn neighboring_voxels(center: VoxelKey, radius: i32) -> Vec<VoxelKey> {
    let mut voxels = Vec::with_capacity(((radius * 2 + 1).pow(3)) as usize);
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            for dz in -radius..=radius {
                voxels.push((center.0 + dx, center.1 + dy, center.2 + dz));
            }
        }
    }
    voxels
}

fn quantize_point(point: [f32; 3], cell_size: f32) -> VoxelKey {
    (
        (point[0] / cell_size).floor() as i32,
        (point[1] / cell_size).floor() as i32,
        (point[2] / cell_size).floor() as i32,
    )
}

fn squared_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    let dz = left[2] - right[2];
    dx * dx + dy * dy + dz * dz
}

fn euclidean_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    squared_distance(left, right).sqrt()
}

fn point_segment_distance_squared(point: [f32; 3], start: [f32; 3], end: [f32; 3]) -> f32 {
    let ab = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let ap = [point[0] - start[0], point[1] - start[1], point[2] - start[2]];
    let ab_len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if ab_len2 <= 1e-12 {
        return squared_distance(point, start);
    }

    let t = ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / ab_len2).clamp(0.0, 1.0);
    let closest = [
        start[0] + ab[0] * t,
        start[1] + ab[1] * t,
        start[2] + ab[2] * t,
    ];
    squared_distance(point, closest)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DuplicateRemovalMode {
    Exact,
    Near,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DuplicateRemovalParams {
    pub mode: DuplicateRemovalMode,
    pub tolerance_mm: f32,
    pub endpoint_tolerance_mm: f32,
    pub min_shared_voxel_fraction: f32,
}

impl Default for DuplicateRemovalParams {
    fn default() -> Self {
        Self {
            mode: DuplicateRemovalMode::Near,
            tolerance_mm: 0.5,
            endpoint_tolerance_mm: 1.0,
            min_shared_voxel_fraction: 1.0,
        }
    }
}

type VoxelKey = (i32, i32, i32);

#[derive(Clone)]
struct RepresentativeDescriptor {
    points: Vec<[f32; 3]>,
    length_mm: f32,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    start: [f32; 3],
    end: [f32; 3],
    start_bucket: VoxelKey,
    end_bucket: VoxelKey,
    voxels: Vec<VoxelKey>,
    segment_hash: SegmentSpatialHash,
}

#[derive(Clone, Default)]
struct SegmentSpatialHash {
    cell_size: f32,
    segments: Vec<([f32; 3], [f32; 3])>,
    cells: HashMap<VoxelKey, Vec<usize>>,
}

impl SegmentSpatialHash {
    fn build(points: &[[f32; 3]], tolerance_mm: f32) -> Self {
        let cell_size = tolerance_mm.max(1e-3);
        let mut segments = Vec::with_capacity(points.len().saturating_sub(1));
        let mut cells = HashMap::<VoxelKey, Vec<usize>>::new();

        for window in points.windows(2) {
            let p0 = window[0];
            let p1 = window[1];
            let segment_index = segments.len();
            segments.push((p0, p1));

            let min = [
                p0[0].min(p1[0]) - tolerance_mm,
                p0[1].min(p1[1]) - tolerance_mm,
                p0[2].min(p1[2]) - tolerance_mm,
            ];
            let max = [
                p0[0].max(p1[0]) + tolerance_mm,
                p0[1].max(p1[1]) + tolerance_mm,
                p0[2].max(p1[2]) + tolerance_mm,
            ];
            let min_key = quantize_point(min, cell_size);
            let max_key = quantize_point(max, cell_size);

            for ix in min_key.0..=max_key.0 {
                for iy in min_key.1..=max_key.1 {
                    for iz in min_key.2..=max_key.2 {
                        cells.entry((ix, iy, iz)).or_default().push(segment_index);
                    }
                }
            }
        }

        Self {
            cell_size,
            segments,
            cells,
        }
    }

    fn point_within_tolerance(&self, point: [f32; 3], tolerance_mm: f32) -> bool {
        if self.segments.is_empty() {
            return false;
        }
        let key = quantize_point(point, self.cell_size);
        let tol2 = tolerance_mm * tolerance_mm;
        self.cells.get(&key).is_some_and(|segments| {
            segments.iter().copied().any(|segment_index| {
                let (start, end) = self.segments[segment_index];
                point_segment_distance_squared(point, start, end) <= tol2
            })
        })
    }
}

/// Compute the intersection: streamlines present in both `a` and `b`.
/// Returns indices into `a`.
pub fn intersection_indices<P: TrxScalar>(a: &TrxFile<P>, b: &TrxFile<P>) -> Vec<usize> {
    let b_set: HashSet<StreamlineKey> = b.streamlines().map(streamline_key).collect();

    a.streamlines()
        .enumerate()
        .filter_map(|(index, streamline)| {
            b_set.contains(&streamline_key(streamline)).then_some(index)
        })
        .collect()
}

/// Compute the difference: streamlines in `a` but not in `b`.
/// Returns indices into `a`.
pub fn difference_indices<P: TrxScalar>(a: &TrxFile<P>, b: &TrxFile<P>) -> Vec<usize> {
    let b_set: HashSet<StreamlineKey> = b.streamlines().map(streamline_key).collect();

    a.streamlines()
        .enumerate()
        .filter_map(|(index, streamline)| {
            (!b_set.contains(&streamline_key(streamline))).then_some(index)
        })
        .collect()
}

/// Intersection: return a new TrxFile with streamlines present in both.
pub fn intersection<P: TrxScalar>(a: &TrxFile<P>, b: &TrxFile<P>) -> Result<TrxFile<P>> {
    let indices = intersection_indices(a, b);
    subset_streamlines(a, &indices)
}

/// Difference: return a new TrxFile with streamlines in `a` but not in `b`.
pub fn difference<P: TrxScalar>(a: &TrxFile<P>, b: &TrxFile<P>) -> Result<TrxFile<P>> {
    let indices = difference_indices(a, b);
    subset_streamlines(a, &indices)
}

/// Union: return a new TrxFile with all unique streamlines from both.
pub fn streamline_union<P: TrxScalar>(a: &TrxFile<P>, b: &TrxFile<P>) -> Result<TrxFile<P>> {
    // Start with all of a, add streamlines from b not already in a
    let a_set: HashSet<StreamlineKey> = a.streamlines().map(streamline_key).collect();

    let mut stream =
        crate::stream::TrxStream::<P>::new(a.header().voxel_to_rasmm, a.header().dimensions);

    // Add all streamlines from a
    for streamline in a.streamlines() {
        stream.push_streamline(streamline);
    }

    // Add unique streamlines from b
    for streamline in b.streamlines() {
        let key = streamline_key(streamline);
        if !a_set.contains(&key) {
            stream.push_streamline(streamline);
        }
    }

    Ok(stream.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tractogram_from_streamlines(streamlines: &[Vec<[f32; 3]>]) -> Tractogram {
        let mut tractogram = Tractogram::new();
        for streamline in streamlines {
            tractogram.push_streamline(streamline).unwrap();
        }
        tractogram
    }

    #[test]
    fn exact_mode_retains_one_representative_for_reversed_duplicates() {
        let tractogram = tractogram_from_streamlines(&[
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ]);

        let params = DuplicateRemovalParams {
            mode: DuplicateRemovalMode::Exact,
            ..DuplicateRemovalParams::default()
        };
        let kept = retain_tractogram_representative_indices(&tractogram, &params);
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn near_mode_retains_one_representative_for_offset_duplicates() {
        let tractogram = tractogram_from_streamlines(&[
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
        ]);

        let kept = retain_tractogram_representative_indices(
            &tractogram,
            &DuplicateRemovalParams {
                mode: DuplicateRemovalMode::Near,
                tolerance_mm: 0.35,
                endpoint_tolerance_mm: 0.5,
                min_shared_voxel_fraction: 1.0,
            },
        );
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn near_mode_preserves_endpoint_mismatched_streamlines() {
        let tractogram = tractogram_from_streamlines(&[
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [3.0, 0.0, 0.0]],
        ]);

        let kept = retain_tractogram_representative_indices(
            &tractogram,
            &DuplicateRemovalParams {
                mode: DuplicateRemovalMode::Near,
                tolerance_mm: 0.35,
                endpoint_tolerance_mm: 0.5,
                min_shared_voxel_fraction: 0.75,
            },
        );
        assert_eq!(kept, vec![0, 1]);
    }

    #[test]
    fn near_mode_preserves_distinct_parallel_streamlines() {
        let tractogram = tractogram_from_streamlines(&[
            vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            vec![[0.0, 1.5, 0.0], [3.0, 1.5, 0.0]],
        ]);

        let kept = retain_tractogram_representative_indices(
            &tractogram,
            &DuplicateRemovalParams {
                mode: DuplicateRemovalMode::Near,
                tolerance_mm: 0.5,
                endpoint_tolerance_mm: 0.75,
                min_shared_voxel_fraction: 0.8,
            },
        );
        assert_eq!(kept, vec![0, 1]);
    }
}
