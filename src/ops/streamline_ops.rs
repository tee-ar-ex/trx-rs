use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::dtype::TrxScalar;
use crate::error::Result;
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
pub fn union<P: TrxScalar>(a: &TrxFile<P>, b: &TrxFile<P>) -> Result<TrxFile<P>> {
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
