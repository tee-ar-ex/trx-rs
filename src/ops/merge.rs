use crate::dtype::TrxScalar;
use crate::error::{Result, TrxError};
use crate::stream::TrxStream;
use crate::trx_file::TrxFile;

/// Merge multiple TRX files into one.
///
/// All inputs must have the same affine and dimensions. Positions, offsets,
/// and DPS/DPV arrays that exist in ALL inputs are concatenated. Groups
/// have their indices remapped.
pub fn merge_trx_shards<P: TrxScalar>(shards: &[&TrxFile<P>]) -> Result<TrxFile<P>> {
    if shards.is_empty() {
        return Err(TrxError::Argument("no shards to merge".into()));
    }

    let first = shards[0];

    // Build merged positions and offsets via TrxStream
    let mut stream = TrxStream::<P>::new(first.header.voxel_to_rasmm, first.header.dimensions);

    for shard in shards {
        for i in 0..shard.nb_streamlines() {
            stream.push_streamline(shard.streamline(i));
        }
    }

    let mut merged = stream.finalize();

    // Merge DPS: only fields present in ALL shards
    let common_dps: Vec<String> = first
        .dps
        .keys()
        .filter(|k| shards.iter().all(|s| s.dps.contains_key(*k)))
        .cloned()
        .collect();

    for name in &common_dps {
        let first_arr = &first.dps[name];
        let mut bytes = Vec::new();
        for shard in shards {
            bytes.extend_from_slice(shard.dps[name].backing.as_bytes());
        }
        merged.dps.insert(
            name.clone(),
            crate::trx_file::DataArray {
                backing: crate::mmap_backing::MmapBacking::Owned(bytes),
                ncols: first_arr.ncols,
                dtype: first_arr.dtype,
            },
        );
    }

    // Merge DPV: only fields present in ALL shards
    let common_dpv: Vec<String> = first
        .dpv
        .keys()
        .filter(|k| shards.iter().all(|s| s.dpv.contains_key(*k)))
        .cloned()
        .collect();

    for name in &common_dpv {
        let first_arr = &first.dpv[name];
        let mut bytes = Vec::new();
        for shard in shards {
            bytes.extend_from_slice(shard.dpv[name].backing.as_bytes());
        }
        merged.dpv.insert(
            name.clone(),
            crate::trx_file::DataArray {
                backing: crate::mmap_backing::MmapBacking::Owned(bytes),
                ncols: first_arr.ncols,
                dtype: first_arr.dtype,
            },
        );
    }

    // Merge groups with index remapping
    let common_groups: Vec<String> = first
        .groups
        .keys()
        .filter(|k| shards.iter().all(|s| s.groups.contains_key(*k)))
        .cloned()
        .collect();

    for name in &common_groups {
        let mut all_members: Vec<u32> = Vec::new();
        let mut streamline_offset: u32 = 0;
        for shard in shards {
            let members: &[u32] = shard.groups[name.as_str()].backing.cast_slice();
            all_members.extend(members.iter().map(|&m| m + streamline_offset));
            streamline_offset += shard.nb_streamlines() as u32;
        }
        let bytes = crate::mmap_backing::vec_to_bytes(all_members);
        merged.groups.insert(
            name.clone(),
            crate::trx_file::DataArray {
                backing: crate::mmap_backing::MmapBacking::Owned(bytes),
                ncols: 1,
                dtype: crate::dtype::DType::UInt32,
            },
        );
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::stream::TrxStream;

    #[test]
    fn merge_two_shards() {
        let mut s1 = TrxStream::<f32>::new(Header::identity_affine(), [100, 100, 100]);
        s1.push_streamline(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        let t1 = s1.finalize();

        let mut s2 = TrxStream::<f32>::new(Header::identity_affine(), [100, 100, 100]);
        s2.push_streamline(&[[2.0, 2.0, 2.0]]);
        s2.push_streamline(&[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]);
        let t2 = s2.finalize();

        let merged = merge_trx_shards(&[&t1, &t2]).unwrap();
        assert_eq!(merged.nb_streamlines(), 3);
        assert_eq!(merged.nb_vertices(), 6);
        assert_eq!(merged.streamline(0), &[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(merged.streamline(1), &[[2.0, 2.0, 2.0]]);
        assert_eq!(
            merged.streamline(2),
            &[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]
        );
    }
}
