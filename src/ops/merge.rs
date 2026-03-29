use crate::dtype::TrxScalar;
use crate::error::{Result, TrxError};
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::stream::TrxStream;
use crate::trx_file::{DataArray, TrxFile};

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
    let mut stream = TrxStream::<P>::new(first.header().voxel_to_rasmm, first.header().dimensions);

    for shard in shards {
        for streamline in shard.streamlines() {
            stream.push_streamline(streamline);
        }
    }

    let mut merged = stream.finalize();

    // Merge DPS: only fields present in ALL shards
    for name in common_field_names(
        first.dps_arrays(),
        shards.iter().map(|shard| shard.dps_arrays()),
    ) {
        let first_arr = &first.dps_arrays()[&name];
        let bytes = concatenate_arrays(shards.iter().map(|shard| &shard.dps_arrays()[&name]));
        merged.dps_arrays_mut().insert(
            name,
            DataArray::from_backing(
                MmapBacking::Owned(bytes),
                first_arr.ncols(),
                first_arr.dtype(),
            ),
        );
    }

    // Merge DPV: only fields present in ALL shards
    for name in common_field_names(
        first.dpv_arrays(),
        shards.iter().map(|shard| shard.dpv_arrays()),
    ) {
        let first_arr = &first.dpv_arrays()[&name];
        let bytes = concatenate_arrays(shards.iter().map(|shard| &shard.dpv_arrays()[&name]));
        merged.dpv_arrays_mut().insert(
            name,
            DataArray::from_backing(
                MmapBacking::Owned(bytes),
                first_arr.ncols(),
                first_arr.dtype(),
            ),
        );
    }

    // Merge groups with index remapping
    for name in common_field_names(
        first.group_arrays(),
        shards.iter().map(|shard| shard.group_arrays()),
    ) {
        let mut all_members: Vec<u32> = Vec::new();
        let mut streamline_offset: u32 = 0;
        for shard in shards {
            let members: &[u32] = shard.group_arrays()[&name].cast_slice();
            all_members.extend(members.iter().map(|&m| m + streamline_offset));
            streamline_offset += shard.nb_streamlines() as u32;
        }
        merged.group_arrays_mut().insert(
            name,
            DataArray::from_backing(
                MmapBacking::Owned(vec_to_bytes(all_members)),
                1,
                crate::dtype::DType::UInt32,
            ),
        );
    }

    for group in common_field_names(
        first.dpg_arrays(),
        shards.iter().map(|shard| shard.dpg_arrays()),
    ) {
        let merged_group = first.dpg_arrays()[&group]
            .iter()
            .map(|(name, arr)| (name.clone(), arr.clone_owned()))
            .collect();
        merged.dpg_arrays_mut().insert(group, merged_group);
    }

    Ok(merged)
}

fn common_field_names<'a, T: 'a>(
    first: &'a std::collections::HashMap<String, T>,
    others: impl Iterator<Item = &'a std::collections::HashMap<String, T>>,
) -> Vec<String> {
    let other_maps: Vec<_> = others.collect();
    first
        .keys()
        .filter(|name| other_maps.iter().all(|map| map.contains_key(*name)))
        .cloned()
        .collect()
}

fn concatenate_arrays<'a>(arrays: impl Iterator<Item = &'a DataArray>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for arr in arrays {
        bytes.extend_from_slice(arr.as_bytes());
    }
    bytes
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
