use half::f16;
use std::collections::HashMap;

use crate::any_trx_file::{AnyTrxFile, PositionsRef};
use crate::dtype::{DType, TrxScalar};
use crate::error::{Result, TrxError};
use crate::header::Header;
use crate::mmap_backing::{vec_to_bytes, MmapBacking};
use crate::trx_file::{DataArray, DataPerGroup, TrxFile, TrxParts};

/// Neutral in-memory streamline representation used for cross-format conversion.
///
/// Positions are stored in world/RASMM-style coordinates as `f32` triplets.
#[derive(Debug)]
pub struct Tractogram {
    header: Header,
    positions: Vec<[f32; 3]>,
    offsets: Vec<u32>,
    groups: HashMap<String, Vec<u32>>,
    dpg: DataPerGroup,
}

impl Clone for Tractogram {
    fn clone(&self) -> Self {
        Self {
            header: self.header.clone(),
            positions: self.positions.clone(),
            offsets: self.offsets.clone(),
            groups: self.groups.clone(),
            dpg: clone_dpg(&self.dpg),
        }
    }
}

impl Tractogram {
    /// Create an empty tractogram with identity metadata.
    pub fn new() -> Self {
        Self::with_header(Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [1, 1, 1],
            nb_streamlines: 0,
            nb_vertices: 0,
            extra: Default::default(),
        })
    }

    /// Create an empty tractogram with a caller-provided header.
    pub fn with_header(mut header: Header) -> Self {
        header.nb_streamlines = 0;
        header.nb_vertices = 0;
        Self {
            header,
            positions: Vec::new(),
            offsets: vec![0],
            groups: HashMap::new(),
            dpg: HashMap::new(),
        }
    }

    /// Build a tractogram by copying streamline geometry from a typed TRX file.
    pub fn from_trx<P: TrxScalar>(trx: &TrxFile<P>) -> Self {
        let positions = trx
            .positions()
            .iter()
            .map(|point| [point[0].to_f32(), point[1].to_f32(), point[2].to_f32()])
            .collect();
        Self {
            header: trx.header().clone(),
            positions,
            offsets: trx.offsets().to_vec(),
            groups: clone_groups(trx.group_arrays()),
            dpg: clone_dpg(trx.dpg_arrays()),
        }
    }

    /// Build a tractogram by copying streamline geometry from a dtype-erased TRX file.
    pub fn from_any_trx(trx: &AnyTrxFile) -> Self {
        let positions = match trx.positions_ref() {
            PositionsRef::F16(data) => data
                .iter()
                .map(|point| [point[0].to_f32(), point[1].to_f32(), point[2].to_f32()])
                .collect(),
            PositionsRef::F32(data) => data.to_vec(),
            PositionsRef::F64(data) => data
                .iter()
                .map(|point| [point[0] as f32, point[1] as f32, point[2] as f32])
                .collect(),
        };

        Self {
            header: trx.header().clone(),
            positions,
            offsets: trx.offsets_vec(),
            groups: trx.groups_owned().into_iter().collect(),
            dpg: trx.with_typed(
                |inner| clone_dpg(inner.dpg_arrays()),
                |inner| clone_dpg(inner.dpg_arrays()),
                |inner| clone_dpg(inner.dpg_arrays()),
            ),
        }
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn set_spatial_metadata(&mut self, voxel_to_rasmm: [[f64; 4]; 4], dimensions: [u64; 3]) {
        self.header.voxel_to_rasmm = voxel_to_rasmm;
        self.header.dimensions = dimensions;
    }

    pub fn extra(&self) -> &HashMap<String, serde_json::Value> {
        &self.header.extra
    }

    pub fn extra_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        &mut self.header.extra
    }

    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn group_names(&self) -> impl Iterator<Item = &str> {
        self.groups.keys().map(String::as_str)
    }

    pub fn group(&self, name: &str) -> Option<&[u32]> {
        self.groups.get(name).map(Vec::as_slice)
    }

    pub fn groups(&self) -> &HashMap<String, Vec<u32>> {
        &self.groups
    }

    pub fn insert_group(&mut self, name: impl Into<String>, members: Vec<u32>) {
        self.groups.insert(name.into(), members);
    }

    pub fn insert_dpg(
        &mut self,
        group: impl Into<String>,
        name: impl Into<String>,
        data: DataArray,
    ) {
        self.dpg
            .entry(group.into())
            .or_default()
            .insert(name.into(), data);
    }

    pub fn dpg(&self) -> &DataPerGroup {
        &self.dpg
    }

    /// Add a streamline to the tractogram.
    pub fn push_streamline(&mut self, points: &[[f32; 3]]) -> Result<()> {
        self.positions.extend_from_slice(points);
        let next_offset = u32::try_from(self.positions.len())
            .map_err(|_| TrxError::Argument("tractogram has more than u32::MAX vertices".into()))?;
        self.offsets.push(next_offset);
        self.header.nb_streamlines += 1;
        self.header.nb_vertices = self.positions.len() as u64;
        Ok(())
    }

    /// Number of streamlines.
    pub fn nb_streamlines(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Number of vertices.
    pub fn nb_vertices(&self) -> usize {
        self.positions.len()
    }

    /// Borrow a single streamline as a position slice.
    pub fn streamline(&self, index: usize) -> &[[f32; 3]] {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        &self.positions[start..end]
    }

    /// Iterate over streamlines.
    pub fn streamlines(&self) -> impl Iterator<Item = &[[f32; 3]]> {
        self.offsets.windows(2).map(|window| {
            let start = window[0] as usize;
            let end = window[1] as usize;
            &self.positions[start..end]
        })
    }

    /// Materialize a TRX file using the requested positions dtype.
    pub fn to_trx(&self, dtype: DType) -> Result<AnyTrxFile> {
        match dtype {
            DType::Float16 => Ok(AnyTrxFile::F16(self.to_trx_typed::<f16>()?)),
            DType::Float32 => Ok(AnyTrxFile::F32(self.to_trx_typed::<f32>()?)),
            DType::Float64 => Ok(AnyTrxFile::F64(self.to_trx_typed::<f64>()?)),
            other => Err(TrxError::DType(format!(
                "TRX positions must be float16, float32, or float64, got {other}"
            ))),
        }
    }

    fn to_trx_typed<P>(&self) -> Result<TrxFile<P>>
    where
        P: TrxScalar + FromF32,
    {
        let positions: Vec<[P; 3]> = self
            .positions
            .iter()
            .map(|point| {
                [
                    P::from_f32(point[0]),
                    P::from_f32(point[1]),
                    P::from_f32(point[2]),
                ]
            })
            .collect();

        let mut header = self.header.clone();
        header.nb_streamlines = self.nb_streamlines() as u64;
        header.nb_vertices = self.nb_vertices() as u64;

        Ok(TrxFile::from_parts(TrxParts {
            header,
            positions_backing: MmapBacking::Owned(vec_to_bytes(positions)),
            offsets_backing: MmapBacking::Owned(vec_to_bytes(self.offsets.clone())),
            dps: HashMap::new(),
            dpv: HashMap::new(),
            groups: groups_to_data_arrays(&self.groups),
            dpg: clone_dpg(&self.dpg),
            tempdir: None,
        }))
    }
}

impl<P: TrxScalar> From<&TrxFile<P>> for Tractogram {
    fn from(value: &TrxFile<P>) -> Self {
        Self::from_trx(value)
    }
}

impl From<&AnyTrxFile> for Tractogram {
    fn from(value: &AnyTrxFile) -> Self {
        Self::from_any_trx(value)
    }
}

impl Default for Tractogram {
    fn default() -> Self {
        Self::new()
    }
}

fn clone_groups(groups: &HashMap<String, DataArray>) -> HashMap<String, Vec<u32>> {
    groups
        .iter()
        .map(|(name, arr)| (name.clone(), arr.cast_slice::<u32>().to_vec()))
        .collect()
}

fn groups_to_data_arrays(groups: &HashMap<String, Vec<u32>>) -> HashMap<String, DataArray> {
    groups
        .iter()
        .map(|(name, members)| {
            (
                name.clone(),
                DataArray::owned_bytes(vec_to_bytes(members.clone()), 1, DType::UInt32),
            )
        })
        .collect()
}

fn clone_dpg(dpg: &DataPerGroup) -> DataPerGroup {
    dpg.iter()
        .map(|(group, entries)| {
            (
                group.clone(),
                entries
                    .iter()
                    .map(|(name, arr)| (name.clone(), arr.clone_owned()))
                    .collect(),
            )
        })
        .collect()
}

trait FromF32 {
    fn from_f32(value: f32) -> Self;
}

impl FromF32 for f16 {
    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }
}

impl FromF32 for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl FromF32 for f64 {
    fn from_f32(value: f32) -> Self {
        value as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamline_push_updates_counts() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .unwrap();
        tractogram.push_streamline(&[[7.0, 8.0, 9.0]]).unwrap();

        assert_eq!(tractogram.nb_streamlines(), 2);
        assert_eq!(tractogram.nb_vertices(), 3);
        assert_eq!(tractogram.offsets, vec![0, 2, 3]);
        assert_eq!(tractogram.streamline(1), &[[7.0, 8.0, 9.0]]);
    }
}
