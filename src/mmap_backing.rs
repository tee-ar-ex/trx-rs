use std::sync::Arc;

use bytemuck::{cast_slice, cast_slice_mut, Pod};
use memmap2::{Mmap, MmapMut};

use crate::error::{Result, TrxError};

/// Convert a `Vec<T: Pod>` into a `Vec<u8>` by copying the raw bytes.
pub fn vec_to_bytes<T: Pod>(v: Vec<T>) -> Vec<u8> {
    cast_slice::<T, u8>(&v).to_vec()
}

/// Owns the backing memory for a TRX data array.
///
/// May be a read-only mmap, a read-write mmap, an owned heap buffer, or a slice of a shared mmap.
pub enum MmapBacking {
    ReadOnly(Mmap),
    ReadWrite(MmapMut),
    SharedSlice {
        mmap: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
    Owned(Vec<u8>),
    OwnedU64(Vec<u64>, usize),
    OwnedU32(Vec<u32>, usize),
}

impl MmapBacking {
    /// Raw bytes view.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            MmapBacking::ReadOnly(m) => m,
            MmapBacking::ReadWrite(m) => m,
            MmapBacking::SharedSlice { mmap, offset, len } => &mmap[*offset..*offset + *len],
            MmapBacking::Owned(v) => v,
            MmapBacking::OwnedU64(v, len) => &bytemuck::cast_slice(v)[..*len],
            MmapBacking::OwnedU32(v, len) => &bytemuck::cast_slice(v)[..*len],
        }
    }

    /// Mutable raw bytes view (only for ReadWrite and Owned).
    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8]> {
        match self {
            MmapBacking::ReadOnly(_) => Err(TrxError::Argument(
                "cannot mutably access read-only mmap".into(),
            )),
            MmapBacking::SharedSlice { .. } => Err(TrxError::Argument(
                "cannot mutably access shared mmap slice".into(),
            )),
            MmapBacking::ReadWrite(m) => Ok(m.as_mut()),
            MmapBacking::Owned(v) => Ok(v.as_mut_slice()),
            MmapBacking::OwnedU64(v, len) => Ok(&mut bytemuck::cast_slice_mut(v)[..*len]),
            MmapBacking::OwnedU32(v, len) => Ok(&mut bytemuck::cast_slice_mut(v)[..*len]),
        }
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Whether the backing is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_mapped(&self) -> bool {
        matches!(
            self,
            MmapBacking::ReadOnly(_) | MmapBacking::ReadWrite(_) | MmapBacking::SharedSlice { .. }
        )
    }

    /// Cast the raw bytes to a typed slice.
    ///
    /// Panics if the bytes are not aligned or the length is not a multiple
    /// of `size_of::<T>()`.
    pub fn cast_slice<T: Pod>(&self) -> &[T] {
        cast_slice(self.as_bytes())
    }

    /// Cast the raw bytes to a mutable typed slice.
    pub fn cast_slice_mut<T: Pod>(&mut self) -> Result<&mut [T]> {
        let bytes = self.as_bytes_mut()?;
        Ok(cast_slice_mut(bytes))
    }
}

impl std::fmt::Debug for MmapBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapBacking::ReadOnly(m) => write!(f, "ReadOnly({} bytes)", m.len()),
            MmapBacking::ReadWrite(m) => write!(f, "ReadWrite({} bytes)", m.len()),
            MmapBacking::SharedSlice { offset, len, .. } => {
                write!(f, "SharedSlice({} bytes at offset {})", len, offset)
            }
            MmapBacking::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
            MmapBacking::OwnedU64(_, len) => write!(f, "OwnedU64({} bytes)", len),
            MmapBacking::OwnedU32(_, len) => write!(f, "OwnedU32({} bytes)", len),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_slice_behavior_and_mapping_status() {
        let mut anon = memmap2::MmapMut::map_anon(16).unwrap();
        anon[0..4].copy_from_slice(&1.0f32.to_ne_bytes());
        anon[4..8].copy_from_slice(&2.0f32.to_ne_bytes());
        let mmap = Arc::new(anon.make_read_only().unwrap());

        let mut backing = MmapBacking::SharedSlice {
            mmap: Arc::clone(&mmap),
            offset: 0,
            len: 8,
        };

        assert!(backing.is_mapped());
        assert_eq!(backing.len(), 8);
        assert_eq!(backing.as_bytes(), &mmap[0..8]);
        assert_eq!(backing.cast_slice::<f32>(), &[1.0f32, 2.0f32]);
        assert!(backing.as_bytes_mut().is_err());

        let owned = MmapBacking::Owned(vec![1, 2, 3]);
        assert!(!owned.is_mapped());
    }
}
