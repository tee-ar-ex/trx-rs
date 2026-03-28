use std::path::Path;

use crate::dtype::TrxScalar;
use crate::error::Result;
use crate::header::Header;
use crate::mmap_backing::MmapBacking;
use crate::trx_file::TrxFile;

/// Incremental builder for constructing a TRX file streamline-by-streamline.
///
/// Buffers positions and offsets in memory, then finalizes into a `TrxFile`
/// or writes directly to disk.
pub struct TrxStream<P: TrxScalar> {
    positions: Vec<[P; 3]>,
    offsets: Vec<u32>,
    header: Header,
}

impl<P: TrxScalar> TrxStream<P> {
    /// Create a new stream builder with the given affine and dimensions.
    pub fn new(voxel_to_rasmm: [[f64; 4]; 4], dimensions: [u64; 3]) -> Self {
        Self {
            positions: Vec::new(),
            offsets: vec![0],
            header: Header {
                voxel_to_rasmm,
                dimensions,
                nb_streamlines: 0,
                nb_vertices: 0,
                extra: Default::default(),
            },
        }
    }

    /// Push a single streamline (slice of 3D points).
    pub fn push_streamline(&mut self, points: &[[P; 3]]) {
        self.positions.extend_from_slice(points);
        self.offsets.push(self.positions.len() as u32);
        self.header.nb_streamlines += 1;
        self.header.nb_vertices += points.len() as u64;
    }

    /// Number of streamlines added so far.
    pub fn nb_streamlines(&self) -> usize {
        self.header.nb_streamlines as usize
    }

    /// Number of vertices added so far.
    pub fn nb_vertices(&self) -> usize {
        self.header.nb_vertices as usize
    }

    /// Finalize into an in-memory `TrxFile`.
    pub fn finalize(self) -> TrxFile<P> {
        let pos_bytes = crate::mmap_backing::vec_to_bytes(self.positions);
        let off_bytes = crate::mmap_backing::vec_to_bytes(self.offsets);

        TrxFile::from_parts(
            self.header,
            MmapBacking::Owned(pos_bytes),
            MmapBacking::Owned(off_bytes),
            Default::default(),
            Default::default(),
            Default::default(),
            None,
        )
    }

    /// Finalize and save to the given path.
    pub fn finalize_to(self, path: &Path) -> Result<()> {
        let trx = self.finalize();
        trx.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_build_and_finalize() {
        let mut stream = TrxStream::<f32>::new(Header::identity_affine(), [100, 100, 100]);

        stream.push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        stream.push_streamline(&[[7.0, 8.0, 9.0]]);

        assert_eq!(stream.nb_streamlines(), 2);
        assert_eq!(stream.nb_vertices(), 3);

        let trx = stream.finalize();
        assert_eq!(trx.nb_streamlines(), 2);
        assert_eq!(trx.nb_vertices(), 3);
        assert_eq!(trx.streamline(0), &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(trx.streamline(1), &[[7.0, 8.0, 9.0]]);
    }
}
