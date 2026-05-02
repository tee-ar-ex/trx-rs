//! Apply spatial transforms to streamline coordinates.
//!
//! Streamlines are stored as a single contiguous `Vec<[f32; 3]>` of
//! vertices in RAS+ mm. Spatially transforming them is a pointwise
//! operation: for each vertex `p`, compute `chain.map_point(p)` and
//! overwrite. There is no notion of "pull"-style resampling — streamlines
//! aren't a sampled image, they're a list of mm coordinates — so the
//! transform's chain must map *source-coords → target-coords* (i.e., the
//! direction the points are moving). For ANTs paired h5s in BIDS naming,
//! that means the *inverse-named* file: e.g. to warp streamlines
//! ACPC → MNI, pass `from-MNI_to-ACPC.h5`.
//!
//! Use [`apply_transform_in_place`] to mutate a [`Tractogram`] in situ,
//! or [`apply_transform`] for the consuming variant. With the `parallel`
//! crate feature enabled (default), the per-point loop is rayon-parallel
//! across streamline vertices.

use itk_transforms_rs::TransformChain;

use crate::tractogram::Tractogram;

/// Apply `chain` to every vertex of `tractogram` in place.
///
/// Streamline boundaries, DPS, DPV, DPG, and groups are preserved
/// unchanged — only `positions` is mutated.
pub fn apply_transform_in_place(tractogram: &mut Tractogram, chain: &TransformChain) {
    let positions = tractogram.positions_mut();
    map_positions(positions, chain);
}

/// Consuming variant of [`apply_transform_in_place`]. Returns the
/// transformed [`Tractogram`].
pub fn apply_transform(mut tractogram: Tractogram, chain: &TransformChain) -> Tractogram {
    apply_transform_in_place(&mut tractogram, chain);
    tractogram
}

#[inline]
fn map_one(p: &mut [f32; 3], chain: &TransformChain) {
    let q = chain.map_point([p[0] as f64, p[1] as f64, p[2] as f64]);
    *p = [q[0] as f32, q[1] as f32, q[2] as f32];
}

#[cfg(feature = "parallel")]
fn map_positions(positions: &mut [[f32; 3]], chain: &TransformChain) {
    use rayon::prelude::*;
    positions.par_iter_mut().for_each(|p| map_one(p, chain));
}

#[cfg(not(feature = "parallel"))]
fn map_positions(positions: &mut [[f32; 3]], chain: &TransformChain) {
    for p in positions.iter_mut() {
        map_one(p, chain);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itk_transforms_rs::{Affine3, TransformChain};
    use nalgebra::Matrix4;

    use crate::header::Header;
    use crate::tractogram::Tractogram;

    fn one_streamline(points: Vec<[f32; 3]>) -> Tractogram {
        let mut t = Tractogram::with_header(Header {
            voxel_to_rasmm: Header::identity_affine(),
            dimensions: [10, 10, 10],
            nb_streamlines: 0,
            nb_vertices: 0,
            extra: Default::default(),
        });
        t.push_streamline(&points).unwrap();
        t
    }

    #[test]
    fn identity_chain_is_noop() {
        let pts = vec![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut t = one_streamline(pts.clone());
        let mut chain = TransformChain::new();
        chain.push_affine(Affine3::identity());
        apply_transform_in_place(&mut t, &chain);
        for (a, b) in t.positions().iter().zip(pts.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn translation_shifts_every_point() {
        let pts = vec![[0.0_f32, 0.0, 0.0], [1.0, 2.0, 3.0]];
        let mut t = one_streamline(pts);

        let mut m = Matrix4::identity();
        m[(0, 3)] = 10.0;
        m[(1, 3)] = -5.0;
        m[(2, 3)] = 100.0;
        let mut chain = TransformChain::new();
        chain.push_affine(Affine3::from_matrix(m));

        apply_transform_in_place(&mut t, &chain);
        let p0 = t.positions()[0];
        let p1 = t.positions()[1];
        assert!((p0[0] - 10.0).abs() < 1e-6);
        assert!((p0[1] + 5.0).abs() < 1e-6);
        assert!((p0[2] - 100.0).abs() < 1e-6);
        assert!((p1[0] - 11.0).abs() < 1e-6);
        assert!((p1[1] + 3.0).abs() < 1e-6);
        assert!((p1[2] - 103.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_z_90_maps_x_to_y() {
        // Rz(90°) sends +x → +y, +y → -x. Matrix in RAS+:
        let theta = std::f64::consts::FRAC_PI_2;
        let (s, c) = (theta.sin(), theta.cos());
        let mut m = Matrix4::identity();
        m[(0, 0)] = c;
        m[(0, 1)] = -s;
        m[(1, 0)] = s;
        m[(1, 1)] = c;
        let mut chain = TransformChain::new();
        chain.push_affine(Affine3::from_matrix(m));

        let pts = vec![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut t = one_streamline(pts);
        apply_transform_in_place(&mut t, &chain);

        let p0 = t.positions()[0];
        let p1 = t.positions()[1];
        assert!(p0[0].abs() < 1e-6 && (p0[1] - 1.0).abs() < 1e-6 && p0[2].abs() < 1e-6);
        assert!((p1[0] + 1.0).abs() < 1e-6 && p1[1].abs() < 1e-6 && p1[2].abs() < 1e-6);
    }

    #[test]
    fn streamline_count_and_offsets_unchanged() {
        let pts = vec![[1.0_f32, 0.0, 0.0]; 17];
        let mut t = one_streamline(pts);
        let n_before = t.positions().len();
        let offsets_before = t.offsets().to_vec();
        let mut chain = TransformChain::new();
        chain.push_affine(Affine3::identity());
        apply_transform_in_place(&mut t, &chain);
        assert_eq!(t.positions().len(), n_before);
        assert_eq!(t.offsets(), offsets_before.as_slice());
    }
}
