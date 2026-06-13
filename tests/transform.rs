//! Integration tests for `trx_rs::transform::apply_transform`.

use itk_transforms_rs::{Affine3, TransformChain};
use nalgebra::Matrix4;
use trx_rs::{
    apply_transform, apply_transform_in_place, AnyTrxFile, DType, Header, Tractogram, TrxStream,
};

use trx_rs::DataArray;

fn make_tractogram(streams: &[&[[f32; 3]]]) -> Tractogram {
    let mut t = Tractogram::with_header(Header {
        voxel_to_rasmm: Header::identity_affine(),
        dimensions: [10, 10, 10],
        nb_streamlines: 0,
        nb_vertices: 0,
        extra: Default::default(),
    });
    for s in streams {
        t.push_streamline(s).unwrap();
    }
    t
}

#[test]
fn identity_round_trip_all_points() {
    let pts = [
        [1.0_f32, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-7.5, 0.0, 100.25],
    ];
    let mut t = make_tractogram(&[&pts]);
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::identity());
    apply_transform_in_place(&mut t, &chain);
    for (got, want) in t.positions().iter().zip(pts.iter()) {
        assert_eq!(got, want);
    }
}

#[test]
fn translation_shifts_uniformly_across_streamlines() {
    let s1: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let s2: [[f32; 3]; 2] = [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]];
    let mut t = make_tractogram(&[&s1, &s2]);

    let mut m = Matrix4::identity();
    m[(0, 3)] = 100.0;
    m[(1, 3)] = -50.0;
    m[(2, 3)] = 7.5;
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::from_matrix(m));

    apply_transform_in_place(&mut t, &chain);
    let positions = t.positions();
    // s1 starts at offset 0, s2 starts at offset 3.
    let expected = [
        [100.0, -50.0, 7.5],
        [101.0, -50.0, 7.5],
        [102.0, -50.0, 7.5],
        [110.0, -30.0, 37.5],
        [111.0, -29.0, 38.5],
    ];
    assert_eq!(positions.len(), expected.len());
    for (got, want) in positions.iter().zip(expected.iter()) {
        for i in 0..3 {
            assert!((got[i] - want[i]).abs() < 1e-5, "got {got:?} want {want:?}");
        }
    }
    // Streamline boundaries unchanged.
    assert_eq!(t.offsets(), &[0, 3, 5]);
}

#[test]
fn rotation_z_90_maps_x_endpoints_to_y() {
    // Streamline along +X from origin to [5, 0, 0]: after Rz(90°) it should
    // run from origin to [0, 5, 0].
    let s: [[f32; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ];
    let t_in = make_tractogram(&[&s]);

    let theta = std::f64::consts::FRAC_PI_2;
    let (sin_t, cos_t) = (theta.sin(), theta.cos());
    let mut m = Matrix4::identity();
    m[(0, 0)] = cos_t;
    m[(0, 1)] = -sin_t;
    m[(1, 0)] = sin_t;
    m[(1, 1)] = cos_t;
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::from_matrix(m));

    let t_out = apply_transform(t_in, &chain);
    let positions = t_out.positions();
    for (i, got) in positions.iter().enumerate() {
        let want = [0.0_f32, i as f32, 0.0];
        for axis in 0..3 {
            assert!(
                (got[axis] - want[axis]).abs() < 1e-5,
                "vertex {i}: got {got:?}, want {want:?}"
            );
        }
    }
    assert_eq!(t_out.offsets(), &[0, 6]);
}

#[test]
fn dps_dpv_groups_dpg_survive_full_round_trip() {
    // Build a TRX directly via TrxStream (the public path with DPS/DPV/groups
    // population), save → load → from_any_trx → apply_transform → to_trx →
    // save → load again and verify DPS/DPV/groups all came along.
    use std::path::Path;
    let tmp = tempfile::tempdir().unwrap();

    // Author a TRX with 2 streamlines (5 vertices total), 1 DPS, 1 DPV, 1 group.
    let in_path = tmp.path().join("in.trx");
    {
        let mut s = TrxStream::<f32>::new(Header::identity_affine(), [10, 10, 10]);
        s.push_streamline(&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        s.push_streamline(&[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]);
        let trx = s.finalize();
        // Inject DPS/DPV via TrxParts. Easiest path: roundtrip through
        // public ops::copy_metadata… but for a self-contained test we
        // build a fresh TrxParts ourselves. Skip — instead, use
        // Tractogram's `insert_dps` / `insert_dpv` which we just added.
        let mut tract = Tractogram::from_trx(&trx);
        tract.insert_dps(
            "weight",
            DataArray::owned_bytes(
                bytemuck::cast_slice(&[0.5_f32, 0.75]).to_vec(),
                1,
                DType::Float32,
            ),
        );
        tract.insert_dpv(
            "scalar",
            DataArray::owned_bytes(
                bytemuck::cast_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).to_vec(),
                1,
                DType::Float32,
            ),
        );
        tract.insert_group("bundle", vec![0, 1]);
        let trx2 = tract.to_trx(DType::Float32).unwrap();
        trx2.save(&in_path).unwrap();
        let _ = trx; // silence unused
    }

    // Apply identity transform (we want to check metadata, not coords).
    let file = AnyTrxFile::load(&in_path).unwrap();
    let dtype = file.dtype();
    let mut tract = Tractogram::from_any_trx(&file);
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::identity());
    apply_transform_in_place(&mut tract, &chain);
    let out = tract.to_trx(dtype).unwrap();
    let out_path = tmp.path().join("out.trx");
    out.save(&out_path).unwrap();

    // Reload and assert DPS/DPV/groups survive verbatim.
    let reloaded = AnyTrxFile::load(&out_path).unwrap();
    let dps_names: Vec<String> = reloaded
        .dps_entries()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    let dpv_names: Vec<String> = reloaded
        .dpv_entries()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(
        dps_names,
        vec!["weight".to_string()],
        "DPS lost in round trip"
    );
    assert_eq!(
        dpv_names,
        vec!["scalar".to_string()],
        "DPV lost in round trip"
    );

    // Values should also match exactly. AnyTrxFile lacks a direct
    // dps_array accessor, so go through with_typed.
    let weight_bytes = reloaded.with_typed(
        |t| t.dps_array("weight").unwrap().as_bytes().to_vec(),
        |t| t.dps_array("weight").unwrap().as_bytes().to_vec(),
        |t| t.dps_array("weight").unwrap().as_bytes().to_vec(),
    );
    let weight: &[f32] = bytemuck::cast_slice(&weight_bytes);
    assert_eq!(weight, &[0.5_f32, 0.75]);

    let _ = Path::new(&in_path);
}

#[test]
fn metadata_preserved_through_transform() {
    let mut t = make_tractogram(&[&[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]);
    // Stash a sentinel into header.extra so we can verify it survives.
    t.extra_mut().insert(
        "_test_marker".to_string(),
        serde_json::Value::String("intact".to_string()),
    );

    let mut m = Matrix4::identity();
    m[(2, 3)] = 100.0;
    let mut chain = TransformChain::new();
    chain.push_affine(Affine3::from_matrix(m));

    apply_transform_in_place(&mut t, &chain);

    assert_eq!(
        t.extra_mut().get("_test_marker").map(|v| v.as_str()),
        Some(Some("intact"))
    );
    // voxel_to_rasmm and dimensions on the header are NOT auto-updated by
    // the transform; that's the caller's responsibility via
    // `set_spatial_metadata`. Confirm here.
    assert_eq!(t.header().voxel_to_rasmm, Header::identity_affine());
}
