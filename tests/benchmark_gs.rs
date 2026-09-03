use std::path::PathBuf;

use trx_rs::{
    header_from_reference, read_tractogram, ConversionOptions, TrxFile, VtkCoordinateMode,
};

fn gs_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gs")
}

fn assert_affine_close(actual: &[[f64; 4]; 4], expected: &[[f64; 4]; 4], eps: f64, label: &str) {
    for r in 0..4 {
        for c in 0..4 {
            let diff = (actual[r][c] - expected[r][c]).abs();
            assert!(
                diff <= eps,
                "Affine mismatch in {label} at [{r}][{c}]: got {}, expected {}, diff {diff} > eps {eps}",
                actual[r][c],
                expected[r][c]
            );
        }
    }
}

fn assert_positions_close(actual: &[[f32; 3]], expected: &[[f32; 3]], eps: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Vertex count mismatch for {label}: got {}, expected {}",
        actual.len(),
        expected.len()
    );

    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        for axis in 0..3 {
            let diff = (a[axis] - e[axis]).abs();
            assert!(
                diff <= eps,
                "Position mismatch in {label} at vertex {index}, axis {axis}: got {}, expected {}, diff {diff} > eps {eps}",
                a[axis],
                e[axis]
            );
        }
    }
}

fn assert_slice_close(actual: &[f32], expected: &[f32], eps: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch for {label}: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= eps,
            "Value mismatch in {label} at index {i}: got {}, expected {}, diff {diff} > eps {eps}",
            a,
            e
        );
    }
}

// ── 1. Header Parity ────────────────────────────────────────────────────────

#[test]
fn test_benchmark_gs_header_parity() {
    let dir = gs_fixtures_dir();
    let nii_ref = dir.join("gs.nii");
    let ref_header = header_from_reference(&nii_ref).expect("failed to load gs.nii header");

    let trx_variants = [
        "gs_from_cpp.trx",
        "gs_from_js.trx",
        "gs_from_py.trx",
        "gs_from_rs.trx",
        "gs_from_zenodo.trx",
    ];

    let trk_path = dir.join("gs.trk");
    let trk_tr = read_tractogram(
        &trk_path,
        &ConversionOptions {
            header: Some(ref_header.clone()),
            ..Default::default()
        },
    )
    .expect("failed to load gs.trk");

    // TRK header dimensions and affine check
    assert_eq!(
        trk_tr.header().dimensions,
        ref_header.dimensions,
        "gs.trk dimensions mismatch"
    );
    assert_affine_close(
        &trk_tr.header().voxel_to_rasmm,
        &ref_header.voxel_to_rasmm,
        1e-4,
        "gs.trk header",
    );

    // TRX variants header check
    for file_name in &trx_variants {
        let path = dir.join(file_name);
        let trx = TrxFile::<f32>::load(&path)
            .unwrap_or_else(|e| panic!("failed to load {file_name}: {e}"));
        let h = trx.header();

        assert_eq!(
            h.dimensions, ref_header.dimensions,
            "Dimensions mismatch for {file_name}"
        );
        assert_affine_close(
            &h.voxel_to_rasmm,
            &ref_header.voxel_to_rasmm,
            1e-4,
            file_name,
        );
    }
}

// ── 2. Data Parity (Streamlines & Geometry) ──────────────────────────────────

#[test]
fn test_benchmark_gs_data_parity() {
    let dir = gs_fixtures_dir();
    let nii_ref = dir.join("gs.nii");
    let ref_header = header_from_reference(&nii_ref).expect("failed to load gs.nii header");

    let opts_with_header = ConversionOptions {
        header: Some(ref_header),
        vtk_coordinate_mode: VtkCoordinateMode::HeaderOrWarn,
        ..Default::default()
    };

    // Baseline: Python TRX variant as reference geometry
    let ref_trx_path = dir.join("gs_from_py.trx");
    let ref_tr = read_tractogram(&ref_trx_path, &ConversionOptions::default())
        .expect("failed to load gs_from_py.trx");
    let ref_positions = ref_tr.positions();
    let ref_offsets = ref_tr.offsets();

    let formats_to_test = [
        ("gs.tck", true),
        ("gs.trk", true),
        ("gs.vtk", true),
        ("gs_from_cpp.trx", false),
        ("gs_from_js.trx", false),
        ("gs_from_rs.trx", false),
        ("gs_from_zenodo.trx", false),
    ];

    for (file_name, requires_ref) in formats_to_test {
        let path = dir.join(file_name);
        let opts = if requires_ref {
            opts_with_header.clone()
        } else {
            ConversionOptions::default()
        };

        let tr = read_tractogram(&path, &opts)
            .unwrap_or_else(|e| panic!("failed to read tractogram {file_name}: {e}"));

        assert_eq!(
            tr.nb_streamlines(),
            ref_tr.nb_streamlines(),
            "Streamline count mismatch for {file_name}"
        );
        assert_eq!(
            tr.nb_vertices(),
            ref_tr.nb_vertices(),
            "Vertex count mismatch for {file_name}"
        );
        assert_eq!(
            tr.offsets(),
            ref_offsets,
            "Offsets mismatch for {file_name}"
        );

        // Position coordinates matching within 1e-3 epsilon
        assert_positions_close(tr.positions(), ref_positions, 1e-3, file_name);
    }
}

// ── 3. Metadata Parity ──────────────────────────────────────────────────────

#[test]
fn test_benchmark_gs_metadata_parity() {
    let dir = gs_fixtures_dir();

    let py_trx_path = dir.join("gs_from_py.trx");
    let ref_trx = TrxFile::<f32>::load(&py_trx_path).expect("failed to load gs_from_py.trx");

    let trx_variants = [
        "gs_from_cpp.trx",
        "gs_from_js.trx",
        "gs_from_rs.trx",
        "gs_from_zenodo.trx",
    ];

    // Read reference metadata keys and data arrays
    let mut ref_dps_keys = ref_trx.dps_names();
    let mut ref_dpv_keys = ref_trx.dpv_names();
    let mut ref_group_keys = ref_trx.group_names();

    ref_dps_keys.sort();
    ref_dpv_keys.sort();
    ref_group_keys.sort();

    for file_name in &trx_variants {
        let path = dir.join(file_name);
        let trx = TrxFile::<f32>::load(&path)
            .unwrap_or_else(|e| panic!("failed to load TRX file {file_name}: {e}"));

        let mut dps_keys = trx.dps_names();
        let mut dpv_keys = trx.dpv_names();
        let mut group_keys = trx.group_names();

        dps_keys.sort();
        dpv_keys.sort();
        group_keys.sort();

        // Compare sorted key names
        assert_eq!(dps_keys, ref_dps_keys, "DPS keys mismatch in {file_name}");
        assert_eq!(dpv_keys, ref_dpv_keys, "DPV keys mismatch in {file_name}");
        assert_eq!(
            group_keys, ref_group_keys,
            "Group keys mismatch in {file_name}"
        );

        // Compare DPS values
        for key in &ref_dps_keys {
            let ref_data = ref_trx.dps::<f32>(key).unwrap();
            let actual_data = trx.dps::<f32>(key).unwrap();
            assert_slice_close(
                actual_data.as_flat_slice(),
                ref_data.as_flat_slice(),
                1e-4,
                &format!("{file_name} dps[{key}]"),
            );
        }

        // Compare DPV values
        for key in &ref_dpv_keys {
            let ref_data = ref_trx.dpv::<f32>(key).unwrap();
            let actual_data = trx.dpv::<f32>(key).unwrap();
            assert_slice_close(
                actual_data.as_flat_slice(),
                ref_data.as_flat_slice(),
                1e-4,
                &format!("{file_name} dpv[{key}]"),
            );
        }

        // Compare Group membership arrays
        for group in &ref_group_keys {
            let ref_members = ref_trx.group(group).unwrap();
            let actual_members = trx.group(group).unwrap();
            assert_eq!(
                actual_members, ref_members,
                "Group '{group}' membership mismatch in {file_name}"
            );
        }
    }
}
