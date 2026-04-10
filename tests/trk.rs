use std::fs;
use std::io::Write;
use std::path::PathBuf;

use flate2::write::GzEncoder;
use flate2::Compression;
use trx_rs::{read_tractogram, AnyTrxFile, ConversionOptions, DType};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/trk")
}

fn fixture(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

fn assert_positions_close(actual: &[[f32; 3]], expected: &[[f32; 3]]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        for axis in 0..3 {
            let diff = (actual[axis] - expected[axis]).abs();
            let tol = 1e-5 + 1e-4 * expected[axis].abs();
            assert!(
                diff <= tol,
                "position mismatch at vertex {index}, axis {axis}: got {}, expected {}",
                actual[axis],
                expected[axis]
            );
        }
    }
}

fn assert_rows_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (row_index, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(actual_row.len(), expected_row.len());
        for (col_index, (actual, expected)) in
            actual_row.iter().zip(expected_row.iter()).enumerate()
        {
            let diff = (actual - expected).abs();
            let tol = 1e-6 + 1e-5 * expected.abs();
            assert!(
                diff <= tol,
                "row {row_index}, col {col_index}: got {actual}, expected {expected}"
            );
        }
    }
}

fn dps_rows(any: &AnyTrxFile, name: &str) -> Vec<Vec<f32>> {
    any.with_typed(
        |trx| {
            trx.dps_array(name)
                .unwrap()
                .typed_view::<f32>()
                .rows()
                .map(|row| row.to_vec())
                .collect()
        },
        |trx| {
            trx.dps_array(name)
                .unwrap()
                .typed_view::<f32>()
                .rows()
                .map(|row| row.to_vec())
                .collect()
        },
        |trx| {
            trx.dps_array(name)
                .unwrap()
                .typed_view::<f32>()
                .rows()
                .map(|row| row.to_vec())
                .collect()
        },
    )
}

fn dpv_rows(any: &AnyTrxFile, name: &str) -> Vec<Vec<f32>> {
    any.with_typed(
        |trx| {
            trx.dpv_array(name)
                .unwrap()
                .typed_view::<f32>()
                .rows()
                .map(|row| row.to_vec())
                .collect()
        },
        |trx| {
            trx.dpv_array(name)
                .unwrap()
                .typed_view::<f32>()
                .rows()
                .map(|row| row.to_vec())
                .collect()
        },
        |trx| {
            trx.dpv_array(name)
                .unwrap()
                .typed_view::<f32>()
                .rows()
                .map(|row| row.to_vec())
                .collect()
        },
    )
}

#[test]
fn simple_trk_direct_import_matches_expected_geometry() {
    let tractogram =
        read_tractogram(&fixture("simple.trk"), &ConversionOptions::default()).unwrap();

    assert_eq!(tractogram.nb_streamlines(), 3);
    assert_eq!(
        tractogram
            .streamlines()
            .map(|streamline| streamline.len())
            .collect::<Vec<_>>(),
        vec![1, 2, 5]
    );

    let expected = vec![
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0],
        [12.0, 13.0, 14.0],
    ];
    assert_positions_close(tractogram.positions(), &expected);
}

#[test]
fn simple_trk_gz_direct_import_matches_expected_geometry() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gz_path = tmp.path().join("simple.trk.gz");
    let bytes = fs::read(fixture("simple.trk")).unwrap();
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&bytes).unwrap();
    fs::write(&gz_path, encoder.finish().unwrap()).unwrap();

    let tractogram = read_tractogram(&gz_path, &ConversionOptions::default()).unwrap();
    assert_eq!(tractogram.nb_streamlines(), 3);
    assert_eq!(tractogram.positions()[0], [0.0, 1.0, 2.0]);
    assert_eq!(tractogram.positions()[7], [12.0, 13.0, 14.0]);
}

#[test]
fn payload_bearing_trk_direct_import_refuses_to_drop_metadata() {
    let err = read_tractogram(&fixture("complex.trk"), &ConversionOptions::default()).unwrap_err();
    assert!(err
        .to_string()
        .contains("convert it to .trx first to preserve metadata"));
}

#[test]
fn malformed_trk_with_missing_affine_fails_cleanly() {
    let tmp = tempfile::TempDir::new().unwrap();
    let bad_path = tmp.path().join("bad.trk");
    let mut bytes = fs::read(fixture("simple.trk")).unwrap();
    bytes[440..504].fill(0);
    fs::write(&bad_path, bytes).unwrap();

    let err = read_tractogram(&bad_path, &ConversionOptions::default()).unwrap_err();
    assert!(err.to_string().contains("vox_to_ras is missing or zero"));
}

#[test]
fn complex_trk_to_trx_conversion_preserves_dpv_and_dps() {
    let tmp = tempfile::TempDir::new().unwrap();
    let out = tmp.path().join("complex.trx");

    trx_rs::convert(&fixture("complex.trk"), &out, &ConversionOptions::default()).unwrap();

    let any = AnyTrxFile::load(&out).unwrap();
    assert_eq!(any.dtype(), DType::Float32);

    let mut dpv_names = any
        .dpv_entries()
        .into_iter()
        .map(|(name, info)| (name, info.ncols, info.nrows))
        .collect::<Vec<_>>();
    dpv_names.sort();
    assert_eq!(
        dpv_names,
        vec![("colors".to_string(), 3, 8), ("fa".to_string(), 1, 8),]
    );

    let mut dps_names = any
        .dps_entries()
        .into_iter()
        .map(|(name, info)| (name, info.ncols, info.nrows))
        .collect::<Vec<_>>();
    dps_names.sort();
    assert_eq!(
        dps_names,
        vec![
            ("mean_colors".to_string(), 3, 3),
            ("mean_curvature".to_string(), 1, 3),
            ("mean_torsion".to_string(), 1, 3),
        ]
    );

    assert_rows_close(
        &dpv_rows(&any, "fa"),
        &[
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.6],
            vec![0.7],
            vec![0.8],
        ],
    );
    assert_rows_close(
        &dpv_rows(&any, "colors"),
        &[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0],
        ],
    );
    assert_rows_close(
        &dps_rows(&any, "mean_curvature"),
        &[vec![1.11], vec![2.11], vec![3.11]],
    );
    assert_rows_close(
        &dps_rows(&any, "mean_torsion"),
        &[vec![1.22], vec![2.22], vec![3.22]],
    );
    assert_rows_close(
        &dps_rows(&any, "mean_colors"),
        &[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
    );
}

#[test]
fn big_endian_trk_to_trx_conversion_preserves_metadata_layout() {
    let tmp = tempfile::TempDir::new().unwrap();
    let out = tmp.path().join("complex_big_endian.trx");

    trx_rs::convert(
        &fixture("complex_big_endian.trk"),
        &out,
        &ConversionOptions::default(),
    )
    .unwrap();

    let any = AnyTrxFile::load(&out).unwrap();
    assert_eq!(any.dpv_entries().len(), 2);
    assert_eq!(any.dps_entries().len(), 3);
}

#[test]
fn standard_and_lps_trk_imports_match() {
    let standard =
        read_tractogram(&fixture("standard.trk"), &ConversionOptions::default()).unwrap();
    let lps = read_tractogram(&fixture("standard.LPS.trk"), &ConversionOptions::default()).unwrap();

    assert_eq!(standard.nb_streamlines(), lps.nb_streamlines());
    assert_positions_close(standard.positions(), lps.positions());
}
