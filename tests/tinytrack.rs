use std::path::{Path, PathBuf};

use trx_rs::mmap_backing::vec_to_bytes;
use trx_rs::trx_file::DataArray;
use trx_rs::{
    read_tractogram, write_tractogram, AnyTrxFile, ConversionOptions, DType, TypedView2D,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn assert_close(actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "value mismatch: got {actual}, expected {expected}, tolerance {tol}"
    );
}

#[test]
fn tinytrack_import_preserves_metadata_and_groups() {
    let path = fixture("tinytrack_small.tt.gz");
    let tractogram = read_tractogram(&path, &ConversionOptions::default()).unwrap();

    assert_eq!(tractogram.nb_streamlines(), 9);
    assert_eq!(tractogram.header().dimensions, [157, 189, 136]);
    assert_eq!(
        tractogram.header().voxel_to_rasmm,
        [
            [-1.0, 0.0, 0.0, 78.0],
            [0.0, -1.0, 0.0, 76.0],
            [0.0, 0.0, 1.0, -50.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    );

    let names: Vec<_> = tractogram.groups().keys().cloned().collect();
    assert!(names.contains(&"Association_ArcuateFasciculusL".to_string()));
    assert!(names.contains(&"Association_ArcuateFasciculusR".to_string()));
    assert!(names.contains(&"Association_FrontalAslantTractL".to_string()));

    let report = tractogram
        .header()
        .extra
        .get("tt_report")
        .and_then(|value| value.as_str())
        .unwrap();
    assert!(!report.is_empty());
    let parameter_id = tractogram
        .header()
        .extra
        .get("tt_parameter_id")
        .and_then(|value| value.as_str())
        .unwrap();
    assert!(!parameter_id.is_empty());

    let color = tractogram
        .dpg()
        .get("Association_ArcuateFasciculusL")
        .and_then(|group| group.get("color"))
        .unwrap();
    assert_eq!(color.ncols(), 3);
    assert_eq!(color.dtype(), DType::UInt8);
}

#[test]
fn tinytrack_to_trx_writes_groups_and_dpg_color() {
    let input = fixture("tinytrack_small.tt.gz");
    let tractogram = read_tractogram(&input, &ConversionOptions::default()).unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let output = tmp.path().join("tinytrack.trx");

    write_tractogram(&output, &tractogram, &ConversionOptions::default()).unwrap();

    let file = std::fs::File::open(&output).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    assert!(archive
        .by_name("groups/Association_ArcuateFasciculusL.uint32")
        .is_ok());
    assert!(archive
        .by_name("dpg/Association_ArcuateFasciculusL/color.3.uint8")
        .is_ok());

    let loaded = AnyTrxFile::load(&output).unwrap();
    loaded.with_typed(
        |trx| {
            assert_group_and_color(
                trx.group("Association_ArcuateFasciculusL").unwrap(),
                trx.dpg("Association_ArcuateFasciculusL", "color").unwrap(),
            )
        },
        |trx| {
            assert_group_and_color(
                trx.group("Association_ArcuateFasciculusL").unwrap(),
                trx.dpg("Association_ArcuateFasciculusL", "color").unwrap(),
            )
        },
        |trx| {
            assert_group_and_color(
                trx.group("Association_ArcuateFasciculusL").unwrap(),
                trx.dpg("Association_ArcuateFasciculusL", "color").unwrap(),
            )
        },
    );

    assert_eq!(
        loaded.header().voxel_to_rasmm,
        [
            [-1.0, 0.0, 0.0, 78.0],
            [0.0, -1.0, 0.0, 76.0],
            [0.0, 0.0, 1.0, -50.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    );
    assert_eq!(loaded.header().dimensions, [157, 189, 136]);
}

fn assert_group_and_color(group: &[u32], color: TypedView2D<'_, u8>) {
    assert!(!group.is_empty());
    assert_eq!(color.ncols(), 3);
    assert_eq!(color.nrows(), 1);
    assert_eq!(color.row(0), &[60, 160, 255]);
}

#[test]
fn one_dimensional_group_and_dpg_entries_omit_explicit_unit_dimension() {
    let mut tractogram = trx_rs::Tractogram::new();
    tractogram.push_streamline(&[[1.0, 2.0, 3.0]]).unwrap();
    tractogram.insert_group("AF_L", vec![0]);
    tractogram.insert_dpg(
        "AF_L",
        "volume",
        DataArray::owned_bytes(vec_to_bytes(vec![7u32]), 1, DType::UInt32),
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let output = tmp.path().join("oned.trx");
    write_tractogram(&output, &tractogram, &ConversionOptions::default()).unwrap();

    let file = std::fs::File::open(&output).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    assert!(archive.by_name("groups/AF_L.uint32").is_ok());
    assert!(archive.by_name("groups/AF_L.1.uint32").is_err());
    assert!(archive.by_name("dpg/AF_L/volume.uint32").is_ok());
    assert!(archive.by_name("dpg/AF_L/volume.1.uint32").is_err());
}

#[test]
fn tinytrack_malformed_payload_errors() {
    let err = read_tractogram(
        &fixture("tinytrack_malformed.tt.gz"),
        &ConversionOptions::default(),
    )
    .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("truncated TT tract")
            || message.contains("truncated TT MAT")
            || message.contains("invalid TT")
    );
}

#[test]
fn tinytrack_group_names_fall_back_without_sidecar() {
    let source = fixture("tinytrack_small.tt.gz");
    let tmp = tempfile::TempDir::new().unwrap();
    let copied = tmp.path().join("nolabels.tt.gz");
    std::fs::copy(&source, &copied).unwrap();

    let tractogram = read_tractogram(&copied, &ConversionOptions::default()).unwrap();
    assert!(tractogram.groups().contains_key("cluster_0"));
    assert!(tractogram.groups().contains_key("cluster_1"));
    assert!(tractogram.groups().contains_key("cluster_2"));
}

#[test]
fn atlas_left_right_bundle_means_have_expected_x_sign() {
    let atlas =
        Path::new("/Applications/dsi_studio-hou.app/Contents/MacOS/atlas/human/human.tt.gz");
    if !atlas.exists() {
        return;
    }

    let tractogram = read_tractogram(atlas, &ConversionOptions::default()).unwrap();
    let left_af = mean_x_for_group(&tractogram, "Association_ArcuateFasciculusL");
    let right_af = mean_x_for_group(&tractogram, "Association_ArcuateFasciculusR");
    let left_fat = mean_x_for_group(&tractogram, "Association_FrontalAslantTractL");
    let right_fat = mean_x_for_group(&tractogram, "Association_FrontalAslantTractR");

    assert!(left_af < 0.0);
    assert!(right_af > 0.0);
    assert!(left_fat < 0.0);
    assert!(right_fat > 0.0);

    assert_close(left_af, -42.77, 2.0);
    assert_close(right_af, 36.73, 2.0);
    assert_close(left_fat, -30.88, 2.0);
    assert_close(right_fat, 29.99, 2.0);
}

fn mean_x_for_group(tractogram: &trx_rs::Tractogram, name: &str) -> f64 {
    let members = tractogram.groups().get(name).unwrap();
    let mut xs = Vec::new();
    for &index in members {
        let streamline = tractogram.streamline(index as usize);
        xs.extend(streamline.iter().map(|point| point[0] as f64));
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}
