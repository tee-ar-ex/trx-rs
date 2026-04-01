use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::process::Command;

use trx_rs::{read_tractogram, ConversionOptions, Header, TrxStream};

fn create_test_trx(path: &std::path::Path) {
    let mut stream = TrxStream::<f32>::new(Header::identity_affine(), [10, 10, 10]);
    stream.push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    stream.push_streamline(&[[7.0, 8.0, 9.0]]);
    stream.finalize().save(path).unwrap();
}

fn create_custom_trx(
    path: &std::path::Path,
    dims: [u64; 3],
    dps_name: Option<&str>,
    include_groups: bool,
) {
    fs::create_dir_all(path).unwrap();
    let header = serde_json::json!({
        "VOXEL_TO_RASMM": Header::identity_affine(),
        "DIMENSIONS": dims,
        "NB_STREAMLINES": 1,
        "NB_VERTICES": 2
    });
    fs::write(path.join("header.json"), serde_json::to_vec(&header).unwrap()).unwrap();
    fs::write(
        path.join("positions.3.float32"),
        bytemuck::cast_slice(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    .unwrap();
    fs::write(path.join("offsets.uint32"), bytemuck::cast_slice(&[0u32, 2u32])).unwrap();
    if let Some(name) = dps_name {
        fs::create_dir_all(path.join("dps")).unwrap();
        let file_name = format!("{name}.float32");
        fs::write(path.join("dps").join(file_name), bytemuck::cast_slice(&[1.5f32])).unwrap();
    }
    if include_groups {
        fs::create_dir_all(path.join("groups")).unwrap();
        fs::write(
            path.join("groups").join("bundle.uint32"),
            bytemuck::cast_slice(&[0u32]),
        )
        .unwrap();
    }
}

#[test]
fn info_help_prints_subcommand_usage() {
    Command::cargo_bin("trx")
        .unwrap()
        .args(["info", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Print a concise summary"));
}

#[test]
fn convert_help_mentions_positions_dtype() {
    Command::cargo_bin("trx")
        .unwrap()
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--positions-dtype"));
}

#[test]
fn manipulate_dtype_rewrites_trx_positions_dtype() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("input.trx");
    let output = tmp.path().join("output.trx");
    create_test_trx(&input);

    Command::cargo_bin("trx")
        .unwrap()
        .args([
            "manipulate-dtype",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--positions-dtype",
            "f16",
        ])
        .assert()
        .success();

    let loaded = trx_rs::AnyTrxFile::load(&output).unwrap();
    assert_eq!(loaded.dtype(), trx_rs::DType::Float16);
    assert_eq!(loaded.nb_streamlines(), 2);
}

#[test]
fn concatenate_merges_two_trx_inputs() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_a = tmp.path().join("a.trx");
    let input_b = tmp.path().join("b.trx");
    let output = tmp.path().join("merged.trx");
    create_test_trx(&input_a);
    create_test_trx(&input_b);

    Command::cargo_bin("trx")
        .unwrap()
        .args([
            "concatenate",
            input_a.to_str().unwrap(),
            input_b.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--positions-dtype",
            "f32",
        ])
        .assert()
        .success();

    let loaded = trx_rs::AnyTrxFile::load(&output).unwrap();
    assert_eq!(loaded.nb_streamlines(), 4);
    assert_eq!(loaded.dtype(), trx_rs::DType::Float32);
}

#[test]
fn concatenate_uses_first_header_and_drops_dpg() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_a = tmp.path().join("a.trx");
    let input_b = tmp.path().join("b.trx");
    let output = tmp.path().join("merged.trx");
    create_custom_trx(&input_a, [10, 20, 30], Some("weights"), true);
    create_custom_trx(&input_b, [99, 99, 99], Some("weights"), false);

    Command::cargo_bin("trx")
        .unwrap()
        .args([
            "concatenate",
            input_a.to_str().unwrap(),
            input_b.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .assert()
        .success();

    let loaded = trx_rs::AnyTrxFile::load(&output).unwrap();
    assert_eq!(loaded.header().dimensions, [10, 20, 30]);
    assert_eq!(loaded.nb_streamlines(), 2);
    assert_eq!(loaded.groups_owned().len(), 1);
    assert!(loaded.dpg_group_entries().is_empty());
}

#[test]
fn concatenate_delete_dps_allows_missing_dps() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input_a = tmp.path().join("a.trx");
    let input_b = tmp.path().join("b.trx");
    let output = tmp.path().join("merged.trx");
    create_custom_trx(&input_a, [10, 20, 30], Some("weights"), false);
    create_custom_trx(&input_b, [10, 20, 30], None, false);

    Command::cargo_bin("trx")
        .unwrap()
        .args([
            "concatenate",
            input_a.to_str().unwrap(),
            input_b.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--delete-dps",
        ])
        .assert()
        .success();

    let loaded = trx_rs::AnyTrxFile::load(&output).unwrap();
    assert!(loaded.dps_entries().is_empty());
}

#[test]
fn concatenate_requires_reference_for_tck() {
    let tmp = tempfile::TempDir::new().unwrap();
    let tck = tmp.path().join("sample.tck");
    let trx = tmp.path().join("sample.trx");
    let output = tmp.path().join("merged.trx");
    create_test_trx(&trx);
    let mut tractogram = trx_rs::Tractogram::new();
    tractogram.push_streamline(&[[1.0, 2.0, 3.0]]).unwrap();
    trx_rs::write_tractogram(&tck, &tractogram, &ConversionOptions::default()).unwrap();

    Command::cargo_bin("trx")
        .unwrap()
        .args([
            "concatenate",
            tck.to_str().unwrap(),
            trx.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--reference is required"));
}

#[test]
fn convert_tck_to_trx_respects_positions_dtype() {
    let tmp = tempfile::TempDir::new().unwrap();
    let tck = tmp.path().join("sample.tck.gz");
    let trx = tmp.path().join("sample.trx");

    let mut tractogram = trx_rs::Tractogram::new();
    tractogram
        .push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .unwrap();
    trx_rs::write_tractogram(&tck, &tractogram, &ConversionOptions::default()).unwrap();

    Command::cargo_bin("trx")
        .unwrap()
        .args([
            "convert",
            tck.to_str().unwrap(),
            trx.to_str().unwrap(),
            "--positions-dtype",
            "f16",
        ])
        .assert()
        .success();

    let loaded = trx_rs::AnyTrxFile::load(&trx).unwrap();
    assert_eq!(loaded.dtype(), trx_rs::DType::Float16);

    let roundtrip = read_tractogram(&trx, &ConversionOptions::default()).unwrap();
    assert_eq!(roundtrip.nb_streamlines(), 1);
}
