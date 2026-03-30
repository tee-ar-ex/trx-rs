use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

use trx_rs::{read_tractogram, ConversionOptions, Header, TrxStream};

fn create_test_trx(path: &std::path::Path) {
    let mut stream = TrxStream::<f32>::new(Header::identity_affine(), [10, 10, 10]);
    stream.push_streamline(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    stream.push_streamline(&[[7.0, 8.0, 9.0]]);
    stream.finalize().save(path).unwrap();
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
