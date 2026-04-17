use std::collections::HashMap;

use trx_rs::io::{directory, zip};
use trx_rs::mmap_backing::vec_to_bytes;
use trx_rs::{AnyTrxFile, DType, DataArray, Tractogram, TrxFile};
use ::zip::{CompressionMethod, ZipArchive};

fn base_trx() -> TrxFile<f32> {
    let mut tractogram = Tractogram::new();
    tractogram
        .push_streamline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        .unwrap();
    tractogram.push_streamline(&[[0.0, 1.0, 0.0]]).unwrap();
    match tractogram.to_trx(DType::Float32).unwrap() {
        AnyTrxFile::F32(trx) => trx,
        _ => unreachable!(),
    }
}

fn sample_dps() -> HashMap<String, DataArray> {
    HashMap::from([(
        "weight".to_string(),
        DataArray::owned_bytes(vec_to_bytes(vec![1.0f32, 2.0f32]), 1, DType::Float32),
    )])
}

fn sample_dpv() -> HashMap<String, DataArray> {
    HashMap::from([(
        "fa".to_string(),
        DataArray::owned_bytes(
            vec_to_bytes(vec![0.1f32, 0.2f32, 0.3f32]),
            1,
            DType::Float32,
        ),
    )])
}

fn sample_groups() -> HashMap<String, Vec<u32>> {
    HashMap::from([("bundle".to_string(), vec![0u32])])
}

fn sample_dpg() -> HashMap<String, HashMap<String, DataArray>> {
    HashMap::from([(
        "bundle".to_string(),
        HashMap::from([(
            "color".to_string(),
            DataArray::owned_bytes(vec![7, 8, 9], 3, DType::UInt8),
        )]),
    )])
}

#[test]
fn zip_archive_mutation_supports_append_replace_and_delete() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sample.trx");
    base_trx().save_to_zip(&path).unwrap();

    zip::append_dps_to_zip(&path, &sample_dps(), CompressionMethod::Stored, false).unwrap();
    zip::append_dpv_to_zip(&path, &sample_dpv(), CompressionMethod::Stored, false).unwrap();
    zip::append_groups_to_zip(&path, &sample_groups(), CompressionMethod::Stored, false).unwrap();
    zip::append_dpg_to_zip(&path, &sample_dpg(), CompressionMethod::Stored, false).unwrap();

    let loaded = TrxFile::<f32>::load(&path).unwrap();
    assert_eq!(loaded.scalar_dps_f32("weight").unwrap(), vec![1.0, 2.0]);
    assert_eq!(loaded.scalar_dpv_f32("fa").unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(loaded.group("bundle").unwrap(), &[0u32]);
    assert_eq!(
        loaded.dpg::<u8>("bundle", "color").unwrap().row(0),
        &[7, 8, 9]
    );

    let replacement = HashMap::from([(
        "weight".to_string(),
        DataArray::owned_bytes(vec_to_bytes(vec![5u16, 6u16]), 1, DType::UInt16),
    )]);
    zip::append_dps_to_zip(&path, &replacement, CompressionMethod::Stored, true).unwrap();
    zip::delete_dpv_from_zip(&path, &["fa"]).unwrap();
    zip::delete_groups_from_zip(&path, &["bundle"]).unwrap();

    let replaced = TrxFile::<f32>::load(&path).unwrap();
    assert_eq!(replaced.dps_info("weight").unwrap().dtype, DType::UInt16);
    assert_eq!(replaced.dpv_names(), Vec::<&str>::new());
    assert_eq!(replaced.group_names(), Vec::<&str>::new());
    assert_eq!(replaced.dpg_group_names(), Vec::<&str>::new());

    let file = std::fs::File::open(&path).unwrap();
    let mut archive = ZipArchive::new(file).unwrap();
    assert!(archive.by_name("dps/weight.float32").is_err());
    assert!(archive.by_name("dps/weight.uint16").is_ok());
    assert!(archive.by_name("dpg/bundle/color.3.uint8").is_err());
}

#[test]
fn directory_mutation_matches_zip_semantics() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sample.trxd");
    base_trx().save_to_directory(&path).unwrap();

    directory::append_dps_to_directory(&path, &sample_dps(), false).unwrap();
    directory::append_groups_to_directory(&path, &sample_groups(), false).unwrap();
    directory::append_dpg_to_directory(&path, &sample_dpg(), false).unwrap();

    let loaded = TrxFile::<f32>::load(&path).unwrap();
    assert_eq!(loaded.scalar_dps_f32("weight").unwrap(), vec![1.0, 2.0]);
    assert_eq!(loaded.group("bundle").unwrap(), &[0u32]);
    assert_eq!(
        loaded.dpg::<u8>("bundle", "color").unwrap().row(0),
        &[7, 8, 9]
    );

    directory::delete_groups_from_directory(&path, &["bundle"]).unwrap();

    let updated = TrxFile::<f32>::load(&path).unwrap();
    assert_eq!(updated.group_names(), Vec::<&str>::new());
    assert_eq!(updated.dpg_group_names(), Vec::<&str>::new());
}

#[test]
fn zip_append_rejects_invalid_row_count() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("invalid.trx");
    base_trx().save_to_zip(&path).unwrap();

    let invalid = HashMap::from([(
        "bad".to_string(),
        DataArray::owned_bytes(vec_to_bytes(vec![1.0f32]), 1, DType::Float32),
    )]);
    let err =
        zip::append_dps_to_zip(&path, &invalid, CompressionMethod::Stored, false).unwrap_err();

    assert!(err.to_string().contains("expected 2"));
}
