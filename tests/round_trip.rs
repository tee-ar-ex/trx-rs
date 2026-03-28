use trx_rs::{Header, TrxFile, TrxStream};

/// Helper: create a test TRX with known data.
fn create_test_trx(num_streamlines: usize, points_per: usize) -> TrxFile<f32> {
    let mut stream = TrxStream::<f32>::new(Header::identity_affine(), [100, 100, 100]);

    for s in 0..num_streamlines {
        let points: Vec<[f32; 3]> = (0..points_per)
            .map(|p| {
                let v = (s * points_per + p) as f32;
                [v, v + 0.1, v + 0.2]
            })
            .collect();
        stream.push_streamline(&points);
    }

    stream.finalize()
}

#[test]
fn directory_round_trip() {
    let original = create_test_trx(5, 10);
    let dir = tempfile::TempDir::new().unwrap();
    let dir_path = dir.path().join("test.trxd");

    original.save_to_directory(&dir_path).unwrap();
    let loaded = TrxFile::<f32>::load(&dir_path).unwrap();

    assert_eq!(loaded.nb_streamlines(), 5);
    assert_eq!(loaded.nb_vertices(), 50);

    for i in 0..5 {
        assert_eq!(loaded.streamline(i), original.streamline(i));
    }
}

#[test]
fn zip_round_trip() {
    let original = create_test_trx(3, 7);
    let dir = tempfile::TempDir::new().unwrap();
    let zip_path = dir.path().join("test.trx");

    original.save_to_zip(&zip_path).unwrap();
    let loaded = TrxFile::<f32>::load(&zip_path).unwrap();

    assert_eq!(loaded.nb_streamlines(), 3);
    assert_eq!(loaded.nb_vertices(), 21);

    for i in 0..3 {
        assert_eq!(loaded.streamline(i), original.streamline(i));
    }
}

#[test]
fn any_trx_file_load() {
    let original = create_test_trx(2, 5);
    let dir = tempfile::TempDir::new().unwrap();
    let zip_path = dir.path().join("test.trx");
    original.save_to_zip(&zip_path).unwrap();

    let any = trx_rs::AnyTrxFile::load(&zip_path).unwrap();
    assert_eq!(any.dtype(), trx_rs::DType::Float32);
    assert_eq!(any.nb_streamlines(), 2);
    assert_eq!(any.nb_vertices(), 10);
}

#[test]
fn streamlines_iterator() {
    let trx = create_test_trx(4, 3);
    let streamlines: Vec<_> = trx.streamlines().collect();
    assert_eq!(streamlines.len(), 4);
    for (i, sl) in streamlines.iter().enumerate() {
        assert_eq!(sl.len(), 3);
        assert_eq!(*sl, trx.streamline(i));
    }
}

#[test]
fn positions_bytes_for_gpu() {
    let trx = create_test_trx(2, 3);
    let bytes = trx.positions_bytes();
    // 6 vertices × 3 floats × 4 bytes = 72 bytes
    assert_eq!(bytes.len(), 72);

    // Verify we can cast back
    let positions: &[[f32; 3]] = bytemuck::cast_slice(bytes);
    assert_eq!(positions.len(), 6);
    assert_eq!(positions, trx.positions());
}

#[test]
fn zip_round_trip_writes_uint32_offsets_by_default() {
    let original = create_test_trx(3, 7);
    let dir = tempfile::TempDir::new().unwrap();
    let zip_path = dir.path().join("test.trx");

    original.save_to_zip(&zip_path).unwrap();

    let file = std::fs::File::open(&zip_path).unwrap();
    let mut archive = zip::ZipArchive::new(file).unwrap();
    assert!(archive.by_name("offsets.1.uint32").is_ok());
    assert!(archive.by_name("offsets.1.uint64").is_err());
}

#[test]
fn directory_round_trip_writes_uint32_offsets_by_default() {
    let original = create_test_trx(3, 7);
    let dir = tempfile::TempDir::new().unwrap();
    let out_dir = dir.path().join("test.trxd");

    original.save_to_directory(&out_dir).unwrap();

    assert!(out_dir.join("offsets.1.uint32").exists());
    assert!(!out_dir.join("offsets.1.uint64").exists());
}
