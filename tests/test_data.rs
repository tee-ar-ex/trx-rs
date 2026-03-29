//! Integration tests using downloaded test data from trx-test-data.
//!
//! Test data is automatically downloaded from GitHub releases on first run
//! and cached in `target/test_data/`. You can also set `TRX_TEST_DATA_DIR`
//! to point at a pre-existing directory.
//!
//! Archives from:
//!   https://github.com/tee-ar-ex/trx-test-data/releases/download/v0.1.0/

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Once;

use trx_rs::{AnyTrxFile, DType, TrxFile};

const BASE_URL: &str = "https://github.com/tee-ar-ex/trx-test-data/releases/download/v0.1.0";
const ARCHIVES: &[&str] = &["gold_standard.zip", "memmap_test_data.zip"];

/// Download and extract test data once, return the cache directory.
fn ensure_test_data() -> PathBuf {
    static INIT: Once = Once::new();

    let cache_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test_data");

    INIT.call_once(|| {
        std::fs::create_dir_all(&cache_dir).expect("failed to create test_data cache dir");

        for archive_name in ARCHIVES {
            let marker = cache_dir.join(format!(".{archive_name}.done"));
            if marker.exists() {
                continue;
            }

            let url = format!("{BASE_URL}/{archive_name}");
            eprintln!("Downloading {url} ...");

            let response = ureq::get(&url)
                .call()
                .expect("failed to download test data");

            let mut bytes = Vec::new();
            response
                .into_body()
                .as_reader()
                .read_to_end(&mut bytes)
                .expect("failed to read response body");

            let cursor = std::io::Cursor::new(bytes);
            let mut archive = zip::ZipArchive::new(cursor).expect("failed to open downloaded zip");

            for i in 0..archive.len() {
                let mut entry = archive.by_index(i).unwrap();
                let out_path = cache_dir.join(entry.name());
                if entry.is_dir() {
                    std::fs::create_dir_all(&out_path).unwrap();
                } else {
                    if let Some(parent) = out_path.parent() {
                        std::fs::create_dir_all(parent).unwrap();
                    }
                    let mut out_file = std::fs::File::create(&out_path).unwrap();
                    std::io::copy(&mut entry, &mut out_file).unwrap();
                }
            }

            std::fs::write(&marker, "").unwrap();
            eprintln!("Extracted {archive_name}");
        }
    });

    cache_dir
}

/// Return the test data root, auto-downloading if needed.
fn test_data_root() -> PathBuf {
    if let Ok(dir) = std::env::var("TRX_TEST_DATA_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    ensure_test_data()
}

fn gold_standard_dir() -> PathBuf {
    let root = test_data_root();
    // Try subdirectory first (when TRX_TEST_DATA_DIR is set), then flat layout
    for candidate in [root.join("gold_standard"), root.clone()] {
        if candidate.join("gs.trx").exists() {
            return candidate;
        }
    }
    root
}

fn memmap_test_data_dir() -> PathBuf {
    let root = test_data_root();
    for candidate in [root.join("memmap_test_data"), root.clone()] {
        if candidate.join("small.trx").exists() {
            return candidate;
        }
    }
    root
}

/// Load the gold standard RASMM coordinates from the text file.
fn load_rasmm_coords(path: &Path) -> Vec<[f32; 3]> {
    let text = std::fs::read_to_string(path).expect("failed to read rasmm coords");
    let values: Vec<f32> = text
        .split_whitespace()
        .map(|s| s.parse::<f32>().expect("failed to parse float"))
        .collect();
    assert_eq!(values.len() % 3, 0, "coordinate count not multiple of 3");
    values.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

// ── Gold Standard Tests ─────────────────────────────────────────────

#[test]
fn gs_load_zip_positions_match_rasmm() {
    let gs_dir = gold_standard_dir();
    let coords = load_rasmm_coords(&gs_dir.join("gs_rasmm_space.txt"));

    let trx = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();
    let positions = trx.positions();

    assert_eq!(
        positions.len(),
        coords.len(),
        "vertex count mismatch: got {}, expected {}",
        positions.len(),
        coords.len()
    );

    for (i, (actual, expected)) in positions.iter().zip(coords.iter()).enumerate() {
        for j in 0..3 {
            let diff = (actual[j] - expected[j]).abs();
            let tol = 1e-6 + 1e-4 * expected[j].abs();
            assert!(
                diff <= tol,
                "position mismatch at vertex {i}, component {j}: got {}, expected {} (diff {diff})",
                actual[j],
                expected[j]
            );
        }
    }
}

#[test]
fn gs_load_dir_positions_match_rasmm() {
    let gs_dir = gold_standard_dir();
    let coords = load_rasmm_coords(&gs_dir.join("gs_rasmm_space.txt"));

    let trx = TrxFile::<f32>::load(&gs_dir.join("gs_fldr.trx")).unwrap();
    let positions = trx.positions();

    assert_eq!(positions.len(), coords.len());

    for (i, (actual, expected)) in positions.iter().zip(coords.iter()).enumerate() {
        for j in 0..3 {
            let diff = (actual[j] - expected[j]).abs();
            let tol = 1e-6 + 1e-4 * expected[j].abs();
            assert!(
                diff <= tol,
                "position mismatch at vertex {i}, component {j}: got {}, expected {}",
                actual[j],
                expected[j]
            );
        }
    }
}

#[test]
fn gs_zip_and_dir_match() {
    let gs_dir = gold_standard_dir();

    let zip_trx = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();
    let dir_trx = TrxFile::<f32>::load(&gs_dir.join("gs_fldr.trx")).unwrap();

    assert_eq!(zip_trx.nb_streamlines(), dir_trx.nb_streamlines());
    assert_eq!(zip_trx.nb_vertices(), dir_trx.nb_vertices());
    assert_eq!(zip_trx.positions(), dir_trx.positions());
    assert_eq!(zip_trx.offsets(), dir_trx.offsets());
}

#[test]
fn gs_header_values() {
    let gs_dir = gold_standard_dir();

    let trx = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();

    assert!(trx.nb_streamlines() > 0);
    assert!(trx.nb_vertices() > 0);
    assert_eq!(trx.header().dimensions.len(), 3);
    assert!(trx.header().dimensions.iter().all(|&d| d > 0));
}

#[test]
fn gs_round_trip_zip() {
    let gs_dir = gold_standard_dir();

    let original = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let out_path = tmp.path().join("gs_roundtrip.trx");

    original.save_to_zip(&out_path).unwrap();
    let reloaded = TrxFile::<f32>::load(&out_path).unwrap();

    assert_eq!(reloaded.nb_streamlines(), original.nb_streamlines());
    assert_eq!(reloaded.nb_vertices(), original.nb_vertices());
    assert_eq!(reloaded.positions(), original.positions());
    assert_eq!(reloaded.offsets(), original.offsets());
}

#[test]
fn gs_round_trip_directory() {
    let gs_dir = gold_standard_dir();

    let original = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let out_path = tmp.path().join("gs_roundtrip_dir");

    original.save_to_directory(&out_path).unwrap();
    let reloaded = TrxFile::<f32>::load(&out_path).unwrap();

    assert_eq!(reloaded.nb_streamlines(), original.nb_streamlines());
    assert_eq!(reloaded.nb_vertices(), original.nb_vertices());
    assert_eq!(reloaded.positions(), original.positions());
}

#[test]
fn gs_multi_round_trip() {
    let gs_dir = gold_standard_dir();
    let coords = load_rasmm_coords(&gs_dir.join("gs_rasmm_space.txt"));

    let mut trx = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();
    let tmp = tempfile::TempDir::new().unwrap();

    // Save and reload 3 times
    for i in 0..3 {
        let out_path = tmp.path().join(format!("gs_rt_{i}.trx"));
        trx.save_to_zip(&out_path).unwrap();
        trx = TrxFile::<f32>::load(&out_path).unwrap();
    }

    let positions = trx.positions();
    assert_eq!(positions.len(), coords.len());
    for (i, (actual, expected)) in positions.iter().zip(coords.iter()).enumerate() {
        for j in 0..3 {
            let diff = (actual[j] - expected[j]).abs();
            let tol = 1e-6 + 1e-4 * expected[j].abs();
            assert!(diff <= tol, "drift at vertex {i}, component {j}");
        }
    }
}

// ── Memmap Test Data Tests ──────────────────────────────────────────

#[test]
fn memmap_load_small_trx() {
    let mm_dir = memmap_test_data_dir();

    let trx = AnyTrxFile::load(&mm_dir.join("small.trx")).unwrap();

    // small.trx is float16 positions
    assert!(
        matches!(trx.dtype(), DType::Float16 | DType::Float32),
        "expected float16 or float32, got {:?}",
        trx.dtype()
    );
    assert!(trx.nb_streamlines() > 0);
    assert!(trx.nb_vertices() > 0);
}

#[test]
fn memmap_load_small_compressed() {
    let mm_dir = memmap_test_data_dir();

    let trx = AnyTrxFile::load(&mm_dir.join("small_compressed.trx")).unwrap();
    assert!(trx.nb_streamlines() > 0);
}

#[test]
fn memmap_load_small_dir() {
    let mm_dir = memmap_test_data_dir();
    let dir_path = mm_dir.join("small_fldr.trx");

    if !dir_path.exists() {
        eprintln!("Skipping: small_fldr.trx not found");
        return;
    }

    let trx = AnyTrxFile::load(&dir_path).unwrap();
    assert!(trx.nb_streamlines() > 0);
}

#[test]
fn memmap_small_zip_and_dir_match() {
    let mm_dir = memmap_test_data_dir();

    let zip_path = mm_dir.join("small.trx");
    let dir_path = mm_dir.join("small_fldr.trx");

    if !dir_path.exists() {
        eprintln!("Skipping: small_fldr.trx not found");
        return;
    }

    let zip_trx = AnyTrxFile::load(&zip_path).unwrap();
    let dir_trx = AnyTrxFile::load(&dir_path).unwrap();

    assert_eq!(zip_trx.nb_streamlines(), dir_trx.nb_streamlines());
    assert_eq!(zip_trx.nb_vertices(), dir_trx.nb_vertices());
}

// ── AnyTrxFile Tests ────────────────────────────────────────────────

#[test]
fn any_trx_detect_dtype_gs() {
    let gs_dir = gold_standard_dir();

    let dtype = trx_rs::any_trx_file::detect_positions_dtype(&gs_dir.join("gs.trx")).unwrap();
    assert_eq!(dtype, DType::Float32);
}

#[test]
fn any_trx_detect_dtype_gs_dir() {
    let gs_dir = gold_standard_dir();

    let dtype = trx_rs::any_trx_file::detect_positions_dtype(&gs_dir.join("gs_fldr.trx")).unwrap();
    assert_eq!(dtype, DType::Float32);
}

#[test]
fn gs_streamline_lengths_consistent() {
    let gs_dir = gold_standard_dir();

    let trx = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();
    let lengths = trx.streamline_lengths();
    let total: usize = lengths.iter().sum();
    assert_eq!(total, trx.nb_vertices());
    assert_eq!(lengths.len(), trx.nb_streamlines());

    // Verify each streamline has the right length
    for (i, &len) in lengths.iter().enumerate() {
        assert_eq!(trx.streamline(i).len(), len);
    }
}

#[test]
fn gs_affine_round_trip() {
    let gs_dir = gold_standard_dir();

    let trx = TrxFile::<f32>::load(&gs_dir.join("gs.trx")).unwrap();

    let tmp = tempfile::TempDir::new().unwrap();
    let out_path = tmp.path().join("gs_affine.trx");
    trx.save_to_zip(&out_path).unwrap();

    let reloaded = TrxFile::<f32>::load(&out_path).unwrap();

    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (trx.header().voxel_to_rasmm[i][j] - reloaded.header().voxel_to_rasmm[i][j]).abs()
                    < 1e-10,
                "affine mismatch at [{i}][{j}]"
            );
        }
    }
}
