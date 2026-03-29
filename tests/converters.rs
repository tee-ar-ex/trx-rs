use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Once;

use trx_rs::{convert, read_tractogram, AnyTrxFile, ConversionOptions, DType, Header, Tractogram};

const BASE_URL: &str = "https://github.com/tee-ar-ex/trx-test-data/releases/download/v0.1.0";
const ARCHIVES: &[&str] = &["gold_standard.zip", "memmap_test_data.zip"];

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
            let response = ureq::get(&url)
                .call()
                .expect("failed to download converter test data");

            let mut bytes = Vec::new();
            response
                .into_body()
                .as_reader()
                .read_to_end(&mut bytes)
                .expect("failed to read converter test data");

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
            std::fs::write(marker, "").unwrap();
        }
    });

    cache_dir
}

fn load_rasmm_coords(path: &Path) -> Vec<[f32; 3]> {
    let text = std::fs::read_to_string(path).expect("failed to read rasmm coords");
    let values: Vec<f32> = text
        .split_whitespace()
        .map(|value| value.parse::<f32>().expect("failed to parse coordinate"))
        .collect();
    values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect()
}

fn gold_standard_dir() -> PathBuf {
    let root = ensure_test_data();
    for candidate in [root.join("gold_standard"), root.clone()] {
        if candidate.join("gs.trx").exists() {
            return candidate;
        }
    }
    root
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

fn tractogram_from_points() -> Tractogram {
    let mut tractogram = Tractogram::with_header(Header {
        voxel_to_rasmm: Header::identity_affine(),
        dimensions: [1, 1, 1],
        nb_streamlines: 0,
        nb_vertices: 0,
        extra: Default::default(),
    });
    tractogram
        .push_streamline(&[[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]])
        .unwrap();
    tractogram.push_streamline(&[[7.0, 8.0, 9.0]]).unwrap();
    tractogram
}

#[test]
fn gs_tck_positions_match_rasmm() {
    let gs_dir = gold_standard_dir();
    let coords = load_rasmm_coords(&gs_dir.join("gs_rasmm_space.txt"));

    let tractogram =
        read_tractogram(&gs_dir.join("gs.tck"), &ConversionOptions::default()).unwrap();
    assert_positions_close(tractogram.positions(), &coords);
}

#[test]
fn gs_vtk_positions_match_rasmm() {
    let gs_dir = gold_standard_dir();
    let coords = load_rasmm_coords(&gs_dir.join("gs_rasmm_space.txt"));

    let tractogram =
        read_tractogram(&gs_dir.join("gs.vtk"), &ConversionOptions::default()).unwrap();
    assert_positions_close(tractogram.positions(), &coords);
}

#[test]
fn trx_to_tck_and_back_preserves_geometry() {
    let gs_dir = gold_standard_dir();
    let tmp = tempfile::TempDir::new().unwrap();
    let tck_path = tmp.path().join("roundtrip.tck");
    let trx_path = tmp.path().join("roundtrip.trx");

    convert(
        &gs_dir.join("gs.trx"),
        &tck_path,
        &ConversionOptions::default(),
    )
    .unwrap();
    convert(&tck_path, &trx_path, &ConversionOptions::default()).unwrap();

    let original = read_tractogram(&gs_dir.join("gs.trx"), &ConversionOptions::default()).unwrap();
    let roundtrip = read_tractogram(&trx_path, &ConversionOptions::default()).unwrap();

    assert_eq!(roundtrip.nb_streamlines(), original.nb_streamlines());
    assert_positions_close(roundtrip.positions(), original.positions());
    assert_eq!(roundtrip.header().dimensions, [1, 1, 1]);
}

#[test]
fn writes_and_reads_tck_gz() {
    let tractogram = tractogram_from_points();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("sample.tck.gz");

    convert_pathless(&tractogram, &path, ConversionOptions::default()).unwrap();
    let loaded = read_tractogram(&path, &ConversionOptions::default()).unwrap();

    assert_eq!(loaded.nb_streamlines(), tractogram.nb_streamlines());
    assert_positions_close(loaded.positions(), tractogram.positions());
}

#[test]
fn writes_trx_as_float16_when_requested() {
    let tractogram = tractogram_from_points();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("sample.trx");
    let options = ConversionOptions {
        trx_positions_dtype: DType::Float16,
        ..Default::default()
    };

    convert_pathless(&tractogram, &path, options).unwrap();
    let loaded = AnyTrxFile::load(&path).unwrap();
    assert_eq!(loaded.dtype(), DType::Float16);
}

#[test]
fn vtk_ascii_round_trip() {
    let tractogram = tractogram_from_points();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("sample.vtk");

    convert_pathless(&tractogram, &path, ConversionOptions::default()).unwrap();
    let loaded = read_tractogram(&path, &ConversionOptions::default()).unwrap();

    assert_eq!(loaded.nb_streamlines(), tractogram.nb_streamlines());
    assert_positions_close(loaded.positions(), tractogram.positions());
}

fn convert_pathless(
    tractogram: &Tractogram,
    output: &Path,
    options: ConversionOptions,
) -> trx_rs::Result<()> {
    trx_rs::write_tractogram(output, tractogram, &options)
}
