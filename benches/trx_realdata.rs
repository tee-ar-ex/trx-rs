//! Benchmarks using real tractography data, mirroring trx-cpp's bench suite.
//!
//! Uses the 10M-streamline HCP reference file. Set `TRX_BENCH_DATA_DIR` to a
//! directory containing `10milHCP_dps-sift2.trx`, or the file is downloaded
//! from figshare on first run (~7.7 GB).
//!
//! Run:
//!   cargo bench --bench trx_realdata
//!
//! Quick smoke test (100k streamlines only):
//!   TRX_BENCH_MAX_STREAMLINES=100000 cargo bench --bench trx_realdata
//!
//! Environment variables:
//!   TRX_BENCH_DATA_DIR          - directory containing the reference .trx file
//!   TRX_BENCH_MAX_STREAMLINES   - cap on streamline counts (default: 10M)
//!   TRX_QUERY_TIMINGS_PATH      - JSONL output for per-slab query timings
//!                                 (default: bench/query_timings.jsonl)
//!   TRX_BENCH_RESULTS_PATH      - JSONL output for all benchmark results
//!                                 (default: bench/bench_results.jsonl)

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

use criterion::{criterion_group, BenchmarkId, Criterion, SamplingMode, Throughput};
use half::f16;

use trx_rs::{TrxFile, TrxStream};

// ── RSS memory tracking (matches trx-cpp's get_current_rss_kb) ───────

/// Get **current** RSS in kilobytes (not peak).
/// macOS: uses task_info(MACH_TASK_BASIC_INFO) for live resident size.
/// Linux: reads VmRSS from /proc/self/status.
#[cfg(target_os = "macos")]
fn get_current_rss_kb() -> u64 {
    // mach2 crate doesn't expose mach_task_basic_info, so we define it inline.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i32,
    }
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: Timeval,
        system_time: Timeval,
        policy: i32,
        suspend_count: i32,
    }
    const MACH_TASK_BASIC_INFO: u32 = 20;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut MachTaskBasicInfo,
            task_info_count: *mut u32,
        ) -> i32;
    }
    unsafe {
        let mut info: MachTaskBasicInfo = std::mem::zeroed();
        let mut count =
            (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info,
            &mut count,
        );
        if kr == 0 {
            info.resident_size / 1024
        } else {
            0
        }
    }
}

#[cfg(target_os = "linux")]
fn get_current_rss_kb() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb) = line.split_whitespace().nth(1) {
                    return kb.parse().unwrap_or(0);
                }
            }
        }
    }
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_current_rss_kb() -> u64 {
    0
}

// ── Constants matching trx-cpp ──────────────────────────────────────

const BENCH_DATA_URL: &str = "https://figshare.com/ndownloader/files/62317780";
const BENCH_DATA_FILENAME: &str = "10milHCP_dps-sift2.trx";

const SLAB_THICKNESS_MM: f64 = 5.0;
const SLAB_COUNT: usize = 20;

// FOV of the reference dataset (HCP, RAS+mm)
const FOV_MIN_Z: f64 = -60.0;
const FOV_MAX_Z: f64 = 75.0;
const FOV_MIN_X: f64 = -70.0;
const FOV_MAX_X: f64 = 70.0;
const FOV_MIN_Y: f64 = -108.0;
const FOV_MAX_Y: f64 = 79.0;

// ── JSONL output ────────────────────────────────────────────────────

fn query_timings_path() -> PathBuf {
    std::env::var("TRX_QUERY_TIMINGS_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("bench/query_timings.jsonl"))
}

fn bench_results_path() -> PathBuf {
    std::env::var("TRX_BENCH_RESULTS_PATH")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("bench/results_rust.json"))
}

/// Append a JSONL line to a file, creating parent directories as needed.
fn append_jsonl(path: &PathBuf, line: &str) {
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap();

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Collected benchmark entries for writing the final Google Benchmark JSON.
static BENCH_ENTRIES: Mutex<Vec<serde_json::Value>> = Mutex::new(Vec::new());

/// Record a benchmark result in Google Benchmark JSON format.
/// Automatically includes current RSS (max_rss_kb) matching trx-cpp's output.
fn record_bench_result(name: &str, time_ms: f64, counters: &[(&str, f64)]) {
    let mut obj = serde_json::Map::new();
    obj.insert("name".into(), name.into());
    obj.insert("real_time".into(), time_ms.into());
    obj.insert("time_unit".into(), "ms".into());
    // Include RSS unless explicitly provided in counters
    if !counters.iter().any(|(k, _)| *k == "max_rss_kb") {
        obj.insert("max_rss_kb".into(), (get_current_rss_kb() as f64).into());
    }
    for &(k, v) in counters {
        obj.insert(k.to_string(), v.into());
    }
    BENCH_ENTRIES
        .lock()
        .unwrap()
        .push(serde_json::Value::Object(obj));
}

/// Write per-slab query timings in the same JSONL format as trx-cpp.
fn write_query_timings(streamlines: usize, timings_ms: &[f64]) {
    let timings_str: Vec<String> = timings_ms.iter().map(|t| format!("{t}")).collect();
    let line = format!(
        concat!(
            "{{",
            "\"streamlines\":{streamlines},",
            "\"group_case\":0,",
            "\"group_count\":0,",
            "\"dps\":0,",
            "\"dpv\":0,",
            "\"slab_thickness_mm\":{slab_thickness},",
            "\"timings_ms\":[{timings}]",
            "}}"
        ),
        streamlines = streamlines,
        slab_thickness = SLAB_THICKNESS_MM,
        timings = timings_str.join(","),
    );
    append_jsonl(&query_timings_path(), &line);
}

/// Write accumulated benchmark results as a Google Benchmark-compatible JSON file.
fn write_results_json() {
    let entries = BENCH_ENTRIES.lock().unwrap();
    if entries.is_empty() {
        return;
    }

    let path = bench_results_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let doc = serde_json::json!({
        "context": {
            "library": "trx-rs",
        },
        "benchmarks": *entries,
    });

    if let Ok(json) = serde_json::to_string_pretty(&doc) {
        let _ = std::fs::write(&path, json);
        eprintln!("Wrote benchmark results to {}", path.display());
    }
}

// ── Data provisioning ───────────────────────────────────────────────

fn bench_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("TRX_BENCH_DATA_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/bench_data")
}

/// Ensure the reference TRX file exists, downloading if necessary.
fn ensure_reference_trx() -> PathBuf {
    static INIT: Once = Once::new();
    let dir = bench_data_dir();
    let path = dir.join(BENCH_DATA_FILENAME);

    INIT.call_once(|| {
        if path.exists() {
            eprintln!("Using existing benchmark data: {}", path.display());
            return;
        }

        std::fs::create_dir_all(&dir).expect("failed to create bench_data dir");

        eprintln!(
            "Downloading benchmark data (~7.7 GB) to {} ...",
            path.display()
        );
        eprintln!("  URL: {BENCH_DATA_URL}");
        eprintln!("  Set TRX_BENCH_DATA_DIR to skip this download.");

        let response = ureq::get(BENCH_DATA_URL)
            .call()
            .expect("failed to download benchmark data");

        let tmp_path = path.with_extension("trx.partial");
        {
            let mut out = std::fs::File::create(&tmp_path).expect("failed to create temp file");
            let mut body = response.into_body();
            let mut reader = body.as_reader();
            std::io::copy(&mut reader, &mut out).expect("failed to write benchmark data");
        }
        std::fs::rename(&tmp_path, &path).expect("failed to rename benchmark data");
        eprintln!("Download complete.");
    });

    path
}

/// Shared reference directory on disk — extracted once, reused across all benchmarks.
/// Returns (dir_path, nb_streamlines, nb_vertices).
fn get_reference_dir() -> &'static (PathBuf, usize, usize) {
    static REF: OnceLock<(PathBuf, usize, usize)> = OnceLock::new();
    REF.get_or_init(|| {
        let path = ensure_reference_trx();
        eprintln!("Extracting reference TRX to directory...");

        // Extract zip to a persistent temp directory.
        let dir = tempfile::TempDir::new().expect("failed to create temp dir");
        #[allow(deprecated)]
        let dir_path = dir.into_path(); // don't delete on drop — we keep it for the run

        let file = std::fs::File::open(&path).expect("failed to open reference trx");
        let mut archive = zip::ZipArchive::new(file).expect("invalid zip");
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).unwrap();
            let entry_path = dir_path.join(entry.name());
            if entry.is_dir() {
                std::fs::create_dir_all(&entry_path).unwrap();
            } else {
                if let Some(parent) = entry_path.parent() {
                    std::fs::create_dir_all(parent).unwrap();
                }
                let mut out_file = std::fs::File::create(&entry_path).unwrap();
                std::io::copy(&mut entry, &mut out_file).unwrap();
            }
        }

        // Read header to get counts.
        let header = trx_rs::Header::from_file(&dir_path.join("header.json")).unwrap();
        let nb_sl = header.nb_streamlines as usize;
        let nb_v = header.nb_vertices as usize;
        eprintln!(
            "  {nb_sl} streamlines, {nb_v} vertices  (dir: {})",
            dir_path.display()
        );
        (dir_path, nb_sl, nb_v)
    })
}

fn max_streamlines() -> usize {
    std::env::var("TRX_BENCH_MAX_STREAMLINES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000)
}

fn streamline_counts() -> Vec<usize> {
    let max = max_streamlines();
    let (_, ref_nb, _) = get_reference_dir();
    [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
        .into_iter()
        .filter(|&c| c <= max && c <= *ref_nb)
        .collect()
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Build AABB slab queries distributed along the Z axis.
fn build_slabs() -> Vec<([f64; 3], [f64; 3])> {
    let z_range = FOV_MAX_Z - FOV_MIN_Z;
    let step = z_range / SLAB_COUNT as f64;

    (0..SLAB_COUNT)
        .map(|i| {
            let z_lo = FOV_MIN_Z + i as f64 * step;
            let z_hi = z_lo + SLAB_THICKNESS_MM;
            ([FOV_MIN_X, FOV_MIN_Y, z_lo], [FOV_MAX_X, FOV_MAX_Y, z_hi])
        })
        .collect()
}

/// Build a prefix subset on disk by writing only the needed prefix of each
/// file from the reference directory. Returns an mmap-backed TrxFile.
///
/// This mirrors trx-cpp's `build_trx_file_on_disk_single()`: the result is
/// mmap-backed, keeping RSS proportional to accessed pages rather than total
/// data size. Unlike a full copy+truncate, we only write the bytes needed
/// for `n` streamlines.
fn build_prefix_on_disk(n: usize) -> TrxFile<f16> {
    let (ref_dir, ref_nb, _) = get_reference_dir();
    let count = n.min(*ref_nb);

    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    std::fs::create_dir_all(&dir).unwrap();

    if count == *ref_nb {
        // Full copy — use hardlinks where possible, fallback to copy.
        copy_dir_recursive(ref_dir, &dir);
        return trx_rs::io::directory::load_from_directory(&dir, Some(tmp)).unwrap();
    }

    // Read offsets from reference to find vertex_cutoff.
    let ref_off_path = find_file_by_prefix(ref_dir, "offsets");
    let ref_off_fname = ref_off_path.file_name().unwrap().to_str().unwrap();
    let off_parsed = trx_rs::io::filename::TrxFilename::parse(ref_off_fname).unwrap();
    let vertex_cutoff = read_offset_at(&ref_off_path, count, off_parsed.dtype);

    // Find reference positions file for dtype info.
    let ref_pos_path = find_file_by_prefix(ref_dir, "positions");
    let ref_pos_fname = ref_pos_path.file_name().unwrap().to_str().unwrap();
    let pos_parsed = trx_rs::io::filename::TrxFilename::parse(ref_pos_fname).unwrap();

    // Write prefix of positions.
    let pos_bytes = vertex_cutoff * pos_parsed.ncols * pos_parsed.dtype.size_of();
    copy_file_prefix(&ref_pos_path, &dir.join(ref_pos_fname), pos_bytes);

    // Write prefix of offsets.
    let off_bytes = (count + 1) * off_parsed.ncols * off_parsed.dtype.size_of();
    copy_file_prefix(&ref_off_path, &dir.join(ref_off_fname), off_bytes);

    // Write prefix of DPS arrays.
    copy_array_dir_prefix(&ref_dir.join("dps"), &dir.join("dps"), count);

    // Write prefix of DPV arrays.
    copy_array_dir_prefix(&ref_dir.join("dpv"), &dir.join("dpv"), vertex_cutoff);

    // Copy groups directory (if it exists) — these need full copy since
    // group membership indices aren't prefix-compatible.
    let ref_groups = ref_dir.join("groups");
    if ref_groups.exists() {
        copy_dir_recursive(&ref_groups, &dir.join("groups"));
    }

    // Write updated header.
    let mut header = trx_rs::Header::from_file(&ref_dir.join("header.json")).unwrap();
    header.nb_streamlines = count as u64;
    header.nb_vertices = vertex_cutoff as u64;
    header.write_to(&dir.join("header.json")).unwrap();

    trx_rs::io::directory::load_from_directory(&dir, Some(tmp)).unwrap()
}

/// Copy the first `n_bytes` from `src` to `dst`.
fn copy_file_prefix(src: &Path, dst: &Path, n_bytes: usize) {
    use std::io::{Read, Write as IoWrite};
    let mut reader = std::fs::File::open(src).unwrap();
    let mut writer = std::fs::File::create(dst).unwrap();
    let mut remaining = n_bytes;
    let mut buf = vec![0u8; 1024 * 1024]; // 1 MB chunks
    while remaining > 0 {
        let to_read = remaining.min(buf.len());
        let n = reader.read(&mut buf[..to_read]).unwrap();
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n]).unwrap();
        remaining -= n;
    }
}

/// Copy prefix rows of all array files from src_dir to dst_dir.
fn copy_array_dir_prefix(src_dir: &Path, dst_dir: &Path, row_count: usize) {
    if !src_dir.exists() {
        return;
    }
    std::fs::create_dir_all(dst_dir).unwrap();
    for entry in std::fs::read_dir(src_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let fname = path.file_name().unwrap().to_str().unwrap();
        if let Ok(parsed) = trx_rs::io::filename::TrxFilename::parse(fname) {
            let n_bytes = row_count * parsed.ncols * parsed.dtype.size_of();
            copy_file_prefix(&path, &dst_dir.join(fname), n_bytes);
        }
    }
}

/// Read the offset value at index `idx` from an offsets file.
fn read_offset_at(path: &Path, idx: usize, dtype: trx_rs::DType) -> usize {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).unwrap();
    let elem_size = dtype.size_of();
    f.seek(SeekFrom::Start((idx * elem_size) as u64)).unwrap();
    match dtype {
        trx_rs::DType::UInt32 => {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf).unwrap();
            u32::from_ne_bytes(buf) as usize
        }
        trx_rs::DType::UInt64 => {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf).unwrap();
            u64::from_ne_bytes(buf) as usize
        }
        _ => panic!("unexpected offsets dtype: {dtype}"),
    }
}

/// Find a file starting with `prefix.` in a directory.
fn find_file_by_prefix(dir: &Path, prefix: &str) -> PathBuf {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(prefix) && name_str.as_bytes().get(prefix.len()) == Some(&b'.') {
            return entry.path();
        }
    }
    panic!("no file starting with '{prefix}.' in {}", dir.display());
}

/// Recursively copy a directory.
fn copy_dir_recursive(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path);
        } else {
            std::fs::copy(&src_path, &dst_path).unwrap();
        }
    }
}

// ── Benchmarks ──────────────────────────────────────────────────────

/// Benchmark: Load a TRX file from zip.
fn bench_load(c: &mut Criterion) {
    let ref_path = ensure_reference_trx();

    let mut group = c.benchmark_group("load");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    group.bench_function("reference_f16_zip", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                let trx = TrxFile::<f16>::load(&ref_path).unwrap();
                let elapsed = start.elapsed();
                let n = trx.nb_streamlines();
                criterion::black_box(n);
                record_bench_result(
                    "BM_TrxLoad/reference_f16_zip",
                    elapsed.as_secs_f64() * 1000.0,
                    &[("streamlines", n as f64)],
                );
                total += elapsed;
            }
            total
        });
    });

    group.finish();
}

/// Benchmark: Save TRX subsets of varying sizes to zip (stored) and directory.
/// Uses native f16 dtype — no conversion.
fn bench_save(c: &mut Criterion) {
    let counts = streamline_counts();

    // -- Zip (stored) saves --
    let mut group = c.benchmark_group("save_zip_stored");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        eprintln!("Building on-disk f16 subset with {count} streamlines for save_zip_stored...");
        let subset = build_prefix_on_disk(count);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("f16", count), &subset, |b, trx| {
            let tmp = tempfile::TempDir::new().unwrap();
            let out_path = tmp.path().join("bench.trx");
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    trx.save_to_zip_stored(&out_path).unwrap();
                    let elapsed = start.elapsed();
                    let file_bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
                    record_bench_result(
                        &format!("BM_TrxFileSize_Float16/{}", trx.nb_streamlines()),
                        elapsed.as_secs_f64() * 1000.0,
                        &[
                            ("streamlines", trx.nb_streamlines() as f64),
                            ("group_case", 0.0),
                            ("group_count", 0.0),
                            ("dps", 0.0),
                            ("dpv", 0.0),
                            ("compression", 1.0),
                            ("positions_dtype", 16.0),
                            ("file_bytes", file_bytes as f64),
                            ("write_ms", elapsed.as_secs_f64() * 1000.0),
                        ],
                    );
                    total += elapsed;
                }
                total
            });
        });
    }

    group.finish();

    // -- Directory saves --
    let mut group = c.benchmark_group("save_directory");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        eprintln!("Building on-disk f16 subset with {count} streamlines for save_directory...");
        let subset = build_prefix_on_disk(count);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("f16", count), &subset, |b, trx| {
            let tmp = tempfile::TempDir::new().unwrap();
            let out_path = tmp.path().join("bench_dir");
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    trx.save_to_directory(&out_path).unwrap();
                    let elapsed = start.elapsed();
                    record_bench_result(
                        &format!("BM_TrxFileSize_Float16/{}", trx.nb_streamlines()),
                        elapsed.as_secs_f64() * 1000.0,
                        &[
                            ("streamlines", trx.nb_streamlines() as f64),
                            ("group_case", 0.0),
                            ("group_count", 0.0),
                            ("dps", 0.0),
                            ("dpv", 0.0),
                            ("compression", 0.0),
                            ("positions_dtype", 16.0),
                            ("write_ms", elapsed.as_secs_f64() * 1000.0),
                        ],
                    );
                    total += elapsed;
                }
                total
            });
        });
    }

    group.finish();
}

/// Benchmark: Streaming translate — load, translate positions +1.0, save.
/// This benchmark necessarily converts to f32 for arithmetic, matching trx-cpp.
fn bench_stream_translate(c: &mut Criterion) {
    let counts = streamline_counts();

    let mut group = c.benchmark_group("stream_translate_write");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        // Build on-disk subset, save to temp zip for the translate benchmark input.
        eprintln!("Building on-disk f16 subset with {count} streamlines for stream_translate...");
        let input_dir = tempfile::TempDir::new().unwrap();
        let input_path = input_dir.path().join("input.trx");
        {
            let subset = build_prefix_on_disk(count);
            subset.save_to_zip_stored(&input_path).unwrap();
        }

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("f16", count), &input_path, |b, path| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();

                    // Load as f16
                    let trx = TrxFile::<f16>::load(path).unwrap();

                    // Translate positions: convert to f32, add 1.0, write as f32
                    let mut stream =
                        TrxStream::<f32>::new(trx.header.voxel_to_rasmm, trx.header.dimensions);
                    for i in 0..trx.nb_streamlines() {
                        let sl = trx.streamline(i);
                        let translated: Vec<[f32; 3]> = sl
                            .iter()
                            .map(|p| {
                                [
                                    f16::to_f32(p[0]) + 1.0,
                                    f16::to_f32(p[1]) + 1.0,
                                    f16::to_f32(p[2]) + 1.0,
                                ]
                            })
                            .collect();
                        stream.push_streamline(&translated);
                    }
                    let result = stream.finalize();

                    // Save
                    let tmp = tempfile::TempDir::new().unwrap();
                    let out_path = tmp.path().join("translated.trx");
                    result.save_to_zip_stored(&out_path).unwrap();

                    let elapsed = start.elapsed();
                    record_bench_result(
                        &format!("BM_TrxStream_TranslateWrite/{count}"),
                        elapsed.as_secs_f64() * 1000.0,
                        &[
                            ("streamlines", count as f64),
                            ("group_case", 0.0),
                            ("group_count", 0.0),
                            ("dps", 0.0),
                            ("dpv", 0.0),
                            ("positions_dtype", 16.0),
                        ],
                    );
                    total += elapsed;
                    criterion::black_box(&out_path);
                }
                total
            });
        });
    }

    group.finish();
}

/// Benchmark: AABB slab queries on subsets of the reference data.
/// Uses native f16 — no conversion needed.
/// AABBs are precomputed once per subset (matching trx-cpp's caching strategy).
fn bench_query_aabb(c: &mut Criterion) {
    let counts = streamline_counts();
    let slabs = build_slabs();

    let mut group = c.benchmark_group("query_aabb_slabs");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        // Measure RSS before dataset setup to capture physical footprint.
        let rss_before_setup = get_current_rss_kb();

        eprintln!("Building on-disk f16 subset with {count} streamlines for query_aabb...");
        let subset = build_prefix_on_disk(count);

        // Precompute AABBs once, matching trx-cpp's build_streamline_aabbs() cache.
        eprintln!("  Precomputing streamline AABBs...");
        let aabb_start = Instant::now();
        let aabbs = trx_rs::ops::subset::build_streamline_aabbs(&subset);
        let aabb_ms = aabb_start.elapsed().as_secs_f64() * 1000.0;

        let rss_after_setup = get_current_rss_kb();
        let dataset_rss_kb = rss_after_setup.saturating_sub(rss_before_setup);
        eprintln!(
            "  AABBs built in {aabb_ms:.1} ms  (dataset footprint: {:.0} MB)",
            dataset_rss_kb as f64 / 1024.0
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("f16", count), &aabbs, |b, aabbs| {
            let mut first_iter = true;
            b.iter(|| {
                // Measure RSS delta during query phase, matching trx-cpp.
                let rss_iter_start = get_current_rss_kb();

                let mut total_hits = 0usize;
                let mut slab_times_ms = Vec::with_capacity(SLAB_COUNT);

                for (min, max) in &slabs {
                    let slab_start = Instant::now();
                    let hits = trx_rs::ops::subset::query_aabb_cached(aabbs, *min, *max);
                    slab_times_ms.push(slab_start.elapsed().as_secs_f64() * 1000.0);
                    total_hits += hits.len();
                }

                let query_rss_delta = get_current_rss_kb().saturating_sub(rss_iter_start);

                if first_iter {
                    write_query_timings(count, &slab_times_ms);

                    let total_ms: f64 = slab_times_ms.iter().sum();
                    let mut sorted = slab_times_ms.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = sorted[sorted.len() / 2];
                    let p95_idx = ((0.95 * sorted.len() as f64).ceil() as usize).saturating_sub(1);
                    let p95 = sorted[p95_idx.min(sorted.len() - 1)];

                    record_bench_result(
                        &format!("BM_TrxQueryAabb_Slabs/{count}"),
                        total_ms,
                        &[
                            ("streamlines", count as f64),
                            ("group_case", 0.0),
                            ("group_count", 0.0),
                            ("dps", 0.0),
                            ("dpv", 0.0),
                            ("positions_dtype", 16.0),
                            ("query_p50_ms", p50),
                            ("query_p95_ms", p95),
                            ("aabb_build_ms", aabb_ms),
                            ("max_rss_kb", query_rss_delta as f64),
                            ("dataset_rss_kb", dataset_rss_kb as f64),
                        ],
                    );
                    first_iter = false;
                }

                criterion::black_box(total_hits)
            });
        });

        drop(subset);
    }

    group.finish();
}

/// Benchmark: Subset extraction (pick every 10th streamline).
fn bench_subset(c: &mut Criterion) {
    let counts = streamline_counts();

    let mut group = c.benchmark_group("subset_streamlines");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        eprintln!("Building on-disk f16 subset with {count} streamlines for subset_bench...");
        let subset = build_prefix_on_disk(count);
        let indices: Vec<usize> = (0..subset.nb_streamlines()).step_by(10).collect();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("f16_every10th", count),
            &(subset, indices),
            |b, (trx, idx)| {
                b.iter(|| {
                    let result = trx_rs::ops::subset::subset_streamlines(trx, idx).unwrap();
                    criterion::black_box(result.nb_streamlines())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Iteration over all streamlines (simulating a processing pipeline).
fn bench_iterate(c: &mut Criterion) {
    let counts = streamline_counts();

    let mut group = c.benchmark_group("iterate_streamlines");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        eprintln!("Building on-disk f16 subset with {count} streamlines for iterate_bench...");
        let subset = build_prefix_on_disk(count);

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("f16", count), &subset, |b, trx| {
            b.iter(|| {
                let mut total_len = 0usize;
                for sl in trx.streamlines() {
                    total_len += sl.len();
                }
                criterion::black_box(total_len)
            });
        });
    }

    group.finish();
}

// ── Criterion groups ────────────────────────────────────────────────

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(30))
        .warm_up_time(Duration::from_secs(3));
    targets =
        bench_load,
        bench_save,
        bench_stream_translate,
        bench_query_aabb,
        bench_subset,
        bench_iterate
}

/// Custom main that writes Google Benchmark-compatible JSON after criterion runs.
fn main() {
    // Remove stale output files from previous runs.
    let _ = std::fs::remove_file(query_timings_path());
    let _ = std::fs::remove_file(bench_results_path());

    benches();

    // Finalize criterion (it writes its own reports).
    Criterion::default().configure_from_args().final_summary();

    // Write our Google Benchmark-compatible JSON.
    write_results_json();
}
