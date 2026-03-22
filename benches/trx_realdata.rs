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
use std::path::PathBuf;
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
    struct Timeval { tv_sec: i64, tv_usec: i32 }
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
        let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
        let kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, &mut info, &mut count);
        if kr == 0 { info.resident_size as u64 / 1024 } else { 0 }
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
fn get_current_rss_kb() -> u64 { 0 }

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

/// Shared reference TRX file — loaded once, reused across all benchmarks.
fn get_reference() -> &'static TrxFile<f16> {
    static REF: OnceLock<TrxFile<f16>> = OnceLock::new();
    REF.get_or_init(|| {
        let path = ensure_reference_trx();
        eprintln!("Loading reference TRX file...");
        let trx = TrxFile::<f16>::load(&path).unwrap();
        eprintln!(
            "  {} streamlines, {} vertices",
            trx.nb_streamlines(),
            trx.nb_vertices()
        );
        trx
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
    [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
        .into_iter()
        .filter(|&c| c <= max)
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
            (
                [FOV_MIN_X, FOV_MIN_Y, z_lo],
                [FOV_MAX_X, FOV_MAX_Y, z_hi],
            )
        })
        .collect()
}

/// Build a prefix subset of the reference as TrxFile<f16>, taking `n` streamlines.
/// Uses subset_streamlines to avoid copying + converting all positions.
fn build_subset_f16(reference: &TrxFile<f16>, n: usize) -> TrxFile<f16> {
    let count = n.min(reference.nb_streamlines());
    let indices: Vec<usize> = (0..count).collect();
    trx_rs::ops::subset::subset_streamlines(reference, &indices).unwrap()
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
    let reference = get_reference();
    let counts = streamline_counts();

    // -- Zip (stored) saves --
    let mut group = c.benchmark_group("save_zip_stored");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        if count > reference.nb_streamlines() {
            continue;
        }

        eprintln!("Building f16 subset with {count} streamlines for save_zip_stored...");
        let subset = build_subset_f16(reference, count);

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
                    let file_bytes =
                        std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
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
        // `subset` dropped here before building the next one
    }

    group.finish();

    // -- Directory saves --
    let mut group = c.benchmark_group("save_directory");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        if count > reference.nb_streamlines() {
            continue;
        }

        eprintln!("Building f16 subset with {count} streamlines for save_directory...");
        let subset = build_subset_f16(reference, count);

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
    let reference = get_reference();
    let counts = streamline_counts();

    let mut group = c.benchmark_group("stream_translate_write");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        if count > reference.nb_streamlines() {
            continue;
        }

        // Build f16 subset, save to temp zip, then drop the in-memory subset.
        eprintln!("Building f16 subset with {count} streamlines for stream_translate...");
        let input_dir = tempfile::TempDir::new().unwrap();
        let input_path = input_dir.path().join("input.trx");
        {
            let subset = build_subset_f16(reference, count);
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
                    let mut stream = TrxStream::<f32>::new(
                        trx.header.voxel_to_rasmm,
                        trx.header.dimensions,
                    );
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
    let reference = get_reference();
    let counts = streamline_counts();
    let slabs = build_slabs();

    let mut group = c.benchmark_group("query_aabb_slabs");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        if count > reference.nb_streamlines() {
            continue;
        }

        eprintln!("Building f16 subset with {count} streamlines for query_aabb...");
        let subset = build_subset_f16(reference, count);

        // Precompute AABBs once, matching trx-cpp's build_streamline_aabbs() cache.
        eprintln!("  Precomputing streamline AABBs...");
        let aabb_start = Instant::now();
        let aabbs = trx_rs::ops::subset::build_streamline_aabbs(&subset);
        let aabb_ms = aabb_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  AABBs built in {aabb_ms:.1} ms");

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("f16", count), &aabbs, |b, aabbs| {
            let mut first_iter = true;
            b.iter(|| {
                let mut total_hits = 0usize;
                let mut slab_times_ms = Vec::with_capacity(SLAB_COUNT);

                for (min, max) in &slabs {
                    let slab_start = Instant::now();
                    let hits = trx_rs::ops::subset::query_aabb_cached(aabbs, *min, *max);
                    slab_times_ms.push(slab_start.elapsed().as_secs_f64() * 1000.0);
                    total_hits += hits.len();
                }

                if first_iter {
                    write_query_timings(count, &slab_times_ms);

                    let total_ms: f64 = slab_times_ms.iter().sum();
                    let mut sorted = slab_times_ms.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p50 = sorted[sorted.len() / 2];
                    let p95_idx =
                        ((0.95 * sorted.len() as f64).ceil() as usize).saturating_sub(1);
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
    let reference = get_reference();
    let counts = streamline_counts();

    let mut group = c.benchmark_group("subset_streamlines");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        if count > reference.nb_streamlines() {
            continue;
        }

        eprintln!("Building f16 subset with {count} streamlines for subset_bench...");
        let subset = build_subset_f16(reference, count);
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
    let reference = get_reference();
    let counts = streamline_counts();

    let mut group = c.benchmark_group("iterate_streamlines");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    for &count in &counts {
        if count > reference.nb_streamlines() {
            continue;
        }

        eprintln!("Building f16 subset with {count} streamlines for iterate_bench...");
        let subset = build_subset_f16(reference, count);

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
    Criterion::default()
        .configure_from_args()
        .final_summary();

    // Write our Google Benchmark-compatible JSON.
    write_results_json();
}
