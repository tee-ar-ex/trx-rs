# trx-rs

A Rust library for reading, writing, and manipulating [TRX](https://github.com/tee-ar-ex/trx-spec) brain tractography files.

TRX is a binary file format for storing streamline tractography data using memory-mapped arrays. It supports multiple coordinate precisions (f16, f32, f64), per-vertex data (DPV), per-streamline data (DPS), and named groups — all backed by zero-copy memory mapping for efficient access to large datasets.

## Features

- **Zero-copy memory mapping** via `memmap2` for efficient access to multi-gigabyte tractography files
- **Multiple precision support** — f16, f32, and f64 positions with runtime dtype detection
- **Full read/write support** for both directory-based and ZIP archive (`.trx`) formats
- **Streaming construction** — build TRX files incrementally with `TrxStream`
- **Ancillary data access** — DPS (per-streamline), DPV (per-vertex), and group membership arrays
- **Set operations** — intersection, union, and difference on streamline sets
- **Connectivity matrices** — compute group-to-group connectivity (count or weighted)
- **Subset and merge** — extract streamlines by index or concatenate multiple files

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
trx-rs = { git = "https://github.com/YOUR_USERNAME/trx-rs" }
```

### Load and iterate streamlines

```rust
use trx_rs::TrxFile;

let trx = TrxFile::<f32>::load("tractogram.trx")?;

println!("{} streamlines, {} vertices", trx.nb_streamlines(), trx.nb_vertices());

for (i, streamline) in trx.streamlines().enumerate() {
    println!("Streamline {i}: {} points", streamline.len());
}
```

### Runtime dtype detection

When the position dtype isn't known at compile time:

```rust
use trx_rs::AnyTrxFile;

let any = AnyTrxFile::load("tractogram.trx")?;

any.with_typed(
    |trx_f16| { /* work with TrxFile<f16> */ },
    |trx_f32| { /* work with TrxFile<f32> */ },
    |trx_f64| { /* work with TrxFile<f64> */ },
);
```

### Access ancillary data

```rust
// Per-streamline scalar (e.g., SIFT2 weights)
let weights = trx.dps::<f32>("weights")?;
for row in weights.rows() {
    println!("weight: {}", row[0]);
}

// Per-vertex scalar (e.g., FA along streamline)
let fa = trx.dpv::<f32>("fa")?;

// Group membership (streamline indices)
let members: &[u32] = trx.group("corticospinal_tract")?;
```

### Build and save

```rust
use trx_rs::TrxStream;

let affine = [[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]];
let dimensions = [128, 128, 80];

let mut stream = TrxStream::<f32>::new(affine, dimensions);
stream.push_streamline(&points);
let trx = stream.finalize();

trx.save("output.trx")?;           // ZIP archive
trx.save_to_directory("output/")?;  // directory format
```

### Operations

```rust
use trx_rs::ops::{subset_streamlines, merge_trx_shards, intersection, difference};

// Extract a subset of streamlines by index
let sub = subset_streamlines(&trx, &[0, 5, 10, 42])?;

// Merge multiple files (must share affine and dimensions)
let merged = merge_trx_shards(&[&shard1, &shard2])?;

// Set operations on streamlines
let common = intersection(&trx_a, &trx_b)?;
let only_a = difference(&trx_a, &trx_b)?;
```

## Supported data types

| Type | Positions | DPS/DPV |
|------|-----------|---------|
| `f16` (half) | Yes | Yes |
| `f32` | Yes | Yes |
| `f64` | Yes | Yes |
| `u8`, `u16`, `u32`, `u64` | - | Yes |
| `i8`, `i16`, `i32`, `i64` | - | Yes |

## Testing

```bash
# Unit tests (synthetic data, fast)
cargo test

# Integration tests (downloads gold-standard test data on first run)
cargo test -- --ignored
```

## Benchmarks

Benchmarks use a real 10M-streamline HCP dataset (~7.7 GB, auto-downloaded from figshare):

```bash
# Quick smoke test with 100k streamlines
TRX_BENCH_MAX_STREAMLINES=100000 cargo bench

# Full benchmark
cargo bench
```

Set `TRX_BENCH_DATA_DIR` to point to a cached copy of the benchmark data.

## License

BSD 2-Clause
