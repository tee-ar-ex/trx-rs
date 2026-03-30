# trx-rs

A Rust library for reading, writing, and manipulating [TRX](https://github.com/tee-ar-ex/trx-spec) brain tractography files, plus format-conversion helpers for common streamline formats.

TRX is a binary file format for storing streamline tractography data using memory-mapped arrays. It supports multiple coordinate precisions (f16, f32, f64), per-vertex data (DPV), per-streamline data (DPS), named groups, and data-per-group (DPG) arrays — all backed by zero-copy memory mapping for efficient access to large datasets.

## Features

- **Zero-copy memory mapping** via `memmap2` for efficient access to multi-gigabyte tractography files
- **Multiple precision support** — f16, f32, and f64 positions with runtime dtype detection
- **Full read/write support** for both directory-based and ZIP archive (`.trx`) formats
- **Format conversion helpers** for `.tck`, `.tck.gz`, `.vtk`, and `.tt.gz -> .trx`
- **Streaming construction** — build TRX files incrementally with `TrxStream`
- **Ancillary data access** — DPS (per-streamline), DPV (per-vertex), group membership arrays, and DPG metadata
- **Set operations** — intersection, union, and difference on streamline sets
- **Connectivity matrices** — compute group-to-group connectivity (count or weighted)
- **Subset and merge** — extract streamlines by index or concatenate multiple files

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
trx-rs = { git = "https://github.com/tee-ar-ex/trx-rs" }
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

// Data-per-group metadata (for example, RGB color stored on a group)
let color = trx.dpg::<u8>("corticospinal_tract", "color")?;
assert_eq!(color.shape(), (1, 3));
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
use trx_rs::ops::merge::merge_trx_shards;
use trx_rs::ops::subset::subset_streamlines;
use trx_rs::ops::streamline_ops::{difference, intersection};

// Extract a subset of streamlines by index
let sub = subset_streamlines(&trx, &[0, 5, 10, 42])?;

// Merge multiple files (must share affine and dimensions)
let merged = merge_trx_shards(&[&shard1, &shard2])?;

// Set operations on streamlines
let common = intersection(&trx_a, &trx_b)?;
let only_a = difference(&trx_a, &trx_b)?;
```

### Format conversion

`trx-rs` currently exposes conversion at the library level:

```rust
use trx_rs::{convert, read_tractogram, write_tractogram, ConversionOptions, DType};

// Convert between file formats.
convert("input.tck.gz".as_ref(), "output.trx".as_ref(), &ConversionOptions::default())?;

// Read a non-TRX tractogram into the neutral in-memory representation.
let tractogram = read_tractogram("bundles.tt.gz".as_ref(), &ConversionOptions::default())?;

// Write TRX with explicit positions dtype.
let options = ConversionOptions {
    trx_positions_dtype: DType::Float16,
    ..Default::default()
};
write_tractogram("bundles_f16.trx".as_ref(), &tractogram, &options)?;
```

Supported conversion formats:

| Format | Read | Write | Notes |
|------|------|------|------|
| `TRX` | Yes | Yes | Directory and `.trx` zip archives |
| `TCK` / `TCK.GZ` | Yes | Yes | Gzipped TCK is supported |
| `VTK` | Yes | Yes | Legacy PolyData `POINTS` + `LINES` subset |
| `TT.GZ` | Yes | No | Import only; TT clusters become TRX groups and TT colors become `dpg/<group>/color.3.uint8` |

TT import behavior:

- TT streamline points are decoded in voxel space and mapped to TRX positions with `trans_to_mni` directly
- TT `cluster` values become TRX groups
- `file.tt.gz.txt` sidecar labels are used for group names when present
- TT colors become `dpg/<group>/color.3.uint8`
- TT `report` and `parameter_id` are stored in `header.extra`

## Building

Build the library and tests:

```bash
cargo build
cargo test --workspace
```

Build an optimized release:

```bash
cargo build --release
```

## Installation

As a library dependency:

```toml
[dependencies]
trx-rs = { git = "https://github.com/tee-ar-ex/trx-rs" }
```

For local development:

```bash
git clone https://github.com/tee-ar-ex/trx-rs
cd trx-rs
cargo build --release
```

## Command-line interface

The repository now ships a `trx` binary under `src/bin/trx.rs`.

Build it locally:

```bash
cargo build --release
./target/release/trx --help
```

Install it into Cargo's bin directory:

```bash
cargo install --path .
trx --help
```

Available subcommands:

- `trx info <input>`
- `trx convert <input> <output> [--positions-dtype f16|f32|f64]`
- `trx concatenate <input-a.trx> <input-b.trx> ... --output <output.trx> [--positions-dtype f16|f32|f64]`
- `trx manipulate-dtype <input.trx> <output.trx> [--positions-dtype f16|f32|f64]`

Examples:

```bash
# Inspect a TRX or supported foreign-format tractogram
trx info bundles.trx

# Convert gzipped TCK to TRX with float16 positions
trx convert bundles.tck.gz bundles.trx --positions-dtype f16

# Rewrite an existing TRX with float32 positions
trx manipulate-dtype input.trx output.trx --positions-dtype f32

# Concatenate TRX files and write float16 output
trx concatenate shard1.trx shard2.trx --output merged.trx --positions-dtype f16
```

Notes:

- `trx convert` uses the library conversion layer and supports `trx`, `tck`, `tck.gz`, `vtk`, and `tt.gz` input
- `.tt.gz` is import-only; writing Tiny Track is not implemented
- `trx concatenate` currently supports TRX inputs and TRX output only
- offsets are always written as `uint32`

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
# Full test suite
cargo test --workspace

# Formatting check
cargo fmt --check

# Lint with warnings denied
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Coverage report (requires cargo-llvm-cov)
cargo llvm-cov --workspace --html
```

Real-data integration tests download the fixture archives on first run and cache
them in `target/test_data/`. Set `TRX_TEST_DATA_DIR` to reuse an existing copy
of the fixture data instead of downloading it again.

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
