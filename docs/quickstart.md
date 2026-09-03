# Quick start

## Supported data types

| Type | Positions | DPS/DPV |
|------|-----------|---------|
| `f16` (half) | Yes | Yes |
| `f32` | Yes | Yes |
| `f64` | Yes | Yes |
| `u8`, `u16`, `u32`, `u64` | - | Yes |
| `i8`, `i16`, `i32`, `i64` | - | Yes |

## Load and iterate streamlines

```rust
use trx_rs::TrxFile;

let trx = TrxFile::<f32>::load("tractogram.trx")?;

println!("{} streamlines, {} vertices", trx.nb_streamlines(), trx.nb_vertices());

for (i, streamline) in trx.streamlines().enumerate() {
    println!("Streamline {i}: {} points", streamline.len());
}
```

## Runtime dtype detection

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

## Access ancillary data

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

## Build and save

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

## API reference

See the <a href="api/rust/trx_rs/index.html">API reference</a> for the full Rustdoc-generated API documentation.
