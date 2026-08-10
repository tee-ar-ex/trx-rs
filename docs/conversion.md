# Format conversion

`trx-rs` supports reading and writing multiple tractogram formats via its library conversion layer.

## Library usage

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

## Supported formats

| Format | Read | Write | Notes |
|------|------|------|------|
| TRX | Yes | Yes | Directory and `.trx` zip archives |
| TCK / TCK.GZ | Yes | Yes | Gzipped TCK is supported |
| VTK | Yes | Yes | Legacy PolyData `POINTS` + `LINES` subset |
| TT.GZ | Yes | No | Import only; TT clusters become TRX groups and TT colors become `dpg/<group>/color.3.uint8` |

## CLI conversion

The `trxrs` binary exposes the same conversion functionality:

```bash
# Convert gzipped TCK to TRX with float16 positions
trxrs convert bundles.tck.gz bundles.trx --positions-dtype f16

# Rewrite an existing TRX with float32 positions
trxrs manipulate-dtype input.trx output.trx --positions-dtype f32

# Concatenate TRX files and write float16 output
trxrs concatenate shard1.trx shard2.trx --output merged.trx --positions-dtype f16
```

## TT import behavior

- TT streamline points are decoded in voxel space and mapped to TRX positions with `trans_to_mni` directly
- TT `cluster` values become TRX groups
- `file.tt.gz.txt` sidecar labels are used for group names when present
- TT colors become `dpg/<group>/color.3.uint8`
- TT `report` and `parameter_id` are stored in `header.extra`
