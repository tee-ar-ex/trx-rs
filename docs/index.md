# trx-rs

A Rust library for reading, writing, and manipulating [TRX](https://github.com/tee-ar-ex/trx-spec) brain tractography files, plus format-conversion helpers for common streamline formats.

```{toctree}
:maxdepth: 2
installation
quickstart
conversion
transform
operations
api
```

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

## Supported data types

| Type | Positions | DPS/DPV |
|------|-----------|---------|
| `f16` (half) | Yes | Yes |
| `f32` | Yes | Yes |
| `f64` | Yes | Yes |
| `u8`, `u16`, `u32`, `u64` | - | Yes |
| `i8`, `i16`, `i32`, `i64` | - | Yes |

## License

BSD 2-Clause.
