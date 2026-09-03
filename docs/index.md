# trx-rs

A Rust library for reading, writing, and manipulating [TRX](https://github.com/tee-ar-ex/trx-spec) brain tractography files, plus format-conversion helpers for common streamline formats.

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

::::{grid} 2
:::{grid-item-card} Getting Started
```{toctree}
:maxdepth: 1
installation
quickstart
```
:::

:::{grid-item-card} User Guide
```{toctree}
:maxdepth: 1
conversion
transform
operations
api
```
:::
::::

