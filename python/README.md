# trxrs

`trxrs` is the Python binding layer for `trx-rs`, the Rust implementation of the
TRX tractography file format.

The package is aimed at Python users who want:

- fast TRX loading through the Rust core
- zero-copy, read-only NumPy views over TRX payload arrays
- a Python surface that stays close to `trx-python` for parity testing and migration

## Status

The current Python package is read-only and TRX-focused. It exposes:

- `load(path)`
- `TrxFile.positions()`
- `TrxFile.offsets()`
- `TrxFile.get_dps(name)`
- `TrxFile.get_dpv(name)`
- `TrxFile.get_group(name)`
- `TrxFile.get_dpg(group, name)`
- metadata and key accessors such as `header`, `dtype`, `dps_keys()`, and `dpg_keys()`

## Design Goals

`trxrs` is intentionally thin. The Rust crate continues to own file parsing,
memory mapping, and dtype handling. The Python layer focuses on:

- preserving zero-copy semantics whenever possible
- keeping the TRX owner alive for the lifetime of derived NumPy views
- matching important `trx-python` behaviors closely enough for parity tests

## Installation

For local development inside the repository:

```bash
cd python
maturin develop
```

If you are using a Conda or Mamba environment, activate it first and install the
build dependencies there:

```bash
mamba activate trx
python -m pip install maturin numpy
cd python
maturin develop
```

## Quick Start

```python
import trxrs

trx = trxrs.load("bundle.trx")

print(trx.nb_streamlines, trx.nb_vertices, trx.dtype)

positions = trx.positions()
offsets = trx.offsets()
weights = trx.get_dps("weights")
groups = trx.get_group("cst")
```

## Documentation

Documentation sources live in `python/docs/`.

The docs are structured so the Python binding can be published as its own Read
the Docs project, separate from the Rust crate documentation.
