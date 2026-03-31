Overview
========

What `trxrs` is
---------------

`trxrs` is a thin Python binding over the Rust crate `trx-rs`.

It exists to expose the Rust loader and its memory-mapped data model to Python
without re-implementing TRX parsing logic in Python.

The current package is intentionally narrow:

- read-only
- TRX-focused
- optimized for zero-copy NumPy access
- shaped to make parity checks against `trx-python` straightforward

Core Principles
---------------

Zero-copy first
~~~~~~~~~~~~~~~

The package is built around direct NumPy views into Rust-owned memory-mapped
buffers. The design goal is to avoid copying numeric payload arrays on normal
read paths.

Rust owns the mapping
~~~~~~~~~~~~~~~~~~~~~

Unlike ``numpy.memmap``, the file mapping is created and owned by the Rust
crate. Python receives array views backed by that mapped memory.

This means:

- the Rust `TrxFile` wrapper remains the owner of the underlying mapping
- Python arrays are non-owning views
- the arrays are marked read-only from Python

Parity-focused surface
~~~~~~~~~~~~~~~~~~~~~~

The public API favors predictable accessors that map cleanly to data exposed by
`trx-python`, even when the internal representations differ.

Current Scope
-------------

The package currently exposes:

- top-level loading through ``trxrs.load(path)``
- geometry access via ``positions()`` and ``offsets()``
- named access to ``dps``, ``dpv``, groups, and ``dpg``
- basic metadata such as ``header``, ``dtype``, ``nb_streamlines``, and
  ``nb_vertices``

Not yet included:

- write support
- non-TRX format conversion APIs
- direct Dipy or nibabel object conversion helpers

