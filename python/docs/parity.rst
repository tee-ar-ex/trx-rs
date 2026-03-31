Parity With `trx-python`
========================

Why Parity Matters
------------------

The current Python bindings exist partly to answer a practical question:

Can the Rust implementation behave closely enough to `trx-python` that the two
can be compared directly in Python CI?

The answer is yes, with a small amount of adaptation where the internal object
models differ.

Parity Philosophy
-----------------

The goal is behavioral parity for the public data access surface, not object-for-
object emulation of every nibabel or Dipy wrapper type.

This means the project aims to match:

- file loading behavior
- positions dtype and values
- offsets semantics exposed to Python
- key sets for DPS, DPV, groups, and DPG
- numeric payload contents

Some internal representations will remain different:

- `trx-python` stores some vertex-associated payloads in `ArraySequence`-style
  containers
- `trxrs` returns direct NumPy arrays for payload accessors
- the Rust crate internally keeps the sentinel offset used by the TRX core
  representation, while the Python-facing accessor follows `trx-python`
  conventions

Parity Testing Strategy
-----------------------

The Python test suite in ``python/tests/test_parity.py``:

- creates a deterministic local TRX fixture
- loads it with both `trxrs` and `trx-python`
- compares geometry, metadata keys, and representative payload arrays
- checks zero-copy expectations on the `trxrs` side

This gives a compact, reproducible parity check without depending on remote test
data.
