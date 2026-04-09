Installation
============

Requirements
------------

For local development you will typically want:

- Python 3.11 or newer
- `numpy`
- `maturin`
- a working Rust toolchain

Local Development Install
-------------------------

From the repository root:

.. code-block:: bash

   cd python
   maturin develop

If you are working inside a Mamba or Conda environment:

.. code-block:: bash

   mamba activate trx
   python -m pip install maturin numpy
   cd python
   maturin develop
