Quick Start
===========

Load A TRX File
---------------

.. code-block:: python

   import trxrs

   trx = trxrs.load("bundle.trx")

   print(trx.nb_streamlines)
   print(trx.nb_vertices)
   print(trx.dtype)

Read Geometry
-------------

.. code-block:: python

   positions = trx.positions()
   offsets = trx.offsets()

   print(positions.shape)
   print(offsets.shape)

`positions()` returns an ``(N, 3)`` NumPy array.

`offsets()` returns streamline start offsets in the Python-facing shape used for
parity with `trx-python`.

Read Named Arrays
-----------------

.. code-block:: python

   weights = trx.get_dps("weights")
   fa = trx.get_dpv("fa")
   members = trx.get_group("bundle")
   color = trx.get_dpg("bundle", "color")

All returned numeric payloads are intended to be read-only NumPy views backed
by the Rust-owned TRX data.

Inspect Available Keys
----------------------

.. code-block:: python

   print(trx.dps_keys())
   print(trx.dpv_keys())
   print(trx.group_keys())
   print(trx.dpg_keys())

