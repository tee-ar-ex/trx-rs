API Reference
=============

Top-level Entry Point
---------------------

.. code-block:: python

   import trxrs
   trx = trxrs.load(path)

``load(path)``
~~~~~~~~~~~~~~

Load a TRX file or TRX directory and return a ``TrxFile`` instance.

`path` may point to:

- a ``.trx`` zip archive
- a TRX directory layout

`TrxFile`
---------

Metadata
~~~~~~~~

``nb_streamlines``
  Number of streamlines.

``nb_vertices``
  Number of vertices across all streamlines.

``dtype``
  Positions dtype as a string such as ``"float16"``, ``"float32"``, or
  ``"float64"``.

``header``
  Header metadata as a Python dictionary.

Geometry
~~~~~~~~

``positions()``
  Return positions as an ``(N, 3)`` NumPy array.

``offsets()``
  Return streamline offsets in the Python-facing parity shape.

``streamline(index)``
  Return one streamline as an ``(M, 3)`` NumPy view.

Named Arrays
~~~~~~~~~~~~

``get_dps(name)``
  Return a data-per-streamline array.

``get_dpv(name)``
  Return a data-per-vertex payload array.

``get_group(name)``
  Return the streamline indices belonging to a named group.

``get_dpg(group, name)``
  Return a named data-per-group array for a specific group.

Key Discovery
~~~~~~~~~~~~~

``dps_keys()``
  Return known DPS field names.

``dpv_keys()``
  Return known DPV field names.

``group_keys()``
  Return known group names.

``dpg_keys()``
  Return a mapping of group name to available DPG field names.

Compatibility Attributes
~~~~~~~~~~~~~~~~~~~~~~~~

For easier interoperability with parity tests and migration code, the package
also exposes dictionary-style compatibility properties:

- ``data_per_streamline``
- ``data_per_vertex``
- ``groups``
- ``data_per_group``

These are convenience views over the same underlying TRX payloads.

