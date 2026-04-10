from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import trxrs

dipy = pytest.importorskip("dipy")
trx_workflows = pytest.importorskip("trx.workflows")
trx_io = pytest.importorskip("trx.io")
tmm = pytest.importorskip("trx.trx_file_memmap")


def _positions_and_offsets(obj) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(obj, "positions"):
        positions = np.asarray(obj.positions())
        offsets = np.asarray(obj.offsets())
    else:
        positions = np.asarray(obj.streamlines._data)
        offsets = np.asarray(obj.streamlines._offsets)
    return positions, offsets


def _streamline_lengths(offsets: np.ndarray, nb_vertices: int) -> np.ndarray:
    full = np.concatenate([np.asarray(offsets, dtype=np.int64), [nb_vertices]])
    return np.diff(full)


def _assert_semantic_match(rust_obj, py_obj) -> None:
    rust_positions, rust_offsets = _positions_and_offsets(rust_obj)
    py_positions, py_offsets = _positions_and_offsets(py_obj)

    assert len(rust_obj) == len(py_obj.streamlines if hasattr(py_obj, "streamlines") else py_obj)
    assert rust_positions.shape == py_positions.shape
    np.testing.assert_allclose(rust_positions, py_positions, rtol=1e-5, atol=1e-5)

    rust_lengths = _streamline_lengths(rust_offsets, rust_positions.shape[0])
    py_lengths = _streamline_lengths(py_offsets, py_positions.shape[0])
    np.testing.assert_array_equal(rust_lengths, py_lengths)


def _load_python(path: Path, reference: Path | None):
    obj = trx_io.load(str(path), None if reference is None else str(reference))
    return obj


def _close_python(obj) -> None:
    if hasattr(obj, "close"):
        obj.close()


def _assert_header_matches_reference(header: dict, reference: Path) -> None:
    img = nib.load(str(reference))
    np.testing.assert_allclose(np.asarray(header["VOXEL_TO_RASMM"]), img.affine)
    np.testing.assert_array_equal(np.asarray(header["DIMENSIONS"]), img.shape[:3])


def test_load_tractogram_uses_nifti_reference(gold_standard_dir: Path) -> None:
    reference = gold_standard_dir / "gs.nii"
    tractogram = trxrs.load_tractogram(gold_standard_dir / "gs.tck", reference=reference)
    _assert_header_matches_reference(tractogram.header, reference)


@pytest.mark.parametrize(
    ("source_name", "target_name", "needs_reference"),
    [
        ("gs.trx", "rust.tck", False),
        ("gs.trx", "rust.vtk", False),
        ("gs.tck", "rust.trx", True),
        ("gs.vtk", "rust.trx", True),
    ],
)
def test_converter_matches_trx_python(
    gold_standard_dir: Path,
    tmp_path: Path,
    source_name: str,
    target_name: str,
    needs_reference: bool,
) -> None:
    source = gold_standard_dir / source_name
    reference = gold_standard_dir / "gs.nii" if needs_reference else None
    compare_reference = gold_standard_dir / "gs.nii" if Path(target_name).suffix != ".trx" else reference
    rust_out = tmp_path / target_name
    py_out = tmp_path / f"py_{target_name}"

    trxrs.convert(source, rust_out, reference=reference)
    trx_workflows.convert_tractogram(
        str(source),
        str(py_out),
        None if reference is None else str(reference),
    )

    rust_obj = (
        trxrs.load(rust_out)
        if rust_out.suffix == ".trx"
        else trxrs.load_tractogram(rust_out, reference=compare_reference)
    )
    py_obj = _load_python(py_out, compare_reference)
    try:
        _assert_semantic_match(rust_obj, py_obj)
        if rust_out.suffix == ".trx" and reference is not None:
            _assert_header_matches_reference(rust_obj.header, reference)
    finally:
        _close_python(py_obj)


def test_trx_tck_trx_roundtrip_matches_trx_python(
    gold_standard_dir: Path,
    tmp_path: Path,
) -> None:
    reference = gold_standard_dir / "gs.nii"
    rust_tck = tmp_path / "rust_roundtrip.tck"
    rust_trx = tmp_path / "rust_roundtrip.trx"
    py_tck = tmp_path / "py_roundtrip.tck"
    py_trx = tmp_path / "py_roundtrip.trx"

    trxrs.convert(gold_standard_dir / "gs.trx", rust_tck)
    trxrs.convert(rust_tck, rust_trx, reference=reference)
    trx_workflows.convert_tractogram(str(gold_standard_dir / "gs.trx"), str(py_tck), None)
    trx_workflows.convert_tractogram(str(py_tck), str(py_trx), str(reference))

    rust_obj = trxrs.load(rust_trx)
    py_obj = tmm.load(str(py_trx))
    try:
        _assert_semantic_match(rust_obj, py_obj)
        _assert_header_matches_reference(rust_obj.header, reference)
    finally:
        py_obj.close()


@pytest.mark.parametrize("positions_dtype", ["float16", "float64"])
def test_trx_output_dtype_matches_trx_python(
    gold_standard_dir: Path,
    tmp_path: Path,
    positions_dtype: str,
) -> None:
    reference = gold_standard_dir / "gs.nii"
    rust_out = tmp_path / f"rust_{positions_dtype}.trx"
    py_out = tmp_path / f"py_{positions_dtype}.trx"

    trxrs.convert(
        gold_standard_dir / "gs.tck",
        rust_out,
        reference=reference,
        positions_dtype=positions_dtype,
    )
    trx_workflows.convert_tractogram(
        str(gold_standard_dir / "gs.tck"),
        str(py_out),
        str(reference),
        pos_dtype=positions_dtype,
    )

    rust_obj = trxrs.load(rust_out)
    py_obj = tmm.load(str(py_out))
    try:
        assert rust_obj.dtype == positions_dtype
        assert py_obj.streamlines._data.dtype.name == positions_dtype
        _assert_semantic_match(rust_obj, py_obj)
        _assert_header_matches_reference(rust_obj.header, reference)
    finally:
        py_obj.close()
