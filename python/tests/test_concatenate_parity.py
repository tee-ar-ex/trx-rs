from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import trxrs

tmm = pytest.importorskip("trx.trx_file_memmap")


def _run_rust_cli(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ["cargo", "run", "--quiet", "--bin", "trx", "--", *args],
        cwd=repo_root,
        check=True,
    )


def _run_python_cli(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "trx.cli", *args], check=True)


def _positions_and_lengths_trxrs(obj) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(obj.positions())
    offsets = np.asarray(obj.offsets(), dtype=np.int64)
    lengths = np.diff(np.concatenate([offsets, [positions.shape[0]]]))
    return positions, lengths


def _positions_and_lengths_py(obj) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(obj.streamlines._data)
    offsets = np.asarray(obj.streamlines._offsets, dtype=np.int64)
    lengths = np.diff(np.concatenate([offsets, [positions.shape[0]]]))
    return positions, lengths


def _assert_cli_outputs_match(rust_path: Path, py_path: Path) -> None:
    rust_obj = trxrs.load(rust_path)
    py_obj = tmm.load(str(py_path))
    try:
        rust_positions, rust_lengths = _positions_and_lengths_trxrs(rust_obj)
        py_positions, py_lengths = _positions_and_lengths_py(py_obj)
        np.testing.assert_allclose(rust_positions, py_positions, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(rust_lengths, py_lengths)
        assert rust_obj.header["DIMENSIONS"] == list(py_obj.header["DIMENSIONS"])
        assert rust_obj.group_keys() == sorted(py_obj.groups.keys())
        assert rust_obj.dpg_keys() == {}
    finally:
        py_obj.close()


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array).tofile(path)


def _write_trx_fixture(dir_path: Path, dimensions: list[int], dps_name: str | None) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    header = {
        "VOXEL_TO_RASMM": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "DIMENSIONS": dimensions,
        "NB_STREAMLINES": 1,
        "NB_VERTICES": 2,
    }
    (dir_path / "header.json").write_text(json.dumps(header), encoding="utf-8")
    _write_array(
        dir_path / "positions.3.float32",
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    )
    _write_array(dir_path / "offsets.uint32", np.array([0, 2], dtype=np.uint32))
    _write_array(dir_path / "groups" / "bundle.uint32", np.array([0], dtype=np.uint32))
    _write_array(
        dir_path / "dpg" / "bundle" / "color.3.uint8",
        np.array([[255, 0, 0]], dtype=np.uint8),
    )
    if dps_name is not None:
        _write_array(dir_path / "dps" / f"{dps_name}.float32", np.array([1.5], dtype=np.float32))


def test_concatenate_cli_matches_python_for_trx_inputs(
    repo_root: Path,
    gold_standard_dir: Path,
    tmp_path: Path,
) -> None:
    rust_out = tmp_path / "rust_concat.trx"
    py_out = tmp_path / "py_concat.trx"

    _run_rust_cli(
        repo_root,
        "concatenate",
        str(gold_standard_dir / "gs.trx"),
        str(gold_standard_dir / "gs.trx"),
        "--output",
        str(rust_out),
    )
    _run_python_cli(
        "concatenate",
        str(gold_standard_dir / "gs.trx"),
        str(gold_standard_dir / "gs.trx"),
        str(py_out),
    )

    _assert_cli_outputs_match(rust_out, py_out)


@pytest.mark.parametrize("source_name", ["gs.tck", "gs.vtk"])
def test_concatenate_cli_matches_python_for_mixed_inputs(
    repo_root: Path,
    gold_standard_dir: Path,
    tmp_path: Path,
    source_name: str,
) -> None:
    reference = gold_standard_dir / "gs.nii"
    rust_out = tmp_path / f"rust_{source_name}.trx"
    py_out = tmp_path / f"py_{source_name}.trx"

    _run_rust_cli(
        repo_root,
        "concatenate",
        str(gold_standard_dir / source_name),
        str(gold_standard_dir / "gs.trx"),
        "--output",
        str(rust_out),
        "--reference",
        str(reference),
        "--delete-dps",
        "--delete-dpv",
    )
    _run_python_cli(
        "concatenate",
        str(gold_standard_dir / source_name),
        str(gold_standard_dir / "gs.trx"),
        str(py_out),
        "--reference",
        str(reference),
        "--delete-dps",
        "--delete-dpv",
    )

    _assert_cli_outputs_match(rust_out, py_out)


def test_concatenate_cli_delete_dps_matches_python(repo_root: Path, tmp_path: Path) -> None:
    fixture_a = tmp_path / "a.trx"
    fixture_b = tmp_path / "b.trx"
    _write_trx_fixture(fixture_a, [10, 20, 30], "weights")
    _write_trx_fixture(fixture_b, [10, 20, 30], None)
    rust_out = tmp_path / "rust_delete_dps.trx"
    py_out = tmp_path / "py_delete_dps.trx"

    _run_rust_cli(
        repo_root,
        "concatenate",
        str(fixture_a),
        str(fixture_b),
        "--output",
        str(rust_out),
        "--delete-dps",
    )
    _run_python_cli(
        "concatenate",
        str(fixture_a),
        str(fixture_b),
        str(py_out),
        "--delete-dps",
    )

    rust_obj = trxrs.load(rust_out)
    py_obj = tmm.load(str(py_out))
    try:
        assert rust_obj.header["DIMENSIONS"] == [10, 20, 30]
        assert rust_obj.dps_keys() == []
        assert sorted(py_obj.data_per_streamline.keys()) == []
        assert rust_obj.group_keys() == sorted(py_obj.groups.keys())
        assert rust_obj.dpg_keys() == {}
    finally:
        py_obj.close()
