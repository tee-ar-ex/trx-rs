import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import trxrs

tmm = pytest.importorskip("trx.trx_file_memmap")


def _write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array).tofile(path)


def _write_fixture(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    header = {
        "VOXEL_TO_RASMM": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "DIMENSIONS": [20, 20, 20],
        "NB_STREAMLINES": 2,
        "NB_VERTICES": 5,
        "CUSTOM": {"name": "fixture"},
    }
    (dir_path / "header.json").write_text(json.dumps(header), encoding="utf-8")

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.array([0, 2, 5], dtype=np.uint32)
    weights = np.array([1.5, 2.5], dtype=np.float32)
    fa = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    color = np.array([[255, 128, 0]], dtype=np.uint8)
    members = np.array([1], dtype=np.uint32)

    _write_array(dir_path / "positions.3.float32", positions)
    _write_array(dir_path / "offsets.uint32", offsets)
    _write_array(dir_path / "dps" / "weights.float32", weights)
    _write_array(dir_path / "dpv" / "fa.float16", fa)
    _write_array(dir_path / "groups" / "bundle.uint32", members)
    _write_array(dir_path / "dpg" / "bundle" / "color.3.uint8", color)


@pytest.fixture
def trx_paths(tmp_path: Path) -> tuple[Path, Path]:
    dir_path = tmp_path / "fixture.trx"
    _write_fixture(dir_path)

    zip_path = tmp_path / "fixture_zip.trx"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for child in dir_path.rglob("*"):
            if child.is_file():
                zf.write(child, child.relative_to(dir_path))
    return dir_path, zip_path


@pytest.mark.parametrize("fixture_index", [0, 1])
def test_python_parity(trx_paths: tuple[Path, Path], fixture_index: int) -> None:
    path = trx_paths[fixture_index]

    rust_obj = trxrs.load(path)
    py_obj = tmm.load(str(path))

    assert len(rust_obj) == len(py_obj)
    assert rust_obj.nb_streamlines == len(py_obj.streamlines)
    assert rust_obj.nb_vertices == len(py_obj.streamlines._data)
    assert rust_obj.dtype == py_obj.streamlines._data.dtype.name

    np.testing.assert_array_equal(rust_obj.positions(), py_obj.streamlines._data)
    np.testing.assert_array_equal(rust_obj.offsets(), py_obj.streamlines._offsets)

    assert rust_obj.dps_keys() == sorted(py_obj.data_per_streamline.keys())
    assert rust_obj.dpv_keys() == sorted(py_obj.data_per_vertex.keys())
    assert rust_obj.group_keys() == sorted(py_obj.groups.keys())
    assert rust_obj.dpg_keys() == {
        key: sorted(value.keys()) for key, value in py_obj.data_per_group.items()
    }

    np.testing.assert_array_equal(
        rust_obj.get_dps("weights"),
        py_obj.data_per_streamline["weights"],
    )
    np.testing.assert_array_equal(
        rust_obj.get_dpv("fa"),
        py_obj.data_per_vertex["fa"]._data,
    )
    np.testing.assert_array_equal(
        rust_obj.get_group("bundle"),
        py_obj.groups["bundle"],
    )
    np.testing.assert_array_equal(
        rust_obj.get_dpg("bundle", "color"),
        py_obj.data_per_group["bundle"]["color"],
    )

    assert rust_obj.header["CUSTOM"] == {"name": "fixture"}
    py_obj.close()


def test_zero_copy_views(trx_paths: tuple[Path, Path]) -> None:
    rust_obj = trxrs.load(trx_paths[0])

    positions_a = rust_obj.positions()
    positions_b = rust_obj.positions()
    weights = rust_obj.get_dps("weights")

    assert np.shares_memory(positions_a, positions_b)
    assert not positions_a.flags["OWNDATA"]
    assert not positions_a.flags["WRITEABLE"]
    assert not weights.flags["OWNDATA"]
    assert not weights.flags["WRITEABLE"]


def test_missing_key_raises_key_error(trx_paths: tuple[Path, Path]) -> None:
    rust_obj = trxrs.load(trx_paths[0])

    with pytest.raises(KeyError):
        rust_obj.get_dps("missing")
