from __future__ import annotations

from pathlib import Path

import numpy as np

import trxrs


def test_tinytrack_load_exposes_groups_and_dpg(tinytrack_fixture: Path) -> None:
    tractogram = trxrs.load_tractogram(tinytrack_fixture)

    assert len(tractogram) == 9
    assert tractogram.nb_streamlines == 9
    assert tractogram.header["DIMENSIONS"] == [157, 189, 136]
    assert tractogram.header["VOXEL_TO_RASMM"] == [
        [-1.0, 0.0, 0.0, 78.0],
        [0.0, -1.0, 0.0, 76.0],
        [0.0, 0.0, 1.0, -50.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    groups = tractogram.group_keys()
    assert "Association_ArcuateFasciculusL" in groups
    assert "Association_ArcuateFasciculusR" in groups
    assert "Association_FrontalAslantTractL" in groups

    color = tractogram.get_dpg("Association_ArcuateFasciculusL", "color")
    assert color.shape == (1, 3)
    assert color.dtype.name == "uint8"
    np.testing.assert_array_equal(color[0], np.array([60, 160, 255], dtype=np.uint8))


def test_tinytrack_to_trx_roundtrip_preserves_groups_and_color(
    tinytrack_fixture: Path,
    tmp_path: Path,
) -> None:
    source = trxrs.load_tractogram(tinytrack_fixture)
    out_path = tmp_path / "tinytrack.trx"
    source.save(out_path)

    reloaded = trxrs.load(out_path)
    np.testing.assert_allclose(reloaded.positions(), source.positions())
    np.testing.assert_array_equal(reloaded.offsets(), source.offsets())
    assert reloaded.header["DIMENSIONS"] == source.header["DIMENSIONS"]
    assert reloaded.header["VOXEL_TO_RASMM"] == source.header["VOXEL_TO_RASMM"]
    assert reloaded.group_keys() == source.group_keys()
    np.testing.assert_array_equal(
        reloaded.get_group("Association_ArcuateFasciculusL"),
        source.get_group("Association_ArcuateFasciculusL"),
    )
    np.testing.assert_array_equal(
        reloaded.get_dpg("Association_ArcuateFasciculusL", "color"),
        source.get_dpg("Association_ArcuateFasciculusL", "color"),
    )
