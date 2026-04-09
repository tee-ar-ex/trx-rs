import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def tinytrack_fixture(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "tinytrack_small.tt.gz"


@pytest.fixture(scope="session")
def gold_standard_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in [
        repo_root / "target" / "test_data" / "gold_standard",
        repo_root / "target" / "test_data",
    ]:
        if (candidate / "gs.trx").exists():
            return candidate

    fetcher = pytest.importorskip("trx.fetcher")
    trx_home = tmp_path_factory.mktemp("trx-home")
    os.environ["TRX_HOME"] = str(trx_home)
    fetcher.fetch_data(fetcher.get_testing_files_dict(), keys=["gold_standard.zip"])
    return Path(fetcher.get_home()) / "gold_standard"
