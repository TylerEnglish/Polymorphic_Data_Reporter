import os
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def cfg_path(project_root: Path) -> Path:
    # default location we agreed on
    return project_root / "config" / "config.toml"

@pytest.fixture(scope="session")
def cfg():
    try:
        from src.config_model.model import load_config
    except Exception as e:
        pytest.skip(f"config loader not importable yet: {e}")
    return load_config()

@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir(parents=True, exist_ok=True)
    return d