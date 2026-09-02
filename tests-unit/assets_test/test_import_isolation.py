import subprocess
import sys
from pathlib import Path

import pytest

from app.assets import manager
from utils.install_util import get_missing_requirements_message


def test_no_assets_imports_without_database_dependencies() -> None:
    code = """
import importlib.abc
import sys
from aiohttp import web

class MissingDatabaseDependencyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"sqlalchemy", "alembic"}:
            raise ImportError(f"blocked dependency: {fullname}")
        return None

sys.meta_path.insert(0, MissingDatabaseDependencyFinder())
import app.assets.manager as manager

assert manager._IMPORT_ERROR is not None
manager.NoAssets(manager.args).register_routes(web.Application(), None)
manager.args.enable_assets = False
assert isinstance(manager.default_asset_manager(), manager.NoAssets)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("import_error", "database_dependencies_available"),
    [(ImportError("sqlalchemy"), True), (None, False)],
)
def test_enabled_assets_exits_without_database_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    import_error: ImportError | None,
    database_dependencies_available: bool,
) -> None:
    monkeypatch.setattr(manager.args, "enable_assets", True)
    monkeypatch.setattr(manager, "_IMPORT_ERROR", import_error)
    monkeypatch.setattr(
        manager,
        "dependencies_available",
        lambda: database_dependencies_available,
        raising=False,
    )

    with pytest.raises(SystemExit):
        manager.default_asset_manager()

    assert "--enable-assets" in caplog.text
    assert get_missing_requirements_message() in caplog.text
    if import_error is not None:
        assert str(import_error) in caplog.text


def test_disabled_assets_do_not_consult_database_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_dependencies() -> bool:
        raise AssertionError("disabled assets must not check database dependencies")

    monkeypatch.setattr(manager.args, "enable_assets", False)
    monkeypatch.setattr(
        manager, "dependencies_available", unavailable_dependencies, raising=False
    )

    assert isinstance(manager.default_asset_manager(), manager.NoAssets)


def test_enabled_asset_hashing_exits_without_blake3(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(manager.args, "enable_assets", True)
    monkeypatch.setattr(manager.args, "enable_asset_hashing", True)
    monkeypatch.setattr(manager, "_IMPORT_ERROR", None)
    monkeypatch.setattr(manager, "dependencies_available", lambda: True)
    monkeypatch.setitem(sys.modules, "blake3", None)

    with pytest.raises(SystemExit):
        manager.default_asset_manager()

    assert "blake3" in caplog.text
    assert "--enable-asset-hashing" in caplog.text


def test_enabled_assets_without_hashing_do_not_consult_blake3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager.args, "enable_assets", True)
    monkeypatch.setattr(manager.args, "enable_asset_hashing", False)
    monkeypatch.setattr(manager, "_IMPORT_ERROR", None)
    monkeypatch.setattr(manager, "dependencies_available", lambda: True)
    monkeypatch.setitem(sys.modules, "blake3", None)

    assert isinstance(manager.default_asset_manager(), manager.AssetsEnabled)
