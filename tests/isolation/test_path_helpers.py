from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from pyisolate.path_helpers import build_child_sys_path, serialize_host_snapshot


def test_serialize_host_snapshot_includes_expected_keys(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "snapshot.json"
    monkeypatch.setenv("EXTRA_FLAG", "1")
    snapshot = serialize_host_snapshot(output_path=output, extra_env_keys=["EXTRA_FLAG"])

    assert "sys_path" in snapshot
    assert "sys_executable" in snapshot
    assert "sys_prefix" in snapshot
    assert "environment" in snapshot
    assert output.exists()
    assert snapshot["environment"].get("EXTRA_FLAG") == "1"

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["sys_path"] == snapshot["sys_path"]


def test_build_child_sys_path_preserves_host_order() -> None:
    host_paths = ["/host/root", "/host/site-packages"]
    extra_paths = ["/node/.venv/lib/python3.12/site-packages"]
    result = build_child_sys_path(host_paths, extra_paths, preferred_root=None)
    assert result == host_paths + extra_paths


def test_build_child_sys_path_inserts_comfy_root_when_missing() -> None:
    host_paths = ["/host/site-packages"]
    comfy_root = os.environ.get("COMFYUI_ROOT") or str(Path.home() / "ComfyUI")
    extra_paths: list[str] = []
    result = build_child_sys_path(host_paths, extra_paths, preferred_root=comfy_root)
    assert result[0] == comfy_root
    assert result[1:] == host_paths


def test_build_child_sys_path_deduplicates_entries(tmp_path: Path) -> None:
    path_a = str(tmp_path / "a")
    path_b = str(tmp_path / "b")
    host_paths = [path_a, path_b]
    extra_paths = [path_a, path_b, str(tmp_path / "c")]
    result = build_child_sys_path(host_paths, extra_paths)
    assert result == [path_a, path_b, str(tmp_path / "c")]


def test_build_child_sys_path_skips_duplicate_comfy_root() -> None:
    comfy_root = os.environ.get("COMFYUI_ROOT") or str(Path.home() / "ComfyUI")
    host_paths = [comfy_root, "/host/other"]
    result = build_child_sys_path(host_paths, extra_paths=[], preferred_root=comfy_root)
    assert result == host_paths


def test_child_import_succeeds_after_path_unification(tmp_path: Path, monkeypatch) -> None:
    host_root = tmp_path / "host"
    utils_pkg = host_root / "utils"
    app_pkg = host_root / "app"
    utils_pkg.mkdir(parents=True)
    app_pkg.mkdir(parents=True)

    (utils_pkg / "__init__.py").write_text("from . import install_util\n", encoding="utf-8")
    (utils_pkg / "install_util.py").write_text("VALUE = 'hello'\n", encoding="utf-8")
    (app_pkg / "__init__.py").write_text("", encoding="utf-8")
    (app_pkg / "frontend_management.py").write_text(
        "from utils import install_util\nVALUE = install_util.VALUE\n",
        encoding="utf-8",
    )

    child_only = tmp_path / "child_only"
    child_only.mkdir()

    target_module = "app.frontend_management"
    for name in [n for n in list(sys.modules) if n.startswith("app") or n.startswith("utils")]:
        sys.modules.pop(name)

    monkeypatch.setattr(sys, "path", [str(child_only)])
    with pytest.raises(ModuleNotFoundError):
        __import__(target_module)

    for name in [n for n in list(sys.modules) if n.startswith("app") or n.startswith("utils")]:
        sys.modules.pop(name)

    unified = build_child_sys_path([], [], preferred_root=str(host_root))
    monkeypatch.setattr(sys, "path", unified)
    module = __import__(target_module, fromlist=["VALUE"])
    assert module.VALUE == "hello"
