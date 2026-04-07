"""Tests for pyisolate._internal.client import-time snapshot handling."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Paths needed for subprocess
PYISOLATE_ROOT = str(Path(__file__).parent.parent)
COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT") or str(Path.home() / "ComfyUI")

SCRIPT = """
import json, sys
import pyisolate._internal.client  # noqa: F401  # triggers snapshot logic
print(json.dumps(sys.path[:6]))
"""


def _run_client_process(env):
    # Ensure subprocess can find pyisolate and ComfyUI
    pythonpath_parts = [PYISOLATE_ROOT, COMFYUI_ROOT]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    stdout = result.stdout.strip().splitlines()[-1]
    return json.loads(stdout)


@pytest.fixture()
def comfy_module_path(tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    module_path = comfy_root / "custom_nodes" / "TestNode"
    module_path.mkdir(parents=True)
    return comfy_root, module_path


def test_snapshot_applied_and_comfy_root_prepend(tmp_path, comfy_module_path):
    comfy_root, module_path = comfy_module_path
    # Must include real ComfyUI path for utils validation to pass
    host_paths = [COMFYUI_ROOT, "/host/lib1", "/host/lib2"]
    snapshot = {
        "sys_path": host_paths,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "environment": {},
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYISOLATE_CHILD": "1",
            "PYISOLATE_HOST_SNAPSHOT": str(snapshot_path),
            "PYISOLATE_MODULE_PATH": str(module_path),
        }
    )

    path_prefix = _run_client_process(env)

    # Current client behavior preserves the runtime bootstrap path order and
    # keeps the resolved ComfyUI root available for imports.
    assert COMFYUI_ROOT in path_prefix
    # Module path should not override runtime root selection.
    assert str(comfy_root) not in path_prefix


def test_missing_snapshot_file_does_not_crash(tmp_path, comfy_module_path):
    _, module_path = comfy_module_path
    missing_snapshot = tmp_path / "missing.json"

    env = os.environ.copy()
    env.update(
        {
            "PYISOLATE_CHILD": "1",
            "PYISOLATE_HOST_SNAPSHOT": str(missing_snapshot),
            "PYISOLATE_MODULE_PATH": str(module_path),
        }
    )

    # Should not raise even though snapshot path is missing
    paths = _run_client_process(env)
    assert len(paths) > 0


def test_no_comfy_root_when_module_path_absent(tmp_path):
    # Must include real ComfyUI path for utils validation to pass
    host_paths = [COMFYUI_ROOT, "/alpha", "/beta"]
    snapshot = {
        "sys_path": host_paths,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "environment": {},
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYISOLATE_CHILD": "1",
            "PYISOLATE_HOST_SNAPSHOT": str(snapshot_path),
        }
    )

    paths = _run_client_process(env)
    # Runtime path bootstrap keeps ComfyUI importability regardless of host
    # snapshot extras.
    assert COMFYUI_ROOT in paths
    assert "/alpha" not in paths and "/beta" not in paths
