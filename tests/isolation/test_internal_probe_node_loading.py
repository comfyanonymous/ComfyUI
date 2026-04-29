from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import nodes
from tests.isolation.stage_internal_probe_node import (
    PROBE_NODE_NAME,
    stage_probe_node,
    staged_probe_node,
)


COMFYUI_ROOT = Path(__file__).resolve().parents[2]
ISOLATION_ROOT = COMFYUI_ROOT / "tests" / "isolation"
PROBE_SOURCE_ROOT = ISOLATION_ROOT / "internal_probe_node"
EXPECTED_NODE_IDS = [
    "InternalIsolationProbeAudio",
    "InternalIsolationProbeImage",
    "InternalIsolationProbeUI3D",
]

CLIENT_SCRIPT = """
import importlib.util
import json
import os
import sys

import pyisolate._internal.client  # noqa: F401  # triggers snapshot bootstrap

module_path = os.environ["PYISOLATE_MODULE_PATH"]
spec = importlib.util.spec_from_file_location(
    "internal_probe_node",
    os.path.join(module_path, "__init__.py"),
    submodule_search_locations=[module_path],
)
module = importlib.util.module_from_spec(spec)
assert spec is not None
assert spec.loader is not None
sys.modules["internal_probe_node"] = module
spec.loader.exec_module(module)
print(
    json.dumps(
        {
            "sys_path": list(sys.path),
            "module_path": module_path,
            "node_ids": sorted(module.NODE_CLASS_MAPPINGS.keys()),
        }
    )
)
"""


def _run_client_process(env: dict[str, str]) -> dict:
    pythonpath_parts = [str(COMFYUI_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", CLIENT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture()
def staged_probe_module(tmp_path: Path) -> tuple[Path, Path]:
    staged_comfy_root = tmp_path / "ComfyUI"
    module_path = staged_comfy_root / "custom_nodes" / "InternalIsolationProbeNode"
    shutil.copytree(PROBE_SOURCE_ROOT, module_path)
    return staged_comfy_root, module_path


@pytest.mark.asyncio
async def test_staged_probe_node_discovered(staged_probe_module: tuple[Path, Path]) -> None:
    _, module_path = staged_probe_module
    class_mappings_snapshot = dict(nodes.NODE_CLASS_MAPPINGS)
    display_name_snapshot = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
    loaded_module_dirs_snapshot = dict(nodes.LOADED_MODULE_DIRS)

    try:
        ignore = set(nodes.NODE_CLASS_MAPPINGS.keys())
        loaded = await nodes.load_custom_node(
            str(module_path), ignore=ignore, module_parent="custom_nodes"
        )

        assert loaded is True
        assert nodes.LOADED_MODULE_DIRS["InternalIsolationProbeNode"] == str(
            module_path.resolve()
        )

        for node_id in EXPECTED_NODE_IDS:
            assert node_id in nodes.NODE_CLASS_MAPPINGS
            node_cls = nodes.NODE_CLASS_MAPPINGS[node_id]
            assert (
                getattr(node_cls, "RELATIVE_PYTHON_MODULE", None)
                == "custom_nodes.InternalIsolationProbeNode"
            )
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(class_mappings_snapshot)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
        nodes.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_snapshot)
        nodes.LOADED_MODULE_DIRS.clear()
        nodes.LOADED_MODULE_DIRS.update(loaded_module_dirs_snapshot)


def test_staged_probe_node_module_path_is_valid_for_child_bootstrap(
    tmp_path: Path, staged_probe_module: tuple[Path, Path]
) -> None:
    staged_comfy_root, module_path = staged_probe_module
    snapshot = {
        "sys_path": [str(COMFYUI_ROOT), "/host/lib1", "/host/lib2"],
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

    payload = _run_client_process(env)

    assert payload["module_path"] == str(module_path)
    assert payload["node_ids"] == EXPECTED_NODE_IDS
    assert str(COMFYUI_ROOT) in payload["sys_path"]
    assert str(staged_comfy_root) not in payload["sys_path"]


def test_stage_probe_node_stages_only_under_explicit_root(tmp_path: Path) -> None:
    comfy_root = tmp_path / "sandbox-root"

    module_path = stage_probe_node(comfy_root)

    assert module_path == comfy_root / "custom_nodes" / PROBE_NODE_NAME
    assert module_path.is_dir()
    assert (module_path / "__init__.py").is_file()
    assert (module_path / "probe_nodes.py").is_file()
    assert (module_path / "pyproject.toml").is_file()


def test_staged_probe_node_context_cleans_up_temp_root() -> None:
    with staged_probe_node() as module_path:
        staging_root = module_path.parents[1]
        assert module_path.name == PROBE_NODE_NAME
        assert module_path.is_dir()
        assert staging_root.is_dir()

    assert not staging_root.exists()


def test_stage_script_requires_explicit_target_root() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(ISOLATION_ROOT / "stage_internal_probe_node.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--target-root" in result.stderr
