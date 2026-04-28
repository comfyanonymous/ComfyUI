from __future__ import annotations

import importlib.util
import json
from pathlib import Path


COMFYUI_ROOT = Path(__file__).resolve().parents[2]
ISOLATION_ROOT = COMFYUI_ROOT / "tests" / "isolation"
PROBE_ROOT = ISOLATION_ROOT / "internal_probe_node"
WORKFLOW_ROOT = ISOLATION_ROOT / "workflows"
TOOLKIT_ROOT = COMFYUI_ROOT / "custom_nodes" / "ComfyUI-IsolationToolkit"

EXPECTED_PROBE_FILES = {
    "__init__.py",
    "probe_nodes.py",
}
EXPECTED_WORKFLOWS = {
    "internal_probe_preview_image_audio.json",
    "internal_probe_ui3d.json",
}
BANNED_REFERENCES = (
    "ComfyUI-IsolationToolkit",
    "toolkit_smoke_playlist",
    "run_isolation_toolkit_smoke.sh",
)


def _text_assets() -> list[Path]:
    return sorted(list(PROBE_ROOT.rglob("*.py")) + list(WORKFLOW_ROOT.glob("internal_probe_*.json")))


def _load_probe_package():
    spec = importlib.util.spec_from_file_location(
        "internal_probe_node",
        PROBE_ROOT / "__init__.py",
        submodule_search_locations=[str(PROBE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inventory_is_minimal_and_isolation_owned():
    assert PROBE_ROOT.is_dir()
    assert WORKFLOW_ROOT.is_dir()
    assert PROBE_ROOT.is_relative_to(ISOLATION_ROOT)
    assert WORKFLOW_ROOT.is_relative_to(ISOLATION_ROOT)
    assert not PROBE_ROOT.is_relative_to(TOOLKIT_ROOT)

    probe_files = {path.name for path in PROBE_ROOT.iterdir() if path.is_file()}
    workflow_files = {path.name for path in WORKFLOW_ROOT.glob("internal_probe_*.json")}

    assert probe_files == EXPECTED_PROBE_FILES
    assert workflow_files == EXPECTED_WORKFLOWS

    module = _load_probe_package()
    mappings = module.NODE_CLASS_MAPPINGS

    assert sorted(mappings.keys()) == [
        "InternalIsolationProbeAudio",
        "InternalIsolationProbeImage",
        "InternalIsolationProbeUI3D",
    ]

    preview_workflow = json.loads(
        (WORKFLOW_ROOT / "internal_probe_preview_image_audio.json").read_text(
            encoding="utf-8"
        )
    )
    ui3d_workflow = json.loads(
        (WORKFLOW_ROOT / "internal_probe_ui3d.json").read_text(encoding="utf-8")
    )

    assert [preview_workflow[node_id]["class_type"] for node_id in ("1", "2")] == [
        "InternalIsolationProbeImage",
        "InternalIsolationProbeAudio",
    ]
    assert [ui3d_workflow[node_id]["class_type"] for node_id in ("1",)] == [
        "InternalIsolationProbeUI3D",
    ]


def test_zero_toolkit_references_in_probe_assets():
    for asset in _text_assets():
        content = asset.read_text(encoding="utf-8")
        for banned in BANNED_REFERENCES:
            assert banned not in content, f"{asset} unexpectedly references {banned}"


def test_replacement_contract_has_zero_toolkit_references():
    contract_assets = [
        *(PROBE_ROOT.rglob("*.py")),
        *WORKFLOW_ROOT.glob("internal_probe_*.json"),
        ISOLATION_ROOT / "stage_internal_probe_node.py",
        ISOLATION_ROOT / "internal_probe_host_policy.toml",
    ]

    for asset in sorted(contract_assets):
        assert asset.exists(), f"Missing replacement-contract asset: {asset}"
        content = asset.read_text(encoding="utf-8")
        for banned in BANNED_REFERENCES:
            assert banned not in content, f"{asset} unexpectedly references {banned}"
