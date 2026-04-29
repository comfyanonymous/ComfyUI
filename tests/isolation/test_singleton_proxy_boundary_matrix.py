from __future__ import annotations

import json

from tests.isolation.singleton_boundary_helpers import (
    capture_minimal_sealed_worker_imports,
    capture_sealed_singleton_imports,
)


def test_minimal_sealed_worker_forbidden_imports() -> None:
    payload = capture_minimal_sealed_worker_imports()

    assert payload["mode"] == "minimal_sealed_worker"
    assert payload["runtime_probe_function"] == "inspect"
    assert payload["forbidden_matches"] == []


def test_torch_share_subset_scope() -> None:
    minimal = capture_minimal_sealed_worker_imports()

    allowed_torch_share_only = {
        "torch",
        "folder_paths",
        "comfy.utils",
        "comfy.model_management",
        "main",
        "comfy.isolation.extension_wrapper",
    }

    assert minimal["forbidden_matches"] == []
    assert all(
        module_name not in minimal["modules"] for module_name in sorted(allowed_torch_share_only)
    )


def test_capture_payload_is_json_serializable() -> None:
    payload = capture_minimal_sealed_worker_imports()

    encoded = json.dumps(payload, sort_keys=True)

    assert "\"minimal_sealed_worker\"" in encoded


def test_folder_paths_child_safe() -> None:
    payload = capture_sealed_singleton_imports()

    assert payload["mode"] == "sealed_singletons"
    assert payload["folder_path"] == "/sandbox/input/demo.png"
    assert payload["temp_dir"] == "/sandbox/temp"
    assert payload["models_dir"] == "/sandbox/models"
    assert payload["forbidden_matches"] == []


def test_utils_child_safe() -> None:
    payload = capture_sealed_singleton_imports()

    progress_calls = [
        call
        for call in payload["rpc_calls"]
        if call["object_id"] == "UtilsProxy" and call["method"] == "progress_bar_hook"
    ]

    assert progress_calls
    assert payload["forbidden_matches"] == []


def test_progress_child_safe() -> None:
    payload = capture_sealed_singleton_imports()

    progress_calls = [
        call
        for call in payload["rpc_calls"]
        if call["object_id"] == "ProgressProxy" and call["method"] == "rpc_set_progress"
    ]

    assert progress_calls
    assert payload["forbidden_matches"] == []
