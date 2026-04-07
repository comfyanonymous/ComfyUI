from __future__ import annotations

from tests.isolation.singleton_boundary_helpers import (
    capture_exact_small_proxy_relay,
    capture_model_management_exact_relay,
    capture_prompt_web_exact_relay,
)


def _transcripts_for(payload: dict[str, object], object_id: str, method: str) -> list[dict[str, object]]:
    return [
        entry
        for entry in payload["transcripts"]
        if entry["object_id"] == object_id and entry["method"] == method
    ]


def test_folder_paths_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    assert payload["forbidden_matches"] == []
    assert payload["models_dir"] == "/sandbox/models"
    assert payload["folder_path"] == "/sandbox/input/demo.png"

    models_dir_calls = _transcripts_for(payload, "FolderPathsProxy", "rpc_get_models_dir")
    annotated_calls = _transcripts_for(payload, "FolderPathsProxy", "rpc_get_annotated_filepath")

    assert models_dir_calls
    assert annotated_calls
    assert all(entry["phase"] != "child_call" or entry["method"] != "rpc_snapshot" for entry in payload["transcripts"])


def test_progress_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    progress_calls = _transcripts_for(payload, "ProgressProxy", "rpc_set_progress")

    assert progress_calls
    host_targets = [entry["target"] for entry in progress_calls if entry["phase"] == "host_invocation"]
    assert host_targets == ["comfy_execution.progress.get_progress_state().update_progress"]
    result_entries = [entry for entry in progress_calls if entry["phase"] == "result"]
    assert result_entries == [{"phase": "result", "object_id": "ProgressProxy", "method": "rpc_set_progress", "result": None}]


def test_utils_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    utils_calls = _transcripts_for(payload, "UtilsProxy", "progress_bar_hook")

    assert utils_calls
    host_targets = [entry["target"] for entry in utils_calls if entry["phase"] == "host_invocation"]
    assert host_targets == ["comfy.utils.PROGRESS_BAR_HOOK"]
    result_entries = [entry for entry in utils_calls if entry["phase"] == "result"]
    assert result_entries
    assert result_entries[0]["result"]["value"] == 2
    assert result_entries[0]["result"]["total"] == 5


def test_helper_proxy_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    helper_calls = _transcripts_for(payload, "HelperProxiesService", "rpc_restore_input_types")

    assert helper_calls
    host_targets = [entry["target"] for entry in helper_calls if entry["phase"] == "host_invocation"]
    assert host_targets == ["comfy.isolation.proxies.helper_proxies.restore_input_types"]
    assert payload["restored_any_type"] == "*"


def test_model_management_exact_relay() -> None:
    payload = capture_model_management_exact_relay()

    model_calls = _transcripts_for(payload, "ModelManagementProxy", "get_torch_device")
    model_calls += _transcripts_for(payload, "ModelManagementProxy", "get_torch_device_name")
    model_calls += _transcripts_for(payload, "ModelManagementProxy", "get_free_memory")

    assert payload["forbidden_matches"] == []
    assert model_calls
    host_targets = [
        entry["target"]
        for entry in payload["transcripts"]
        if entry["phase"] == "host_invocation"
    ]
    assert host_targets == [
        "comfy.model_management.get_torch_device",
        "comfy.model_management.get_torch_device_name",
        "comfy.model_management.get_free_memory",
    ]


def test_model_management_capability_preserved() -> None:
    payload = capture_model_management_exact_relay()

    assert payload["device"] == "cpu"
    assert payload["device_type"] == "cpu"
    assert payload["device_name"] == "cpu"
    assert payload["free_memory"] == 34359738368


def test_prompt_server_exact_relay() -> None:
    payload = capture_prompt_web_exact_relay()

    prompt_calls = _transcripts_for(payload, "PromptServerService", "ui_send_progress_text")
    prompt_calls += _transcripts_for(payload, "PromptServerService", "register_route_rpc")

    assert payload["forbidden_matches"] == []
    assert prompt_calls
    host_targets = [
        entry["target"]
        for entry in payload["transcripts"]
        if entry["object_id"] == "PromptServerService" and entry["phase"] == "host_invocation"
    ]
    assert host_targets == [
        "server.PromptServer.instance.send_progress_text",
        "server.PromptServer.instance.routes.add_route",
    ]


def test_web_directory_exact_relay() -> None:
    payload = capture_prompt_web_exact_relay()

    web_calls = _transcripts_for(payload, "WebDirectoryProxy", "get_web_file")

    assert web_calls
    host_targets = [entry["target"] for entry in web_calls if entry["phase"] == "host_invocation"]
    assert host_targets == ["comfy.isolation.proxies.web_directory_proxy.WebDirectoryProxy.get_web_file"]
    assert payload["web_file"]["content_type"] == "application/javascript"
    assert payload["web_file"]["content"] == "console.log('deo');"
