import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
pyisolate_root = repo_root.parent / "pyisolate"
if pyisolate_root.exists():
    sys.path.insert(0, str(pyisolate_root))

from comfy.isolation.adapter import ComfyUIAdapter
from pyisolate._internal.sandbox import build_bwrap_command
from pyisolate._internal.sandbox_detect import RestrictionModel
from pyisolate._internal.serialization_registry import SerializerRegistry


def test_identifier():
    adapter = ComfyUIAdapter()
    assert adapter.identifier == "comfyui"


def test_get_path_config_valid():
    adapter = ComfyUIAdapter()
    path = os.path.join("/opt", "ComfyUI", "custom_nodes", "demo")
    cfg = adapter.get_path_config(path)
    assert cfg is not None
    assert cfg["preferred_root"].endswith("ComfyUI")
    assert "custom_nodes" in cfg["additional_paths"][0]


def test_get_path_config_invalid():
    adapter = ComfyUIAdapter()
    assert adapter.get_path_config("/random/path") is None


def test_provide_rpc_services():
    adapter = ComfyUIAdapter()
    services = adapter.provide_rpc_services()
    names = {s.__name__ for s in services}
    assert "PromptServerService" in names
    assert "FolderPathsProxy" in names


def test_register_serializers():
    adapter = ComfyUIAdapter()
    registry = SerializerRegistry.get_instance()
    registry.clear()

    adapter.register_serializers(registry)
    assert registry.has_handler("ModelPatcher")
    assert registry.has_handler("CLIP")
    assert registry.has_handler("VAE")

    registry.clear()


def test_child_temp_directory_fence_uses_private_tmp(tmp_path):
    adapter = ComfyUIAdapter()
    child_script = textwrap.dedent(
        """
        from pathlib import Path

        child_temp = Path("/tmp/comfyui_temp")
        child_temp.mkdir(parents=True, exist_ok=True)
        scratch = child_temp / "child_only.txt"
        scratch.write_text("child-only", encoding="utf-8")
        print(f"CHILD_TEMP={child_temp}")
        print(f"CHILD_FILE={scratch}")
        """
    )
    fake_folder_paths = types.SimpleNamespace(
        temp_directory="/host/tmp/should_not_survive",
        folder_names_and_paths={},
        extension_mimetypes_cache={},
        filename_list_cache={},
    )

    class FolderPathsProxy:
        def get_temp_directory(self):
            return "/host/tmp/should_not_survive"

    original_folder_paths = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = fake_folder_paths
    try:
        os.environ["PYISOLATE_CHILD"] = "1"
        adapter.handle_api_registration(FolderPathsProxy, rpc=None)
    finally:
        os.environ.pop("PYISOLATE_CHILD", None)
        if original_folder_paths is not None:
            sys.modules["folder_paths"] = original_folder_paths
        else:
            sys.modules.pop("folder_paths", None)

    import tempfile as _tf
    expected_temp = os.path.join(_tf.gettempdir(), "comfyui_temp")
    assert fake_folder_paths.temp_directory == expected_temp

    host_child_file = Path(expected_temp) / "child_only.txt"
    if host_child_file.exists():
        host_child_file.unlink()

    cmd = build_bwrap_command(
        python_exe=sys.executable,
        module_path=str(repo_root / "custom_nodes" / "ComfyUI-IsolationToolkit"),
        venv_path=str(repo_root / ".venv"),
        uds_address=str(tmp_path / "adapter.sock"),
        allow_gpu=False,
        restriction_model=RestrictionModel.NONE,
        sandbox_config={"writable_paths": ["/dev/shm"], "readonly_paths": [], "network": False},
        adapter=adapter,
    )
    assert "--tmpfs" in cmd and "/tmp" in cmd
    assert ["--bind", "/tmp", "/tmp"] not in [cmd[i : i + 3] for i in range(len(cmd) - 2)]

    command_tail = cmd[-3:]
    assert command_tail[1:] == ["-m", "pyisolate._internal.uds_client"]
    cmd = cmd[:-3] + [sys.executable, "-c", child_script]

    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "CHILD_TEMP=/tmp/comfyui_temp" in completed.stdout
    assert not host_child_file.exists(), "Child temp file leaked into host /tmp"
