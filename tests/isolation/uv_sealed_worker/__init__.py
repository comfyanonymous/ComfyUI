# pylint: disable=import-outside-toplevel,import-error
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _artifact_dir() -> Path | None:
    raw = os.environ.get("PYISOLATE_ARTIFACT_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_artifact(name: str, content: str) -> None:
    artifact_dir = _artifact_dir()
    if artifact_dir is None:
        return
    (artifact_dir / name).write_text(content, encoding="utf-8")


def _contains_tensor_marker(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("__type__") == "TensorValue":
            return True
        return any(_contains_tensor_marker(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor_marker(v) for v in value)
    return False


class InspectRuntimeNode:
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "BOOLEAN",
        "BOOLEAN",
        "STRING",
        "STRING",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "path_dump",
        "boltons_origin",
        "saw_comfy_root",
        "imported_comfy_wrapper",
        "comfy_module_dump",
        "report",
        "saw_user_site",
    )
    FUNCTION = "inspect"
    CATEGORY = "PyIsolated/SealedWorker"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {}}

    def inspect(self) -> tuple[str, str, bool, bool, str, str, bool]:
        import boltons

        path_dump = "\n".join(sys.path)
        comfy_root = "/home/johnj/ComfyUI"
        saw_comfy_root = any(
            entry == comfy_root
            or entry.startswith(f"{comfy_root}/comfy")
            or entry.startswith(f"{comfy_root}/.venv")
            for entry in sys.path
        )
        imported_comfy_wrapper = "comfy.isolation.extension_wrapper" in sys.modules
        comfy_module_dump = "\n".join(
            sorted(name for name in sys.modules if name.startswith("comfy"))
        )
        saw_user_site = any("/.local/lib/" in entry for entry in sys.path)
        boltons_origin = getattr(boltons, "__file__", "<missing>")

        report_lines = [
            "UV sealed worker runtime probe",
            f"boltons_origin={boltons_origin}",
            f"saw_comfy_root={saw_comfy_root}",
            f"imported_comfy_wrapper={imported_comfy_wrapper}",
            f"saw_user_site={saw_user_site}",
        ]
        report = "\n".join(report_lines)

        _write_artifact("child_bootstrap_paths.txt", path_dump)
        _write_artifact("child_import_trace.txt", comfy_module_dump)
        _write_artifact("child_dependency_dump.txt", boltons_origin)
        logger.warning("][ UV sealed runtime probe executed")
        logger.warning("][ boltons origin: %s", boltons_origin)

        return (
            path_dump,
            boltons_origin,
            saw_comfy_root,
            imported_comfy_wrapper,
            comfy_module_dump,
            report,
            saw_user_site,
        )


class BoltonsSlugifyNode:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("slug", "boltons_origin")
    FUNCTION = "slugify_text"
    CATEGORY = "PyIsolated/SealedWorker"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {"text": ("STRING", {"default": "Sealed Worker Rocks"})}}

    def slugify_text(self, text: str) -> tuple[str, str]:
        import boltons
        from boltons.strutils import slugify

        slug = slugify(text)
        origin = getattr(boltons, "__file__", "<missing>")
        logger.warning("][ boltons slugify: %r -> %r", text, slug)
        return slug, origin


class FilesystemBarrierNode:
    RETURN_TYPES = ("STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = (
        "report",
        "outside_blocked",
        "module_mutation_blocked",
        "artifact_write_ok",
    )
    FUNCTION = "probe"
    CATEGORY = "PyIsolated/SealedWorker"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {}}

    def probe(self) -> tuple[str, bool, bool, bool]:
        artifact_dir = _artifact_dir()
        artifact_write_ok = False
        if artifact_dir is not None:
            probe_path = artifact_dir / "filesystem_barrier_probe.txt"
            probe_path.write_text("artifact write ok\n", encoding="utf-8")
            artifact_write_ok = probe_path.exists()

        module_target = Path(__file__).with_name(
            "mutated_from_child_should_not_exist.txt"
        )
        module_mutation_blocked = False
        try:
            module_target.write_text("mutation should fail\n", encoding="utf-8")
        except Exception:
            module_mutation_blocked = True
        else:
            module_target.unlink(missing_ok=True)

        outside_target = Path("/home/johnj/mysolate/.uv_sealed_worker_escape_probe")
        outside_blocked = False
        try:
            outside_target.write_text("escape should fail\n", encoding="utf-8")
        except Exception:
            outside_blocked = True
        else:
            outside_target.unlink(missing_ok=True)

        report_lines = [
            "UV sealed worker filesystem barrier probe",
            f"artifact_write_ok={artifact_write_ok}",
            f"module_mutation_blocked={module_mutation_blocked}",
            f"outside_blocked={outside_blocked}",
        ]
        report = "\n".join(report_lines)
        _write_artifact("filesystem_barrier_report.txt", report)
        logger.warning("][ filesystem barrier probe executed")
        return report, outside_blocked, module_mutation_blocked, artifact_write_ok


class EchoTensorNode:
    RETURN_TYPES = ("TENSOR", "BOOLEAN")
    RETURN_NAMES = ("tensor", "saw_json_tensor")
    FUNCTION = "echo"
    CATEGORY = "PyIsolated/SealedWorker"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {"tensor": ("TENSOR",)}}

    def echo(self, tensor: Any) -> tuple[Any, bool]:
        saw_json_tensor = _contains_tensor_marker(tensor)
        logger.warning("][ tensor echo json_marker=%s", saw_json_tensor)
        return tensor, saw_json_tensor


class EchoLatentNode:
    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("latent", "saw_json_tensor")
    FUNCTION = "echo_latent"
    CATEGORY = "PyIsolated/SealedWorker"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {"latent": ("LATENT",)}}

    def echo_latent(self, latent: Any) -> tuple[Any, bool]:
        saw_json_tensor = _contains_tensor_marker(latent)
        logger.warning("][ latent echo json_marker=%s", saw_json_tensor)
        return latent, saw_json_tensor


NODE_CLASS_MAPPINGS = {
    "UVSealedRuntimeProbe": InspectRuntimeNode,
    "UVSealedBoltonsSlugify": BoltonsSlugifyNode,
    "UVSealedFilesystemBarrier": FilesystemBarrierNode,
    "UVSealedTensorEcho": EchoTensorNode,
    "UVSealedLatentEcho": EchoLatentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UVSealedRuntimeProbe": "UV Sealed Runtime Probe",
    "UVSealedBoltonsSlugify": "UV Sealed Boltons Slugify",
    "UVSealedFilesystemBarrier": "UV Sealed Filesystem Barrier",
    "UVSealedTensorEcho": "UV Sealed Tensor Echo",
    "UVSealedLatentEcho": "UV Sealed Latent Echo",
}
