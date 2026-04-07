# pylint: disable=import-outside-toplevel,import-error
from __future__ import annotations

import logging
import os
import sys
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
        "runtime_report",
        "saw_comfy_root",
        "imported_comfy_wrapper",
        "comfy_module_dump",
        "python_exe",
        "saw_user_site",
    )
    FUNCTION = "inspect"
    CATEGORY = "PyIsolated/SealedWorker"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {}}

    def inspect(self) -> tuple[str, str, bool, bool, str, str, bool]:
        import cfgrib
        import eccodes
        import xarray as xr

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
        python_exe = sys.executable

        runtime_lines = [
            "Conda sealed worker runtime probe",
            f"python_exe={python_exe}",
            f"xarray_origin={getattr(xr, '__file__', '<missing>')}",
            f"cfgrib_origin={getattr(cfgrib, '__file__', '<missing>')}",
            f"eccodes_origin={getattr(eccodes, '__file__', '<missing>')}",
            f"saw_comfy_root={saw_comfy_root}",
            f"imported_comfy_wrapper={imported_comfy_wrapper}",
            f"saw_user_site={saw_user_site}",
        ]
        runtime_report = "\n".join(runtime_lines)

        _write_artifact("child_bootstrap_paths.txt", path_dump)
        _write_artifact("child_import_trace.txt", comfy_module_dump)
        _write_artifact("child_dependency_dump.txt", runtime_report)
        logger.warning("][ Conda sealed runtime probe executed")
        logger.warning("][ conda python executable: %s", python_exe)
        logger.warning(
            "][ conda dependency origins: xarray=%s cfgrib=%s eccodes=%s",
            getattr(xr, "__file__", "<missing>"),
            getattr(cfgrib, "__file__", "<missing>"),
            getattr(eccodes, "__file__", "<missing>"),
        )

        return (
            path_dump,
            runtime_report,
            saw_comfy_root,
            imported_comfy_wrapper,
            comfy_module_dump,
            python_exe,
            saw_user_site,
        )


class OpenWeatherDatasetNode:
    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("sum_value", "grib_path", "dependency_report")
    FUNCTION = "open_dataset"
    CATEGORY = "PyIsolated/SealedWorker"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {}}

    def open_dataset(self) -> tuple[float, str, str]:
        import eccodes
        import xarray as xr

        artifact_dir = _artifact_dir()
        if artifact_dir is None:
            artifact_dir = Path(os.environ.get("HOME", ".")) / "pyisolate_artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

        grib_path = artifact_dir / "toolkit_weather_fixture.grib2"

        gid = eccodes.codes_grib_new_from_samples("GRIB2")
        for key, value in [
            ("gridType", "regular_ll"),
            ("Nx", 2),
            ("Ny", 2),
            ("latitudeOfFirstGridPointInDegrees", 1.0),
            ("longitudeOfFirstGridPointInDegrees", 0.0),
            ("latitudeOfLastGridPointInDegrees", 0.0),
            ("longitudeOfLastGridPointInDegrees", 1.0),
            ("iDirectionIncrementInDegrees", 1.0),
            ("jDirectionIncrementInDegrees", 1.0),
            ("jScansPositively", 0),
            ("shortName", "t"),
            ("typeOfLevel", "surface"),
            ("level", 0),
            ("date", 20260315),
            ("time", 0),
            ("step", 0),
        ]:
            eccodes.codes_set(gid, key, value)

        eccodes.codes_set_values(gid, [1.0, 2.0, 3.0, 4.0])
        with grib_path.open("wb") as handle:
            eccodes.codes_write(gid, handle)
        eccodes.codes_release(gid)

        dataset = xr.open_dataset(grib_path, engine="cfgrib")
        sum_value = float(dataset["t"].sum().item())
        dependency_report = "\n".join(
            [
                f"dataset_sum={sum_value}",
                f"grib_path={grib_path}",
                "xarray_engine=cfgrib",
            ]
        )
        _write_artifact("weather_dependency_report.txt", dependency_report)
        logger.warning("][ cfgrib import ok")
        logger.warning("][ xarray open_dataset engine=cfgrib path=%s", grib_path)
        logger.warning("][ conda weather dataset sum=%s", sum_value)
        return sum_value, str(grib_path), dependency_report


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
        logger.warning("][ conda latent echo json_marker=%s", saw_json_tensor)
        return latent, saw_json_tensor


NODE_CLASS_MAPPINGS = {
    "CondaSealedRuntimeProbe": InspectRuntimeNode,
    "CondaSealedOpenWeatherDataset": OpenWeatherDatasetNode,
    "CondaSealedLatentEcho": EchoLatentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CondaSealedRuntimeProbe": "Conda Sealed Runtime Probe",
    "CondaSealedOpenWeatherDataset": "Conda Sealed Open Weather Dataset",
    "CondaSealedLatentEcho": "Conda Sealed Latent Echo",
}
