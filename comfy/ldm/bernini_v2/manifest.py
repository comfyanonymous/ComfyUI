"""Compatibility checks for Bernini model packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_FORMATS = {
    "bernini_v2_safetensors_sharded",
    "bernini_v2_int8_tensorwise_convrot",
}
REQUIRED_COMPONENTS = {
    "wan_high",
    "wan_low",
    "mllm",
    "t5_text_encoder",
    "connector",
    "mask_tokens",
    "vit_decoder",
}
MAX_SCHEMA_VERSION = 3


def load_repack_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"invalid outputs in {manifest_path}")
    missing = REQUIRED_COMPONENTS - set(outputs)
    if missing:
        raise ValueError(f"model package is missing components: {sorted(missing)}")
    model_format = payload.get("format")
    if model_format is not None and model_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported Bernini model package format: {model_format!r}")
    schema = int(payload.get("schema_version", 1))
    if schema < 1 or schema > MAX_SCHEMA_VERSION:
        raise ValueError(f"unsupported Bernini model package schema_version: {schema}")
    storage_dtype = payload.get("storage_dtype", "preserve")
    if storage_dtype not in {"preserve", "bfloat16", "float16"}:
        raise ValueError(f"unsupported Bernini storage_dtype: {storage_dtype!r}")
    return payload
