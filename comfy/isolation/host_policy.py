# pylint: disable=logging-fstring-interpolation
from __future__ import annotations

import logging
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, List, TypedDict

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

HOST_POLICY_PATH_ENV = "COMFY_HOST_POLICY_PATH"
VALID_SANDBOX_MODES = frozenset({"required", "disabled"})
FORBIDDEN_WRITABLE_PATHS = frozenset({"/tmp"})


class HostSecurityPolicy(TypedDict):
    sandbox_mode: str
    allow_network: bool
    writable_paths: List[str]
    readonly_paths: List[str]
    sealed_worker_ro_import_paths: List[str]
    whitelist: Dict[str, str]


DEFAULT_POLICY: HostSecurityPolicy = {
    "sandbox_mode": "required",
    "allow_network": False,
    "writable_paths": ["/dev/shm"],
    "readonly_paths": [],
    "sealed_worker_ro_import_paths": [],
    "whitelist": {},
}


def _default_policy() -> HostSecurityPolicy:
    return {
        "sandbox_mode": DEFAULT_POLICY["sandbox_mode"],
        "allow_network": DEFAULT_POLICY["allow_network"],
        "writable_paths": list(DEFAULT_POLICY["writable_paths"]),
        "readonly_paths": list(DEFAULT_POLICY["readonly_paths"]),
        "sealed_worker_ro_import_paths": list(DEFAULT_POLICY["sealed_worker_ro_import_paths"]),
        "whitelist": dict(DEFAULT_POLICY["whitelist"]),
    }


def _normalize_writable_paths(paths: list[object]) -> list[str]:
    normalized_paths: list[str] = []
    for raw_path in paths:
        # Host-policy paths are contract-style POSIX paths; keep representation
        # stable across Windows/Linux so tests and config behavior stay consistent.
        normalized_path = str(PurePosixPath(str(raw_path).replace("\\", "/")))
        if normalized_path in FORBIDDEN_WRITABLE_PATHS:
            continue
        normalized_paths.append(normalized_path)
    return normalized_paths


def _load_whitelist_file(file_path: Path, config_path: Path) -> Dict[str, str]:
    if not file_path.is_absolute():
        file_path = config_path.parent / file_path
    if not file_path.exists():
        logger.warning("whitelist_file %s not found, skipping.", file_path)
        return {}
    entries: Dict[str, str] = {}
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries[line] = "*"
    logger.debug("Loaded %d whitelist entries from %s", len(entries), file_path)
    return entries


def _normalize_sealed_worker_ro_import_paths(raw_paths: object) -> list[str]:
    if not isinstance(raw_paths, list):
        raise ValueError(
            "tool.comfy.host.sealed_worker_ro_import_paths must be a list of absolute paths."
        )

    normalized_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                "tool.comfy.host.sealed_worker_ro_import_paths entries must be non-empty strings."
            )
        normalized_path = str(PurePosixPath(raw_path.replace("\\", "/")))
        # Accept both POSIX absolute paths (/home/...) and Windows drive-letter paths (D:/...)
        is_absolute = normalized_path.startswith("/") or (
            len(normalized_path) >= 3 and normalized_path[1] == ":" and normalized_path[2] == "/"
        )
        if not is_absolute:
            raise ValueError(
                "tool.comfy.host.sealed_worker_ro_import_paths entries must be absolute paths."
            )
        if normalized_path not in seen:
            seen.add(normalized_path)
            normalized_paths.append(normalized_path)

    return normalized_paths


def load_host_policy(comfy_root: Path) -> HostSecurityPolicy:
    config_override = os.environ.get(HOST_POLICY_PATH_ENV)
    config_path = Path(config_override) if config_override else comfy_root / "pyproject.toml"
    policy = _default_policy()

    if not config_path.exists():
        logger.debug("Host policy file missing at %s, using defaults.", config_path)
        return policy

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.warning(
            "Failed to parse host policy from %s, using defaults.",
            config_path,
            exc_info=True,
        )
        return policy

    tool_config = data.get("tool", {}).get("comfy", {}).get("host", {})
    if not isinstance(tool_config, dict):
        logger.debug("No [tool.comfy.host] section found, using defaults.")
        return policy

    sandbox_mode = tool_config.get("sandbox_mode")
    if isinstance(sandbox_mode, str):
        normalized_sandbox_mode = sandbox_mode.strip().lower()
        if normalized_sandbox_mode in VALID_SANDBOX_MODES:
            policy["sandbox_mode"] = normalized_sandbox_mode
        else:
            logger.warning(
                "Invalid host sandbox_mode %r in %s, using default %r.",
                sandbox_mode,
                config_path,
                DEFAULT_POLICY["sandbox_mode"],
            )

    if "allow_network" in tool_config:
        policy["allow_network"] = bool(tool_config["allow_network"])

    if "writable_paths" in tool_config:
        policy["writable_paths"] = _normalize_writable_paths(tool_config["writable_paths"])

    if "readonly_paths" in tool_config:
        policy["readonly_paths"] = [str(p) for p in tool_config["readonly_paths"]]

    if "sealed_worker_ro_import_paths" in tool_config:
        policy["sealed_worker_ro_import_paths"] = _normalize_sealed_worker_ro_import_paths(
            tool_config["sealed_worker_ro_import_paths"]
        )

    whitelist_file = tool_config.get("whitelist_file")
    if isinstance(whitelist_file, str):
        policy["whitelist"].update(_load_whitelist_file(Path(whitelist_file), config_path))

    whitelist_raw = tool_config.get("whitelist")
    if isinstance(whitelist_raw, dict):
        policy["whitelist"].update({str(k): str(v) for k, v in whitelist_raw.items()})

    os.environ["PYISOLATE_SANDBOX_MODE"] = policy["sandbox_mode"]

    logger.debug(
        "Loaded Host Policy: %d whitelisted nodes, Sandbox=%s, Network=%s",
        len(policy["whitelist"]),
        policy["sandbox_mode"],
        policy["allow_network"],
    )
    return policy


__all__ = ["HostSecurityPolicy", "load_host_policy", "DEFAULT_POLICY"]
