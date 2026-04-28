# pylint: disable=cyclic-import,import-outside-toplevel,redefined-outer-name
from __future__ import annotations

import logging
import os
import inspect
import sys
import types
import platform
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pyisolate
from pyisolate import ExtensionManager, ExtensionManagerConfig
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .manifest_loader import is_cache_valid, load_from_cache, save_to_cache
from .host_policy import load_host_policy

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _register_web_directory(extension_name: str, node_dir: Path) -> None:
    """Register an isolated extension's web directory on the host side."""
    import nodes

    # Method 1: pyproject.toml [tool.comfy] web field
    pyproject = node_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            with pyproject.open("rb") as f:
                data = tomllib.load(f)
            web_dir_name = data.get("tool", {}).get("comfy", {}).get("web")
            if web_dir_name:
                web_dir_path = str(node_dir / web_dir_name)
                if os.path.isdir(web_dir_path):
                    nodes.EXTENSION_WEB_DIRS[extension_name] = web_dir_path
                    logger.debug(
                        "][ Registered web dir for isolated %s: %s",
                        extension_name,
                        web_dir_path,
                    )
                    return
        except Exception:
            pass

    # Method 2: __init__.py WEB_DIRECTORY constant (parse without importing)
    init_file = node_dir / "__init__.py"
    if init_file.exists():
        try:
            source = init_file.read_text()
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("WEB_DIRECTORY"):
                    # Parse: WEB_DIRECTORY = "./web" or WEB_DIRECTORY = "web"
                    _, _, value = stripped.partition("=")
                    value = value.strip().strip("\"'")
                    if value:
                        web_dir_path = str((node_dir / value).resolve())
                        if os.path.isdir(web_dir_path):
                            nodes.EXTENSION_WEB_DIRS[extension_name] = web_dir_path
                            logger.debug(
                                "][ Registered web dir for isolated %s: %s",
                                extension_name,
                                web_dir_path,
                            )
                            return
        except Exception:
            pass


def _get_extension_type(execution_model: str) -> type[Any]:
    if execution_model == "sealed_worker":
        return pyisolate.SealedNodeExtension

    from .extension_wrapper import ComfyNodeExtension

    return ComfyNodeExtension


async def _stop_extension_safe(extension: Any, extension_name: str) -> None:
    try:
        stop_result = extension.stop()
        if inspect.isawaitable(stop_result):
            await stop_result
    except Exception:
        logger.debug("][ %s stop failed", extension_name, exc_info=True)


def _normalize_dependency_spec(dep: str, base_paths: list[Path]) -> str:
    req, sep, marker = dep.partition(";")
    req = req.strip()
    marker_suffix = f";{marker}" if sep else ""

    def _resolve_local_path(local_path: str) -> Path | None:
        for base in base_paths:
            candidate = (base / local_path).resolve()
            if candidate.exists():
                return candidate
        return None

    if req.startswith("./") or req.startswith("../"):
        resolved = _resolve_local_path(req)
        if resolved is not None:
            return f"{resolved}{marker_suffix}"

    if req.startswith("file://"):
        raw = req[len("file://") :]
        if raw.startswith("./") or raw.startswith("../"):
            resolved = _resolve_local_path(raw)
            if resolved is not None:
                return f"file://{resolved}{marker_suffix}"

    return dep


def _dependency_name_from_spec(dep: str) -> str | None:
    stripped = dep.strip()
    if not stripped or stripped == "-e" or stripped.startswith("-e "):
        return None
    if stripped.startswith(("/", "./", "../", "file://")):
        return None

    try:
        return canonicalize_name(Requirement(stripped).name)
    except InvalidRequirement:
        return None


def _parse_cuda_wheels_config(
    tool_config: dict[str, object], dependencies: list[str]
) -> dict[str, object] | None:
    raw_config = tool_config.get("cuda_wheels")
    if raw_config is None:
        return None
    if not isinstance(raw_config, dict):
        raise ExtensionLoadError("[tool.comfy.isolation.cuda_wheels] must be a table")

    index_url = raw_config.get("index_url")
    index_urls = raw_config.get("index_urls")
    if index_urls is not None:
        if not isinstance(index_urls, list) or not all(
            isinstance(u, str) and u.strip() for u in index_urls
        ):
            raise ExtensionLoadError(
                "[tool.comfy.isolation.cuda_wheels.index_urls] must be a list of non-empty strings"
            )
    elif not isinstance(index_url, str) or not index_url.strip():
        raise ExtensionLoadError(
            "[tool.comfy.isolation.cuda_wheels.index_url] must be a non-empty string"
        )

    packages = raw_config.get("packages")
    if not isinstance(packages, list) or not all(
        isinstance(package_name, str) and package_name.strip()
        for package_name in packages
    ):
        raise ExtensionLoadError(
            "[tool.comfy.isolation.cuda_wheels.packages] must be a list of non-empty strings"
        )

    declared_dependencies = {
        dependency_name
        for dep in dependencies
        if (dependency_name := _dependency_name_from_spec(dep)) is not None
    }
    normalized_packages = [canonicalize_name(package_name) for package_name in packages]
    missing = [
        package_name
        for package_name in normalized_packages
        if package_name not in declared_dependencies
    ]
    if missing:
        missing_joined = ", ".join(sorted(missing))
        raise ExtensionLoadError(
            "[tool.comfy.isolation.cuda_wheels.packages] references undeclared dependencies: "
            f"{missing_joined}"
        )

    package_map = raw_config.get("package_map", {})
    if not isinstance(package_map, dict):
        raise ExtensionLoadError(
            "[tool.comfy.isolation.cuda_wheels.package_map] must be a table"
        )

    normalized_package_map: dict[str, str] = {}
    for dependency_name, index_package_name in package_map.items():
        if not isinstance(dependency_name, str) or not dependency_name.strip():
            raise ExtensionLoadError(
                "[tool.comfy.isolation.cuda_wheels.package_map] keys must be non-empty strings"
            )
        if not isinstance(index_package_name, str) or not index_package_name.strip():
            raise ExtensionLoadError(
                "[tool.comfy.isolation.cuda_wheels.package_map] values must be non-empty strings"
            )
        canonical_dependency_name = canonicalize_name(dependency_name)
        if canonical_dependency_name not in normalized_packages:
            raise ExtensionLoadError(
                "[tool.comfy.isolation.cuda_wheels.package_map] can only override packages listed in "
                "[tool.comfy.isolation.cuda_wheels.packages]"
            )
        normalized_package_map[canonical_dependency_name] = index_package_name.strip()

    result: dict = {
        "packages": normalized_packages,
        "package_map": normalized_package_map,
    }
    if index_urls is not None:
        result["index_urls"] = [u.rstrip("/") + "/" for u in index_urls]
    else:
        result["index_url"] = index_url.rstrip("/") + "/"
    return result


def get_enforcement_policy() -> Dict[str, bool]:
    return {
        "force_isolated": os.environ.get("PYISOLATE_ENFORCE_ISOLATED") == "1",
        "force_sandbox": os.environ.get("PYISOLATE_ENFORCE_SANDBOX") == "1",
    }


class ExtensionLoadError(RuntimeError):
    pass


def register_dummy_module(extension_name: str, node_dir: Path) -> None:
    normalized_name = extension_name.replace("-", "_").replace(".", "_")
    if normalized_name not in sys.modules:
        dummy_module = types.ModuleType(normalized_name)
        dummy_module.__file__ = str(node_dir / "__init__.py")
        dummy_module.__path__ = [str(node_dir)]
        dummy_module.__package__ = normalized_name
        sys.modules[normalized_name] = dummy_module


def _is_stale_node_cache(cached_data: Dict[str, Dict]) -> bool:
    for details in cached_data.values():
        if not isinstance(details, dict):
            return True
        if details.get("is_v3") and "schema_v1" not in details:
            return True
    return False


async def load_isolated_node(
    node_dir: Path,
    manifest_path: Path,
    logger: logging.Logger,
    build_stub_class: Callable[[str, Dict[str, object], Any], type],
    venv_root: Path,
    extension_managers: List[ExtensionManager],
) -> List[Tuple[str, str, type]]:
    try:
        with manifest_path.open("rb") as handle:
            manifest_data = tomllib.load(handle)
    except Exception as e:
        logger.warning(f"][ Failed to parse {manifest_path}: {e}")
        return []

    # Parse [tool.comfy.isolation]
    tool_config = manifest_data.get("tool", {}).get("comfy", {}).get("isolation", {})
    can_isolate = tool_config.get("can_isolate", False)
    share_torch = tool_config.get("share_torch", False)
    package_manager = tool_config.get("package_manager", "uv")
    is_conda = package_manager == "conda"
    execution_model = tool_config.get("execution_model")
    if execution_model is None:
        execution_model = "sealed_worker" if is_conda else "host-coupled"

    if "sealed_host_ro_paths" in tool_config:
        raise ValueError(
            "Manifest field 'sealed_host_ro_paths' is not allowed. "
            "Configure [tool.comfy.host].sealed_worker_ro_import_paths in host policy."
        )

    # Conda-specific manifest fields
    conda_channels: list[str] = (
        tool_config.get("conda_channels", []) if is_conda else []
    )
    conda_dependencies: list[str] = (
        tool_config.get("conda_dependencies", []) if is_conda else []
    )
    conda_platforms: list[str] = (
        tool_config.get("conda_platforms", []) if is_conda else []
    )
    conda_python: str = (
        tool_config.get("conda_python", "*") if is_conda else "*"
    )

    # Parse [project] dependencies
    project_config = manifest_data.get("project", {})
    dependencies = project_config.get("dependencies", [])
    if not isinstance(dependencies, list):
        dependencies = []

    # Get extension name (default to folder name if not in project.name)
    extension_name = project_config.get("name", node_dir.name)

    # LOGIC: Isolation Decision
    policy = get_enforcement_policy()
    isolated = can_isolate or policy["force_isolated"]

    if not isolated:
        return []

    import folder_paths

    base_paths = [Path(folder_paths.base_path), node_dir]
    dependencies = [
        _normalize_dependency_spec(dep, base_paths) if isinstance(dep, str) else dep
        for dep in dependencies
    ]
    cuda_wheels = _parse_cuda_wheels_config(tool_config, dependencies)

    manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
    extension_type = _get_extension_type(execution_model)
    manager: ExtensionManager = pyisolate.ExtensionManager(
        extension_type, manager_config
    )
    extension_managers.append(manager)

    host_policy = load_host_policy(Path(folder_paths.base_path))

    sandbox_config = {}
    is_linux = platform.system() == "Linux"

    if is_conda:
        share_torch = False
        share_cuda_ipc = False
    else:
        share_cuda_ipc = share_torch and is_linux

    if is_linux and isolated:
        sandbox_config = {
            "network": host_policy["allow_network"],
            "writable_paths": host_policy["writable_paths"],
            "readonly_paths": host_policy["readonly_paths"],
        }

    extension_config: dict = {
        "name": extension_name,
        "module_path": str(node_dir),
        "isolated": True,
        "dependencies": dependencies,
        "share_torch": share_torch,
        "share_cuda_ipc": share_cuda_ipc,
        "sandbox_mode": host_policy["sandbox_mode"],
        "sandbox": sandbox_config,
    }

    share_torch_no_deps = tool_config.get("share_torch_no_deps", [])
    if share_torch_no_deps:
        if not isinstance(share_torch_no_deps, list) or not all(
            isinstance(dep, str) and dep.strip() for dep in share_torch_no_deps
        ):
            raise ExtensionLoadError(
                "[tool.comfy.isolation.share_torch_no_deps] must be a list of non-empty strings"
            )
        extension_config["share_torch_no_deps"] = share_torch_no_deps

    _is_sealed = execution_model == "sealed_worker"
    _is_sandboxed = host_policy["sandbox_mode"] != "disabled" and is_linux
    logger.info(
        "][ Loading isolated node: %s (torch_share [%s], sealed [%s], sandboxed [%s])",
        extension_name,
        "x" if share_torch else " ",
        "x" if _is_sealed else " ",
        "x" if _is_sandboxed else " ",
    )

    if cuda_wheels is not None:
        extension_config["cuda_wheels"] = cuda_wheels

    extra_index_urls = tool_config.get("extra_index_urls", [])
    if extra_index_urls:
        if not isinstance(extra_index_urls, list) or not all(
            isinstance(u, str) and u.strip() for u in extra_index_urls
        ):
            raise ExtensionLoadError(
                "[tool.comfy.isolation.extra_index_urls] must be a list of non-empty strings"
            )
        extension_config["extra_index_urls"] = extra_index_urls

    # Conda-specific keys
    if is_conda:
        extension_config["package_manager"] = "conda"
        extension_config["conda_channels"] = conda_channels
        extension_config["conda_dependencies"] = conda_dependencies
        extension_config["conda_python"] = conda_python
        find_links = tool_config.get("find_links", [])
        if find_links:
            extension_config["find_links"] = find_links
        if conda_platforms:
            extension_config["conda_platforms"] = conda_platforms

    if execution_model != "host-coupled":
        extension_config["execution_model"] = execution_model
    if execution_model == "sealed_worker":
        policy_ro_paths = host_policy.get("sealed_worker_ro_import_paths", [])
        if isinstance(policy_ro_paths, list) and policy_ro_paths:
            extension_config["sealed_host_ro_paths"] = list(policy_ro_paths)
        # Sealed workers keep the host RPC service inventory even when the
        # child resolves no API classes locally.

    extension = manager.load_extension(extension_config)
    register_dummy_module(extension_name, node_dir)

    # Register host-side event handlers via adapter
    from .adapter import ComfyUIAdapter
    ComfyUIAdapter.register_host_event_handlers(extension)

    # Register web directory on the host — only when sandbox is disabled.
    # In sandbox mode, serving untrusted JS to the browser is not safe.
    if host_policy["sandbox_mode"] == "disabled":
        _register_web_directory(extension_name, node_dir)

    # Register for proxied web serving — the child's web dir may have
    # content that doesn't exist on the host (e.g., pip-installed viewer
    # bundles). The WebDirectoryCache will lazily fetch via RPC.
    from .proxies.web_directory_proxy import WebDirectoryProxy, get_web_directory_cache
    cache = get_web_directory_cache()
    cache.register_proxy(extension_name, WebDirectoryProxy())

    # Try cache first (lazy spawn)
    if is_cache_valid(node_dir, manifest_path, venv_root):
        cached_data = load_from_cache(node_dir, venv_root)
        if cached_data:
            if _is_stale_node_cache(cached_data):
                pass
            else:
                try:
                    flushed = await extension.flush_pending_routes()
                    logger.info("][ %s flushed %d routes", extension_name, flushed)
                except Exception as exc:
                    logger.warning("][ %s route flush failed: %s", extension_name, exc)
                specs: List[Tuple[str, str, type]] = []
                for node_name, details in cached_data.items():
                    stub_cls = build_stub_class(node_name, details, extension)
                    specs.append(
                        (node_name, details.get("display_name", node_name), stub_cls)
                    )
                return specs
    # Cache miss - spawn process and get metadata

    try:
        remote_nodes: Dict[str, str] = await extension.list_nodes()
    except Exception as exc:
        logger.warning(
            "][ %s metadata discovery failed, skipping isolated load: %s",
            extension_name,
            exc,
        )
        await _stop_extension_safe(extension, extension_name)
        return []

    if not remote_nodes:
        logger.debug("][ %s exposed no isolated nodes; skipping", extension_name)
        await _stop_extension_safe(extension, extension_name)
        return []

    specs: List[Tuple[str, str, type]] = []
    cache_data: Dict[str, Dict] = {}

    for node_name, display_name in remote_nodes.items():
        try:
            details = await extension.get_node_details(node_name)
        except Exception as exc:
            logger.warning(
                "][ %s failed to load metadata for %s, skipping node: %s",
                extension_name,
                node_name,
                exc,
            )
            continue
        details["display_name"] = display_name
        cache_data[node_name] = details
        stub_cls = build_stub_class(node_name, details, extension)
        specs.append((node_name, display_name, stub_cls))

    if not specs:
        logger.warning(
            "][ %s produced no usable nodes after metadata scan; skipping",
            extension_name,
        )
        await _stop_extension_safe(extension, extension_name)
        return []

    # Save metadata to cache for future runs
    save_to_cache(node_dir, venv_root, cache_data, manifest_path)
    logger.debug(f"][ {extension_name} metadata cached")

    # Re-check web directory AFTER child has populated it
    if host_policy["sandbox_mode"] == "disabled":
        _register_web_directory(extension_name, node_dir)

    # Flush any routes the child buffered during module import — must happen
    # before router freeze and before we kill the child process.
    try:
        flushed = await extension.flush_pending_routes()
        logger.info("][ %s flushed %d routes", extension_name, flushed)
    except Exception as exc:
        logger.warning("][ %s route flush failed: %s", extension_name, exc)

    # EJECT: Kill process after getting metadata (will respawn on first execution)
    await _stop_extension_safe(extension, extension_name)

    return specs


__all__ = ["ExtensionLoadError", "register_dummy_module", "load_isolated_node"]
