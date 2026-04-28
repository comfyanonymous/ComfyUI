# pylint: disable=consider-using-from-import,cyclic-import,global-statement,global-variable-not-assigned,import-outside-toplevel,logging-fstring-interpolation
from __future__ import annotations
import asyncio
import inspect
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING
_IMPORT_TORCH = os.environ.get("PYISOLATE_IMPORT_TORCH", "1") == "1"

load_isolated_node = None
find_manifest_directories = None
build_stub_class = None
get_class_types_for_extension = None
scan_shm_forensics = None
start_shm_forensics = None

if _IMPORT_TORCH:
    import folder_paths
    from .extension_loader import load_isolated_node
    from .manifest_loader import find_manifest_directories
    from .runtime_helpers import build_stub_class, get_class_types_for_extension
    from .shm_forensics import scan_shm_forensics, start_shm_forensics

if TYPE_CHECKING:
    from pyisolate import ExtensionManager
    from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "]["
isolated_node_timings: List[tuple[float, Path, int]] = []

if _IMPORT_TORCH:
    PYISOLATE_VENV_ROOT = Path(folder_paths.base_path) / ".pyisolate_venvs"
    PYISOLATE_VENV_ROOT.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
_WORKFLOW_BOUNDARY_MIN_FREE_VRAM_BYTES = 2 * 1024 * 1024 * 1024
_MODEL_PATCHER_IDLE_TIMEOUT_MS = 120000


def initialize_proxies() -> None:
    from .child_hooks import is_child_process

    is_child = is_child_process()

    if is_child:
        from .child_hooks import initialize_child_process

        initialize_child_process()
    else:
        from .host_hooks import initialize_host_process

        initialize_host_process()
        if start_shm_forensics is not None:
            start_shm_forensics()


@dataclass(frozen=True)
class IsolatedNodeSpec:
    node_name: str
    display_name: str
    stub_class: type
    module_path: Path


_ISOLATED_NODE_SPECS: List[IsolatedNodeSpec] = []
_CLAIMED_PATHS: Set[Path] = set()
_ISOLATION_SCAN_ATTEMPTED = False
_EXTENSION_MANAGERS: List["ExtensionManager"] = []
_RUNNING_EXTENSIONS: Dict[str, "ComfyNodeExtension"] = {}
_ISOLATION_BACKGROUND_TASK: Optional["asyncio.Task[List[IsolatedNodeSpec]]"] = None
_EARLY_START_TIME: Optional[float] = None


def start_isolation_loading_early(loop: "asyncio.AbstractEventLoop") -> None:
    global _ISOLATION_BACKGROUND_TASK, _EARLY_START_TIME
    if _ISOLATION_BACKGROUND_TASK is not None:
        return
    _EARLY_START_TIME = time.perf_counter()
    _ISOLATION_BACKGROUND_TASK = loop.create_task(initialize_isolation_nodes())


async def await_isolation_loading() -> List[IsolatedNodeSpec]:
    global _ISOLATION_BACKGROUND_TASK, _EARLY_START_TIME
    if _ISOLATION_BACKGROUND_TASK is not None:
        specs = await _ISOLATION_BACKGROUND_TASK
        return specs
    return await initialize_isolation_nodes()


async def initialize_isolation_nodes() -> List[IsolatedNodeSpec]:
    global _ISOLATED_NODE_SPECS, _ISOLATION_SCAN_ATTEMPTED, _CLAIMED_PATHS

    if _ISOLATED_NODE_SPECS:
        return _ISOLATED_NODE_SPECS

    if _ISOLATION_SCAN_ATTEMPTED:
        return []

    _ISOLATION_SCAN_ATTEMPTED = True
    if find_manifest_directories is None or load_isolated_node is None or build_stub_class is None:
        return []
    manifest_entries = find_manifest_directories()
    _CLAIMED_PATHS = {entry[0].resolve() for entry in manifest_entries}

    if not manifest_entries:
        return []

    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"
    concurrency_limit = max(1, (os.cpu_count() or 4) // 2)
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def load_with_semaphore(
        node_dir: Path, manifest: Path
    ) -> List[IsolatedNodeSpec]:
        async with semaphore:
            load_start = time.perf_counter()
            spec_list = await load_isolated_node(
                node_dir,
                manifest,
                logger,
                lambda name, info, extension: build_stub_class(
                    name,
                    info,
                    extension,
                    _RUNNING_EXTENSIONS,
                    logger,
                ),
                PYISOLATE_VENV_ROOT,
                _EXTENSION_MANAGERS,
            )
            spec_list = [
                IsolatedNodeSpec(
                    node_name=node_name,
                    display_name=display_name,
                    stub_class=stub_cls,
                    module_path=node_dir,
                )
                for node_name, display_name, stub_cls in spec_list
            ]
            isolated_node_timings.append(
                (time.perf_counter() - load_start, node_dir, len(spec_list))
            )
            return spec_list

    tasks = [
        load_with_semaphore(node_dir, manifest)
        for node_dir, manifest in manifest_entries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    specs: List[IsolatedNodeSpec] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(
                "%s Isolated node failed during startup; continuing: %s",
                LOG_PREFIX,
                result,
            )
            continue
        specs.extend(result)

    _ISOLATED_NODE_SPECS = specs
    return list(_ISOLATED_NODE_SPECS)


def _get_class_types_for_extension(extension_name: str) -> Set[str]:
    """Get all node class types (node names) belonging to an extension."""
    extension = _RUNNING_EXTENSIONS.get(extension_name)
    if not extension:
        return set()

    ext_path = Path(extension.module_path)
    class_types = set()
    for spec in _ISOLATED_NODE_SPECS:
        if spec.module_path.resolve() == ext_path.resolve():
            class_types.add(spec.node_name)

    return class_types


async def notify_execution_graph(needed_class_types: Set[str], caches: list | None = None) -> None:
    """Evict running extensions not needed for current execution.

    When *caches* is provided, cache entries for evicted extensions' node
    class_types are invalidated to prevent stale ``RemoteObjectHandle``
    references from surviving in the output cache.
    """
    await wait_for_model_patcher_quiescence(
        timeout_ms=_MODEL_PATCHER_IDLE_TIMEOUT_MS,
        fail_loud=True,
        marker="ISO:notify_graph_wait_idle",
    )

    evicted_class_types: Set[str] = set()

    async def _stop_extension(
        ext_name: str, extension: "ComfyNodeExtension", reason: str
    ) -> None:
        # Collect class_types BEFORE stopping so we can invalidate cache entries.
        ext_class_types = _get_class_types_for_extension(ext_name)
        evicted_class_types.update(ext_class_types)
        logger.info("%s ISO:eject_start ext=%s reason=%s", LOG_PREFIX, ext_name, reason)
        logger.debug("%s ISO:stop_start ext=%s", LOG_PREFIX, ext_name)
        stop_result = extension.stop()
        if inspect.isawaitable(stop_result):
            await stop_result
        _RUNNING_EXTENSIONS.pop(ext_name, None)
        logger.debug("%s ISO:stop_done ext=%s", LOG_PREFIX, ext_name)
        if scan_shm_forensics is not None:
            scan_shm_forensics("ISO:stop_extension", refresh_model_context=True)

    if scan_shm_forensics is not None:
        scan_shm_forensics("ISO:notify_graph_start", refresh_model_context=True)
    isolated_class_types_in_graph = needed_class_types.intersection(
        {spec.node_name for spec in _ISOLATED_NODE_SPECS}
    )
    graph_uses_isolation = bool(isolated_class_types_in_graph)
    logger.debug(
        "%s ISO:notify_graph_start running=%d needed=%d",
        LOG_PREFIX,
        len(_RUNNING_EXTENSIONS),
        len(needed_class_types),
    )
    if graph_uses_isolation:
        for ext_name, extension in list(_RUNNING_EXTENSIONS.items()):
            ext_class_types = _get_class_types_for_extension(ext_name)

            # If NONE of this extension's nodes are in the execution graph -> evict.
            if not ext_class_types.intersection(needed_class_types):
                await _stop_extension(
                    ext_name,
                    extension,
                    "isolated custom_node not in execution graph, evicting",
                )
    else:
        logger.debug(
            "%s ISO:notify_graph_skip_evict running=%d reason=no isolated nodes in graph",
            LOG_PREFIX,
            len(_RUNNING_EXTENSIONS),
        )

    # Isolated child processes add steady VRAM pressure; reclaim host-side models
    # at workflow boundaries so subsequent host nodes (e.g. CLIP encode) keep headroom.
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        if getattr(device, "type", None) == "cuda":
            required = max(
                model_management.minimum_inference_memory(),
                _WORKFLOW_BOUNDARY_MIN_FREE_VRAM_BYTES,
            )
            free_before = model_management.get_free_memory(device)
            if free_before < required and _RUNNING_EXTENSIONS and graph_uses_isolation:
                for ext_name, extension in list(_RUNNING_EXTENSIONS.items()):
                    await _stop_extension(
                        ext_name,
                        extension,
                        f"boundary low-vram restart (free={int(free_before)} target={int(required)})",
                    )
            if model_management.get_free_memory(device) < required:
                model_management.unload_all_models()
            model_management.cleanup_models_gc()
            model_management.cleanup_models()
            if model_management.get_free_memory(device) < required:
                model_management.free_memory(required, device, for_dynamic=False)
                model_management.soft_empty_cache()
    except Exception:
        logger.debug(
            "%s workflow-boundary host VRAM relief failed", LOG_PREFIX, exc_info=True
        )
    finally:
        # Invalidate cached outputs for evicted extensions so stale
        # RemoteObjectHandle references are not served from cache.
        if evicted_class_types and caches:
            total_invalidated = 0
            for cache in caches:
                if hasattr(cache, "invalidate_by_class_types"):
                    total_invalidated += cache.invalidate_by_class_types(
                        evicted_class_types
                    )
            if total_invalidated > 0:
                logger.info(
                    "%s ISO:cache_invalidated count=%d class_types=%s",
                    LOG_PREFIX,
                    total_invalidated,
                    evicted_class_types,
                )
        scan_shm_forensics("ISO:notify_graph_done", refresh_model_context=True)
        logger.debug(
            "%s ISO:notify_graph_done running=%d", LOG_PREFIX, len(_RUNNING_EXTENSIONS)
        )


async def flush_running_extensions_transport_state() -> int:
    await wait_for_model_patcher_quiescence(
        timeout_ms=_MODEL_PATCHER_IDLE_TIMEOUT_MS,
        fail_loud=True,
        marker="ISO:flush_transport_wait_idle",
    )
    total_flushed = 0
    for ext_name, extension in list(_RUNNING_EXTENSIONS.items()):
        flush_fn = getattr(extension, "flush_transport_state", None)
        if not callable(flush_fn):
            continue
        try:
            flushed = await flush_fn()
            if isinstance(flushed, int):
                total_flushed += flushed
                if flushed > 0:
                    logger.debug(
                        "%s %s workflow-end flush released=%d",
                        LOG_PREFIX,
                        ext_name,
                        flushed,
                    )
        except Exception:
            logger.debug(
                "%s %s workflow-end flush failed", LOG_PREFIX, ext_name, exc_info=True
            )
    scan_shm_forensics(
        "ISO:flush_running_extensions_transport_state", refresh_model_context=True
    )
    return total_flushed


async def wait_for_model_patcher_quiescence(
    timeout_ms: int = _MODEL_PATCHER_IDLE_TIMEOUT_MS,
    *,
    fail_loud: bool = False,
    marker: str = "ISO:wait_model_patcher_idle",
) -> bool:
    try:
        from comfy.isolation.model_patcher_proxy_registry import ModelPatcherRegistry

        registry = ModelPatcherRegistry()
        start = time.perf_counter()
        idle = await registry.wait_all_idle(timeout_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if idle:
            logger.debug(
                "%s %s idle=1 timeout_ms=%d elapsed_ms=%.3f",
                LOG_PREFIX,
                marker,
                timeout_ms,
                elapsed_ms,
            )
            return True

        states = await registry.get_all_operation_states()
        logger.error(
            "%s %s idle_timeout timeout_ms=%d elapsed_ms=%.3f states=%s",
            LOG_PREFIX,
            marker,
            timeout_ms,
            elapsed_ms,
            states,
        )
        if fail_loud:
            raise TimeoutError(
                f"ModelPatcherRegistry did not quiesce within {timeout_ms} ms"
            )
        return False
    except Exception:
        if fail_loud:
            raise
        logger.debug("%s %s failed", LOG_PREFIX, marker, exc_info=True)
        return False


def get_claimed_paths() -> Set[Path]:
    return _CLAIMED_PATHS


def update_rpc_event_loops(loop: "asyncio.AbstractEventLoop | None" = None) -> None:
    """Update all active RPC instances with the current event loop.

    This MUST be called at the start of each workflow execution to ensure
    RPC calls are scheduled on the correct event loop. This handles the case
    where asyncio.run() creates a new event loop for each workflow.

    Args:
        loop: The event loop to use. If None, uses asyncio.get_running_loop().
    """
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

    update_count = 0

    # Update RPCs from ExtensionManagers
    for manager in _EXTENSION_MANAGERS:
        if not hasattr(manager, "extensions"):
            continue
        for name, extension in manager.extensions.items():
            if hasattr(extension, "rpc") and extension.rpc is not None:
                if hasattr(extension.rpc, "update_event_loop"):
                    extension.rpc.update_event_loop(loop)
                    update_count += 1
                    logger.debug(f"{LOG_PREFIX}Updated loop on extension '{name}'")

    # Also update RPCs from running extensions (they may have direct RPC refs)
    for name, extension in _RUNNING_EXTENSIONS.items():
        if hasattr(extension, "rpc") and extension.rpc is not None:
            if hasattr(extension.rpc, "update_event_loop"):
                extension.rpc.update_event_loop(loop)
                update_count += 1
                logger.debug(f"{LOG_PREFIX}Updated loop on running extension '{name}'")

    if update_count > 0:
        logger.debug(f"{LOG_PREFIX}Updated event loop on {update_count} RPC instances")
    else:
        logger.debug(
            f"{LOG_PREFIX}No RPC instances found to update (managers={len(_EXTENSION_MANAGERS)}, running={len(_RUNNING_EXTENSIONS)})"
        )


__all__ = [
    "LOG_PREFIX",
    "initialize_proxies",
    "initialize_isolation_nodes",
    "start_isolation_loading_early",
    "await_isolation_loading",
    "notify_execution_graph",
    "flush_running_extensions_transport_state",
    "wait_for_model_patcher_quiescence",
    "get_claimed_paths",
    "update_rpc_event_loops",
    "IsolatedNodeSpec",
    "get_class_types_for_extension",
]
