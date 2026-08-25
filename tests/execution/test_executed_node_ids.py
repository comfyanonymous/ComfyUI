from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

import comfy.options


_original_args_parsing = comfy.options.args_parsing
_original_argv = sys.argv
try:
    comfy.options.args_parsing = True
    sys.argv = [sys.argv[0], "--cpu"]
    nodes = importlib.import_module("nodes")
    execution = importlib.import_module("execution")
finally:
    sys.argv = _original_argv
    comfy.options.args_parsing = _original_args_parsing

graph_utils = importlib.import_module("comfy_execution.graph_utils")
caching = importlib.import_module("comfy_execution.caching")
progress = importlib.import_module("comfy_execution.progress")
model_management = importlib.import_module("comfy.model_management")
latent_preview = importlib.import_module("latent_preview")
singleton = importlib.import_module("comfy_api.internal.singleton")
cli_args = importlib.import_module("comfy.cli_args")

pytestmark = pytest.mark.asyncio

_TESTING_PACK = Path(__file__).parent / "testing_nodes" / "testing-pack"
_CACHE_ARGS = {"lru": 32, "ram": 0.0, "ram_inactive": 0.0}


class _Server:
    def __init__(self):
        self.client_id = None
        self.last_node_id = None

    def send_sync(self, *_args):
        return None


class _EvictableOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": "evictable-output"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "emit"
    CATEGORY = "Testing/Nodes"
    OUTPUT_NODE = True

    def emit(self, value):
        return {"ui": {"values": [value]}, "result": (value,)}


def _snapshot_globals():
    graph_builder = graph_utils.GraphBuilder
    return {
        "node_classes": tuple(nodes.NODE_CLASS_MAPPINGS.items()),
        "node_display_names": tuple(nodes.NODE_DISPLAY_NAME_MAPPINGS.items()),
        "extension_web_dirs": tuple(nodes.EXTENSION_WEB_DIRS.items()),
        "loaded_module_dirs": tuple(nodes.LOADED_MODULE_DIRS.items()),
        "graph_builder": (
            graph_builder._default_prefix_root,
            graph_builder._default_prefix_call_index,
            graph_builder._default_prefix_graph_index,
        ),
        "progress_registry": progress.global_progress_registry,
        "preview_method": cli_args.args.preview_method,
        "interrupt_state": model_management.processing_interrupted(),
        "singleton_instances": tuple(singleton.SingletonMetaclass._instances.items()),
        "sys_modules": tuple(sys.modules.items()),
    }


def _restore_globals(state):
    for mapping, key in (
        (nodes.NODE_CLASS_MAPPINGS, "node_classes"),
        (nodes.NODE_DISPLAY_NAME_MAPPINGS, "node_display_names"),
        (nodes.EXTENSION_WEB_DIRS, "extension_web_dirs"),
        (nodes.LOADED_MODULE_DIRS, "loaded_module_dirs"),
        (singleton.SingletonMetaclass._instances, "singleton_instances"),
    ):
        mapping.clear()
        mapping.update(state[key])

    graph_builder = graph_utils.GraphBuilder
    (
        graph_builder._default_prefix_root,
        graph_builder._default_prefix_call_index,
        graph_builder._default_prefix_graph_index,
    ) = state["graph_builder"]
    progress.__dict__["global_progress_registry"] = state["progress_registry"]
    cli_args.args.preview_method = state["preview_method"]
    nodes.interrupt_processing(state["interrupt_state"])

    original_modules = dict(state["sys_modules"])
    for module_name in tuple(sys.modules):
        if module_name not in original_modules:
            del sys.modules[module_name]
    for module_name, module in state["sys_modules"]:
        sys.modules[module_name] = module


def _identity_bytes(state):
    mapping_keys = {
        "node_classes",
        "node_display_names",
        "extension_web_dirs",
        "loaded_module_dirs",
        "singleton_instances",
        "sys_modules",
    }
    fingerprints = {}
    for name, value in state.items():
        if name in mapping_keys:
            fingerprints[name] = b"\n".join(
                f"{key!r}\0{id(item)}".encode() for key, item in value
            )
        elif name in {"progress_registry", "preview_method"}:
            fingerprints[name] = f"{value!r}\0{id(value)}".encode()
        else:
            fingerprints[name] = repr(value).encode()
    return fingerprints


@pytest_asyncio.fixture
async def loaded_test_nodes():
    state = _snapshot_globals()
    loaded = await nodes.load_custom_node(str(_TESTING_PACK))
    assert loaded
    try:
        yield
    finally:
        _restore_globals(state)


def _executor(cache_type):
    return execution.PromptExecutor(
        _Server(), cache_type=cache_type, cache_args=_CACHE_ARGS
    )


def _expanded_prompt():
    return {
        "expand": {
            "class_type": "TestExecutedNodeIdsExpander",
            "inputs": {"value": "stable-child"},
        }
    }


async def _run_expanded(executor, prompt_id):
    await executor.execute_async(_expanded_prompt(), prompt_id, {}, ["expand"])
    assert executor.success
    output_ids = tuple(executor.history_result["outputs"])
    assert len(output_ids) == 1
    return output_ids[0]


async def test_first_expanded_child_is_in_final_executed_ids(loaded_test_nodes):
    executor = _executor(execution.CacheType.CLASSIC)

    child_id = await _run_expanded(executor, "first")

    assert child_id in executor.executed_node_ids


async def test_classic_cached_expanded_child_is_absent_from_final_executed_ids(
    loaded_test_nodes,
):
    executor = _executor(execution.CacheType.CLASSIC)
    child_id = await _run_expanded(executor, "classic-first")

    cached_child_id = await _run_expanded(executor, "classic-second")

    assert cached_child_id == child_id
    assert cached_child_id in executor.history_result["outputs"]
    assert cached_child_id not in executor.executed_node_ids


async def test_lru_cached_expanded_child_is_absent_from_final_executed_ids(
    loaded_test_nodes,
):
    executor = _executor(execution.CacheType.LRU)
    child_id = await _run_expanded(executor, "lru-first")

    cached_child_id = await _run_expanded(executor, "lru-second")

    assert cached_child_id == child_id
    assert cached_child_id in executor.history_result["outputs"]
    assert cached_child_id not in executor.executed_node_ids


async def test_ram_pressure_retained_entry_is_cached_on_rerun(loaded_test_nodes):
    executor = _executor(execution.CacheType.RAM_PRESSURE)
    child_id = await _run_expanded(executor, "ram-retained-first")

    cached_child_id = await _run_expanded(executor, "ram-retained-second")

    assert cached_child_id == child_id
    assert cached_child_id in executor.history_result["outputs"]
    assert cached_child_id not in executor.executed_node_ids


async def test_ram_pressure_evicted_entry_re_executes(
    loaded_test_nodes,
    monkeypatch,
):
    executor = _executor(execution.CacheType.RAM_PRESSURE)
    monkeypatch.setitem(
        nodes.NODE_CLASS_MAPPINGS,
        "TestExecutedNodeIdsEvictableOutput",
        _EvictableOutput,
    )
    prompt = {
        "evictable": {
            "class_type": "TestExecutedNodeIdsEvictableOutput",
            "inputs": {"value": "stable-output"},
        }
    }
    await executor.execute_async(prompt, "ram-evicted-first", {}, ["evictable"])
    assert executor.success
    assert "evictable" in executor.executed_node_ids

    outputs = executor.caches.outputs
    monkeypatch.setattr(
        caching.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=0),
    )
    freed = outputs.ram_release(1, free_active=True, min_entry_size=0)
    assert freed > 0
    assert outputs.active_evictions
    assert outputs.full_evictions

    await executor.execute_async(prompt, "ram-evicted-second", {}, ["evictable"])

    assert executor.success
    assert "evictable" in executor.history_result["outputs"]
    assert "evictable" in executor.executed_node_ids


async def test_cache_none_expanded_child_is_in_final_executed_ids(loaded_test_nodes):
    executor = _executor(execution.CacheType.NONE)
    child_id = await _run_expanded(executor, "none-first")

    second_child_id = await _run_expanded(executor, "none-second")

    assert second_child_id == child_id
    assert second_child_id in executor.executed_node_ids


async def test_second_prompt_clears_ids_before_blocked_node_completes(
    loaded_test_nodes,
):
    executor = _executor(execution.CacheType.CLASSIC)
    await _run_expanded(executor, "priming")
    blocking = nodes.NODE_CLASS_MAPPINGS["TestExecutedNodeIdsBlocking"]
    blocking.started_event = asyncio.Event()
    blocking.release_event = asyncio.Event()
    prompt = {"block": {"class_type": "TestExecutedNodeIdsBlocking", "inputs": {}}}

    task = asyncio.create_task(executor.execute_async(prompt, "blocked", {}, ["block"]))
    try:
        _ = await asyncio.wait_for(blocking.started_event.wait(), timeout=2)
        assert executor.executed_node_ids == frozenset()
    finally:
        blocking.release_event.set()
        await task


async def test_reset_clears_executed_node_ids(loaded_test_nodes):
    executor = _executor(execution.CacheType.CLASSIC)
    await _run_expanded(executor, "before-reset")

    executor.reset()

    assert executor.executed_node_ids == frozenset()


async def test_two_fixture_lifecycles_restore_all_globals():
    baseline = _identity_bytes(_snapshot_globals())

    for _ in range(2):
        state = _snapshot_globals()
        try:
            loaded = await nodes.load_custom_node(str(_TESTING_PACK))
            assert loaded
        finally:
            _restore_globals(state)
        assert _identity_bytes(_snapshot_globals()) == baseline
