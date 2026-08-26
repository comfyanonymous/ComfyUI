"""Re-entrant ``execution.execute()`` coverage for the S10.5 cache-purity invariant.

S10.5: the outputs cache must never contain asset ids. Asset ids are per-delivery;
caching one would replay a stale id. ``execute()`` stores the RAW ``output_ui`` in
the cache while an enriched deep COPY (carrying ids) flows to ``ui_outputs`` /
history / ``send_sync``.

The open question this pins: can an **async node re-entering** ``execute()`` capture
an already-enriched value into ``cache_ui_value`` and leak it into the cache?

Unlike ``test_enrich_output.py`` (pure-adapter unit tests, deliberately free of the
heavy ``execution`` import), this file drives the REAL ``execution.execute()`` across
a genuine async suspend/resume so it actually traverses the ``pending_async_nodes``
re-entry branch. ``execution`` -> ``comfy.model_management`` does GPU detection at
import time, so we force ``args.cpu = True`` before importing and skip if even the
CPU import is unavailable (rather than assert a path we never traversed).
"""
import asyncio
import os
import sys
import tempfile
import types

import pytest

_BASE = os.path.join(tempfile.gettempdir(), "execute-reentry-test-base")


def _find_ids(value) -> list:
    """Recursively collect every ``id`` value under nested dicts / lists / tuples."""
    found: list = []
    if isinstance(value, dict):
        for key, sub in value.items():
            if key == "id":
                found.append(sub)
            found.extend(_find_ids(sub))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_find_ids(item))
    return found


class _Server:
    def __init__(self, client_id=None) -> None:
        self.client_id = client_id
        self.last_node_id = None
        self.sent: list = []

    def send_sync(self, event, payload, client_id) -> None:
        self.sent.append((event, payload))


class _AsyncDictCache:
    """Minimal async cache store: we assert on the value handed to ``set``."""

    def __init__(self) -> None:
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value) -> None:
        self.store[key] = value

    async def ensure_subcache_for(self, unique_id, node_ids):
        return self


class _Caches:
    def __init__(self) -> None:
        self.outputs = _AsyncDictCache()
        self.objects = _AsyncDictCache()
        self.all = [self.outputs, self.objects]


class _ExecutionList:
    def cache_update(self, unique_id, entry) -> None:
        pass

    def add_external_block(self, unique_id):
        return lambda: None

    def get_cache(self, a, b):
        return None


class _NoProgress:
    def start_progress(self, *a, **k) -> None:
        pass

    def finish_progress(self, *a, **k) -> None:
        pass


class _AsyncUINode:
    """Async output node that stays pending past one event-loop turn."""

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "test"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    async def run(self):
        # Awaiting a real (non-zero) sleep guarantees the task is NOT done after
        # the scheduler's ``await asyncio.sleep(0)`` probe, so execute() suspends
        # it into ``pending_async_nodes`` and returns PENDING (entry 1).
        await asyncio.sleep(0.02)
        return {"ui": {"images": [{"filename": "async.png", "subfolder": "", "type": "output"}]}}


@pytest.fixture
def execution_env(monkeypatch):
    """Import the real ``execution`` (CPU-forced) and wire enrichment fakes.

    Enrichment is made to actually mint an id (fresh per-test counter) so the test
    proves a real DIVERGENCE - enriched copy carries an id, cache does not - rather
    than a vacuous "no ids anywhere".
    """
    try:
        from comfy.cli_args import args
        monkeypatch.setattr(args, "cpu", True, raising=False)
        import execution
        import folder_paths
        import nodes
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"execution module could not be imported in CPU mode: {exc!r}")

    monkeypatch.setattr(args, "enable_assets", True, raising=False)

    counter = {"n": 0}

    def _register(abs_path, job_id=None):
        counter["n"] += 1
        return types.SimpleNamespace(id=f"asset-{counter['n']}", job_id=job_id)

    monkeypatch.setitem(
        sys.modules,
        "app.assets.services.ingest",
        types.SimpleNamespace(register_executed_output=_register, register_cached_output=_register),
    )
    os.makedirs(_BASE, exist_ok=True)
    monkeypatch.setattr(folder_paths, "get_directory_by_type", lambda t: _BASE)
    monkeypatch.setattr(execution, "get_progress_state", lambda: _NoProgress())
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "AsyncUINode", _AsyncUINode)

    return execution


async def _drive_async_reentry(execution):
    """Run the two execute() entries for one async output node; return observations."""
    from comfy_execution.graph import DynamicPrompt

    unique_id = "1"
    with open(os.path.join(_BASE, "async.png"), "wb") as f:
        f.write(b"x")

    dynprompt = DynamicPrompt({unique_id: {"class_type": "AsyncUINode", "inputs": {}}})
    caches = _Caches()
    server = _Server(client_id=None)
    exec_list = _ExecutionList()
    pending_subgraph_results: dict = {}
    pending_async_nodes: dict = {}
    ui_outputs: dict = {}
    executed: set = set()

    common = (server, dynprompt, caches, unique_id, {}, executed, "job-1", exec_list)

    # Entry 1: node suspends into pending_async_nodes, returns PENDING.
    r1, _, _ = await execution.execute(*common, pending_subgraph_results, pending_async_nodes, ui_outputs)
    ui_outputs_had_uid_after_entry1 = unique_id in ui_outputs

    # Let the suspended node task finish, then drain the completion callback task.
    tasks = [t for t in pending_async_nodes.get(unique_id, []) if isinstance(t, asyncio.Task)]
    if tasks:
        await asyncio.gather(*tasks)
    for _ in range(3):
        await asyncio.sleep(0)

    # Entry 2: re-entry consumes the completed task, enriches, and caches.
    r2, _, _ = await execution.execute(*common, pending_subgraph_results, pending_async_nodes, ui_outputs)

    cached = caches.outputs.store.get(unique_id)
    return {
        "r1": r1,
        "r2": r2,
        "ui_outputs_had_uid_after_entry1": ui_outputs_had_uid_after_entry1,
        "ui_ids": _find_ids(ui_outputs.get(unique_id)),
        "cache_entry": cached,
        "cache_ids": _find_ids(cached.ui) if cached is not None else None,
    }


def test_async_reentry_keeps_cache_id_free(execution_env):
    """Given an async output node whose delivery is enriched with an asset id,
    When execute() suspends it (entry 1) and re-enters to finish it (entry 2),
    Then the outputs cache stores an id-free ui (S10.5) even though the enriched
    copy published to ui_outputs carries the id."""
    execution = execution_env
    from execution import ExecutionResult

    obs = asyncio.run(_drive_async_reentry(execution))

    # The re-entry actually happened: suspend then resume to success.
    assert obs["r1"] == ExecutionResult.PENDING
    assert obs["r2"] == ExecutionResult.SUCCESS
    # WHY it is safe: the async branch returns PENDING *before* writing ui_outputs,
    # so on re-entry ui_outputs.get(unique_id) is None and cannot seed cache_ui_value.
    assert obs["ui_outputs_had_uid_after_entry1"] is False

    # Divergence is real (not a vacuous no-ids-anywhere): the published copy is enriched.
    assert obs["ui_ids"] == ["asset-1"]

    # The invariant: the cached ui contains NO asset id anywhere.
    assert obs["cache_entry"] is not None
    assert obs["cache_ids"] == []
