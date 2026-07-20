import asyncio
from types import SimpleNamespace

import torch

import nodes
from comfy_execution.caching import BasicCache, CacheKeySetID
from comfy_execution.graph import DynamicPrompt
from comfy_execution.progress import WebUIProgressHandler, get_progress_state, reset_progress_state
from comfy_execution.utils import get_current_client_id, has_current_client_id, reset_current_client_id, set_current_client_id
from execution import CacheSet, PromptExecutor, _send_cached_ui


class Server:
    def __init__(self):
        self.client_id = "shared-server-value"
        self.messages = []
        self.last_node_id = None

    def send_sync(self, event, data, client_id):
        self.messages.append((event, data, client_id))


def test_prompt_executor_messages_use_executor_client_id():
    server = Server()
    first = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    second = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    first.client_id = "first-client"
    second.client_id = "second-client"

    first.add_message("event", {"prompt_id": "first"}, broadcast=False)
    second.add_message("event", {"prompt_id": "second"}, broadcast=False)

    assert [message[2] for message in server.messages] == ["first-client", "second-client"]
    assert server.client_id == "shared-server-value"


def test_cached_ui_is_recorded_without_a_connected_client():
    server = Server()
    ui_outputs = {}
    cached = SimpleNamespace(ui={"output": {"images": []}, "meta": {"node_id": "node"}})

    _send_cached_ui(server, None, "node", "node", cached, "prompt", ui_outputs)

    assert ui_outputs == {"node": cached.ui}
    assert server.messages == []


def test_progress_registry_is_isolated_between_async_prompt_tasks():
    async def prompt(prompt_id, ready, release):
        reset_progress_state(prompt_id, DynamicPrompt({}))
        ready.set()
        await release.wait()
        return get_progress_state().prompt_id

    async def run():
        first_ready = asyncio.Event()
        second_ready = asyncio.Event()
        release = asyncio.Event()
        first = asyncio.create_task(prompt("first", first_ready, release))
        second = asyncio.create_task(prompt("second", second_ready, release))
        await asyncio.gather(first_ready.wait(), second_ready.wait())
        release.set()
        return await asyncio.gather(first, second)

    assert asyncio.run(run()) == ["first", "second"]


def test_webui_progress_handler_uses_prompt_client_id():
    server = Server()
    first = WebUIProgressHandler(server, "first-client")
    second = WebUIProgressHandler(server, "second-client")

    first._send_progress_state("first", {})
    second._send_progress_state("second", {})

    assert [message[2] for message in server.messages] == ["first-client", "second-client"]


def test_explicit_anonymous_client_does_not_broadcast_progress():
    server = Server()
    handler = WebUIProgressHandler(server, None)
    handler._send_progress_state("anonymous", {})
    assert server.messages == []

    assert not has_current_client_id()
    token = set_current_client_id(None)
    try:
        assert has_current_client_id()
        assert get_current_client_id() is None
    finally:
        reset_current_client_id(token)


def test_legacy_webui_progress_handler_uses_server_client_id():
    server = Server()
    handler = WebUIProgressHandler(server)

    handler._send_progress_state("legacy", {})

    assert [message[2] for message in server.messages] == ["shared-server-value"]


def test_shared_cache_uses_task_local_prompt_signatures():
    async def run():
        cache = BasicCache(CacheKeySetID)
        both_ready = asyncio.Event()
        ready_count = 0

        async def prompt(class_type, value):
            nonlocal ready_count
            dynprompt = DynamicPrompt({"same-id": {"class_type": class_type, "inputs": {}}})
            await cache.set_prompt(dynprompt, ["same-id"], None)
            cache.set_local("same-id", value)
            ready_count += 1
            if ready_count == 2:
                both_ready.set()
            await both_ready.wait()
            result = cache.get_local("same-id")
            cache.release_prompt()
            return result

        return await asyncio.gather(prompt("First", "first"), prompt("Second", "second"))

    assert asyncio.run(run()) == ["first", "second"]


def test_cooperative_executors_do_not_share_stateful_node_instances():
    instances = []
    both_ready = None
    ready = 0

    class StatefulProbe:
        def __init__(self):
            instances.append(self)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        RETURN_TYPES = ()
        FUNCTION = "run"

        async def run(self):
            nonlocal ready
            ready += 1
            if ready == 2:
                both_ready.set()
            await both_ready.wait()
            return ()

    async def run():
        nonlocal both_ready
        both_ready = asyncio.Event()
        server = Server()
        server.prompt_queue = SimpleNamespace(cooperative=True, is_cancelled=lambda prompt_id: False)
        cache_args = {"lru": 0, "ram": 0, "ram_inactive": 0}
        shared_outputs = CacheSet(cache_args=cache_args).outputs
        prompt = {"same-id": {"class_type": "StatefulInterleaveProbe", "inputs": {}}}
        first = PromptExecutor(server, cache_args=cache_args, shared_outputs=shared_outputs)
        second = PromptExecutor(server, cache_args=cache_args, shared_outputs=shared_outputs)
        with torch.inference_mode():
            await asyncio.gather(
                first.execute_async(prompt, "first", {}, ["same-id"]),
                second.execute_async(prompt, "second", {}, ["same-id"]),
            )

    nodes.NODE_CLASS_MAPPINGS["StatefulInterleaveProbe"] = StatefulProbe
    try:
        asyncio.run(run())
    finally:
        del nodes.NODE_CLASS_MAPPINGS["StatefulInterleaveProbe"]
    assert len(instances) == 2
    assert instances[0] is not instances[1]


def test_concurrent_cooperative_executors_do_not_clear_global_interrupt(monkeypatch):
    interrupt_calls = []
    monkeypatch.setattr(nodes, "interrupt_processing", lambda value=True: interrupt_calls.append(value))

    async def run():
        server = Server()
        server.prompt_queue = SimpleNamespace(cooperative=True, is_cancelled=lambda prompt_id: False)
        first = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        second = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        with torch.inference_mode():
            await asyncio.gather(
                first.execute_async({}, "first", {}, []),
                second.execute_async({}, "second", {}, []),
            )

    asyncio.run(run())
    assert interrupt_calls == []


def test_inference_mode_stays_enabled_across_interleaved_cooperative_tasks():
    observations = []
    both_ready = None
    ready = 0

    class Probe:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        RETURN_TYPES = ()
        FUNCTION = "run"

        async def run(self):
            nonlocal ready
            observations.append(torch.is_inference_mode_enabled())
            ready += 1
            if ready == 2:
                both_ready.set()
            await both_ready.wait()
            observations.append(torch.is_inference_mode_enabled())
            return ()

    async def run():
        nonlocal both_ready
        both_ready = asyncio.Event()
        server = Server()
        server.prompt_queue = SimpleNamespace(cooperative=True, is_cancelled=lambda prompt_id: False)
        prompt = {"probe": {"class_type": "InferenceModeInterleaveProbe", "inputs": {}}}
        first = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        second = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        with torch.inference_mode():
            await asyncio.gather(
                first.execute_async(prompt, "first", {}, ["probe"]),
                second.execute_async(prompt, "second", {}, ["probe"]),
            )
            assert torch.is_inference_mode_enabled()

    nodes.NODE_CLASS_MAPPINGS["InferenceModeInterleaveProbe"] = Probe
    try:
        asyncio.run(run())
    finally:
        del nodes.NODE_CLASS_MAPPINGS["InferenceModeInterleaveProbe"]
    assert observations == [True, True, True, True]
    assert not torch.is_inference_mode_enabled()


def test_serial_prompt_executor_still_owns_inference_mode():
    observations = []

    class Probe:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        RETURN_TYPES = ()
        FUNCTION = "run"

        def run(self):
            observations.append(torch.is_inference_mode_enabled())
            return ()

    async def run():
        server = Server()
        server.prompt_queue = SimpleNamespace(cooperative=False)
        prompt = {"probe": {"class_type": "InferenceModeSerialProbe", "inputs": {}}}
        executor = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        await executor.execute_async(prompt, "serial", {}, ["probe"])

    nodes.NODE_CLASS_MAPPINGS["InferenceModeSerialProbe"] = Probe
    try:
        asyncio.run(run())
    finally:
        del nodes.NODE_CLASS_MAPPINGS["InferenceModeSerialProbe"]
    assert observations == [True]
    assert not torch.is_inference_mode_enabled()


def test_classic_executor_clears_last_node_id_when_cancelled():
    started = asyncio.Event()

    class WaitForCancellation:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        RETURN_TYPES = ()
        FUNCTION = "run"

        async def run(self):
            started.set()
            await asyncio.Event().wait()

    async def run():
        server = Server()
        executor = PromptExecutor(server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
        prompt = {"probe": {"class_type": "ClassicCancellationProbe", "inputs": {}}}
        task = asyncio.create_task(executor.execute_async(prompt, "classic", {"client_id": "client"}, ["probe"]))
        await started.wait()
        assert server.last_node_id == "probe"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            assert False, "Cancelling the executor task must raise CancelledError"
        assert server.last_node_id is None

    nodes.NODE_CLASS_MAPPINGS["ClassicCancellationProbe"] = WaitForCancellation
    try:
        asyncio.run(run())
    finally:
        del nodes.NODE_CLASS_MAPPINGS["ClassicCancellationProbe"]
