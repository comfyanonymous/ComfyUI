import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest_argv = sys.argv
sys.argv = [sys.argv[0], "--cpu"]
try:
    import main  # noqa: E402
finally:
    sys.argv = pytest_argv


PROMPT_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


class StopWorker(Exception):
    pass


class SingleItemQueue:
    def __init__(self, item):
        self.item = item
        self.get_count = 0
        self.task_done = Mock()

    def get(self, timeout):
        self.get_count += 1
        if self.get_count == 1:
            return self.item, 1
        raise StopWorker

    def get_flags(self):
        return {}


@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop)
    thread.start()
    assert started.wait(timeout=1)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    loop.close()


@pytest.fixture
def worker_setup(monkeypatch, loop_thread):
    sensitive = {"auth_token_comfy_org": "private"}
    prompt = {"1": {"class_type": "TestNode", "inputs": {}}}
    extra_data = {"client_id": "client-1"}
    outputs = ["1"]
    item = (0, PROMPT_ID, prompt, extra_data, outputs, sensitive)
    queue = SingleItemQueue(item)
    executor = Mock()
    executor.history_result = {"outputs": {}}
    executor.success = True
    executor.status_messages = []
    executor_factory = Mock(return_value=executor)
    pause = Mock()
    server_instance = SimpleNamespace(
        loop=loop_thread,
        prompt_execution_start_hook=None,
        prompt_execution_complete_hook=None,
        last_prompt_id=None,
        client_id=None,
        send_sync=Mock(),
    )

    monkeypatch.setattr(main, "args", SimpleNamespace(
        cache_classic=False,
        cache_none=True,
        cache_lru=0,
        cache_ram=[],
    ))
    monkeypatch.setattr(main.execution, "PromptExecutor", executor_factory)
    monkeypatch.setattr(main.asset_seeder, "pause", pause)
    monkeypatch.setattr(main.asset_seeder, "is_disabled", Mock(return_value=True))

    return SimpleNamespace(
        queue=queue,
        executor=executor,
        pause=pause,
        server=server_instance,
        sensitive=sensitive,
        prompt=prompt,
        extra_data=extra_data,
        outputs=outputs,
    )


def run_one_prompt(setup):
    with pytest.raises(StopWorker):
        main.prompt_worker(setup.queue, setup.server)


def test_unset_hooks_preserve_native_execution(worker_setup):
    run_one_prompt(worker_setup)

    worker_setup.pause.assert_called_once_with()
    worker_setup.executor.execute.assert_called_once_with(
        worker_setup.prompt,
        PROMPT_ID,
        {"client_id": "client-1", "auth_token_comfy_org": "private"},
        worker_setup.outputs,
    )
    worker_setup.queue.task_done.assert_called_once()


def test_completion_is_not_called_without_a_successful_start_hook(worker_setup):
    completed = Mock()

    async def complete_hook(context):
        completed(context)

    worker_setup.server.prompt_execution_complete_hook = complete_hook

    run_one_prompt(worker_setup)

    worker_setup.executor.execute.assert_called_once()
    completed.assert_not_called()


def test_hooks_run_on_server_loop_in_order_with_exact_private_context(worker_setup):
    events = []
    contexts = []
    loops = []

    async def start_hook(context):
        loops.append(asyncio.get_running_loop())
        contexts.append(context)
        events.append("start")

    async def complete_hook(context):
        loops.append(asyncio.get_running_loop())
        contexts.append(context)
        events.append("complete")

    worker_setup.server.prompt_execution_start_hook = start_hook
    worker_setup.server.prompt_execution_complete_hook = complete_hook
    worker_setup.executor.execute.side_effect = lambda *args: events.append("execute")

    run_one_prompt(worker_setup)

    assert events == ["start", "execute", "complete"]
    assert loops == [worker_setup.server.loop, worker_setup.server.loop]
    assert len(contexts) == 2
    for context in contexts:
        assert set(context) == {"prompt_id", "sensitive"}
        assert context["prompt_id"] == PROMPT_ID
        assert context["sensitive"] is worker_setup.sensitive


def test_native_execution_blocks_until_start_hook_returns(worker_setup):
    hook_entered = threading.Event()
    release_hook = threading.Event()
    worker_error = []

    async def start_hook(context):
        hook_entered.set()
        await asyncio.to_thread(release_hook.wait)

    def run_worker():
        try:
            main.prompt_worker(worker_setup.queue, worker_setup.server)
        except StopWorker:
            pass
        except BaseException as error:
            worker_error.append(error)

    worker_setup.server.prompt_execution_start_hook = start_hook
    thread = threading.Thread(target=run_worker)
    thread.start()

    assert hook_entered.wait(timeout=1)
    worker_setup.pause.assert_not_called()
    worker_setup.executor.execute.assert_not_called()

    release_hook.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert worker_error == []
    worker_setup.executor.execute.assert_called_once()


def test_completion_runs_once_after_ordinary_execution_error(worker_setup):
    completed = Mock()

    async def start_hook(context):
        return None

    async def complete_hook(context):
        completed(context)

    def ordinary_error(*args):
        worker_setup.executor.success = False
        worker_setup.executor.status_messages = ["execution_interrupted"]

    worker_setup.server.prompt_execution_start_hook = start_hook
    worker_setup.server.prompt_execution_complete_hook = complete_hook
    worker_setup.executor.execute.side_effect = ordinary_error

    run_one_prompt(worker_setup)

    completed.assert_called_once()
    assert worker_setup.queue.task_done.call_args.kwargs["status"].status_str == "error"


def test_completion_runs_once_when_native_execution_raises(worker_setup):
    execution_error = RuntimeError("unexpected execution failure")
    completed = Mock()

    async def start_hook(context):
        return None

    async def complete_hook(context):
        completed(context)

    worker_setup.server.prompt_execution_start_hook = start_hook
    worker_setup.server.prompt_execution_complete_hook = complete_hook
    worker_setup.executor.execute.side_effect = execution_error

    with pytest.raises(RuntimeError) as raised:
        main.prompt_worker(worker_setup.queue, worker_setup.server)

    assert raised.value is execution_error
    completed.assert_called_once()
    worker_setup.queue.task_done.assert_not_called()


def test_start_hook_failure_skips_execution_and_completion(worker_setup):
    start_error = RuntimeError("lease unavailable")
    completed = Mock()

    async def start_hook(context):
        raise start_error

    async def complete_hook(context):
        completed(context)

    worker_setup.server.prompt_execution_start_hook = start_hook
    worker_setup.server.prompt_execution_complete_hook = complete_hook

    with pytest.raises(RuntimeError) as raised:
        main.prompt_worker(worker_setup.queue, worker_setup.server)

    assert raised.value is start_error
    worker_setup.pause.assert_not_called()
    worker_setup.executor.execute.assert_not_called()
    completed.assert_not_called()
    worker_setup.queue.task_done.assert_not_called()


def test_completion_hook_failure_propagates(worker_setup):
    completion_error = RuntimeError("lease release failed")

    async def start_hook(context):
        return None

    async def complete_hook(context):
        raise completion_error

    worker_setup.server.prompt_execution_start_hook = start_hook
    worker_setup.server.prompt_execution_complete_hook = complete_hook

    with pytest.raises(RuntimeError) as raised:
        main.prompt_worker(worker_setup.queue, worker_setup.server)

    assert raised.value is completion_error
    worker_setup.executor.execute.assert_called_once()
    worker_setup.queue.task_done.assert_not_called()


def test_refused_start_hook_fails_the_prompt_and_keeps_the_worker_alive(worker_setup):
    completed = Mock()
    start_error = RuntimeError("image bundle validation failed")

    async def start_hook(context):
        raise start_error

    async def complete_hook(context):
        completed(context)

    worker_setup.server.prompt_execution_start_hook = start_hook
    worker_setup.server.prompt_execution_complete_hook = complete_hook

    run_one_prompt(worker_setup)

    worker_setup.pause.assert_not_called()
    worker_setup.executor.execute.assert_not_called()
    completed.assert_not_called()
    status = worker_setup.queue.task_done.call_args.kwargs["status"]
    assert status.status_str == "error"
    assert status.completed is False
    assert worker_setup.queue.task_done.call_args.kwargs["history_result"] == {}


def test_worker_survives_refused_leases_and_executes_the_next_prompt(worker_setup):
    executed = []
    refused = True

    async def start_hook(context):
        if refused:
            raise RuntimeError("refused once")
        executed.append(context["prompt_id"])

    worker_setup.server.prompt_execution_start_hook = start_hook
    worker_setup.executor.execute.side_effect = (
        lambda prompt, prompt_id, *args: executed.append(f"executed-{prompt_id}")
    )

    class TwoItemQueue(SingleItemQueue):
        def __init__(self, first, second):
            self.items = [(first, 1), (second, 2)]
            self.index = 0

        def get(self, timeout):
            self.index += 1
            if self.index == 1:
                return self.items[0]
            if self.index == 2:
                return self.items[1]
            raise StopWorker

    second_id = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    queue = TwoItemQueue(
        (0, PROMPT_ID, {"1": {}}, {}, ["1"], {}),
        (0, second_id := second_prompt_id(), {"1": {}}, {}, ["1"], {}),
    )
    worker_setup.queue = queue

    run_one_prompt(worker_setup)

    assert executed == [second_id]
    assert queue.task_done.call_count == 2


def second_prompt_id():
    return "cccccccc-dddd-4eee-8fff-000000000001"
