import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

import server


PROMPT_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


@pytest.fixture
def prompt_server(monkeypatch):
    prompt_queue = Mock()
    node_replace_manager = Mock()

    monkeypatch.setattr(server, "UserManager", Mock(return_value=Mock()))
    monkeypatch.setattr(server, "ModelFileManager", Mock(return_value=Mock()))
    monkeypatch.setattr(server, "CustomNodeManager", Mock(return_value=Mock()))
    monkeypatch.setattr(server, "SubgraphManager", Mock(return_value=Mock()))
    monkeypatch.setattr(server, "NodeReplaceManager", Mock(return_value=node_replace_manager))
    monkeypatch.setattr(server, "InternalRoutes", Mock(return_value=Mock()))
    monkeypatch.setattr(server.execution, "PromptQueue", Mock(return_value=prompt_queue))
    monkeypatch.setattr(server.FrontendManager, "init_frontend", Mock(return_value="/tmp"))
    monkeypatch.setattr(server, "register_assets_routes", Mock())
    monkeypatch.setattr(server.asset_seeder, "disable", Mock())

    loop = asyncio.new_event_loop()
    prompt_server = server.PromptServer(loop)
    prompt_route = next(route for route in prompt_server.routes if route.method == "POST" and route.path == "/prompt")
    yield prompt_server, prompt_route.handler, prompt_queue, node_replace_manager
    loop.close()


def prompt_request(body, headers=None):
    request = Mock()
    request.json = AsyncMock(return_value=body)
    request.headers = headers or {}
    return request


def response_json(response):
    return json.loads(response.body)


def route_handler(prompt_server, method, path):
    return next(route.handler for route in prompt_server.routes if route.method == method and route.path == path)


def queue_item(prompt_id=PROMPT_ID, sensitive=None, number=0):
    return (number, prompt_id, {}, {}, [], sensitive or {})


def test_prompt_queue_capture_operations_return_exact_removed_items():
    queue_server = Mock()
    prompt_queue = server.execution.PromptQueue(queue_server)
    first = queue_item(number=1)
    second = queue_item("bbbbbbbb-cccc-4ddd-8eee-ffffffffffff", number=2)
    prompt_queue.queue = [first, second]

    removed = prompt_queue.delete_queue_item_with_item(lambda item: item[1] == PROMPT_ID)

    assert removed is first
    assert prompt_queue.queue == [second]
    assert prompt_queue.delete_queue_item(lambda item: item[1] == "unknown") is False

    cleared = prompt_queue.wipe_queue_with_items()

    assert cleared == [second]
    assert cleared[0] is second
    assert prompt_queue.queue == []
    assert prompt_queue.wipe_queue() is None


@pytest.mark.asyncio
async def test_valid_prompt_without_hook_enqueues_once_unchanged(prompt_server, monkeypatch):
    prompt_server, post_prompt, prompt_queue, _ = prompt_server
    prompt = {"1": {"class_type": "TestNode", "inputs": {}}}
    outputs = ["1"]
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(True, None, outputs, {})))
    monkeypatch.setattr(server.time, "time", Mock(return_value=123.456))

    response = await post_prompt(prompt_request({
        "prompt": prompt,
        "prompt_id": PROMPT_ID,
        "number": 7,
        "extra_data": {"custom": "value"},
        "client_id": "client-1",
    }))

    assert prompt_server.prompt_admission_hook is None
    assert prompt_server.prompt_execution_start_hook is None
    assert prompt_server.prompt_execution_complete_hook is None
    assert response.status == 200
    assert response_json(response) == {"prompt_id": PROMPT_ID, "number": 7.0, "node_errors": {}}
    prompt_queue.put.assert_called_once_with((
        7.0,
        PROMPT_ID,
        prompt,
        {"custom": "value", "client_id": "client-1", "create_time": 123456},
        outputs,
        {},
    ))


@pytest.mark.asyncio
async def test_invalid_prompt_never_invokes_hook_or_enqueues(prompt_server, monkeypatch):
    prompt_server, post_prompt, prompt_queue, _ = prompt_server
    error = {"type": "prompt_outputs_failed_validation", "message": "invalid", "details": "", "extra_info": {}}
    prompt_server.prompt_admission_hook = AsyncMock()
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(False, error, [], {"1": {}})))

    response = await post_prompt(prompt_request({"prompt": {}, "prompt_id": PROMPT_ID}))

    assert response.status == 400
    assert response_json(response) == {"error": error, "node_errors": {"1": {}}}
    prompt_server.prompt_admission_hook.assert_not_awaited()
    prompt_queue.put.assert_not_called()


@pytest.mark.asyncio
async def test_async_hook_is_awaited_once_before_enqueue_with_full_context(prompt_server, monkeypatch):
    prompt_server, post_prompt, prompt_queue, node_replace_manager = prompt_server
    prompt_server.number = 5
    prompt = {"1": {"class_type": "TestNode", "inputs": {}}}
    outputs = ["1"]
    calls = []
    context_seen = None

    async def admission_hook(context):
        nonlocal context_seen
        await asyncio.sleep(0)
        calls.append("hook")
        context_seen = context

    prompt_server.prompt_admission_hook = AsyncMock(side_effect=admission_hook)
    prompt_queue.put.side_effect = lambda item: calls.append("put")
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(True, None, outputs, {})))
    monkeypatch.setattr(server.time, "time", Mock(return_value=123.456))

    response = await post_prompt(prompt_request({
        "prompt": prompt,
        "prompt_id": PROMPT_ID,
        "front": True,
        "extra_data": {"auth_token_comfy_org": "secret"},
        "client_id": "client-1",
    }, headers={"Comfy-Usage-Source": "halo"}))

    assert response.status == 200
    assert calls == ["hook", "put"]
    prompt_server.prompt_admission_hook.assert_awaited_once()
    node_replace_manager.apply_replacements.assert_called_once_with(prompt)
    assert context_seen == {
        "prompt_id": PROMPT_ID,
        "prompt": prompt,
        "extra_data": {"client_id": "client-1", "comfy_usage_source": "halo", "create_time": 123456},
        "sensitive": {"auth_token_comfy_org": "secret"},
        "number": -5,
        "front": True,
        "queue_controls": {"front": True},
        "outputs_to_execute": outputs,
    }


@pytest.mark.asyncio
async def test_hook_rejection_returns_prompt_error_without_enqueue(prompt_server, monkeypatch):
    prompt_server, post_prompt, prompt_queue, _ = prompt_server
    error = {"type": "halo_admission_rejected", "message": "not admitted", "details": "", "extra_info": {}}
    prompt_server.prompt_admission_hook = AsyncMock(return_value=error)
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(True, None, ["1"], {"1": {}})))

    response = await post_prompt(prompt_request({"prompt": {}, "prompt_id": PROMPT_ID}))

    assert response.status == 400
    assert response_json(response) == {"error": error, "node_errors": {"1": {}}}
    prompt_server.prompt_admission_hook.assert_awaited_once()
    prompt_queue.put.assert_not_called()


@pytest.mark.asyncio
async def test_hook_exception_propagates_without_enqueue(prompt_server, monkeypatch):
    prompt_server, post_prompt, prompt_queue, _ = prompt_server
    prompt_server.prompt_admission_hook = AsyncMock(side_effect=RuntimeError("hook failed"))
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(True, None, ["1"], {})))

    with pytest.raises(RuntimeError, match="hook failed"):
        await post_prompt(prompt_request({"prompt": {}, "prompt_id": PROMPT_ID}))

    prompt_server.prompt_admission_hook.assert_awaited_once()
    prompt_queue.put.assert_not_called()


@pytest.mark.asyncio
async def test_successful_hook_preserves_prompt_response_and_enqueue(prompt_server, monkeypatch):
    prompt_server, post_prompt, prompt_queue, _ = prompt_server
    prompt_server.prompt_admission_hook = AsyncMock(return_value=None)
    outputs = ["1"]
    node_errors = {"2": {"warning": "retained"}}
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(True, None, outputs, node_errors)))

    response = await post_prompt(prompt_request({"prompt": {}, "prompt_id": PROMPT_ID, "number": 3}))

    assert response.status == 200
    assert response_json(response) == {"prompt_id": PROMPT_ID, "number": 3.0, "node_errors": node_errors}
    prompt_server.prompt_admission_hook.assert_awaited_once()
    assert prompt_server.prompt_admission_hook.await_args.args[0]["queue_controls"] == {"number": 3}
    prompt_queue.put.assert_called_once()
    assert prompt_queue.put.call_args.args[0][:3] == (3.0, PROMPT_ID, {})


@pytest.mark.asyncio
async def test_enqueue_failure_after_admission_is_compensated_once(prompt_server, monkeypatch):
    prompt_server, post_prompt, prompt_queue, _ = prompt_server
    enqueue_error = RuntimeError("enqueue failed")
    prompt_server.prompt_admission_hook = AsyncMock(return_value=None)
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.put.side_effect = enqueue_error
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(True, None, ["1"], {})))

    with pytest.raises(RuntimeError) as raised:
        await post_prompt(prompt_request({
            "prompt": {},
            "prompt_id": PROMPT_ID,
            "extra_data": {"auth_token_comfy_org": "private"},
        }))

    assert raised.value is enqueue_error
    prompt_server.prompt_cancellation_hook.assert_awaited_once_with({
        "prompt_id": PROMPT_ID,
        "sensitive": {"auth_token_comfy_org": "private"},
        "reason": "enqueue_failed",
    })
    assert prompt_queue.put.call_count == 1


@pytest.mark.asyncio
async def test_enqueue_and_compensation_failures_remain_diagnosable(prompt_server, monkeypatch, caplog):
    prompt_server, post_prompt, prompt_queue, _ = prompt_server
    enqueue_error = RuntimeError("enqueue failed")
    cancellation_error = RuntimeError("cancellation failed")
    prompt_server.prompt_admission_hook = AsyncMock(return_value=None)
    prompt_server.prompt_cancellation_hook = AsyncMock(side_effect=cancellation_error)
    prompt_queue.put.side_effect = enqueue_error
    monkeypatch.setattr(server.execution, "validate_prompt", AsyncMock(return_value=(True, None, ["1"], {})))

    with pytest.raises(RuntimeError) as raised:
        await post_prompt(prompt_request({
            "prompt": {},
            "prompt_id": PROMPT_ID,
            "extra_data": {"auth_token_comfy_org": "private"},
        }))

    assert raised.value is cancellation_error
    assert raised.value.__context__ is enqueue_error
    assert "private" not in str(raised.value)
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_queue_delete_without_hook_preserves_success_response(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    prompt_server.prompt_cancellation_hook = None
    prompt_queue.delete_queue_item_with_item.return_value = queue_item()
    post_queue = route_handler(prompt_server, "POST", "/queue")

    response = await post_queue(prompt_request({"delete": [PROMPT_ID]}))

    assert response.status == 200
    prompt_queue.delete_queue_item_with_item.assert_called_once()


@pytest.mark.asyncio
async def test_queue_clear_without_hook_preserves_success_response(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    prompt_server.prompt_cancellation_hook = None
    prompt_queue.wipe_queue_with_items.return_value = [queue_item()]
    post_queue = route_handler(prompt_server, "POST", "/queue")

    response = await post_queue(prompt_request({"clear": True}))

    assert response.status == 200
    prompt_queue.wipe_queue_with_items.assert_called_once_with()


@pytest.mark.asyncio
async def test_queue_delete_awaits_exact_private_context_once(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    sensitive = {"auth_token_comfy_org": "private"}
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.delete_queue_item_with_item.return_value = queue_item(sensitive=sensitive)
    post_queue = route_handler(prompt_server, "POST", "/queue")

    response = await post_queue(prompt_request({"delete": [PROMPT_ID]}))

    assert response.status == 200
    prompt_server.prompt_cancellation_hook.assert_awaited_once_with({
        "prompt_id": PROMPT_ID,
        "sensitive": sensitive,
        "reason": "queue_delete",
    })
    assert response.body is None


@pytest.mark.asyncio
async def test_queue_clear_awaits_once_for_every_removed_item(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    second_id = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    removed = [queue_item(sensitive={"ticket": "one"}), queue_item(second_id, {"ticket": "two"}, 1)]
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.wipe_queue_with_items.return_value = removed
    post_queue = route_handler(prompt_server, "POST", "/queue")

    response = await post_queue(prompt_request({"clear": True}))

    assert response.status == 200
    assert prompt_server.prompt_cancellation_hook.await_count == 2
    assert [call.args[0] for call in prompt_server.prompt_cancellation_hook.await_args_list] == [
        {"prompt_id": PROMPT_ID, "sensitive": {"ticket": "one"}, "reason": "queue_clear"},
        {"prompt_id": second_id, "sensitive": {"ticket": "two"}, "reason": "queue_clear"},
    ]


@pytest.mark.asyncio
async def test_queue_delete_noop_does_not_invoke_cancellation(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.delete_queue_item_with_item.return_value = None
    post_queue = route_handler(prompt_server, "POST", "/queue")

    response = await post_queue(prompt_request({"delete": ["unknown"]}))

    assert response.status == 200
    prompt_server.prompt_cancellation_hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_queue_callback_failure_surfaces_without_reinsertion_or_private_leak(prompt_server, caplog):
    prompt_server, _, prompt_queue, _ = prompt_server
    sensitive = {"ticket": "do-not-leak"}
    prompt_server.prompt_cancellation_hook = AsyncMock(side_effect=RuntimeError("cancellation failed"))
    prompt_queue.delete_queue_item_with_item.return_value = queue_item(sensitive=sensitive)
    post_queue = route_handler(prompt_server, "POST", "/queue")

    with pytest.raises(RuntimeError, match="cancellation failed"):
        await post_queue(prompt_request({"delete": [PROMPT_ID]}))

    prompt_queue.put.assert_not_called()
    assert "do-not-leak" not in caplog.text


@pytest.mark.asyncio
async def test_queue_clear_invokes_every_removed_item_when_one_callback_fails(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    removed = [queue_item(), queue_item("bbbbbbbb-cccc-4ddd-8eee-ffffffffffff")]
    prompt_server.prompt_cancellation_hook = AsyncMock(
        side_effect=[RuntimeError("first cancellation failed"), None]
    )
    prompt_queue.wipe_queue_with_items.return_value = removed
    post_queue = route_handler(prompt_server, "POST", "/queue")

    with pytest.raises(RuntimeError, match="first cancellation failed"):
        await post_queue(prompt_request({"clear": True}))

    assert prompt_server.prompt_cancellation_hook.await_count == 2
    prompt_queue.put.assert_not_called()


@pytest.mark.asyncio
async def test_single_job_cancel_awaits_pending_callback(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    item = queue_item()
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.get_current_queue.return_value = ([], [item])
    prompt_queue.get_history.return_value = {}
    prompt_queue.delete_queue_item_with_item.return_value = item
    cancel_job = route_handler(prompt_server, "POST", "/api/jobs/{job_id}/cancel")
    request = Mock(match_info={"job_id": PROMPT_ID})

    response = await cancel_job(request)

    assert response_json(response) == {"cancelled": True}
    prompt_server.prompt_cancellation_hook.assert_awaited_once_with({
        "prompt_id": PROMPT_ID,
        "sensitive": {},
        "reason": "queue_delete",
    })


@pytest.mark.asyncio
async def test_batch_job_cancel_awaits_each_pending_callback(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    second_id = "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb"
    items = [queue_item(), queue_item(second_id)]
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.get_current_queue.side_effect = [([], items), ([], [items[1]])]
    prompt_queue.get_history.return_value = {}
    prompt_queue.delete_queue_item_with_item.side_effect = items
    cancel_jobs = route_handler(prompt_server, "POST", "/api/jobs/cancel")

    response = await cancel_jobs(prompt_request({"job_ids": [PROMPT_ID, second_id]}))

    assert response_json(response) == {"cancelled": True}
    assert prompt_server.prompt_cancellation_hook.await_count == 2
    assert [call.args[0]["prompt_id"] for call in prompt_server.prompt_cancellation_hook.await_args_list] == [
        PROMPT_ID,
        second_id,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_state,history", [(([], []), {}), (([], []), {PROMPT_ID: {}})])
async def test_unknown_and_completed_job_cancel_do_not_invoke_callback(prompt_server, queue_state, history):
    prompt_server, _, prompt_queue, _ = prompt_server
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.get_current_queue.return_value = queue_state
    prompt_queue.get_history.return_value = history
    cancel_job = route_handler(prompt_server, "POST", "/api/jobs/{job_id}/cancel")

    response = await cancel_job(Mock(match_info={"job_id": PROMPT_ID}))

    assert response_json(response) == {"cancelled": False}
    prompt_server.prompt_cancellation_hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_running_job_cancel_never_uses_queued_callback(prompt_server):
    prompt_server, _, prompt_queue, _ = prompt_server
    prompt_server.prompt_cancellation_hook = AsyncMock(return_value=None)
    prompt_queue.get_current_queue.return_value = ([queue_item()], [])
    prompt_queue.get_history.return_value = {}
    prompt_queue.interrupt_if_running.return_value = True
    cancel_job = route_handler(prompt_server, "POST", "/api/jobs/{job_id}/cancel")

    response = await cancel_job(Mock(match_info={"job_id": PROMPT_ID}))

    assert response_json(response) == {"cancelled": True}
    prompt_queue.interrupt_if_running.assert_called_once_with(PROMPT_ID)
    prompt_server.prompt_cancellation_hook.assert_not_awaited()
