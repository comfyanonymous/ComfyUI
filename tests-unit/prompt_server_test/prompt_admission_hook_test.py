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
