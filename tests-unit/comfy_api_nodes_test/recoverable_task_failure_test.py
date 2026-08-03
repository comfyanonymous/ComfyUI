import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy_api_nodes.util.client as client  # noqa: E402
import comfy_api_nodes.nodes_bytedance as nodes_bytedance  # noqa: E402
from comfy_api_nodes.util import ApiEndpoint, TaskFailedError  # noqa: E402
from comfy_api_nodes.nodes_bytedance import _poll_seedance2_task  # noqa: E402
from comfy_execution.graph_utils import RecoverableNodeError  # noqa: E402

MODERATION_CODE = "OutputVideoSensitiveContentDetected.PolicyViolation"


class _DummyNode:
    hidden = None


@pytest.mark.asyncio
async def test_poll_op_raw_raises_task_failed_error_with_payload(monkeypatch):
    payload = {"id": "task-1", "status": "failed", "error": {"code": MODERATION_CODE, "message": "moderated"}}

    async def fake_sync_op_raw(*args, **kwargs):
        return payload

    monkeypatch.setattr(client, "sync_op_raw", fake_sync_op_raw)

    with pytest.raises(TaskFailedError) as exc_info:
        await client.poll_op_raw(
            _DummyNode,
            poll_endpoint=ApiEndpoint(path="/test/task-1"),
            status_extractor=lambda r: r.get("status"),
        )

    assert exc_info.value.response == payload
    assert MODERATION_CODE in str(exc_info.value)


@pytest.mark.asyncio
async def test_poll_op_raw_wraps_generic_errors(monkeypatch):
    async def fake_sync_op_raw(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(client, "sync_op_raw", fake_sync_op_raw)

    with pytest.raises(Exception) as exc_info:
        await client.poll_op_raw(
            _DummyNode,
            poll_endpoint=ApiEndpoint(path="/test/task-1"),
            status_extractor=lambda r: r.get("status"),
        )

    assert not isinstance(exc_info.value, TaskFailedError)
    assert "Polling aborted due to error" in str(exc_info.value)


def _patch_poll_op_failure(monkeypatch, payload):
    async def fake_poll_op(*args, **kwargs):
        raise TaskFailedError(f"Task failed: {payload}", payload)

    monkeypatch.setattr(nodes_bytedance, "poll_op", fake_poll_op)


@pytest.mark.asyncio
async def test_seedance2_moderation_code_is_recoverable(monkeypatch):
    payload = {"id": "task-1", "status": "failed", "error": {"code": MODERATION_CODE, "message": "moderated"}}
    _patch_poll_op_failure(monkeypatch, payload)

    with pytest.raises(RecoverableNodeError) as exc_info:
        await _poll_seedance2_task(_DummyNode, "task-1", None)

    assert isinstance(exc_info.value.__cause__, TaskFailedError)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_field", [
    {"code": "InternalError", "message": "server error"},
    {"code": MODERATION_CODE.lower(), "message": "case variant must not match"},
    {"code": "OutputVideoSensitiveContentDetected", "message": "prefix must not match"},
    {"message": "no code"},
    "not-a-dict",
    None,
])
async def test_seedance2_other_failures_stay_terminal(monkeypatch, error_field):
    payload = {"id": "task-1", "status": "failed"}
    if error_field is not None:
        payload["error"] = error_field
    _patch_poll_op_failure(monkeypatch, payload)

    with pytest.raises(TaskFailedError):
        await _poll_seedance2_task(_DummyNode, "task-1", None)


@pytest.mark.asyncio
async def test_seedance2_success_passthrough(monkeypatch):
    sentinel = object()

    async def fake_poll_op(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(nodes_bytedance, "poll_op", fake_poll_op)

    assert await _poll_seedance2_task(_DummyNode, "task-1", None) is sentinel
