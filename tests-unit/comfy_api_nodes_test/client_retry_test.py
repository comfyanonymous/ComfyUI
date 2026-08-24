from unittest.mock import MagicMock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.util import client as client_module  # noqa: E402
from comfy_api_nodes.util.client import ApiEndpoint, sync_op_raw  # noqa: E402


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body
        self.headers = {}

    async def json(self):
        return self._body

    async def text(self):
        return str(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    """A new instance is created per retry attempt (see _request_base), so the
    response iterator must be shared across instances, not per-instance."""

    def __init__(self, responses_iter):
        self._responses = responses_iter

    async def request(self, method, url, **kwargs):
        return next(self._responses)

    async def close(self):
        pass


def _patch_session(monkeypatch, responses):
    responses_iter = iter(responses)
    monkeypatch.setattr(
        client_module.aiohttp, "ClientSession", lambda *a, **kw: _FakeSession(responses_iter)
    )
    monkeypatch.setattr(client_module.request_logger, "log_request_response", lambda **kw: None)


@pytest.mark.asyncio
async def test_transient_401_is_retried(monkeypatch):
    """A transient 401 followed by a 200 should succeed instead of failing the whole request."""
    responses = [
        _FakeResponse(401, {"message": "Invalid Comfy API key"}),
        _FakeResponse(200, {"ok": True}),
    ]
    _patch_session(monkeypatch, responses)

    result = await sync_op_raw(
        MagicMock(),
        ApiEndpoint("https://example.com/foo", method="GET"),
        timeout=5,
        max_retries=3,
        retry_delay=0.01,
        retry_backoff=1.0,
        monitor_progress=False,
    )

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_persistent_401_still_fails(monkeypatch):
    """A genuinely invalid key must still raise, after exhausting retries."""
    responses = [_FakeResponse(401, {"message": "Invalid Comfy API key"})] * 10
    _patch_session(monkeypatch, responses)

    with pytest.raises(Exception, match="Unauthorized"):
        await sync_op_raw(
            MagicMock(),
            ApiEndpoint("https://example.com/foo", method="GET"),
            timeout=5,
            max_retries=2,
            retry_delay=0.01,
            retry_backoff=1.0,
            monitor_progress=False,
        )
