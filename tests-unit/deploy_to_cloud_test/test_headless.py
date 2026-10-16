"""Tests for the headless-use marker and the used_headlessly / deploy_to_cloud feature flags"""

import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution  # noqa: E402
import folder_paths  # noqa: E402
import server  # noqa: E402
from app import deploy_to_cloud  # noqa: E402
from comfy_api.feature_flags import get_server_features  # noqa: E402


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    path = tmp_path / "user"
    monkeypatch.setattr(folder_paths, "user_directory", str(path))
    return path


def marker(user_dir):
    return user_dir / deploy_to_cloud.HEADLESS_MARKER


def test_mark_creates_the_marker_in_the_user_directory(user_dir):
    deploy_to_cloud.mark_used_headlessly()
    assert marker(user_dir).is_file()
    assert deploy_to_cloud.headless_marker_path() == str(marker(user_dir))


def test_mark_is_idempotent(user_dir):
    deploy_to_cloud.mark_used_headlessly()
    deploy_to_cloud.mark_used_headlessly()
    assert os.listdir(user_dir) == [deploy_to_cloud.HEADLESS_MARKER]


def test_mark_swallows_os_error_and_warns_once(user_dir, monkeypatch, caplog):
    def fail(*a, **kw):
        raise PermissionError("read-only")
    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(deploy_to_cloud, "_marker_warning_logged", False)
    deploy_to_cloud.mark_used_headlessly()
    deploy_to_cloud.mark_used_headlessly()
    assert not marker(user_dir).exists()
    assert [r.getMessage() for r in caplog.records].count(f"Could not write headless marker {marker(user_dir)}: read-only") == 1


def test_used_headlessly_flag_flips_after_marking(user_dir):
    assert get_server_features()["used_headlessly"] is False
    deploy_to_cloud.mark_used_headlessly()
    assert get_server_features()["used_headlessly"] is True


@pytest.mark.parametrize("offline,expected", [(False, True), (True, False)])
def test_deploy_to_cloud_flag_follows_offline(monkeypatch, offline, expected):
    monkeypatch.setattr(args, "offline", offline)
    assert get_server_features()["deploy_to_cloud"] is expected


def test_offline_does_not_hide_used_headlessly(user_dir, monkeypatch):
    monkeypatch.setattr(args, "offline", True)
    deploy_to_cloud.mark_used_headlessly()
    features = get_server_features()
    assert features["used_headlessly"] is True
    assert features["deploy_to_cloud"] is False


def test_cli_feature_flag_cannot_override_live_flags(tmp_path):
    code = (
        "import comfy.options; comfy.options.enable_args_parsing(); "
        "from comfy_api.feature_flags import SERVER_FEATURE_FLAGS, get_server_features; "
        "f = get_server_features(); "
        "print('used_headlessly' in SERVER_FEATURE_FLAGS, 'deploy_to_cloud' in SERVER_FEATURE_FLAGS, "
        "f['used_headlessly'], f['deploy_to_cloud'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, "--base-directory", str(tmp_path), "--offline",
         "--feature-flag", "used_headlessly=true", "--feature-flag", "deploy_to_cloud=true"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.split()[-4:] == ["False", "False", "False", "False"]


@pytest.fixture
def validation(monkeypatch):
    result = {"valid": True}

    async def validate_prompt(prompt_id, prompt, partial_execution_targets):
        if result["valid"]:
            return (True, None, ["1"], {})
        return (False, {"type": "invalid", "message": "bad"}, [], {})
    monkeypatch.setattr(execution, "validate_prompt", validate_prompt)
    return result


@pytest_asyncio.fixture
async def client(validation):
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.prompt_queue = MagicMock()
    for route in prompt_server.routes:
        if isinstance(route, web.RouteDef) and route.path == "/prompt" and route.method == "POST":
            prompt_server.app.router.add_post("/api/prompt", route.handler)
    async with TestClient(TestServer(prompt_server.app)) as c:
        c.prompt_queue = prompt_server.prompt_queue
        yield c


PROMPT = {"1": {"class_type": "PrimitiveInt", "inputs": {"value": 1}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("body,headless", [
    ({}, True),
    ({"client_id": ""}, True),
    ({"client_id": None}, True),
    ({"client_id": "abc"}, False),
    ({"client_id": "", "extra_data": {"comfy_usage_source": "comfyui-frontend"}}, False),
    ({"extra_data": {"comfy_usage_source": "comfyui-frontend"}}, False),
    ({"client_id": "", "extra_data": {"comfy_usage_source": "my-script"}}, True),
])
async def test_post_prompt_marks_only_without_client_id(client, user_dir, body, headless):
    resp = await client.post("/api/prompt", json={"prompt": PROMPT, **body})
    assert resp.status == 200
    client.prompt_queue.put.assert_called_once()
    assert marker(user_dir).exists() is headless


@pytest.mark.asyncio
async def test_invalid_prompt_does_not_mark(client, user_dir, validation):
    validation["valid"] = False
    resp = await client.post("/api/prompt", json={"prompt": PROMPT})
    assert resp.status == 400
    client.prompt_queue.put.assert_not_called()
    assert not marker(user_dir).exists()


@pytest.mark.asyncio
async def test_marker_write_failure_still_queues_prompt(client, user_dir, monkeypatch):
    def fail(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(deploy_to_cloud.os, "makedirs", fail)
    resp = await client.post("/api/prompt", json={"prompt": PROMPT})
    assert resp.status == 200
    assert "prompt_id" in await resp.json()
    client.prompt_queue.put.assert_called_once()
    assert not marker(user_dir).exists()
