"""Regression test for /system_stats leaking full sys.argv (including secrets
passed as CLI flags, such as --extra-model-paths-config paths) to any
unauthenticated client."""

import asyncio
import sys

import pytest
from aiohttp import web

from comfy.cli_args import args as cli_args

# Must be set before importing server/nodes, which probe cli_args.cpu at
# import time to pick a torch device. Restored after this module's tests
# via _restore_cli_args so other test modules aren't affected.
_original_cpu = cli_args.cpu
_original_front_end_root = cli_args.front_end_root
cli_args.cpu = True
cli_args.front_end_root = "."

import server


@pytest.fixture(autouse=True, scope="module")
def _restore_cli_args():
    yield
    cli_args.cpu = _original_cpu
    cli_args.front_end_root = _original_front_end_root


async def _get_system_stats_argv(aiohttp_client):
    loop = asyncio.get_running_loop()
    prompt_server = server.PromptServer(loop)
    route = next(r for r in prompt_server.routes if getattr(r, "path", None) == "/system_stats")

    app = web.Application()
    app.add_routes([route])
    client = await aiohttp_client(app)

    resp = await client.get("/system_stats")
    assert resp.status == 200

    data = await resp.json()
    return data["system"]["argv"]


@pytest.mark.asyncio
async def test_system_stats_does_not_leak_argv(aiohttp_client, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--extra-model-paths-config", "/secret/models.yaml", "--output-directory", "/secret/renders"],
    )

    assert await _get_system_stats_argv(aiohttp_client) == ["main.py"]


@pytest.mark.asyncio
async def test_system_stats_argv_empty(aiohttp_client, monkeypatch):
    monkeypatch.setattr(sys, "argv", [])

    assert await _get_system_stats_argv(aiohttp_client) == []
