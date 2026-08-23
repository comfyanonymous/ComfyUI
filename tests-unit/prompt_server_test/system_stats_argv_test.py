"""Regression test for /system_stats leaking full sys.argv (including secrets
passed as CLI flags, such as --extra-model-paths-config paths) to any
unauthenticated client."""

import asyncio
import sys

import pytest
from aiohttp import web

from comfy.cli_args import args as cli_args

cli_args.cpu = True
cli_args.front_end_root = "."

import server


@pytest.mark.asyncio
async def test_system_stats_does_not_leak_argv(aiohttp_client, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--extra-model-paths-config", "/secret/models.yaml", "--output-directory", "/secret/renders"],
    )

    loop = asyncio.get_running_loop()
    prompt_server = server.PromptServer(loop)
    route = next(r for r in prompt_server.routes if getattr(r, "path", None) == "/system_stats")

    app = web.Application()
    app.add_routes([route])
    client = await aiohttp_client(app)

    resp = await client.get("/system_stats")
    assert resp.status == 200

    data = await resp.json()
    assert data["system"]["argv"] == ["main.py"]
