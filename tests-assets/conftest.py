import asyncio
import contextlib
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import aiohttp
import pytest
import pytest_asyncio
import subprocess


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_base_dirs(root: Path) -> None:
    for sub in ("models", "custom_nodes", "input", "output", "temp", "user"):
        (root / sub).mkdir(parents=True, exist_ok=True)


async def _wait_http_ready(base: str, session: aiohttp.ClientSession, timeout: float = 90.0) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            async with session.get(base + "/api/assets") as r:
                if r.status in (200, 400):
                    return
        except Exception as e:
            last_err = e
        await asyncio.sleep(0.25)
    raise RuntimeError(f"ComfyUI HTTP did not become ready: {last_err}")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def comfy_tmp_base_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="comfyui-assets-tests-"))
    _make_base_dirs(tmp)
    yield tmp
    with contextlib.suppress(Exception):
        for p in sorted(tmp.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
        for p in sorted(tmp.glob("**/*"), reverse=True):
            with contextlib.suppress(Exception):
                p.rmdir()
        tmp.rmdir()


@pytest.fixture(scope="session")
def comfy_url_and_proc(comfy_tmp_base_dir: Path):
    """
    Boot ComfyUI subprocess with:
      - sandbox base dir
      - sqlite memory DB
      - autoscan disabled
    Returns (base_url, process, port)
    """
    port = _free_port()
    db_url = "sqlite+aiosqlite:///:memory:"

    logs_dir = comfy_tmp_base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_log = open(logs_dir / "stdout.log", "w", buffering=1)
    err_log = open(logs_dir / "stderr.log", "w", buffering=1)

    comfy_root = Path(__file__).resolve().parent.parent
    if not (comfy_root / "main.py").is_file():
        raise FileNotFoundError(f"main.py not found under {comfy_root}")

    proc = subprocess.Popen(
        args=[
            sys.executable,
            "main.py",
            f"--base-directory={str(comfy_tmp_base_dir)}",
            f"--database-url={db_url}",
            "--disable-assets-autoscan",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=out_log,
        stderr=err_log,
        cwd=str(comfy_root),
        env={**os.environ},
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        async def _probe():
            async with aiohttp.ClientSession() as s:
                await _wait_http_ready(base_url, s, timeout=90.0)

        asyncio.run(_probe())
        yield base_url, proc, port
    except Exception as e:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=10)
        raise RuntimeError(f"ComfyUI did not become ready: {e}")

    if proc and proc.poll() is None:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=15)
    out_log.close()
    err_log.close()


@pytest_asyncio.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        yield s


@pytest.fixture
def api_base(comfy_url_and_proc) -> str:
    base_url, _proc, _port = comfy_url_and_proc
    return base_url


async def _post_multipart_asset(
    session: aiohttp.ClientSession,
    base: str,
    *,
    name: str,
    tags: list[str],
    meta: dict,
    data: bytes,
    extra_fields: dict | None = None,
) -> tuple[int, dict]:
    form = aiohttp.FormData()
    form.add_field("file", data, filename=name, content_type="application/octet-stream")
    form.add_field("tags", json.dumps(tags))
    form.add_field("name", name)
    form.add_field("user_metadata", json.dumps(meta))
    if extra_fields:
        for k, v in extra_fields.items():
            form.add_field(k, v)
    async with session.post(base + "/api/assets", data=form) as r:
        body = await r.json()
        return r.status, body


@pytest.fixture
def make_asset_bytes() -> Callable[[str, int], bytes]:
    def _make(name: str, size: int = 8192) -> bytes:
        seed = sum(ord(c) for c in name) % 251
        return bytes((i * 31 + seed) % 256 for i in range(size))
    return _make


@pytest_asyncio.fixture
async def asset_factory(http: aiohttp.ClientSession, api_base: str):
    """
    Returns create(name, tags, meta, data) -> response dict
    Tracks created ids and deletes them after the test.
    """
    created: list[str] = []

    async def create(name: str, tags: list[str], meta: dict, data: bytes) -> dict:
        status, body = await _post_multipart_asset(http, api_base, name=name, tags=tags, meta=meta, data=data)
        assert status in (200, 201), body
        created.append(body["id"])
        return body

    yield create

    # cleanup by id
    for aid in created:
        with contextlib.suppress(Exception):
            async with http.delete(f"{api_base}/api/assets/{aid}") as r:
                await r.read()


@pytest_asyncio.fixture
async def seeded_asset(request: pytest.FixtureRequest, http: aiohttp.ClientSession, api_base: str) -> dict:
    """
    Upload one asset into models/checkpoints/unit-tests/<name>.
    Returns response dict with id, asset_hash, tags, etc.
    """
    name = "unit_1_example.safetensors"
    p = getattr(request, "param", {}) or {}
    tags: Optional[list[str]] = p.get("tags")
    if tags is None:
        tags = ["models", "checkpoints", "unit-tests", "alpha"]
    meta = {"purpose": "test", "epoch": 1, "flags": ["x", "y"], "nullable": None}
    form = aiohttp.FormData()
    form.add_field("file", b"A" * 4096, filename=name, content_type="application/octet-stream")
    form.add_field("tags", json.dumps(tags))
    form.add_field("name", name)
    form.add_field("user_metadata", json.dumps(meta))
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 201, body
        return body


@pytest_asyncio.fixture(autouse=True)
async def autoclean_unit_test_assets(http: aiohttp.ClientSession, api_base: str):
    """Ensure isolation by removing all AssetInfo rows tagged with 'unit-tests' after each test."""
    yield

    while True:
        async with http.get(
            api_base + "/api/assets",
            params={"include_tags": "unit-tests", "limit": "500", "sort": "name"},
        ) as r:
            body = await r.json()
            if r.status != 200:
                break
            ids = [a["id"] for a in body.get("assets", [])]
        if not ids:
            break
        for aid in ids:
            with contextlib.suppress(Exception):
                async with http.delete(f"{api_base}/api/assets/{aid}") as dr:
                    await dr.read()
