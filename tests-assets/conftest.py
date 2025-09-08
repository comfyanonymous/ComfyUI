import asyncio
import contextlib
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Callable

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
    # cleanup in a best-effort way; ComfyUI should not keep files open in this dir
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
    port = 8500  # _free_port()
    db_url = "sqlite+aiosqlite:///:memory:"

    # stdout/stderr capturing for debugging if something goes wrong
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


@pytest.fixture
def make_asset_bytes() -> Callable[[str], bytes]:
    def _make(name: str) -> bytes:
        # Generate deterministic small content variations based on name
        seed = sum(ord(c) for c in name) % 251
        data = bytes((i * 31 + seed) % 256 for i in range(8192))
        return data
    return _make


async def _upload_asset(session: aiohttp.ClientSession, base: str, *, name: str, tags: list[str], meta: dict) -> dict:
    make_asset_bytes = bytes((i % 251) for i in range(4096))
    form = aiohttp.FormData()
    form.add_field("file", make_asset_bytes, filename=name, content_type="application/octet-stream")
    form.add_field("tags", json.dumps(tags))
    form.add_field("name", name)
    form.add_field("user_metadata", json.dumps(meta))
    async with session.post(base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status in (200, 201), body
        return body


@pytest_asyncio.fixture
async def seeded_asset(http: aiohttp.ClientSession, api_base: str) -> dict:
    """
    Upload one asset into models/checkpoints/unit-tests/<name>.
    Returns response dict with id, asset_hash, tags, etc.
    """
    name = "unit_1_example.safetensors"
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
