"""Unit tests for the server-side model download subsystem.

Covers the pieces that don't require talking to a real network:

  - path parsing & allowlist (pure functions)
  - DownloadServer registry lifecycle (in-memory state)
  - API routes via aiohttp_client + folder_paths/probe_url patches

Streaming downloads themselves are exercised indirectly — the route-level
tests stub out the network probe so we can verify the gating logic in
``download_models`` without making real HTTP calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock

import pytest
from aiohttp import web

from app.model_downloader.allowlist import is_url_allowed
from app.model_downloader.api.routes import register_routes
from app.model_downloader.download_server import DownloadServer
from app.model_downloader.gated_detection import MetadataProbeResult
from app.model_downloader.paths import (
    InvalidModelId,
    parse_model_id,
    resolve_destination,
    resolve_existing,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def model_root(tmp_path):
    """A fake ``models/`` root with two registered folder types."""
    loras_dir = tmp_path / "loras"
    checkpoints_dir = tmp_path / "checkpoints"
    loras_dir.mkdir()
    checkpoints_dir.mkdir()
    return tmp_path, loras_dir, checkpoints_dir


@pytest.fixture
def patched_folder_paths(model_root):
    """Point folder_paths at our fake roots for the duration of one test."""
    _root, loras_dir, checkpoints_dir = model_root
    mapping = {
        "loras": ([str(loras_dir)], {".safetensors"}),
        "checkpoints": ([str(checkpoints_dir)], {".safetensors"}),
    }
    with patch(
        "folder_paths.folder_names_and_paths", mapping
    ), patch(
        "folder_paths.get_folder_paths",
        side_effect=lambda name: mapping.get(name, ([], set()))[0],
    ):
        yield mapping


@pytest.fixture
def fresh_download_server():
    """Reset the module-level singleton between tests so registry state
    doesn't leak across tests sharing the singleton."""
    from app.model_downloader.download_server import DOWNLOAD_SERVER

    DOWNLOAD_SERVER.reset_for_tests()
    yield DOWNLOAD_SERVER
    DOWNLOAD_SERVER.reset_for_tests()


@pytest.fixture
def app(patched_folder_paths, fresh_download_server):
    app = web.Application()
    register_routes(app)
    return app


# --------------------------------------------------------------------------- #
# Pure helpers: allowlist + path parsing
# --------------------------------------------------------------------------- #


def test_allowlist_accepts_hf_safetensors():
    assert is_url_allowed("https://huggingface.co/x/y/resolve/main/z.safetensors")


def test_allowlist_accepts_civitai_pth():
    assert is_url_allowed("https://civitai.com/api/download/models/123.pth")


def test_allowlist_rejects_unknown_host():
    assert not is_url_allowed("https://example.com/x.safetensors")


def test_allowlist_rejects_api_path_on_hf():
    # On an allowlisted host but not pointing at a model file.
    assert not is_url_allowed("https://huggingface.co/api/models")


def test_allowlist_rejects_non_https_except_localhost():
    assert not is_url_allowed("http://huggingface.co/x/y.safetensors")
    assert is_url_allowed("http://localhost:8000/x.safetensors")


def test_parse_model_id_valid(patched_folder_paths):
    assert parse_model_id("loras/foo.safetensors") == ("loras", "foo.safetensors")


def test_parse_model_id_rejects_traversal(patched_folder_paths):
    with pytest.raises(InvalidModelId):
        parse_model_id("../etc/passwd")


def test_parse_model_id_rejects_unknown_folder(patched_folder_paths):
    with pytest.raises(InvalidModelId):
        parse_model_id("nope/x.safetensors")


def test_parse_model_id_rejects_double_slash(patched_folder_paths):
    with pytest.raises(InvalidModelId):
        parse_model_id("loras/sub/x.safetensors")


def test_resolve_existing_returns_path_when_present(model_root, patched_folder_paths):
    _root, loras_dir, _ = model_root
    target = loras_dir / "foo.safetensors"
    target.write_bytes(b"x")
    assert resolve_existing("loras/foo.safetensors") == str(target)


def test_resolve_existing_returns_none_when_absent(patched_folder_paths):
    assert resolve_existing("loras/missing.safetensors") is None


def test_resolve_destination_returns_tmp_pair(model_root, patched_folder_paths):
    _root, loras_dir, _ = model_root
    final, tmp = resolve_destination("loras/foo.safetensors", epoch=7)
    assert final == str(loras_dir / "foo.safetensors")
    # Temp path embeds the session epoch (so cancel+retry can't collide on it)
    # and uses the subsystem-specific suffix the startup sweep matches.
    assert tmp == f"{final}.7.comfy-download.tmp"


# --------------------------------------------------------------------------- #
# DownloadServer registry: lifecycle, races, cancellation epoch semantics
# --------------------------------------------------------------------------- #


def test_register_is_exclusive():
    server = DownloadServer()
    s1 = server.try_register("loras/x.safetensors", "https://huggingface.co/a")
    s2 = server.try_register("loras/x.safetensors", "https://huggingface.co/b")
    assert s1 is not None
    assert s2 is None
    assert server.is_downloading("loras/x.safetensors")


def test_cancel_removes_session():
    server = DownloadServer()
    server.try_register("loras/x.safetensors", "https://huggingface.co/a")
    assert server.cancel("loras/x.safetensors") is True
    assert not server.is_downloading("loras/x.safetensors")


def test_cancel_returns_false_when_absent():
    server = DownloadServer()
    assert server.cancel("loras/never.safetensors") is False


def test_finish_only_clears_matching_epoch():
    """If a session is cancelled and a new one for the same id is
    registered, ``finish`` from the original worker must not evict the
    newer session."""
    server = DownloadServer()
    s_old = server.try_register("loras/x.safetensors", "u1")
    server.cancel("loras/x.safetensors")
    s_new = server.try_register("loras/x.safetensors", "u2")
    assert s_new is not None and s_new.epoch != s_old.epoch
    # Old worker's late finish() is a no-op:
    server.finish(s_old)
    assert server.is_downloading("loras/x.safetensors")
    server.finish(s_new)
    assert not server.is_downloading("loras/x.safetensors")


def test_is_active_follows_cancellation():
    server = DownloadServer()
    s = server.try_register("loras/x.safetensors", "u")
    assert server.is_active(s)
    server.cancel("loras/x.safetensors")
    assert not server.is_active(s)


def test_update_progress_tracks_fraction():
    server = DownloadServer()
    s = server.try_register("loras/x.safetensors", "u")
    server.update_progress(s, 50, 100)
    snap = server.snapshot()["loras/x.safetensors"]
    assert snap.bytes_downloaded == 50
    assert snap.total_bytes == 100
    assert snap.progress == 0.5


def test_update_progress_with_unknown_total_keeps_progress_none():
    server = DownloadServer()
    s = server.try_register("loras/x.safetensors", "u")
    server.update_progress(s, 50, None)
    assert server.snapshot()["loras/x.safetensors"].progress is None


def test_cleanup_orphan_tmp_files(model_root):
    """Orphan temp left by a crashed download must be swept on first use,
    while unrelated *.tmp files in the model dir are left untouched."""
    _root, loras_dir, _ = model_root
    orphan = loras_dir / "stale.safetensors.3.comfy-download.tmp"
    orphan.write_bytes(b"partial")
    unrelated = loras_dir / "someothertool.tmp"
    unrelated.write_bytes(b"not ours")
    mapping = {"loras": ([str(loras_dir)], {".safetensors"})}
    with patch("folder_paths.folder_names_and_paths", mapping), patch(
        "folder_paths.get_folder_paths",
        side_effect=lambda name: mapping.get(name, ([], set()))[0],
    ):
        server = DownloadServer()
        assert orphan.exists(), "sweep must not run at construction time"
        server.sweep_orphan_tmp_files()
        assert not orphan.exists()
        assert unrelated.exists(), "unrelated .tmp must not be swept"
        # Idempotent — a second call is a cheap no-op.
        server.sweep_orphan_tmp_files()


# --------------------------------------------------------------------------- #
# Route: POST /api/models-availability-status
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_availability_partitions_correctly(
    aiohttp_client, app, model_root, fresh_download_server
):
    _root, loras_dir, _ = model_root
    (loras_dir / "present.safetensors").write_bytes(b"x")
    fresh_download_server.try_register(
        "loras/inflight.safetensors", "http://localhost:8000/x.safetensors"
    )
    client = await aiohttp_client(app)

    # Stub probes — we're testing state assignment, not network calls.
    with patch(
        "app.model_downloader.api.routes.probe_url",
        new=AsyncMock(return_value=MetadataProbeResult(
            file_size=None, is_hf_downloadable=None,
        )),
    ):
        body = {
            "models": {
                "loras/present.safetensors": "http://localhost:8000/p.safetensors",
                "loras/missing.safetensors": "http://localhost:8000/m.safetensors",
                "loras/inflight.safetensors": "http://localhost:8000/x.safetensors",
            }
        }
        resp = await client.post("/api/models-availability-status", json=body)
    assert resp.status == 200
    data = await resp.json()
    models = data["models"]
    assert models["loras/present.safetensors"]["state"] == "available"
    assert models["loras/missing.safetensors"]["state"] == "missing"
    assert models["loras/inflight.safetensors"]["state"] == "downloading"
    assert "hf_auth" in data


@pytest.mark.asyncio
async def test_availability_invalid_id_classified_as_missing(aiohttp_client, app):
    client = await aiohttp_client(app)
    with patch(
        "app.model_downloader.api.routes.probe_url",
        new=AsyncMock(return_value=MetadataProbeResult(
            file_size=None, is_hf_downloadable=None,
        )),
    ):
        resp = await client.post(
            "/api/models-availability-status",
            json={"models": {"../etc/passwd": "http://localhost:8000/x.safetensors"}},
        )
    assert resp.status == 200
    data = await resp.json()
    assert data["models"]["../etc/passwd"]["state"] == "missing"


# --------------------------------------------------------------------------- #
# Route: POST /api/download-models  — precondition gating
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_download_rejects_url_not_in_allowlist(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/download-models",
        json={"models": {"loras/x.safetensors": "https://evil.com/x.safetensors"}},
    )
    assert resp.status == 400
    err = (await resp.json())["error"]
    assert err["code"] == "URL_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_download_rejects_already_available(
    aiohttp_client, app, model_root
):
    _root, loras_dir, _ = model_root
    (loras_dir / "x.safetensors").write_bytes(b"x")
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/download-models",
        json={"models": {
            "loras/x.safetensors": "https://huggingface.co/a/b/resolve/main/x.safetensors"
        }},
    )
    assert resp.status == 409
    assert (await resp.json())["error"]["code"] == "ALREADY_AVAILABLE"


@pytest.mark.asyncio
async def test_download_rejects_already_downloading(
    aiohttp_client, app, fresh_download_server
):
    fresh_download_server.try_register(
        "loras/x.safetensors", "https://huggingface.co/u.safetensors"
    )
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/download-models",
        json={"models": {
            "loras/x.safetensors": "https://huggingface.co/a/b/resolve/main/x.safetensors"
        }},
    )
    assert resp.status == 409
    assert (await resp.json())["error"]["code"] == "ALREADY_DOWNLOADING"


@pytest.mark.asyncio
async def test_download_rejects_gated_model(aiohttp_client, app):
    client = await aiohttp_client(app)
    with patch(
        "app.model_downloader.api.routes.probe_url",
        new=AsyncMock(return_value=MetadataProbeResult(file_size=None, is_hf_downloadable=False)),
    ):
        resp = await client.post(
            "/api/download-models",
            json={"models": {
                "loras/x.safetensors": "https://huggingface.co/g/r/resolve/main/x.safetensors"
            }},
        )
    assert resp.status == 400
    assert (await resp.json())["error"]["code"] == "MODEL_NOT_DOWNLOADABLE"


@pytest.mark.asyncio
async def test_download_rejects_invalid_model_id(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/download-models",
        json={"models": {"../etc/passwd": "https://huggingface.co/x.safetensors"}},
    )
    assert resp.status == 400
    assert (await resp.json())["error"]["code"] == "INVALID_MODEL_ID"


@pytest.mark.asyncio
async def test_download_atomic_failure_does_not_register_partial(
    aiohttp_client, app, model_root, fresh_download_server
):
    """If one model in a batch fails, none get registered."""
    _root, loras_dir, _ = model_root
    (loras_dir / "already.safetensors").write_bytes(b"x")
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/download-models",
        json={
            "models": {
                "loras/already.safetensors":
                    "https://huggingface.co/a/b/resolve/main/already.safetensors",
                "loras/new.safetensors":
                    "https://huggingface.co/a/b/resolve/main/new.safetensors",
            }
        },
    )
    assert resp.status == 409
    # The "new" model should not have been registered as part of the
    # failed batch.
    assert not fresh_download_server.is_downloading("loras/new.safetensors")


@pytest.mark.asyncio
async def test_download_schedules_when_all_preconditions_pass(
    aiohttp_client, app, fresh_download_server
):
    """Verify the precondition pass, registration pass, and async
    scheduling all wire up correctly. We patch the streamer to avoid
    real HTTP while still letting the route execute end-to-end."""
    started = asyncio.Event()
    finish_signal = asyncio.Event()

    async def fake_stream(session):
        started.set()
        await finish_signal.wait()
        from app.model_downloader.download_server import DOWNLOAD_SERVER
        DOWNLOAD_SERVER.finish(session)
        return "/dev/null"

    with patch(
        "app.model_downloader.api.routes.probe_url",
        new=AsyncMock(return_value=MetadataProbeResult(file_size=42, is_hf_downloadable=True)),
    ), patch(
        "app.model_downloader.downloader.stream_to_disk", new=fake_stream
    ):
        client = await aiohttp_client(app)
        resp = await client.post(
            "/api/download-models",
            json={"models": {
                "loras/new.safetensors":
                    "https://huggingface.co/a/b/resolve/main/new.safetensors"
            }},
        )
        assert resp.status == 202
        body = await resp.json()
        assert body["accepted"] is True
        assert body["scheduled"] == ["loras/new.safetensors"]
        # Wait for the worker to actually start.
        await asyncio.wait_for(started.wait(), timeout=2.0)
        assert fresh_download_server.is_downloading("loras/new.safetensors")
        finish_signal.set()


# --------------------------------------------------------------------------- #
# Route: POST /api/cancel-model-download-session
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancel_removes_active_session(
    aiohttp_client, app, fresh_download_server
):
    fresh_download_server.try_register(
        "loras/x.safetensors", "https://huggingface.co/u.safetensors"
    )
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/cancel-model-download-session",
        json={"model_id": "loras/x.safetensors"},
    )
    assert resp.status == 200
    assert (await resp.json())["cancelled"] is True
    assert not fresh_download_server.is_downloading("loras/x.safetensors")


@pytest.mark.asyncio
async def test_cancel_returns_404_when_none(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/cancel-model-download-session",
        json={"model_id": "loras/nothing.safetensors"},
    )
    assert resp.status == 404
    assert (await resp.json())["error"]["code"] == "NOT_DOWNLOADING"


# --------------------------------------------------------------------------- #
# Unified availability response embeds metadata per id
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_availability_embeds_metadata(aiohttp_client, app):
    """``file_size`` + ``is_hf_downloadable`` come back on the same
    request as the state — no separate metadata endpoint."""
    results = {
        "https://huggingface.co/a/b/resolve/main/free.safetensors":
            MetadataProbeResult(file_size=1024, is_hf_downloadable=True),
        "https://huggingface.co/g/r/resolve/main/gated.safetensors":
            MetadataProbeResult(file_size=None, is_hf_downloadable=False),
    }

    async def fake_probe(url):
        return results[url]

    with patch(
        "app.model_downloader.api.routes.probe_url", new=fake_probe
    ):
        client = await aiohttp_client(app)
        resp = await client.post(
            "/api/models-availability-status",
            json={
                "models": {
                    "loras/free.safetensors":
                        "https://huggingface.co/a/b/resolve/main/free.safetensors",
                    "loras/gated.safetensors":
                        "https://huggingface.co/g/r/resolve/main/gated.safetensors",
                }
            },
        )
    assert resp.status == 200
    models = (await resp.json())["models"]
    assert models["loras/free.safetensors"]["file_size"] == 1024
    assert models["loras/free.safetensors"]["is_hf_downloadable"] is True
    assert models["loras/gated.safetensors"]["file_size"] is None
    assert models["loras/gated.safetensors"]["is_hf_downloadable"] is False
