"""Integration tests for the download engine against a local aiohttp server.

Covers single-stream and segmented transfers, deterministic resume from a
partial file, and cancel rollback. Async tests are driven via ``asyncio.run``
so no pytest-asyncio plugin is required.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import uuid

import pytest
from aiohttp import web

from comfy.cli_args import args
from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database import queries
from app.model_downloader.engine.job import DownloadJob, JobSpec
from app.model_downloader.net.session import close_session
from app.model_downloader.security import paths

PAYLOAD_ETAG = '"v1"'


def _payload(n: int) -> bytes:
    return bytes((i * 37 + 11) % 256 for i in range(n))


def _safetensors_payload(total: int) -> bytes:
    """A structurally valid ``.safetensors`` blob of exactly ``total`` bytes.

    Success-path tests download to ``.safetensors`` destinations, which the
    engine now structurally validates before the atomic rename, so their
    payloads must parse as real safetensors (header length + JSON header +
    data region whose size matches the declared ``data_offsets``).
    """
    def _header(data_len: int) -> bytes:
        return json.dumps(
            {"w": {"dtype": "U8", "shape": [data_len], "data_offsets": [0, data_len]}}
        ).encode("utf-8")

    # The header's byte length depends on the digit count of ``data_len``, so
    # iterate until ``total == 8 + len(header) + data_len`` is self-consistent.
    data_len = total - 8 - len(_header(total))
    for _ in range(8):
        header = _header(data_len)
        new_data_len = total - 8 - len(header)
        if new_data_len == data_len:
            break
        data_len = new_data_len
    assert data_len >= 0, "total too small for a safetensors payload"
    header = _header(data_len)
    body = bytes((i * 37 + 11) % 256 for i in range(data_len))
    return struct.pack("<Q", len(header)) + header + body


def _range_handler(payload: bytes):
    async def handler(request: web.Request) -> web.Response:
        rng = request.headers.get("Range")
        if rng:
            spec = rng.split("=", 1)[1]
            s, _, e = spec.partition("-")
            start = int(s)
            end = int(e) if e else len(payload) - 1
            chunk = payload[start : end + 1]
            return web.Response(
                status=206,
                body=chunk,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "Accept-Ranges": "bytes",
                    "ETag": PAYLOAD_ETAG,
                },
            )
        return web.Response(
            status=200, body=payload, headers={"Accept-Ranges": "bytes", "ETag": PAYLOAD_ETAG}
        )

    return handler


def _content_disposition_handler(payload: bytes, filename: str):
    """A range-capable server that only reveals its filename via a header.

    Models a Civitai-style ``/api/download/...`` endpoint: the URL path has no
    extension, and the real filename (hence extension) lives in the response
    ``Content-Disposition`` header.
    """

    async def handler(request: web.Request) -> web.Response:
        headers = {
            "Accept-Ranges": "bytes",
            "ETag": PAYLOAD_ETAG,
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
        rng = request.headers.get("Range")
        if rng:
            spec = rng.split("=", 1)[1]
            s, _, e = spec.partition("-")
            start = int(s)
            end = int(e) if e else len(payload) - 1
            chunk = payload[start : end + 1]
            return web.Response(
                status=206,
                body=chunk,
                headers={**headers, "Content-Range": f"bytes {start}-{end}/{len(payload)}"},
            )
        return web.Response(status=200, body=payload, headers=headers)

    return handler


def _noranges_handler(payload: bytes):
    async def handler(request: web.Request) -> web.Response:
        # Always full body, never advertises Accept-Ranges -> single-stream.
        return web.Response(status=200, body=payload)

    return handler


def _slow_handler(payload: bytes, chunk: int = 16384, delay: float = 0.01):
    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200, headers={"Content-Length": str(len(payload))}
        )
        await resp.prepare(request)
        for i in range(0, len(payload), chunk):
            await resp.write(payload[i : i + chunk])
            await asyncio.sleep(delay)
        await resp.write_eof()
        return resp

    return handler


def _overflow_range_handler(payload: bytes, extra: int = 256 * 1024):
    """A non-conforming 206 server that returns MORE than the requested range."""

    async def handler(request: web.Request) -> web.Response:
        rng = request.headers.get("Range")
        if rng:
            spec = rng.split("=", 1)[1]
            s, _, e = spec.partition("-")
            start = int(s)
            end = int(e) if e else len(payload) - 1
            # Maliciously overrun: append extra bytes past the requested end.
            body = payload[start : end + 1] + bytes(extra)
            return web.Response(
                status=206,
                body=body,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "Accept-Ranges": "bytes",
                    "ETag": PAYLOAD_ETAG,
                },
            )
        return web.Response(
            status=200, body=payload, headers={"Accept-Ranges": "bytes", "ETag": PAYLOAD_ETAG}
        )

    return handler


def _short_range_handler(payload: bytes, drop: int = 64 * 1024):
    """A 206 server that returns fewer bytes than requested for later segments.

    Simulates a server cleanly closing a range connection early. The response
    is internally consistent (Content-Length matches the short body), so the
    client sees no error and the segment just ends short, leaving a zero-filled
    hole in the preallocated file.
    """

    async def handler(request: web.Request) -> web.Response:
        rng = request.headers.get("Range")
        if rng:
            spec = rng.split("=", 1)[1]
            s, _, e = spec.partition("-")
            start = int(s)
            end = int(e) if e else len(payload) - 1
            chunk = payload[start : end + 1]
            if start > 0 and len(chunk) > drop:
                chunk = chunk[:-drop]  # truncate a non-first segment
            return web.Response(
                status=206,
                body=chunk,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "Accept-Ranges": "bytes",
                    "ETag": PAYLOAD_ETAG,
                },
            )
        return web.Response(
            status=200, body=payload, headers={"Accept-Ranges": "bytes", "ETag": PAYLOAD_ETAG}
        )

    return handler


def _unbounded_handler(total: int, chunk: int = 16384):
    """A 200 stream with no Content-Length / Accept-Ranges (unknown length)."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(status=200)
        await resp.prepare(request)
        sent = 0
        while sent < total:
            await resp.write(bytes(min(chunk, total - sent)))
            sent += chunk
        await resp.write_eof()
        return resp

    return handler


async def _serve(handler):
    app = web.Application()
    app.router.add_route("*", "/{name:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def _insert(model_id: str, url: str, status: str = DownloadStatus.QUEUED) -> tuple[str, str, str]:
    final_path, temp_path = paths.resolve_destination(model_id)
    download_id = str(uuid.uuid4())
    queries.insert_download(
        {
            "id": download_id,
            "url": url,
            "model_id": model_id,
            "dest_path": final_path,
            "temp_path": temp_path,
            "status": status,
        }
    )
    return download_id, final_path, temp_path


# ----- single-stream -----


def test_single_stream_download(model_root):
    payload = _safetensors_payload(300_000)

    async def _run():
        await close_session()
        runner, port = await _serve(_noranges_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, _temp = _insert("loras/single.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/single.safetensors",
                dest_path=final_path, temp_path=_temp,
            ))
            status = await job.run()
            assert status == DownloadStatus.COMPLETED, queries.get_download(did).error
            assert os.path.exists(final_path)
            assert open(final_path, "rb").read() == payload
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


# ----- segmented -----


def test_segmented_download(model_root):
    payload = _safetensors_payload(4 * 1024 * 1024)  # 4 MiB -> multiple segments

    async def _run():
        await close_session()
        runner, port = await _serve(_range_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/seg.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/seg.safetensors",
                dest_path=final_path, temp_path=temp,
            ))
            status = await job.run()
            assert status == DownloadStatus.COMPLETED, queries.get_download(did).error
            assert open(final_path, "rb").read() == payload
            # More than one segment row was planned.
            assert len(queries.list_segments(did)) > 1
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


# ----- deterministic resume from a partial file -----


def test_resume_from_partial(model_root):
    payload = _safetensors_payload(512 * 1024)  # < 1 MiB -> single segment

    async def _run():
        await close_session()
        runner, port = await _serve(_range_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/resume.safetensors", url)
            # Simulate a prior partial: first 200 KiB already written, offset persisted.
            prefix = 200 * 1024
            os.makedirs(os.path.dirname(temp), exist_ok=True)
            with open(temp, "wb") as f:
                f.write(payload[:prefix])
            queries.update_download(did, bytes_done=prefix, etag=PAYLOAD_ETAG)

            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/resume.safetensors",
                dest_path=final_path, temp_path=temp, etag=PAYLOAD_ETAG,
            ))
            status = await job.run()
            assert status == DownloadStatus.COMPLETED, queries.get_download(did).error
            assert open(final_path, "rb").read() == payload
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


# ----- cancel rollback -----


def test_cancel_rollback(model_root, monkeypatch):
    monkeypatch.setattr(args, "download_chunk_size", 16384, raising=False)
    payload = _payload(1024 * 1024)

    async def _run():
        await close_session()
        runner, port = await _serve(_slow_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/cancel.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/cancel.safetensors",
                dest_path=final_path, temp_path=temp,
            ))
            task = asyncio.ensure_future(job.run())
            # Wait until some bytes have been written, then cancel.
            for _ in range(200):
                await asyncio.sleep(0.01)
                if job.state.bytes_done > 0:
                    break
            job.request_cancel()
            status = await task
            assert status == DownloadStatus.CANCELLED
            assert not os.path.exists(temp)
            assert not os.path.exists(final_path)
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


# ----- size-bound enforcement (malicious / non-conforming hosts) -----


def test_segment_overflow_aborts(model_root):
    """A 206 returning more than the requested range must not overrun."""
    payload = _payload(4 * 1024 * 1024)  # large enough to segment

    async def _run():
        await close_session()
        runner, port = await _serve(_overflow_range_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/overflow.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/overflow.safetensors",
                dest_path=final_path, temp_path=temp,
            ))
            status = await job.run()
            assert status == DownloadStatus.FAILED
            assert not os.path.exists(final_path)
            assert not os.path.exists(temp)
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


def test_short_segment_fails_closed(model_root):
    """A segment that ends short must fail, not be accepted as complete.

    The file is preallocated to total_bytes, so the on-disk size still equals
    total even with a zero-filled hole; completeness must be judged per-segment.
    """
    payload = _safetensors_payload(4 * 1024 * 1024)  # large enough to segment

    async def _run():
        await close_session()
        runner, port = await _serve(_short_range_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/short.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/short.safetensors",
                dest_path=final_path, temp_path=temp,
            ))
            status = await job.run()
            assert status == DownloadStatus.FAILED, queries.get_download(did).error
            assert "incomplete" in (queries.get_download(did).error or "")
            assert not os.path.exists(final_path)
            assert not os.path.exists(temp)
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


def test_structural_validation_rejects_corrupt(model_root):
    """A correctly sized but structurally invalid file fails closed (not retried).

    Regression for the dead structural gate: validation must key off the
    destination extension, not the ``.part`` temp suffix.
    """
    payload = _payload(300_000)  # right size, but not a valid safetensors blob

    async def _run():
        await close_session()
        runner, port = await _serve(_noranges_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/corrupt.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/corrupt.safetensors",
                dest_path=final_path, temp_path=temp,
            ))
            status = await job.run()
            assert status == DownloadStatus.FAILED, queries.get_download(did).error
            assert not os.path.exists(final_path)
            assert not os.path.exists(temp)
            # Failed closed at first attempt, not re-queued as retryable.
            assert queries.get_download(did).attempts == 0
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


def test_rejects_oversized_known_download(model_root, monkeypatch):
    """A file whose advertised size exceeds the cap is rejected at probe."""
    monkeypatch.setattr(args, "download_max_bytes", 100_000, raising=False)
    payload = _payload(300_000)

    async def _run():
        await close_session()
        runner, port = await _serve(_noranges_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/toobig.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/toobig.safetensors",
                dest_path=final_path, temp_path=temp,
            ))
            status = await job.run()
            assert status == DownloadStatus.FAILED
            assert not os.path.exists(final_path)
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


def test_unknown_length_capped_by_max_bytes(model_root, monkeypatch):
    """An unbounded unknown-length stream is capped by --download-max-bytes."""
    monkeypatch.setattr(args, "download_max_bytes", 100_000, raising=False)
    monkeypatch.setattr(args, "download_chunk_size", 16384, raising=False)

    async def _run():
        await close_session()
        runner, port = await _serve(_unbounded_handler(2 * 1024 * 1024))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did, final_path, temp = _insert("loras/unbounded.safetensors", url)
            job = DownloadJob(JobSpec(
                download_id=did, url=url, model_id="loras/unbounded.safetensors",
                dest_path=final_path, temp_path=temp,
            ))
            status = await job.run()
            assert status == DownloadStatus.FAILED
            assert not os.path.exists(final_path)
            assert not os.path.exists(temp)
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


# ----- manager + scheduler end-to-end -----


def test_manager_enqueue_to_completion(model_root):
    payload = _safetensors_payload(2 * 1024 * 1024)

    async def _run():
        await close_session()
        from app.model_downloader.manager import DOWNLOAD_MANAGER

        runner, port = await _serve(_range_handler(payload))
        try:
            url = f"http://127.0.0.1:{port}/model.safetensors"
            did = await DOWNLOAD_MANAGER.enqueue(url, "loras/e2e.safetensors")
            # Wait for completion.
            final_path, _ = paths.resolve_destination("loras/e2e.safetensors")
            for _ in range(500):
                await asyncio.sleep(0.02)
                row = queries.get_download(did)
                if row.status in DownloadStatus.TERMINAL:
                    break
            row = queries.get_download(did)
            assert row.status == DownloadStatus.COMPLETED, row.error
            assert open(final_path, "rb").read() == payload
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


def test_manager_rejects_disallowed_url(model_root):
    async def _run():
        from app.model_downloader.manager import DOWNLOAD_MANAGER, DownloadError

        with pytest.raises(DownloadError) as ei:
            await DOWNLOAD_MANAGER.enqueue(
                "https://evil.example.com/x.safetensors", "loras/bad.safetensors"
            )
        assert ei.value.code == "URL_NOT_ALLOWED"

    asyncio.run(_run())


def test_manager_resolves_extensionless_url(model_root):
    """An allowlisted URL with no extension in its path is resolved from the
    response, and the stored file adopts the resolved extension."""
    payload = _safetensors_payload(1 * 1024 * 1024)

    async def _run():
        await close_session()
        from app.model_downloader.manager import DOWNLOAD_MANAGER

        runner, port = await _serve(
            _content_disposition_handler(payload, "RealModel.safetensors")
        )
        try:
            # No extension in the path (Civitai-style) and none in the model_id.
            url = f"http://127.0.0.1:{port}/api/download/models/12345"
            did = await DOWNLOAD_MANAGER.enqueue(url, "loras/my_civitai_model")

            row = queries.get_download(did)
            # The resolved extension was appended to the model_id + destination.
            assert row.model_id == "loras/my_civitai_model.safetensors"
            assert row.dest_path.endswith("my_civitai_model.safetensors")

            final_path, _ = paths.resolve_destination(
                "loras/my_civitai_model.safetensors"
            )
            for _ in range(500):
                await asyncio.sleep(0.02)
                row = queries.get_download(did)
                if row.status in DownloadStatus.TERMINAL:
                    break
            row = queries.get_download(did)
            assert row.status == DownloadStatus.COMPLETED, row.error
            assert open(final_path, "rb").read() == payload
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


def test_manager_overrides_extension_from_resolution(model_root):
    """A model_id carrying a different known extension is corrected to match
    the resolved URL's extension."""
    payload = _safetensors_payload(256 * 1024)

    async def _run():
        await close_session()
        from app.model_downloader.manager import DOWNLOAD_MANAGER

        runner, port = await _serve(
            _content_disposition_handler(payload, "weights.safetensors")
        )
        try:
            url = f"http://127.0.0.1:{port}/api/download/models/777"
            # Caller guessed .ckpt; resolution says .safetensors -> corrected.
            did = await DOWNLOAD_MANAGER.enqueue(url, "loras/guessed.ckpt")
            row = queries.get_download(did)
            assert row.model_id == "loras/guessed.safetensors"
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())


def test_manager_rejects_non_model_resolution(model_root):
    """A URL that resolves to a non-model file is rejected, not downloaded."""

    async def _run():
        await close_session()
        from app.model_downloader.manager import DOWNLOAD_MANAGER, DownloadError

        runner, port = await _serve(
            _content_disposition_handler(b"not a model", "installer.zip")
        )
        try:
            url = f"http://127.0.0.1:{port}/api/download/models/999"
            with pytest.raises(DownloadError) as ei:
                await DOWNLOAD_MANAGER.enqueue(url, "loras/whatever")
            assert ei.value.code == "URL_NOT_ALLOWED"
        finally:
            await runner.cleanup()
            await close_session()

    asyncio.run(_run())
