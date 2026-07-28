"""The per-download worker.

One :class:`DownloadJob` drives a single file from probe to verified, cataloged
completion. It supports cooperative pause / resume / cancel, segmented
multi-connection transfer with positioned writes, and a verification gate
(size + structural + optional sha256) before the atomic rename into place.

Control is cooperative: external callers flip ``_control`` via
:meth:`request_pause` / :meth:`request_cancel`; segment loops observe it between
chunks and raise, which unwinds cleanly and persists resume offsets.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from comfy.cli_args import args
from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database import queries
from app.model_downloader.engine.planner import (
    effective_segment_count,
    plan_segments,
)
from app.model_downloader.engine.writer import FileWriter
from app.model_downloader.net.http import open_validated, redact_url
from app.model_downloader.net.probe import gated_error_message, probe
from app.model_downloader.verify import checksum, dedup, structural

_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
_PERSIST_INTERVAL = 2.0  # seconds between throttled progress persists


class Paused(Exception):
    pass


class Cancelled(Exception):
    pass


class RemoteChanged(Exception):
    """The remote file changed under a resume (got 200 where 206 expected)."""


class RetryableError(Exception):
    pass


class FatalError(Exception):
    """Non-retryable: 4xx, checksum mismatch, structural failure, gated, etc."""


@dataclass
class SegmentRuntime:
    idx: int
    start: int
    end: int  # inclusive; may be -1 for unknown-size single stream
    bytes_done: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class RuntimeState:
    download_id: str
    model_id: str
    url: str
    priority: int
    status: str
    total_bytes: Optional[int] = None
    bytes_done: int = 0
    error: Optional[str] = None
    segments: list[SegmentRuntime] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    _last_bytes: int = 0
    _last_time: float = field(default_factory=time.monotonic)
    speed_bps: float = 0.0

    @property
    def progress(self) -> Optional[float]:
        if not self.total_bytes:
            return None
        return min(1.0, self.bytes_done / self.total_bytes)

    @property
    def eta_seconds(self) -> Optional[float]:
        if not self.total_bytes or self.speed_bps <= 0:
            return None
        remaining = max(0, self.total_bytes - self.bytes_done)
        return remaining / self.speed_bps


@dataclass
class JobSpec:
    download_id: str
    url: str
    model_id: str
    dest_path: str
    temp_path: str
    priority: int = 0
    expected_sha256: Optional[str] = None
    allow_any_extension: bool = False
    etag: Optional[str] = None
    attempts: int = 0


class DownloadJob:
    def __init__(
        self, spec: JobSpec, notify_cb: Optional[Callable[[str], None]] = None
    ) -> None:
        self.spec = spec
        self._notify = notify_cb
        self._control = "run"  # run | pause | cancel
        self.state = RuntimeState(
            download_id=spec.download_id,
            model_id=spec.model_id,
            url=spec.url,
            priority=spec.priority,
            status=DownloadStatus.QUEUED,
        )
        self._writer: Optional[FileWriter] = None
        self._etag: Optional[str] = spec.etag
        self._last_persist = 0.0

    # ----- external control -----

    def request_pause(self) -> None:
        if self._control == "run":
            self._control = "pause"

    def request_cancel(self) -> None:
        self._control = "cancel"

    def _check_control(self) -> None:
        if self._control == "cancel":
            raise Cancelled()
        if self._control == "pause":
            raise Paused()

    # ----- lifecycle -----

    async def run(self) -> str:
        """Run to a terminal/paused state; returns the final status string."""
        await self._set_status(DownloadStatus.ACTIVE, error=None)
        try:
            pr = await self._probe_and_plan()
            await self._transfer(pr)
            await self._finalize()
            await self._set_status(DownloadStatus.COMPLETED)
        except Paused:
            await self._persist_progress(force=True)
            await self._set_status(DownloadStatus.PAUSED)
        except Cancelled:
            await self._close_writer()
            self._remove_temp()
            await self._set_status(DownloadStatus.CANCELLED)
        except RemoteChanged:
            await self._reset_for_restart()
            await self._set_status(
                DownloadStatus.QUEUED, error="remote file changed; restarting"
            )
        except RetryableError as e:
            await self._persist_progress(force=True)
            await self._set_status(DownloadStatus.QUEUED, error=str(e))
        except FatalError as e:
            await self._close_writer()
            self._remove_temp()
            await self._set_status(DownloadStatus.FAILED, error=str(e))
        except Exception as e:  # unexpected -> treat as retryable
            logging.warning(
                "[model_downloader] %s unexpected error: %s",
                self.spec.model_id, e, exc_info=True,
            )
            await self._persist_progress(force=True)
            await self._set_status(DownloadStatus.QUEUED, error=f"{type(e).__name__}: {e}")
        finally:
            await self._close_writer()
        return self.state.status

    # ----- probe + plan -----

    async def _probe_and_plan(self):
        pr = await probe(self.spec.url)
        if not pr.ok:
            if pr.gated:
                raise FatalError(gated_error_message(self.spec.url, pr))
            if pr.status == 0 or pr.status in _RETRYABLE_STATUSES:
                raise RetryableError(pr.error or "probe failed")
            raise FatalError(pr.error or f"probe returned HTTP {pr.status}")

        max_bytes = self._max_download_bytes()
        if max_bytes is not None and pr.total_bytes is not None and pr.total_bytes > max_bytes:
            raise FatalError(
                f"file size {pr.total_bytes} exceeds the maximum allowed "
                f"download size {max_bytes} (--download-max-bytes)"
            )

        self._etag = pr.etag or self._etag
        self.state.total_bytes = pr.total_bytes
        await asyncio.to_thread(
            queries.update_download,
            self.spec.download_id,
            final_url=pr.final_url,
            total_bytes=pr.total_bytes,
            accept_ranges=pr.accept_ranges,
            etag=pr.etag,
            last_modified=pr.last_modified,
        )

        seg_count = effective_segment_count(
            pr.total_bytes, pr.accept_ranges, max(1, args.download_segments)
        )
        existing = await asyncio.to_thread(queries.list_segments, self.spec.download_id)
        can_resume_segmented = (
            seg_count > 1
            and existing
            and pr.total_bytes is not None
            and existing[-1].end_offset == pr.total_bytes - 1
        )
        if can_resume_segmented and not self._segmented_part_valid(pr.total_bytes):
            # The persisted per-segment offsets describe bytes in a preallocated
            # .part that is now gone or the wrong size (e.g. the partial of a
            # failed download was swept on restart, or removed by a fatal
            # error). Trusting them would skip already-"complete" segments and
            # leave zero-filled holes. Discard the offsets and re-plan fresh.
            logging.info(
                "[model_downloader] %s discarding segmented resume offsets "
                "(preallocated .part missing or wrong size); restarting",
                self.spec.model_id,
            )
            self._remove_temp()
            await asyncio.to_thread(
                queries.replace_segments, self.spec.download_id, []
            )
            await asyncio.to_thread(
                queries.update_download, self.spec.download_id, bytes_done=0
            )
            existing = []
            can_resume_segmented = False

        if can_resume_segmented:
            # Resume an existing segmented plan.
            self.state.segments = [
                SegmentRuntime(s.idx, s.start_offset, s.end_offset, s.bytes_done)
                for s in existing
            ]
        elif seg_count > 1 and pr.total_bytes is not None:
            plans = plan_segments(pr.total_bytes, seg_count)
            await asyncio.to_thread(
                queries.replace_segments,
                self.spec.download_id,
                [
                    {"idx": p.idx, "start_offset": p.start, "end_offset": p.end, "bytes_done": 0}
                    for p in plans
                ],
            )
            self.state.segments = [SegmentRuntime(p.idx, p.start, p.end, 0) for p in plans]
        else:
            # Single-stream: one logical segment; bytes_done tracked on the row.
            row = await asyncio.to_thread(queries.get_download, self.spec.download_id)
            resume_from = row.bytes_done if row else 0
            end = (pr.total_bytes - 1) if pr.total_bytes else -1
            # ``row.bytes_done`` may be the SUM of per-segment offsets from a
            # prior segmented run (a preallocated, non-contiguous .part). A
            # single-stream resume writes a contiguous prefix, so the offset is
            # only trustworthy when the on-disk file is exactly that many
            # contiguous bytes. This guards the case where a download that ran
            # segmented now resolves to one segment (server dropped
            # Accept-Ranges, or --download-segments was lowered between runs):
            # resuming over non-contiguous data would corrupt the output.
            if resume_from > 0 and not self._contiguous_prefix_valid(resume_from):
                logging.info(
                    "[model_downloader] %s discarding untrusted resume offset "
                    "%d (on-disk .part not a contiguous prefix); restarting",
                    self.spec.model_id, resume_from,
                )
                resume_from = 0
                self._remove_temp()
                if await asyncio.to_thread(queries.list_segments, self.spec.download_id):
                    await asyncio.to_thread(
                        queries.replace_segments, self.spec.download_id, []
                    )
                await asyncio.to_thread(
                    queries.update_download, self.spec.download_id, bytes_done=0
                )
            self.state.segments = [SegmentRuntime(0, 0, end, resume_from)]
        self._recompute_bytes_done()
        return pr

    # ----- transfer -----

    async def _transfer(self, pr) -> None:
        self._writer = FileWriter(self.spec.temp_path)
        await self._writer.open()

        segmented = len(self.state.segments) > 1
        if segmented and self.state.total_bytes:
            await self._writer.preallocate(self.state.total_bytes)
            await self._run_segmented()
        else:
            await self._run_single()

        await self._writer.flush()

    async def _run_segmented(self) -> None:
        pending = [
            asyncio.ensure_future(self._run_segment(seg))
            for seg in self.state.segments
            if seg.bytes_done < seg.length
        ]
        if not pending:
            return
        done, not_done = await asyncio.wait(
            pending, return_when=asyncio.FIRST_EXCEPTION
        )
        first_exc: Optional[BaseException] = None
        for task in done:
            exc = task.exception()
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None:
            for task in not_done:
                task.cancel()
            await asyncio.gather(*not_done, return_exceptions=True)
            raise first_exc

    async def _run_segment(self, seg: SegmentRuntime) -> None:
        offset = seg.start + seg.bytes_done
        headers = {
            "Range": f"bytes={offset}-{seg.end}",
            "Accept-Encoding": "identity",
        }
        if self._etag:
            headers["If-Range"] = self._etag
        async with open_validated(
            "GET", self.spec.url, headers=headers
        ) as (resp, _final):
            if resp.status == 200:
                # Server ignored the range -> remote changed / no resume support.
                raise RemoteChanged()
            if resp.status not in (206,):
                self._raise_for_status(resp.status)
            async for chunk in resp.content.iter_chunked(args.download_chunk_size):
                self._check_control()
                # Never write past this segment's planned range: a
                # non-conforming 206 that returns more than the requested
                # bytes would otherwise overrun adjacent segments and the
                # preallocated file. Cap the write and abort on overflow.
                remaining = seg.length - seg.bytes_done
                if remaining <= 0:
                    raise FatalError(
                        f"segment {seg.idx}: server returned more than the "
                        f"requested {seg.length} bytes"
                    )
                overflow = len(chunk) > remaining
                if overflow:
                    chunk = chunk[:remaining]
                await self._writer.write_at(offset, chunk)
                offset += len(chunk)
                seg.bytes_done += len(chunk)
                self._recompute_bytes_done()
                await self._persist_progress()
                if overflow:
                    raise FatalError(
                        f"segment {seg.idx}: server returned more than the "
                        f"requested {seg.length} bytes"
                    )

    async def _run_single(self) -> None:
        seg = self.state.segments[0]
        offset = seg.bytes_done  # resume from here for single-stream
        headers = {"Accept-Encoding": "identity"}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
            if self._etag:
                headers["If-Range"] = self._etag
        async with open_validated(
            "GET", self.spec.url, headers=headers
        ) as (resp, _final):
            if offset > 0 and resp.status == 200:
                # Resume not honoured -> start over from the beginning. Truncate
                # the existing partial so stale trailing bytes from the prior
                # attempt cannot survive past the new (possibly shorter) end.
                offset = 0
                seg.bytes_done = 0
                self.state.bytes_done = 0
                await self._writer.truncate(0)
            elif offset > 0 and resp.status != 206:
                self._raise_for_status(resp.status)
            elif offset == 0 and resp.status != 200:
                self._raise_for_status(resp.status)
            # Byte ceiling for this stream: the known total when the server
            # reported a size, otherwise the configured maximum download size.
            # Without a bound, a non-conforming response or an unknown-length
            # stream (end == -1) that never closes could fill the disk (DoS).
            limit = (seg.end + 1) if seg.end >= 0 else self._max_download_bytes()
            async for chunk in resp.content.iter_chunked(args.download_chunk_size):
                self._check_control()
                overflow = False
                if limit is not None:
                    remaining = limit - offset
                    if remaining <= 0:
                        raise FatalError(
                            f"download exceeded the maximum size {limit} bytes"
                        )
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                        overflow = True
                await self._writer.write_at(offset, chunk)
                offset += len(chunk)
                seg.bytes_done = offset
                self.state.bytes_done = offset
                await self._persist_progress()
                if overflow:
                    raise FatalError(
                        f"download exceeded the maximum size {limit} bytes"
                    )

    def _max_download_bytes(self) -> Optional[int]:
        """Configured maximum download size in bytes, or ``None`` if disabled."""
        cap = getattr(args, "download_max_bytes", 0)
        return cap if cap and cap > 0 else None

    def _raise_for_status(self, status: int) -> None:
        if status in (401, 403):
            raise FatalError(
                f"{redact_url(self.spec.url)} returned {status}; authenticate this "
                f"host via /api/download/auth or set its API key env var."
            )
        if status in _RETRYABLE_STATUSES:
            raise RetryableError(f"HTTP {status}")
        raise FatalError(f"unexpected HTTP {status}")

    # ----- finalize / verify (PRD section 8.4) -----

    async def _finalize(self) -> None:
        self._check_control()
        await self._close_writer()
        await self._set_status(DownloadStatus.VERIFYING)

        total = self.state.total_bytes
        segmented = len(self.state.segments) > 1
        if segmented:
            # The .part was preallocated to total_bytes, so its on-disk size is
            # not evidence of completeness: a segment that ends short (truncated
            # 206 / server closes mid-range) leaves a zero-filled hole while the
            # file size still equals total. Verify each segment wrote its full
            # planned range, and trust the byte counter (== sum of segments)
            # rather than os.path.getsize for the total check.
            for seg in self.state.segments:
                if seg.bytes_done != seg.length:
                    raise FatalError(
                        f"segment {seg.idx} incomplete: wrote {seg.bytes_done} "
                        f"of {seg.length} bytes"
                    )
            observed = self.state.bytes_done
        else:
            # Single-stream writes a contiguous prefix, so the on-disk size is
            # an independent witness of how much actually landed.
            observed = os.path.getsize(self.spec.temp_path)
        if total is not None and observed != total:
            raise FatalError(
                f"size mismatch: wrote {observed} of {total} bytes"
            )

        # Structural gate (cheap, no full read) then optional sha256 (full read).
        # Both failures are non-retryable (a truncated/corrupt or mismatched file
        # will not heal on retry), so surface them as FatalError rather than
        # letting the plain Exceptions fall through to the retryable handler.
        # ``temp_path`` carries the ``.part`` suffix; pass ``dest_path`` so the
        # structural check detects the real file format instead of skipping it.
        try:
            await asyncio.to_thread(
                structural.validate, self.spec.temp_path, self.spec.dest_path
            )
            if self.spec.expected_sha256:
                await asyncio.to_thread(
                    checksum.verify_sha256,
                    self.spec.temp_path,
                    self.spec.expected_sha256,
                )
        except (structural.StructuralError, checksum.ChecksumError) as e:
            raise FatalError(str(e)) from e

        os.makedirs(os.path.dirname(self.spec.dest_path), exist_ok=True)
        os.replace(self.spec.temp_path, self.spec.dest_path)
        logging.info(
            "[model_downloader] completed %s (%d bytes)",
            self.spec.model_id, observed,
        )
        # Catalog into the assets system (blake3 dedup identity). Best-effort.
        await dedup.register_completed(self.spec.dest_path)

    # ----- helpers -----

    def _recompute_bytes_done(self) -> None:
        self.state.bytes_done = sum(s.bytes_done for s in self.state.segments)
        now = time.monotonic()
        dt = now - self.state._last_time
        if dt >= 0.5:
            self.state.speed_bps = (self.state.bytes_done - self.state._last_bytes) / dt
            self.state._last_bytes = self.state.bytes_done
            self.state._last_time = now

    async def _persist_progress(self, force: bool = False) -> None:
        # Both the DB write and the websocket notify are gated by the same
        # throttle: persisting hits SQLite, and notifying broadcasts to every
        # client, so doing either per-chunk (small --download-chunk-size or
        # many concurrent segments) would overwhelm both. Skip entirely inside
        # the window; the next persist (or a forced one) ships the latest bytes.
        now = time.monotonic()
        if not force and now - self._last_persist < _PERSIST_INTERVAL:
            return
        self._last_persist = now
        # SQLite is blocking; run it off the event loop per the queries module
        # contract so progress persists don't stall the web server.
        await asyncio.to_thread(self._write_progress)
        if self._notify:
            self._notify(self.spec.download_id)

    def _write_progress(self) -> None:
        queries.update_download(self.spec.download_id, bytes_done=self.state.bytes_done)
        for seg in self.state.segments:
            if seg.end >= seg.start:  # skip unknown-size sentinel
                queries.update_segment_progress(
                    self.spec.download_id, seg.idx, seg.bytes_done
                )

    async def _reset_for_restart(self) -> None:
        await self._close_writer()
        self._remove_temp()
        for seg in self.state.segments:
            seg.bytes_done = 0
        self.state.bytes_done = 0
        await asyncio.to_thread(
            queries.update_download, self.spec.download_id, bytes_done=0
        )
        if await asyncio.to_thread(queries.list_segments, self.spec.download_id):
            await asyncio.to_thread(
                queries.replace_segments, self.spec.download_id, []
            )

    async def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                await self._writer.close()
            except Exception:
                logging.debug("[model_downloader] writer close error", exc_info=True)
            self._writer = None

    def _segmented_part_valid(self, total_bytes: int) -> bool:
        """True when the temp file is the preallocated segmented ``.part``.

        A segmented transfer preallocates the .part to ``total_bytes`` up front
        and tracks how much of each range landed via per-segment offsets. Those
        offsets are only trustworthy when the file they describe is still on
        disk at its full preallocated size. A missing file (swept after a
        failure, removed on a fatal error, deleted by hand) or a wrong-sized one
        means the persisted offsets no longer correspond to real bytes and must
        not be resumed over. Doing so would skip "complete" segments and leave
        zero-filled holes that pass the size-only verification gate.
        """
        try:
            return os.path.getsize(self.spec.temp_path) == total_bytes
        except OSError:
            return False

    def _contiguous_prefix_valid(self, prefix_len: int) -> bool:
        """True when the temp file is exactly ``prefix_len`` contiguous bytes.

        Single-stream resume appends sequentially, so a valid resume point
        implies the .part size equals the persisted offset. A larger file (e.g.
        one preallocated to ``total_bytes`` by a previous segmented run) or a
        missing/short file means the persisted offset is not a trustworthy
        contiguous prefix and must not be resumed over.
        """
        try:
            return os.path.getsize(self.spec.temp_path) == prefix_len
        except OSError:
            return False

    def _remove_temp(self) -> None:
        try:
            os.remove(self.spec.temp_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            logging.warning(
                "[model_downloader] could not remove %s: %s", self.spec.temp_path, e
            )

    async def _set_status(self, status: str, error: Optional[str] = None) -> None:
        # ``error`` is authoritative: passing None clears any prior failure
        # text so transitions out of a failure state (retry/success) don't
        # leave stale messages on RuntimeState or in the persisted row.
        self.state.status = status
        self.state.error = error
        fields = {"status": status, "bytes_done": self.state.bytes_done, "error": error}
        if status == DownloadStatus.QUEUED:
            fields["attempts"] = self.spec.attempts + 1
            self.spec.attempts += 1
        await asyncio.to_thread(queries.update_download, self.spec.download_id, **fields)
        if self._notify:
            self._notify(self.spec.download_id)
