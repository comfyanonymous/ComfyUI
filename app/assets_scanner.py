import asyncio
import contextlib
import logging
import os
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, Optional, Sequence

import folder_paths

from . import assets_manager
from .api import schemas_out
from ._assets_helpers import get_comfy_models_folders
from .database.db import create_session
from .database.services import (
    check_fs_asset_exists_quick,
    list_cache_states_under_prefixes,
    add_missing_tag_for_asset_hash,
    remove_missing_tag_for_asset_hash,
)

LOGGER = logging.getLogger(__name__)

RootType = Literal["models", "input", "output"]
ALLOWED_ROOTS: tuple[RootType, ...] = ("models", "input", "output")

SLOW_HASH_CONCURRENCY = 1


@dataclass
class ScanProgress:
    scan_id: str
    root: RootType
    status: Literal["scheduled", "running", "completed", "failed", "cancelled"] = "scheduled"
    scheduled_at: float = field(default_factory=lambda: time.time())
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    discovered: int = 0
    processed: int = 0
    slow_queue_total: int = 0
    slow_queue_finished: int = 0
    file_errors: list[dict] = field(default_factory=list)  # {"path","message","phase","at"}

    # Internal diagnostics for logs
    _fast_total_seen: int = 0
    _fast_clean: int = 0


@dataclass
class SlowQueueState:
    queue: asyncio.Queue
    workers: list[asyncio.Task] = field(default_factory=list)
    closed: bool = False


RUNNING_TASKS: dict[RootType, asyncio.Task] = {}
PROGRESS_BY_ROOT: dict[RootType, ScanProgress] = {}
SLOW_STATE_BY_ROOT: dict[RootType, SlowQueueState] = {}


async def start_background_assets_scan():
    await fast_reconcile_and_kickoff(progress_cb=_console_cb)


def current_statuses() -> schemas_out.AssetScanStatusResponse:
    scans = []
    for root in ALLOWED_ROOTS:
        prog = PROGRESS_BY_ROOT.get(root)
        if not prog:
            continue
        scans.append(_scan_progress_to_scan_status_model(prog))
    return schemas_out.AssetScanStatusResponse(scans=scans)


async def schedule_scans(roots: Sequence[str]) -> schemas_out.AssetScanStatusResponse:
    """Schedule scans for the provided roots; returns progress snapshots.

    Rules:
      - Only roots in {models, input, output} are accepted.
      - If a root is already scanning, we do NOT enqueue another one. Status returned as-is.
      - Otherwise a new task is created and started immediately.
      - Files with zero size are skipped.
    """
    normalized: list[RootType] = []
    seen = set()
    for r in roots or []:
        rr = r.strip().lower()
        if rr in ALLOWED_ROOTS and rr not in seen:
            normalized.append(rr)  # type: ignore
            seen.add(rr)
    if not normalized:
        normalized = list(ALLOWED_ROOTS)  # schedule all by default

    results: list[ScanProgress] = []
    for root in normalized:
        if root in RUNNING_TASKS and not RUNNING_TASKS[root].done():
            results.append(PROGRESS_BY_ROOT[root])
            continue

        prog = ScanProgress(scan_id=_new_scan_id(root), root=root, status="scheduled")
        PROGRESS_BY_ROOT[root] = prog
        SLOW_STATE_BY_ROOT[root] = SlowQueueState(queue=asyncio.Queue())
        RUNNING_TASKS[root] = asyncio.create_task(
            _pipeline_for_root(root, prog, progress_cb=None),
            name=f"asset-scan:{root}",
        )
        results.append(prog)
    return _status_response_for(results)


async def fast_reconcile_and_kickoff(
    roots: Optional[Sequence[str]] = None,
    *,
    progress_cb: Optional[Callable[[str, str, int, bool, dict], None]] = None,
) -> schemas_out.AssetScanStatusResponse:
    """
    Startup helper: do the fast pass now (so we know queue size),
    start slow hashing in the background, return immediately.
    """
    normalized = [*ALLOWED_ROOTS] if not roots else [r for r in roots if r in ALLOWED_ROOTS]
    snaps: list[ScanProgress] = []

    for root in normalized:
        if root in RUNNING_TASKS and not RUNNING_TASKS[root].done():
            snaps.append(PROGRESS_BY_ROOT[root])
            continue

        prog = ScanProgress(scan_id=_new_scan_id(root), root=root, status="scheduled")
        PROGRESS_BY_ROOT[root] = prog
        state = SlowQueueState(queue=asyncio.Queue())
        SLOW_STATE_BY_ROOT[root] = state

        prog.status = "running"
        prog.started_at = time.time()
        try:
            await _fast_reconcile_into_queue(root, prog, state, progress_cb=progress_cb)
        except Exception as e:
            _append_error(prog, phase="fast", path="", message=str(e))
            prog.status = "failed"
            prog.finished_at = time.time()
            LOGGER.exception("Fast reconcile failed for %s", root)
            snaps.append(prog)
            continue

        _start_slow_workers(root, prog, state, progress_cb=progress_cb)
        RUNNING_TASKS[root] = asyncio.create_task(
            _await_workers_then_finish(root, prog, state, progress_cb=progress_cb),
            name=f"asset-hash:{root}",
        )
        snaps.append(prog)
    return _status_response_for(snaps)


def _status_response_for(progresses: list[ScanProgress]) -> schemas_out.AssetScanStatusResponse:
    return schemas_out.AssetScanStatusResponse(scans=[_scan_progress_to_scan_status_model(p) for p in progresses])


def _scan_progress_to_scan_status_model(progress: ScanProgress) -> schemas_out.AssetScanStatus:
    return schemas_out.AssetScanStatus(
        scan_id=progress.scan_id,
        root=progress.root,
        status=progress.status,
        scheduled_at=_ts_to_iso(progress.scheduled_at),
        started_at=_ts_to_iso(progress.started_at),
        finished_at=_ts_to_iso(progress.finished_at),
        discovered=progress.discovered,
        processed=progress.processed,
        slow_queue_total=progress.slow_queue_total,
        slow_queue_finished=progress.slow_queue_finished,
        file_errors=[
            schemas_out.AssetScanError(
                path=e.get("path", ""),
                message=e.get("message", ""),
                phase=e.get("phase", "slow"),
                at=e.get("at"),
            )
            for e in (progress.file_errors or [])
        ],
    )


async def _pipeline_for_root(
    root: RootType,
    prog: ScanProgress,
    progress_cb: Optional[Callable[[str, str, int, bool, dict], None]],
) -> None:
    state = SLOW_STATE_BY_ROOT.get(root) or SlowQueueState(queue=asyncio.Queue())
    SLOW_STATE_BY_ROOT[root] = state

    prog.status = "running"
    prog.started_at = time.time()

    try:
        await _fast_reconcile_into_queue(root, prog, state, progress_cb=progress_cb)
        _start_slow_workers(root, prog, state, progress_cb=progress_cb)
        await _await_workers_then_finish(root, prog, state, progress_cb=progress_cb)
    except asyncio.CancelledError:
        prog.status = "cancelled"
        raise
    except Exception as exc:
        _append_error(prog, phase="slow", path="", message=str(exc))
        prog.status = "failed"
        prog.finished_at = time.time()
        LOGGER.exception("Asset scan failed for %s", root)
    finally:
        RUNNING_TASKS.pop(root, None)


async def _fast_reconcile_into_queue(
    root: RootType,
    prog: ScanProgress,
    state: SlowQueueState,
    *,
    progress_cb: Optional[Callable[[str, str, int, bool, dict], None]],
) -> None:
    """
    Enumerate files, set 'discovered' to total files seen, increment 'processed' for fast-matched files,
    and queue the rest for slow hashing.
    """
    if root == "models":
        files = _collect_models_files()
        preset_discovered = _count_nonzero_in_list(files)
        files_iter = asyncio.Queue()
        for p in files:
            await files_iter.put(p)
        await files_iter.put(None)  # sentinel for our local draining loop
    elif root == "input":
        base = folder_paths.get_input_directory()
        preset_discovered = _count_files_in_tree(os.path.abspath(base), only_nonzero=True)
        files_iter = await _queue_tree_files(base)
    elif root == "output":
        base = folder_paths.get_output_directory()
        preset_discovered = _count_files_in_tree(os.path.abspath(base), only_nonzero=True)
        files_iter = await _queue_tree_files(base)
    else:
        raise RuntimeError(f"Unsupported root: {root}")

    prog.discovered = int(preset_discovered or 0)

    queued = 0
    checked = 0
    clean = 0

    async with await create_session() as sess:
        while True:
            item = await files_iter.get()
            files_iter.task_done()
            if item is None:
                break

            abs_path = item
            checked += 1

            # Stat; skip empty/unreadable
            try:
                st = os.stat(abs_path, follow_symlinks=True)
                if not st.st_size:
                    continue
                size_bytes = int(st.st_size)
                mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
            except OSError as e:
                _append_error(prog, phase="fast", path=abs_path, message=str(e))
                continue

            try:
                known = await check_fs_asset_exists_quick(
                    sess,
                    file_path=abs_path,
                    size_bytes=size_bytes,
                    mtime_ns=mtime_ns,
                )
            except Exception as e:
                _append_error(prog, phase="fast", path=abs_path, message=str(e))
                known = False

            if known:
                clean += 1
                prog.processed += 1
            else:
                await state.queue.put(abs_path)
                queued += 1
                prog.slow_queue_total += 1

            if progress_cb:
                progress_cb(root, "fast", prog.processed, False, {
                    "checked": checked,
                    "clean": clean,
                    "queued": queued,
                    "discovered": prog.discovered,
                })

    prog._fast_total_seen = checked
    prog._fast_clean = clean

    if progress_cb:
        progress_cb(root, "fast", prog.processed, True, {
            "checked": checked,
            "clean": clean,
            "queued": queued,
            "discovered": prog.discovered,
        })

    await _reconcile_missing_tags_for_root(root, prog)
    state.closed = True


async def _reconcile_missing_tags_for_root(root: RootType, prog: ScanProgress) -> None:
    """
    For every AssetCacheState under the root's base directories:
      - if at least one recorded file_path exists for a hash -> remove 'missing'
      - if none of the recorded file_paths exist for a hash -> add 'missing'
    """
    if root == "models":
        bases: list[str] = []
        for _bucket, paths in get_comfy_models_folders():
            bases.extend(paths)
    elif root == "input":
        bases = [folder_paths.get_input_directory()]
    else:
        bases = [folder_paths.get_output_directory()]

    try:
        async with await create_session() as sess:
            states = await list_cache_states_under_prefixes(sess, prefixes=bases)

            present: set[str] = set()
            missing: set[str] = set()

            for s in states:
                try:
                    if os.path.isfile(s.file_path):
                        present.add(s.asset_hash)
                    else:
                        missing.add(s.asset_hash)
                except Exception as e:
                    _append_error(prog, phase="fast", path=s.file_path, message=f"stat error: {e}")

            only_missing = missing - present

            for h in present:
                with contextlib.suppress(Exception):
                    await remove_missing_tag_for_asset_hash(sess, asset_hash=h)

            for h in only_missing:
                with contextlib.suppress(Exception):
                    await add_missing_tag_for_asset_hash(sess, asset_hash=h, origin="automatic")

            await sess.commit()
    except Exception as e:
        _append_error(prog, phase="fast", path="", message=f"missing-tag reconcile failed: {e}")


def _start_slow_workers(
    root: RootType,
    prog: ScanProgress,
    state: SlowQueueState,
    *,
    progress_cb: Optional[Callable[[str, str, int, bool, dict], None]],
) -> None:
    if state.workers:
        return

    async def _worker(_worker_id: int):
        while True:
            item = await state.queue.get()
            try:
                if item is None:
                    return
                try:
                    await asyncio.to_thread(assets_manager.populate_db_with_asset, item)
                except Exception as e:
                    _append_error(prog, phase="slow", path=item, message=str(e))
                finally:
                    # Slow queue finished for this item; also counts toward overall processed
                    prog.slow_queue_finished += 1
                    prog.processed += 1
                    if progress_cb:
                        progress_cb(root, "slow", prog.processed, False, {
                            "slow_queue_finished": prog.slow_queue_finished,
                            "slow_queue_total": prog.slow_queue_total,
                        })
            finally:
                state.queue.task_done()

    state.workers = [asyncio.create_task(_worker(i), name=f"asset-hash:{root}:{i}") for i in range(SLOW_HASH_CONCURRENCY)]

    async def _close_when_empty():
        # When the fast phase closed the queue, push sentinels to end workers
        while not state.closed:
            await asyncio.sleep(0.05)
        for _ in range(SLOW_HASH_CONCURRENCY):
            await state.queue.put(None)

    asyncio.create_task(_close_when_empty())


async def _await_workers_then_finish(
    root: RootType,
    prog: ScanProgress,
    state: SlowQueueState,
    *,
    progress_cb: Optional[Callable[[str, str, int, bool, dict], None]],
) -> None:
    if state.workers:
        await asyncio.gather(*state.workers, return_exceptions=True)
    prog.finished_at = time.time()
    prog.status = "completed"
    if progress_cb:
        progress_cb(root, "slow", prog.processed, True, {
            "slow_queue_finished": prog.slow_queue_finished,
            "slow_queue_total": prog.slow_queue_total,
        })


def _collect_models_files() -> list[str]:
    """Collect absolute file paths from configured model buckets under models_dir."""
    out: list[str] = []
    for folder_name, bases in get_comfy_models_folders():
        rel_files = folder_paths.get_filename_list(folder_name) or []
        for rel_path in rel_files:
            abs_path = folder_paths.get_full_path(folder_name, rel_path)
            if not abs_path:
                continue
            abs_path = os.path.abspath(abs_path)
            # ensure within allowed bases
            allowed = False
            for b in bases:
                base_abs = os.path.abspath(b)
                with contextlib.suppress(Exception):
                    if os.path.commonpath([abs_path, base_abs]) == base_abs:
                        allowed = True
                        break
            if allowed:
                out.append(abs_path)
    return out


def _count_files_in_tree(base_abs: str, *, only_nonzero: bool = False) -> int:
    if not os.path.isdir(base_abs):
        return 0
    total = 0
    for dirpath, _subdirs, filenames in os.walk(base_abs, topdown=True, followlinks=False):
        if not only_nonzero:
            total += len(filenames)
        else:
            for name in filenames:
                with contextlib.suppress(OSError):
                    st = os.stat(os.path.join(dirpath, name), follow_symlinks=True)
                    if st.st_size:
                        total += 1
    return total


def _count_nonzero_in_list(paths: list[str]) -> int:
    cnt = 0
    for p in paths:
        with contextlib.suppress(OSError):
            st = os.stat(p, follow_symlinks=True)
            if st.st_size:
                cnt += 1
    return cnt


async def _queue_tree_files(base_dir: str) -> asyncio.Queue:
    """
    Walk base_dir in a worker thread and return a queue prefilled with all paths,
    terminated by a single None sentinel for the draining loop in fast reconcile.
    """
    q: asyncio.Queue = asyncio.Queue()
    base_abs = os.path.abspath(base_dir)
    if not os.path.isdir(base_abs):
        await q.put(None)
        return q

    def _walk_list():
        paths: list[str] = []
        for dirpath, _subdirs, filenames in os.walk(base_abs, topdown=True, followlinks=False):
            for name in filenames:
                paths.append(os.path.abspath(os.path.join(dirpath, name)))
        return paths

    for p in await asyncio.to_thread(_walk_list):
        await q.put(p)
    await q.put(None)
    return q


def _append_error(prog: ScanProgress, *, phase: Literal["fast", "slow"], path: str, message: str) -> None:
    prog.file_errors.append({
        "path": path,
        "message": message,
        "phase": phase,
        "at": _ts_to_iso(time.time()),
    })


def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    # interpret ts as seconds since epoch UTC and return naive UTC (consistent with other models)
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None).isoformat()
    except Exception:
        return None


def _new_scan_id(root: RootType) -> str:
    return f"scan-{root}-{uuid.uuid4().hex[:8]}"


def _console_cb(root: str, phase: str, total_processed: int, finished: bool, e: dict):
    if phase == "fast":
        if finished:
            logging.info(
                "[assets][%s] fast done: processed=%s/%s queued=%s",
                root,
                total_processed,
                e["discovered"],
                e["queued"],
            )
        elif e.get("checked", 0) % 1000 == 0:  # do not spam with fast progress
            logging.info(
                "[assets][%s] fast progress: processed=%s/%s",
                root,
                total_processed,
                e["discovered"],
            )
    elif phase == "slow":
        if finished:
            if e.get("slow_queue_finished", 0) or e.get("slow_queue_total", 0):
                logging.info(
                    "[assets][%s] slow done: %s/%s",
                    root,
                    e.get("slow_queue_finished", 0),
                    e.get("slow_queue_total", 0),
                )
        elif e.get('slow_queue_finished', 0) % 3 == 0:
            logging.info(
                "[assets][%s] slow progress: %s/%s",
                root,
                e.get("slow_queue_finished", 0),
                e.get("slow_queue_total", 0),
            )
