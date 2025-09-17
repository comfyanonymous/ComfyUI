import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import sqlalchemy as sa

import folder_paths

from ._assets_helpers import (
    collect_models_files,
    compute_relative_filename,
    get_comfy_models_folders,
    get_name_and_tags_from_asset_path,
    list_tree,
    new_scan_id,
    prefixes_for_root,
    ts_to_iso,
)
from .api import schemas_in, schemas_out
from .database.db import create_session
from .database.helpers import (
    add_missing_tag_for_asset_id,
    ensure_tags_exist,
    escape_like_prefix,
    fast_asset_file_check,
    remove_missing_tag_for_asset_id,
    seed_from_paths_batch,
)
from .database.models import Asset, AssetCacheState, AssetInfo
from .database.services import (
    compute_hash_and_dedup_for_cache_state,
    list_cache_states_by_asset_id,
    list_cache_states_with_asset_under_prefixes,
    list_unhashed_candidates_under_prefixes,
    list_verify_candidates_under_prefixes,
)

LOGGER = logging.getLogger(__name__)

SLOW_HASH_CONCURRENCY = 1


@dataclass
class ScanProgress:
    scan_id: str
    root: schemas_in.RootType
    status: Literal["scheduled", "running", "completed", "failed", "cancelled"] = "scheduled"
    scheduled_at: float = field(default_factory=lambda: time.time())
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    discovered: int = 0
    processed: int = 0
    file_errors: list[dict] = field(default_factory=list)


@dataclass
class SlowQueueState:
    queue: asyncio.Queue
    workers: list[asyncio.Task] = field(default_factory=list)
    closed: bool = False


RUNNING_TASKS: dict[schemas_in.RootType, asyncio.Task] = {}
PROGRESS_BY_ROOT: dict[schemas_in.RootType, ScanProgress] = {}
SLOW_STATE_BY_ROOT: dict[schemas_in.RootType, SlowQueueState] = {}


def current_statuses() -> schemas_out.AssetScanStatusResponse:
    scans = []
    for root in schemas_in.ALLOWED_ROOTS:
        prog = PROGRESS_BY_ROOT.get(root)
        if not prog:
            continue
        scans.append(_scan_progress_to_scan_status_model(prog))
    return schemas_out.AssetScanStatusResponse(scans=scans)


async def schedule_scans(roots: list[schemas_in.RootType]) -> schemas_out.AssetScanStatusResponse:
    results: list[ScanProgress] = []
    for root in roots:
        if root in RUNNING_TASKS and not RUNNING_TASKS[root].done():
            results.append(PROGRESS_BY_ROOT[root])
            continue

        prog = ScanProgress(scan_id=new_scan_id(root), root=root, status="scheduled")
        PROGRESS_BY_ROOT[root] = prog
        state = SlowQueueState(queue=asyncio.Queue())
        SLOW_STATE_BY_ROOT[root] = state
        RUNNING_TASKS[root] = asyncio.create_task(
            _run_hash_verify_pipeline(root, prog, state),
            name=f"asset-scan:{root}",
        )
        results.append(prog)
    return _status_response_for(results)


async def sync_seed_assets(roots: list[schemas_in.RootType]) -> None:
    t_total = time.perf_counter()
    created = 0
    skipped_existing = 0
    paths: list[str] = []
    try:
        existing_paths: set[str] = set()
        for r in roots:
            try:
                survivors = await _fast_db_consistency_pass(r, collect_existing_paths=True, update_missing_tags=True)
                if survivors:
                    existing_paths.update(survivors)
            except Exception as ex:
                LOGGER.exception("fast DB reconciliation failed for %s: %s", r, ex)

        if "models" in roots:
            paths.extend(collect_models_files())
        if "input" in roots:
            paths.extend(list_tree(folder_paths.get_input_directory()))
        if "output" in roots:
            paths.extend(list_tree(folder_paths.get_output_directory()))

        specs: list[dict] = []
        tag_pool: set[str] = set()
        for p in paths:
            ap = os.path.abspath(p)
            if ap in existing_paths:
                skipped_existing += 1
                continue
            try:
                st = os.stat(ap, follow_symlinks=True)
            except OSError:
                continue
            if not st.st_size:
                continue
            name, tags = get_name_and_tags_from_asset_path(ap)
            specs.append(
                {
                    "abs_path": ap,
                    "size_bytes": st.st_size,
                    "mtime_ns": getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)),
                    "info_name": name,
                    "tags": tags,
                    "fname": compute_relative_filename(ap),
                }
            )
            for t in tags:
                tag_pool.add(t)

        if not specs:
            return
        async with await create_session() as sess:
            if tag_pool:
                await ensure_tags_exist(sess, tag_pool, tag_type="user")

            result = await seed_from_paths_batch(sess, specs=specs, owner_id="")
            created += result["inserted_infos"]
            await sess.commit()
    finally:
        LOGGER.info(
            "Assets scan(roots=%s) completed in %.3fs (created=%d, skipped_existing=%d, total_seen=%d)",
            roots,
            time.perf_counter() - t_total,
            created,
            skipped_existing,
            len(paths),
        )


def _status_response_for(progresses: list[ScanProgress]) -> schemas_out.AssetScanStatusResponse:
    return schemas_out.AssetScanStatusResponse(scans=[_scan_progress_to_scan_status_model(p) for p in progresses])


def _scan_progress_to_scan_status_model(progress: ScanProgress) -> schemas_out.AssetScanStatus:
    return schemas_out.AssetScanStatus(
        scan_id=progress.scan_id,
        root=progress.root,
        status=progress.status,
        scheduled_at=ts_to_iso(progress.scheduled_at),
        started_at=ts_to_iso(progress.started_at),
        finished_at=ts_to_iso(progress.finished_at),
        discovered=progress.discovered,
        processed=progress.processed,
        file_errors=[
            schemas_out.AssetScanError(
                path=e.get("path", ""),
                message=e.get("message", ""),
                at=e.get("at"),
            )
            for e in (progress.file_errors or [])
        ],
    )


async def _run_hash_verify_pipeline(root: schemas_in.RootType, prog: ScanProgress, state: SlowQueueState) -> None:
    prog.status = "running"
    prog.started_at = time.time()
    try:
        prefixes = prefixes_for_root(root)

        await _fast_db_consistency_pass(root)

        # collect candidates from DB
        async with await create_session() as sess:
            verify_ids = await list_verify_candidates_under_prefixes(sess, prefixes=prefixes)
            unhashed_ids = await list_unhashed_candidates_under_prefixes(sess, prefixes=prefixes)
        # dedupe: prioritize verification first
        seen = set()
        ordered: list[int] = []
        for lst in (verify_ids, unhashed_ids):
            for sid in lst:
                if sid not in seen:
                    seen.add(sid)
                    ordered.append(sid)

        prog.discovered = len(ordered)

        # queue up work
        for sid in ordered:
            await state.queue.put(sid)
        state.closed = True
        _start_state_workers(root, prog, state)
        await _await_state_workers_then_finish(root, prog, state)
    except asyncio.CancelledError:
        prog.status = "cancelled"
        raise
    except Exception as exc:
        _append_error(prog, path="", message=str(exc))
        prog.status = "failed"
        prog.finished_at = time.time()
        LOGGER.exception("Asset scan failed for %s", root)
    finally:
        RUNNING_TASKS.pop(root, None)


async def _reconcile_missing_tags_for_root(root: schemas_in.RootType, prog: ScanProgress) -> None:
    """
    Detect missing files quickly and toggle 'missing' tag per asset_id.

    Rules:
      - Only hashed assets (assets.hash != NULL) participate in missing tagging.
      - We consider ALL cache states of the asset (across roots) before tagging.
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
            # state + hash + size for the current root
            rows = await list_cache_states_with_asset_under_prefixes(sess, prefixes=bases)

            # Track fast_ok within the scanned root and whether the asset is hashed
            by_asset: dict[str, dict[str, bool]] = {}
            for state, a_hash, size_db in rows:
                aid = state.asset_id
                acc = by_asset.get(aid)
                if acc is None:
                    acc = {"any_fast_ok_here": False, "hashed": (a_hash is not None), "size_db": int(size_db or 0)}
                    by_asset[aid] = acc
                try:
                    if acc["hashed"]:
                        st = os.stat(state.file_path, follow_symlinks=True)
                        if fast_asset_file_check(mtime_db=state.mtime_ns, size_db=acc["size_db"], stat_result=st):
                            acc["any_fast_ok_here"] = True
                except FileNotFoundError:
                    pass
                except OSError as e:
                    _append_error(prog, path=state.file_path, message=str(e))

            # Decide per asset, considering ALL its states (not just this root)
            for aid, acc in by_asset.items():
                try:
                    if not acc["hashed"]:
                        # Never tag seed assets as missing
                        continue

                    any_fast_ok_global = acc["any_fast_ok_here"]
                    if not any_fast_ok_global:
                        # Check other states outside this root
                        others = await list_cache_states_by_asset_id(sess, asset_id=aid)
                        for st in others:
                            try:
                                any_fast_ok_global = fast_asset_file_check(
                                    mtime_db=st.mtime_ns,
                                    size_db=acc["size_db"],
                                    stat_result=os.stat(st.file_path, follow_symlinks=True),
                                )
                            except OSError:
                                continue

                    if any_fast_ok_global:
                        await remove_missing_tag_for_asset_id(sess, asset_id=aid)
                    else:
                        await add_missing_tag_for_asset_id(sess, asset_id=aid, origin="automatic")
                except Exception as ex:
                    _append_error(prog, path="", message=f"reconcile {aid[:8]}: {ex}")

            await sess.commit()
    except Exception as e:
        _append_error(prog, path="", message=f"reconcile failed: {e}")


def _start_state_workers(root: schemas_in.RootType, prog: ScanProgress, state: SlowQueueState) -> None:
    if state.workers:
        return

    async def _worker(_wid: int):
        while True:
            sid = await state.queue.get()
            try:
                if sid is None:
                    return
                try:
                    async with await create_session() as sess:
                        # Optional: fetch path for better error messages
                        st = await sess.get(AssetCacheState, sid)
                        try:
                            await compute_hash_and_dedup_for_cache_state(sess, state_id=sid)
                            await sess.commit()
                        except Exception as e:
                            path = st.file_path if st else f"state:{sid}"
                            _append_error(prog, path=path, message=str(e))
                            raise
                except Exception:
                    pass
                finally:
                    prog.processed += 1
            finally:
                state.queue.task_done()

    state.workers = [
        asyncio.create_task(_worker(i), name=f"asset-hash:{root}:{i}")
        for i in range(SLOW_HASH_CONCURRENCY)
    ]

    async def _close_when_ready():
        while not state.closed:
            await asyncio.sleep(0.05)
        for _ in range(SLOW_HASH_CONCURRENCY):
            await state.queue.put(None)

    asyncio.create_task(_close_when_ready())


async def _await_state_workers_then_finish(
    root: schemas_in.RootType, prog: ScanProgress, state: SlowQueueState
) -> None:
    if state.workers:
        await asyncio.gather(*state.workers, return_exceptions=True)
    await _reconcile_missing_tags_for_root(root, prog)
    prog.finished_at = time.time()
    prog.status = "completed"


def _append_error(prog: ScanProgress, *, path: str, message: str) -> None:
    prog.file_errors.append({
        "path": path,
        "message": message,
        "at": ts_to_iso(time.time()),
    })


async def _fast_db_consistency_pass(
    root: schemas_in.RootType,
    *,
    collect_existing_paths: bool = False,
    update_missing_tags: bool = False,
) -> Optional[set[str]]:
    """Fast DB+FS pass for a root:
      - Toggle needs_verify per state using fast check
      - For hashed assets with at least one fast-ok state in this root: delete stale missing states
      - For seed assets with all states missing: delete Asset and its AssetInfos
      - Optionally add/remove 'missing' tags based on fast-ok in this root
      - Optionally return surviving absolute paths
    """
    prefixes = prefixes_for_root(root)
    if not prefixes:
        return set() if collect_existing_paths else None

    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_like_prefix(base)
        conds.append(AssetCacheState.file_path.like(escaped + "%", escape=esc))

    async with await create_session() as sess:
        rows = (
            await sess.execute(
                sa.select(
                    AssetCacheState.id,
                    AssetCacheState.file_path,
                    AssetCacheState.mtime_ns,
                    AssetCacheState.needs_verify,
                    AssetCacheState.asset_id,
                    Asset.hash,
                    Asset.size_bytes,
                )
                .join(Asset, Asset.id == AssetCacheState.asset_id)
                .where(sa.or_(*conds))
                .order_by(AssetCacheState.asset_id.asc(), AssetCacheState.id.asc())
            )
        ).all()

        by_asset: dict[str, dict] = {}
        for sid, fp, mtime_db, needs_verify, aid, a_hash, a_size in rows:
            acc = by_asset.get(aid)
            if acc is None:
                acc = {"hash": a_hash, "size_db": int(a_size or 0), "states": []}
                by_asset[aid] = acc

            fast_ok = False
            try:
                exists = True
                fast_ok = fast_asset_file_check(
                    mtime_db=mtime_db,
                    size_db=acc["size_db"],
                    stat_result=os.stat(fp, follow_symlinks=True),
                )
            except FileNotFoundError:
                exists = False
            except OSError:
                exists = False

            acc["states"].append({
                "sid": sid,
                "fp": fp,
                "exists": exists,
                "fast_ok": fast_ok,
                "needs_verify": bool(needs_verify),
            })

        to_set_verify: list[int] = []
        to_clear_verify: list[int] = []
        stale_state_ids: list[int] = []
        survivors: set[str] = set()

        for aid, acc in by_asset.items():
            a_hash = acc["hash"]
            states = acc["states"]
            any_fast_ok = any(s["fast_ok"] for s in states)
            all_missing = all(not s["exists"] for s in states)

            for s in states:
                if not s["exists"]:
                    continue
                if s["fast_ok"] and s["needs_verify"]:
                    to_clear_verify.append(s["sid"])
                if not s["fast_ok"] and not s["needs_verify"]:
                    to_set_verify.append(s["sid"])

            if a_hash is None:
                if states and all_missing:  # remove seed Asset completely, if no valid AssetCache exists
                    await sess.execute(sa.delete(AssetInfo).where(AssetInfo.asset_id == aid))
                    asset = await sess.get(Asset, aid)
                    if asset:
                        await sess.delete(asset)
                else:
                    for s in states:
                        if s["exists"]:
                            survivors.add(os.path.abspath(s["fp"]))
                continue

            if any_fast_ok:  # if Asset has at least one valid AssetCache record, remove any invalid AssetCache records
                for s in states:
                    if not s["exists"]:
                        stale_state_ids.append(s["sid"])
                if update_missing_tags:
                    with contextlib.suppress(Exception):
                        await remove_missing_tag_for_asset_id(sess, asset_id=aid)
            elif update_missing_tags:
                with contextlib.suppress(Exception):
                    await add_missing_tag_for_asset_id(sess, asset_id=aid, origin="automatic")

            for s in states:
                if s["exists"]:
                    survivors.add(os.path.abspath(s["fp"]))

        if stale_state_ids:
            await sess.execute(sa.delete(AssetCacheState).where(AssetCacheState.id.in_(stale_state_ids)))
        if to_set_verify:
            await sess.execute(
                sa.update(AssetCacheState)
                .where(AssetCacheState.id.in_(to_set_verify))
                .values(needs_verify=True)
            )
        if to_clear_verify:
            await sess.execute(
                sa.update(AssetCacheState)
                .where(AssetCacheState.id.in_(to_clear_verify))
                .values(needs_verify=False)
            )
        await sess.commit()
        return survivors if collect_existing_paths else None
