import asyncio
import logging
import os
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional, Sequence

import folder_paths

from . import assets_manager
from .api import schemas_out
from ._assets_helpers import get_comfy_models_folders

LOGGER = logging.getLogger(__name__)

RootType = Literal["models", "input", "output"]
ALLOWED_ROOTS: tuple[RootType, ...] = ("models", "input", "output")

# We run at most one scan per root; overall max parallelism is therefore 3
# We also bound per-scan ingestion concurrency to avoid swamping threads/DB
DEFAULT_PER_SCAN_CONCURRENCY = 1


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
    errors: int = 0
    last_error: Optional[str] = None

    # Optional details for diagnostics (e.g., files per bucket)
    details: dict[str, int] = field(default_factory=dict)


RUNNING_TASKS: dict[RootType, asyncio.Task] = {}
PROGRESS_BY_ROOT: dict[RootType, ScanProgress] = {}


def _new_scan_id(root: RootType) -> str:
    return f"scan-{root}-{uuid.uuid4().hex[:8]}"


def current_statuses() -> schemas_out.AssetScanStatusResponse:
    return schemas_out.AssetScanStatusResponse(
        scans=[
            schemas_out.AssetScanStatus(
                scan_id=s.scan_id,
                root=s.root,
                status=s.status,
                scheduled_at=_ts_to_iso(s.scheduled_at),
                started_at=_ts_to_iso(s.started_at),
                finished_at=_ts_to_iso(s.finished_at),
                discovered=s.discovered,
                processed=s.processed,
                errors=s.errors,
                last_error=s.last_error,
            )
            for s in [PROGRESS_BY_ROOT[r] for r in ALLOWED_ROOTS if r in PROGRESS_BY_ROOT]
        ]
    )


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
        if not isinstance(r, str):
            continue
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

        task = asyncio.create_task(_run_scan_for_root(root, prog), name=f"asset-scan:{root}")
        RUNNING_TASKS[root] = task
        results.append(prog)

    return schemas_out.AssetScanStatusResponse(
        scans=[
            schemas_out.AssetScanStatus(
                scan_id=s.scan_id,
                root=s.root,
                status=s.status,
                scheduled_at=_ts_to_iso(s.scheduled_at),
                started_at=_ts_to_iso(s.started_at),
                finished_at=_ts_to_iso(s.finished_at),
                discovered=s.discovered,
                processed=s.processed,
                errors=s.errors,
                last_error=s.last_error,
            )
            for s in results
        ]
    )


async def _run_scan_for_root(root: RootType, prog: ScanProgress) -> None:
    prog.started_at = time.time()
    prog.status = "running"
    try:
        if root == "models":
            await _scan_models(prog)
        elif root == "input":
            base = folder_paths.get_input_directory()
            await _scan_directory_tree(base, root, prog)
        elif root == "output":
            base = folder_paths.get_output_directory()
            await _scan_directory_tree(base, root, prog)
        else:
            raise RuntimeError(f"Unsupported root: {root}")
        prog.status = "completed"
    except asyncio.CancelledError:
        prog.status = "cancelled"
        raise
    except Exception as exc:
        LOGGER.exception("Asset scan failed for %s", root)
        prog.status = "failed"
        prog.errors += 1
        prog.last_error = str(exc)
    finally:
        prog.finished_at = time.time()
        t = RUNNING_TASKS.get(root)
        if t and t.done():
            RUNNING_TASKS.pop(root, None)


async def _scan_models(prog: ScanProgress) -> None:
    """
    Scan all configured model buckets from folder_paths.folder_names_and_paths,
    restricted to entries whose base paths lie under folder_paths.models_dir
    (per get_comfy_models_folders). We trust those mappings and do not try to
    infer anything else here.
    """
    targets: list[tuple[str, list[str]]] = get_comfy_models_folders()

    plans: list[str] = []  # absolute file paths to ingest
    per_bucket: dict[str, int] = {}

    for folder_name, bases in targets:
        rel_files = folder_paths.get_filename_list(folder_name) or []
        count_valid = 0

        for rel_path in rel_files:
            abs_path = folder_paths.get_full_path(folder_name, rel_path)
            if not abs_path:
                continue
            abs_path = os.path.abspath(abs_path)

            # Extra safety: ensure file is inside one of the allowed base paths
            allowed = False
            for base in bases:
                base_abs = os.path.abspath(base)
                try:
                    common = os.path.commonpath([abs_path, base_abs])
                except ValueError:
                    common = ""  # Different drives on Windows
                if common == base_abs:
                    allowed = True
                    break
            if not allowed:
                LOGGER.warning("Skipping file outside models base: %s", abs_path)
                continue

            try:
                if not os.path.getsize(abs_path):
                    continue  # skip empty files
            except OSError as e:
                LOGGER.warning("Could not stat %s: %s – skipping", abs_path, e)
                continue

            plans.append(abs_path)
            count_valid += 1

        if count_valid:
            per_bucket[folder_name] = per_bucket.get(folder_name, 0) + count_valid

    prog.discovered = len(plans)
    for k, v in per_bucket.items():
        prog.details[k] = prog.details.get(k, 0) + v

    if not plans:
        LOGGER.info("Model scan %s: nothing to ingest", prog.scan_id)
        return

    sem = asyncio.Semaphore(DEFAULT_PER_SCAN_CONCURRENCY)
    tasks: list[asyncio.Task] = []

    for abs_path in plans:
        async def worker(fp_abs: str = abs_path):
            try:
                # Offload sync ingestion into a thread; populate_db_with_asset
                # derives name and tags from the path using _assets_helpers.
                await asyncio.to_thread(assets_manager.populate_db_with_asset, fp_abs)
            except Exception as e:
                prog.errors += 1
                prog.last_error = str(e)
                LOGGER.debug("Error ingesting %s: %s", fp_abs, e)
            finally:
                prog.processed += 1
                sem.release()

        await sem.acquire()
        tasks.append(asyncio.create_task(worker()))

    if tasks:
        await asyncio.gather(*tasks)
    LOGGER.info(
        "Model scan %s finished: discovered=%d processed=%d errors=%d",
        prog.scan_id, prog.discovered, prog.processed, prog.errors
    )


def _count_files_in_tree(base_abs: str) -> int:
    if not os.path.isdir(base_abs):
        return 0
    total = 0
    for _dirpath, _subdirs, filenames in os.walk(base_abs, topdown=True, followlinks=False):
        total += len(filenames)
    return total


async def _scan_directory_tree(base_dir: str, root: RootType, prog: ScanProgress) -> None:
    """
    Generic scanner for input/output roots. We pass only the absolute path to
    populate_db_with_asset and let it derive the relative name and tags.
    """
    base_abs = os.path.abspath(base_dir)
    if not os.path.isdir(base_abs):
        LOGGER.info("Scan root %s skipped: base directory missing: %s", root, base_abs)
        return

    prog.discovered = _count_files_in_tree(base_abs)

    sem = asyncio.Semaphore(DEFAULT_PER_SCAN_CONCURRENCY)
    tasks: list[asyncio.Task] = []
    for dirpath, _subdirs, filenames in os.walk(base_abs, topdown=True, followlinks=False):
        for name in filenames:
            abs_path = os.path.abspath(os.path.join(dirpath, name))

            # Safety: ensure within base
            try:
                if os.path.commonpath([abs_path, base_abs]) != base_abs:
                    LOGGER.warning("Skipping path outside root %s: %s", root, abs_path)
                    continue
            except ValueError:
                continue

            # Skip empty files and handle stat errors
            try:
                if not os.path.getsize(abs_path):
                    continue
            except OSError as e:
                LOGGER.warning("Could not stat %s: %s – skipping", abs_path, e)
                continue

            async def worker(fp_abs: str = abs_path):
                try:
                    await asyncio.to_thread(assets_manager.populate_db_with_asset, fp_abs)
                except Exception as e:
                    prog.errors += 1
                    prog.last_error = str(e)
                finally:
                    prog.processed += 1
                    sem.release()

            await sem.acquire()
            tasks.append(asyncio.create_task(worker()))

    if tasks:
        await asyncio.gather(*tasks)

    LOGGER.info(
        "%s scan %s finished: discovered=%d processed=%d errors=%d",
        root.capitalize(), prog.scan_id, prog.discovered, prog.processed, prog.errors
    )


def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    # interpret ts as seconds since epoch UTC and return naive UTC (consistent with other models)
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None).isoformat()
    except Exception:
        return None
