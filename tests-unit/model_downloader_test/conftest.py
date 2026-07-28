"""Shared fixtures for the model download manager tests.

These run in-process (no ComfyUI subprocess): a file-backed SQLite DB is
initialized once, a temp model folder is registered with ``folder_paths``, and
the shared aiohttp session is reset between tests so each async test gets a
session bound to its own event loop.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest


def _drain_scheduler_tasks(scheduler) -> None:
    """Cancel and await live scheduler tasks so none outlive the test.

    Uses the actual task handles rather than only clearing ``_tasks``: each
    per-test event loop is created by ``asyncio.run``, so a task left behind by
    a crashed/aborted test would otherwise keep its coroutine alive. We cancel
    every live task and, when its loop is still usable, run it to completion to
    let the cancellation propagate before dropping the reference.
    """
    for task in list(scheduler._tasks.values()):
        if task is None:
            continue
        loop = task.get_loop()
        if task.done() or loop.is_closed():
            continue
        task.cancel()
        if not loop.is_running():
            try:
                loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
            except Exception:
                pass
    scheduler._tasks.clear()


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    import app.database.db as db
    from comfy.cli_args import args

    fd, db_path = tempfile.mkstemp(suffix="-dlmgr-test.sqlite3")
    os.close(fd)
    args.database_url = f"sqlite:///{db_path}"
    db.init_db()
    yield
    try:
        os.remove(db_path)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _reset_runtime():
    """Reset module singletons that hold event-loop-bound or cross-test state."""
    import app.model_downloader.net.session as ns
    from app.model_downloader.scheduler import SCHEDULER

    ns._session = None
    _drain_scheduler_tasks(SCHEDULER)
    SCHEDULER._jobs.clear()
    SCHEDULER._backoff_until.clear()
    SCHEDULER._started = False
    yield
    _drain_scheduler_tasks(SCHEDULER)
    ns._session = None


@pytest.fixture
def model_root(tmp_path):
    """Register a temp 'loras' model folder and return its absolute path."""
    import folder_paths

    root = tmp_path / "loras"
    root.mkdir(parents=True, exist_ok=True)
    saved = folder_paths.folder_names_and_paths.get("loras")
    folder_paths.folder_names_and_paths["loras"] = (
        [str(root)],
        {".safetensors", ".sft", ".ckpt", ".pt", ".pth"},
    )
    yield str(root)
    if saved is not None:
        folder_paths.folder_names_and_paths["loras"] = saved
    else:
        folder_paths.folder_names_and_paths.pop("loras", None)
