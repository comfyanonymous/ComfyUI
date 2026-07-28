"""Unit tests for ``DownloadManager.delete`` and ``DownloadManager.clear``.

Deleting a terminal row must remove it from history for good (so it does not
reappear on the next ``list``), leave live rows untouched, and clean up any
leftover ``.part`` temp file without touching the finished model file.

``clear()`` is the bulk variant: it removes all terminal rows atomically, skips
live ones, and returns the count of rows deleted.

Async methods are driven via ``asyncio.run`` so no pytest-asyncio plugin is
required.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database import queries
from app.model_downloader.manager import DOWNLOAD_MANAGER, DownloadError


def _insert(download_id: str, status: str, *, temp_path: str = "/tmp/none.part") -> None:
    queries.insert_download(
        {
            "id": download_id,
            "url": "https://huggingface.co/org/model.safetensors",
            "model_id": "loras/model.safetensors",
            "dest_path": "/tmp/model.safetensors",
            "temp_path": temp_path,
            "status": status,
            "priority": 0,
        }
    )


def test_delete_removes_terminal_row_from_history():
    _insert("done", DownloadStatus.COMPLETED)

    asyncio.run(DOWNLOAD_MANAGER.delete("done"))

    assert queries.get_download("done") is None


def test_delete_refuses_live_row():
    _insert("live", DownloadStatus.QUEUED)

    with pytest.raises(DownloadError) as excinfo:
        asyncio.run(DOWNLOAD_MANAGER.delete("live"))

    assert excinfo.value.code == "DOWNLOAD_ACTIVE"
    assert queries.get_download("live") is not None


def test_delete_missing_row_raises_not_found():
    with pytest.raises(DownloadError) as excinfo:
        asyncio.run(DOWNLOAD_MANAGER.delete("nope"))

    assert excinfo.value.code == "NOT_FOUND"


def test_delete_removes_leftover_temp_file(tmp_path):
    partial = tmp_path / "model.safetensors.part"
    partial.write_bytes(b"partial")
    _insert("failed", DownloadStatus.FAILED, temp_path=str(partial))

    asyncio.run(DOWNLOAD_MANAGER.delete("failed"))

    assert not os.path.exists(partial)
    assert queries.get_download("failed") is None


# ----- clear -----


def test_clear_removes_all_terminal_rows():
    _insert("c-done", DownloadStatus.COMPLETED)
    _insert("c-fail", DownloadStatus.FAILED)
    _insert("c-canc", DownloadStatus.CANCELLED)

    deleted = asyncio.run(DOWNLOAD_MANAGER.clear())

    assert deleted == 3
    assert queries.get_download("c-done") is None
    assert queries.get_download("c-fail") is None
    assert queries.get_download("c-canc") is None


def test_clear_skips_live_rows():
    _insert("cl-queued", DownloadStatus.QUEUED)
    _insert("cl-paused", DownloadStatus.PAUSED)
    _insert("cl-done", DownloadStatus.COMPLETED)

    deleted = asyncio.run(DOWNLOAD_MANAGER.clear())

    assert deleted == 1
    assert queries.get_download("cl-queued") is not None
    assert queries.get_download("cl-paused") is not None
    assert queries.get_download("cl-done") is None


def test_clear_returns_zero_when_nothing_to_delete():
    _insert("cl-only-live", DownloadStatus.QUEUED)

    deleted = asyncio.run(DOWNLOAD_MANAGER.clear())

    assert deleted == 0
    assert queries.get_download("cl-only-live") is not None


def test_clear_removes_leftover_temp_files(tmp_path):
    partial = tmp_path / "clear_partial.part"
    partial.write_bytes(b"partial data")
    finished = tmp_path / "finished.safetensors"
    finished.write_bytes(b"real model weights")

    _insert("cl-part", DownloadStatus.FAILED, temp_path=str(partial))
    # The finished file is not the temp_path; temp_path for a completed download
    # no longer exists (already renamed), so use a non-existent path here to
    # verify clear() tolerates a missing temp file without raising.
    _insert("cl-comp", DownloadStatus.COMPLETED, temp_path=str(tmp_path / "gone.part"))

    asyncio.run(DOWNLOAD_MANAGER.clear())

    # Leftover .part from the failed download is cleaned up.
    assert not partial.exists()
    # Finished model file is never touched.
    assert finished.exists()


def test_clear_empty_db_returns_zero():
    deleted = asyncio.run(DOWNLOAD_MANAGER.clear())
    assert deleted == 0
