"""Positioned, off-loop file writes.

Network I/O stays on the event loop; every blocking disk op (preallocate,
positioned write, fsync) is run in a bounded thread pool via
``run_in_executor`` so downloads never stall inference or the web server.

A single file descriptor is opened for the whole download. Segments write to
their own offsets with ``os.pwrite`` — which is offset-addressed and atomic
per call, so concurrent segment writers need no extra locking. Per-chunk
fsync is avoided; we fsync once at completion.

``os.pwrite`` is unavailable on Windows, so there we fall back to
``os.lseek`` + ``os.write`` guarded by a per-writer lock (the seek/write pair
is not atomic, so concurrent segment writers must be serialized).
"""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# One shared, bounded pool for all download disk I/O.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dl-writer")

_HAS_PWRITE = hasattr(os, "pwrite")

# On Windows ``os.open`` defaults to text mode, which translates every ``\n``
# byte into ``\r\n`` on write and corrupts binary payloads (the file grows by
# one byte per 0x0A). ``O_BINARY`` disables that translation; it does not exist
# on POSIX, where the default is already binary.
_O_BINARY = getattr(os, "O_BINARY", 0)


class FileWriter:
    """Owns the ``.part`` file descriptor for one download."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd: Optional[int] = None
        # Serializes lseek+write on platforms without os.pwrite (Windows).
        self._seek_lock = threading.Lock()

    def _open(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | _O_BINARY, 0o644)

    async def open(self) -> None:
        await asyncio.get_running_loop().run_in_executor(_EXECUTOR, self._open)

    async def preallocate(self, size: int) -> None:
        """Grow the file to ``size`` so segments write to their offsets."""
        if self._fd is None or size <= 0:
            return
        await asyncio.get_running_loop().run_in_executor(
            _EXECUTOR, os.ftruncate, self._fd, size
        )

    async def truncate(self, size: int = 0) -> None:
        """Truncate the file to ``size`` bytes (default: empty it)."""
        if self._fd is None:
            return
        await asyncio.get_running_loop().run_in_executor(
            _EXECUTOR, os.ftruncate, self._fd, size
        )

    def _pwrite_all(self, data: bytes, offset: int) -> None:
        """A positioned write may write fewer bytes than requested (signal
        interruption, near-ENOSPC); loop until every byte lands so we never
        leave a gap while the caller advances by the full chunk length.

        Uses ``os.pwrite`` where available (offset-addressed, atomic per call).
        On Windows it falls back to ``os.lseek`` + ``os.write`` under a lock,
        since that pair is not atomic across concurrent segment writers."""
        assert self._fd is not None, "writer not opened"
        view = memoryview(data)
        written = 0
        total = len(view)
        while written < total:
            if _HAS_PWRITE:
                n = os.pwrite(self._fd, view[written:], offset + written)
            else:
                with self._seek_lock:
                    os.lseek(self._fd, offset + written, os.SEEK_SET)
                    n = os.write(self._fd, view[written:])
            if n == 0:
                raise OSError(
                    f"positioned write wrote 0 bytes at offset {offset + written} "
                    f"({written}/{total} bytes written)"
                )
            written += n

    async def write_at(self, offset: int, data: bytes) -> None:
        assert self._fd is not None, "writer not opened"
        await asyncio.get_running_loop().run_in_executor(
            _EXECUTOR, self._pwrite_all, data, offset
        )

    async def flush(self) -> None:
        if self._fd is None:
            return
        await asyncio.get_running_loop().run_in_executor(_EXECUTOR, os.fsync, self._fd)

    async def close(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        await asyncio.get_running_loop().run_in_executor(_EXECUTOR, os.close, fd)
