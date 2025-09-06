import asyncio
import os
from typing import IO, Union

from blake3 import blake3

DEFAULT_CHUNK = 8 * 1024 * 1024  # 8 MiB


def _hash_file_obj_sync(file_obj: IO[bytes], chunk_size: int) -> str:
    """Hash an already-open binary file object by streaming in chunks.
    - Seeks to the beginning before reading (if supported).
    - Restores the original position afterward (if tell/seek are supported).
    """
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK

    orig_pos = None
    if hasattr(file_obj, "tell"):
        orig_pos = file_obj.tell()

    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        h = blake3()
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
        return h.hexdigest()
    finally:
        if hasattr(file_obj, "seek") and orig_pos is not None:
            file_obj.seek(orig_pos)


def blake3_hash_sync(
    fp: Union[str, bytes, os.PathLike[str], os.PathLike[bytes], IO[bytes]],
    chunk_size: int = DEFAULT_CHUNK,
) -> str:
    """Returns a BLAKE3 hex digest for ``fp``, which may be:
      - a filename (str/bytes) or PathLike
      - an open binary file object

    If ``fp`` is a file object, it must be opened in **binary** mode and support
    ``read``, ``seek``, and ``tell``. The function will seek to the start before
    reading and will attempt to restore the original position afterward.
    """
    if hasattr(fp, "read"):
        return _hash_file_obj_sync(fp, chunk_size)

    with open(os.fspath(fp), "rb") as f:
        return _hash_file_obj_sync(f, chunk_size)


async def blake3_hash(
    fp: Union[str, bytes, os.PathLike[str], os.PathLike[bytes], IO[bytes]],
    chunk_size: int = DEFAULT_CHUNK,
) -> str:
    """Async wrapper for ``blake3_hash_sync``.
    Uses a worker thread so the event loop remains responsive.
    """
    # If it is a path, open inside the worker thread to keep I/O off the loop.
    if hasattr(fp, "read"):
        return await asyncio.to_thread(blake3_hash_sync, fp, chunk_size)

    def _worker() -> str:
        with open(os.fspath(fp), "rb") as f:
            return _hash_file_obj_sync(f, chunk_size)

    return await asyncio.to_thread(_worker)
