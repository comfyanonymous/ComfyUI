from blake3 import blake3
from typing import IO
import os
import asyncio


DEFAULT_CHUNK = 8 * 1024 *1024 # 8MB

# NOTE: this allows hashing different representations of a file-like object
def blake3_hash(
    fp: str | IO[bytes],
    chunk_size: int = DEFAULT_CHUNK,
) -> str:
    """
    Returns a BLAKE3 hex digest for ``fp``, which may be:
      - a filename (str/bytes) or PathLike
      - an open binary file object
    If ``fp`` is a file object, it must be opened in **binary** mode and support
    ``read``, ``seek``, and ``tell``. The function will seek to the start before
    reading and will attempt to restore the original position afterward.
    """
    # duck typing to check if input is a file-like object
    if hasattr(fp, "read"):
        return _hash_file_obj(fp, chunk_size)

    with open(os.fspath(fp), "rb") as f:
        return _hash_file_obj(f, chunk_size)


async def blake3_hash_async(
    fp: str | IO[bytes],
    chunk_size: int = DEFAULT_CHUNK,
) -> str:
    """Async wrapper for ``blake3_hash_sync``.
    Uses a worker thread so the event loop remains responsive.
    """
    # If it is a path, open inside the worker thread to keep I/O off the loop.
    if hasattr(fp, "read"):
        return await asyncio.to_thread(blake3_hash, fp, chunk_size)

    def _worker() -> str:
        with open(os.fspath(fp), "rb") as f:
            return _hash_file_obj(f, chunk_size)

    return await asyncio.to_thread(_worker)


def _hash_file_obj(file_obj: IO, chunk_size: int = DEFAULT_CHUNK) -> str:
    """
    Hash an already-open binary file object by streaming in chunks.
    - Seeks to the beginning before reading (if supported).
    - Restores the original position afterward (if tell/seek are supported).
    """
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK

    # in case file object is already open and not at the beginning, track so can be restored after hashing
    orig_pos = file_obj.tell()

    try:
        # seek to the beginning before reading
        if orig_pos != 0:
            file_obj.seek(0)

        h = blake3()
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
        return h.hexdigest()
    finally:
        # restore original position in file object, if needed
        if orig_pos != 0:
            file_obj.seek(orig_pos)
