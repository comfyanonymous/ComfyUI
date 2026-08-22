"""Stable file snapshot hashing."""

from __future__ import annotations

import os
from dataclasses import dataclass

from blake3 import blake3


@dataclass(frozen=True, slots=True)
class _Snapshot:
    dev: int
    ino: int
    mtime_ns: int
    size: int


def _snapshot(stat_result: os.stat_result) -> _Snapshot:
    return _Snapshot(
        dev=stat_result.st_dev,
        ino=stat_result.st_ino,
        mtime_ns=stat_result.st_mtime_ns,
        size=stat_result.st_size,
    )


def snapshot_hash(path: str, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    """Return a BLAKE3 digest only when all path and descriptor snapshots match."""
    pre_stat = _snapshot(os.stat(path))
    hasher = blake3()
    with open(path, "rb") as file:
        open_stat = _snapshot(os.fstat(file.fileno()))
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
        post_hash_stat = _snapshot(os.fstat(file.fileno()))
    try:
        post_stat = _snapshot(os.stat(path))
    except FileNotFoundError:
        return None
    if len({pre_stat, open_stat, post_hash_stat, post_stat}) != 1:
        return None
    return hasher.hexdigest()
