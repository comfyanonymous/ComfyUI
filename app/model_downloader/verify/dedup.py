"""Dedup + catalog handoff — reuse the assets system.

We do NOT build a parallel indexer. "Do I already have it?" is answered by
``resolve_existing`` (path) at enqueue time and, where a hash is known, by the
assets blake3 catalog. After a completed download we register the file
through the assets ingest path so it is cataloged and (eventually) hashed by
the existing enrichment worker.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional


def _register_sync(abs_path: str) -> Optional[str]:
    """Register a finished file into the assets catalog. Returns asset hash."""
    try:
        from app.assets.services.ingest import register_file_in_place
    except Exception as e:  # assets package import failure — non-fatal
        logging.debug("[model_downloader] assets ingest unavailable: %s", e)
        return None
    try:
        result = register_file_in_place(abs_path, name=os.path.basename(abs_path), tags=[])
        return result.asset.hash if result and result.asset else None
    except Exception as e:
        # The file is already safely on disk; cataloging is best-effort.
        logging.warning(
            "[model_downloader] could not register %s into assets catalog: %s",
            abs_path, e,
        )
        return None


async def register_completed(abs_path: str) -> Optional[str]:
    """Catalog a completed download via the assets system (off the event loop)."""
    return await asyncio.to_thread(_register_sync, abs_path)


def _find_by_hash_sync(blake3_hex: str) -> Optional[str]:
    try:
        from app.assets.services.asset_management import get_asset_by_hash
    except Exception:
        return None
    asset = get_asset_by_hash("blake3:" + blake3_hex)
    return asset.hash if asset is not None else None


async def find_existing_by_hash(blake3_hex: str) -> Optional[str]:
    """Pure DB lookup — never triggers hashing on the hot path."""
    return await asyncio.to_thread(_find_by_hash_sync, blake3_hex)
