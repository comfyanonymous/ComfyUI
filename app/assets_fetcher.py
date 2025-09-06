from __future__ import annotations
import asyncio
import os
import tempfile
from typing import Optional
import mimetypes
import aiohttp

from .storage.hashing import blake3_hash_sync
from .database.db import create_session
from .database.services import ingest_fs_asset, list_cache_states_by_asset_hash
from .resolvers import resolve_asset
from ._assets_helpers import resolve_destination_from_tags, ensure_within_base

_FETCH_LOCKS: dict[str, asyncio.Lock] = {}


def _sanitize_filename(name: str) -> str:
    return os.path.basename((name or "").strip()) or "file"


async def ensure_asset_cached(
    asset_hash: str,
    *,
    preferred_name: Optional[str] = None,
    tags_hint: Optional[list[str]] = None,
) -> str:
    """
    Ensure there is a verified local file for asset_hash in the correct Comfy folder.

    Fast path:
      - If any cache_state row has a file_path that exists, return it immediately.
        Preference order is the oldest ID first for stability.

    Slow path:
      - Resolve remote location + placement tags.
      - Download to the correct folder, verify hash, move into place.
      - Ingest identity + cache state so future fast passes can skip hashing.
    """
    lock = _FETCH_LOCKS.setdefault(asset_hash, asyncio.Lock())
    async with lock:
        # 1) If we already have any cache_state path present on disk, use it (oldest-first)
        async with await create_session() as sess:
            states = await list_cache_states_by_asset_hash(sess, asset_hash=asset_hash)
            for s in states:
                if s and s.file_path and os.path.isfile(s.file_path):
                    return s.file_path

        # 2) Resolve remote location + placement hints (must include valid tags)
        res = await resolve_asset(asset_hash)
        if not res:
            raise FileNotFoundError(f"No resolver/locations for {asset_hash}")

        placement_tags = tags_hint or res.tags
        if not placement_tags:
            raise ValueError(f"Resolver did not provide placement tags for {asset_hash}")

        name_hint = res.filename or preferred_name or asset_hash.replace(":", "_")
        safe_name = _sanitize_filename(name_hint)

        # 3) Map tags -> destination (strict: raises if invalid root or models category)
        base_dir, subdirs = resolve_destination_from_tags(placement_tags)  # may raise
        dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
        os.makedirs(dest_dir, exist_ok=True)

        final_path = os.path.abspath(os.path.join(dest_dir, safe_name))
        ensure_within_base(final_path, base_dir)

        # 4) If target path exists, try to reuse; else delete invalid cache
        if os.path.exists(final_path) and os.path.isfile(final_path):
            existing_digest = blake3_hash_sync(final_path)
            if f"blake3:{existing_digest}" == asset_hash:
                size_bytes = os.path.getsize(final_path)
                mtime_ns = getattr(os.stat(final_path), "st_mtime_ns", int(os.path.getmtime(final_path) * 1_000_000_000))
                async with await create_session() as sess:
                    await ingest_fs_asset(
                        sess,
                        asset_hash=asset_hash,
                        abs_path=final_path,
                        size_bytes=size_bytes,
                        mtime_ns=mtime_ns,
                        mime_type=None,
                        info_name=None,
                        tags=(),
                    )
                    await sess.commit()
                return final_path
            else:
                # Invalid cache: remove before re-downloading
                os.remove(final_path)

        # 5) Download to temp next to destination
        timeout = aiohttp.ClientTimeout(total=60 * 30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(res.download_url, headers=dict(res.headers)) as resp:
                resp.raise_for_status()
                cl = resp.headers.get("Content-Length")
                if res.expected_size and cl and int(cl) != int(res.expected_size):
                    raise ValueError("server Content-Length does not match expected size")
                with tempfile.NamedTemporaryFile("wb", delete=False, dir=dest_dir) as tmp:
                    tmp_path = tmp.name
                    async for chunk in resp.content.iter_chunked(8 * 1024 * 1024):
                        if chunk:
                            tmp.write(chunk)

        # 6) Verify content hash
        digest = blake3_hash_sync(tmp_path)
        canonical = f"blake3:{digest}"
        if canonical != asset_hash:
            try:
                os.remove(tmp_path)
            finally:
                raise ValueError(f"Hash mismatch: expected {asset_hash}, got {canonical}")

        # 7) Atomically move into place
        if os.path.exists(final_path):
            os.remove(final_path)
        os.replace(tmp_path, final_path)

        # 8) Record identity + cache state (+ mime type)
        size_bytes = os.path.getsize(final_path)
        mtime_ns = getattr(os.stat(final_path), "st_mtime_ns", int(os.path.getmtime(final_path) * 1_000_000_000))
        mime_type = mimetypes.guess_type(safe_name, strict=False)[0]
        async with await create_session() as sess:
            await ingest_fs_asset(
                sess,
                asset_hash=asset_hash,
                abs_path=final_path,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                mime_type=mime_type,
                info_name=None,
                tags=(),
            )
            await sess.commit()

        return final_path
