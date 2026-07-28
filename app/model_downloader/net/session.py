"""Lazily-created shared :class:`aiohttp.ClientSession`.

A single session reuses TLS handshakes and TCP connections across the probe
and the many segment GETs to the same host (HuggingFace is the dominant
case), which is a large speedup on cold connections and exactly the
connection-reuse strategy that lets us match aria2c.

The connector uses :class:`ValidatingResolver` so every connection — initial
or post-redirect — is screened for private/special-use IPs at connect time.
TLS is pinned to certifi's CA bundle because the OS trust store is not wired
up on some Python installs (python.org macOS, slim containers).
"""

from __future__ import annotations

import asyncio
import ssl
from typing import Optional

import aiohttp

try:
    import certifi
    _CA_FILE = certifi.where()
except Exception:  # pragma: no cover - certifi is a transitive dep of aiohttp
    _CA_FILE = None

from comfy.cli_args import args
from app.model_downloader.security.ssrf import ValidatingResolver

_session: Optional[aiohttp.ClientSession] = None
_lock = asyncio.Lock()


def ssl_context() -> ssl.SSLContext:
    if _CA_FILE is not None:
        return ssl.create_default_context(cafile=_CA_FILE)
    return ssl.create_default_context()


async def get_session() -> aiohttp.ClientSession:
    """Return the shared session, creating it on first use."""
    global _session
    if _session is not None and not _session.closed:
        return _session
    async with _lock:
        if _session is None or _session.closed:
            connector = aiohttp.TCPConnector(
                limit_per_host=max(1, getattr(args, "download_max_connections_per_host", 16)),
                ssl=ssl_context(),
                resolver=ValidatingResolver(),
            )
            _session = aiohttp.ClientSession(connector=connector)
    return _session


async def close_session() -> None:
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


def parse_int_header(value: Optional[str]) -> Optional[int]:
    """Parse a non-negative integer header value, or None if bad/absent."""
    if not value:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None
