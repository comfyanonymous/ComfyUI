"""Lazy module-level aiohttp ClientSession.

A single shared session means TLS handshakes are reused across HEAD probes
and the subsequent GETs to the same host (HuggingFace is the dominant
case), which is a noticeable speedup on cold connections.

We deliberately don't close the session at process exit — aiohttp's
warning about unclosed sessions is benign at shutdown, and adding atexit
plumbing buys nothing because the OS reclaims the sockets anyway. The
session lifetime is the lifetime of the Python process.
"""

from __future__ import annotations

import asyncio
import ssl
from typing import Optional

import aiohttp
import certifi


# Larger per-host pool than aiohttp's default (=100 total / =0 per host)
# so concurrent gated probes + a download to the same host don't queue.
_CONNECTOR_LIMIT_PER_HOST = 8

_session: Optional[aiohttp.ClientSession] = None
_lock = asyncio.Lock()


def ssl_context() -> ssl.SSLContext:
    """TLS context pinned to certifi's CA bundle.
    aiohttp's default context uses the OS trust store, which isn't wired up
    on some Python installs (python.org macOS, slim containers) — there TLS
    to huggingface.co fails with CERTIFICATE_VERIFY_FAILED.
    """
    return ssl.create_default_context(cafile=certifi.where())


async def get_session() -> aiohttp.ClientSession:
    """Return the shared session, creating it on first call."""
    global _session
    if _session is not None and not _session.closed:
        return _session
    async with _lock:
        if _session is None or _session.closed:
            connector = aiohttp.TCPConnector(
                limit_per_host=_CONNECTOR_LIMIT_PER_HOST,
                ssl=ssl_context(),
            )
            _session = aiohttp.ClientSession(connector=connector)
    return _session


def parse_content_length(value: Optional[str]) -> Optional[int]:
    """Parse a byte-count header value, or None if absent/malformed/negative."""
    if not value:
        return None
    try:
        n = int(value)
    except ValueError:
        return None
    return n if n >= 0 else None
