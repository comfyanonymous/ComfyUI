"""Small wrappers around psutil memory queries with conservative fallbacks."""

import logging
import os
from collections import namedtuple

import psutil

_FallbackVMem = namedtuple("svmem", ["total", "available"])
_virtual_memory_warned = False


def _fallback_total_memory():
    """Return total system memory without using psutil, or 0 if unavailable."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 0


def virtual_memory(default_total=None, default_available=None):
    """Return psutil virtual memory, falling back to supplied values on RuntimeError."""
    global _virtual_memory_warned
    try:
        return psutil.virtual_memory()
    except RuntimeError as e:
        if not _virtual_memory_warned:
            logging.warning("psutil.virtual_memory() failed; using fallback memory values: %s", e)
            _virtual_memory_warned = True

        total = default_total if default_total is not None else _fallback_total_memory()
        available = default_available if default_available is not None else total
        return _FallbackVMem(total=total, available=available)


def virtual_memory_available(default=None):
    """Return available system memory, or default when psutil cannot report it."""
    return virtual_memory(default_available=default).available


def virtual_memory_total(default=None):
    """Return total system memory, or default when psutil cannot report it."""
    return virtual_memory(default_total=default).total
