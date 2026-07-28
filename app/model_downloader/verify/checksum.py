"""Hub-checksum verification = SHA256.

Only used to confirm a download matches a *provided* ``expected_sha256``. It
is NOT the dedup key (that is blake3, owned by the assets system). The full
sequential read happens at most once, here, only when a checksum was supplied.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional

_CHUNK = 8 * 1024 * 1024

InterruptCheck = Callable[[], bool]


class ChecksumError(Exception):
    """The computed SHA256 did not match the expected value."""


def sha256_file(path: str, interrupt_check: Optional[InterruptCheck] = None) -> Optional[str]:
    """Stream the file and return its lowercase hex SHA256.

    Returns ``None`` if interrupted via ``interrupt_check``.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if interrupt_check is not None and interrupt_check():
                return None
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(
    path: str, expected: str, interrupt_check: Optional[InterruptCheck] = None
) -> None:
    """Raise :class:`ChecksumError` unless the file's SHA256 matches ``expected``."""
    actual = sha256_file(path, interrupt_check)
    if actual is None:
        return  # interrupted; caller will re-verify on resume
    if actual.lower() != expected.lower():
        raise ChecksumError(
            f"sha256 mismatch: expected {expected.lower()}, got {actual.lower()}"
        )
