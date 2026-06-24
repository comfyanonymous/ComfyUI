"""On-disk persistence for the HuggingFace OAuth token.

The token shape mirrors what HF returns on the token exchange: an
``access_token``, an optional ``refresh_token``, the absolute epoch at
which the access token expires, and the granted scope. We persist
this so logging in once survives ComfyUI restarts under the internal
``__hf_auth`` system-user directory; the file is mode ``0600`` so only
the OS user can read it.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from dataclasses import asdict, dataclass
from typing import Optional

import folder_paths


# Treat a token as expired this many seconds before its server-reported
# ``expires_at`` so we don't try to use a token mid-request only for it
# to flip stale between auth_check and the actual GET.
EXPIRY_BUFFER_SECS = 60

TOKEN_FILENAME = "hf_auth_token.json"


@dataclass
class Token:
    """One OAuth token + the metadata we need to use it."""
    access_token: str
    refresh_token: Optional[str]
    expires_at: float  # absolute epoch seconds
    scope: str = ""

    def is_valid(self) -> bool:
        """True iff we can use this token right now."""
        return (
            bool(self.access_token)
            and (self.expires_at - time.time() > EXPIRY_BUFFER_SECS)
        )


def _token_dir() -> str:
    return folder_paths.get_system_user_directory("hf_auth")


def _token_path() -> str:
    return os.path.join(_token_dir(), TOKEN_FILENAME)


def load_token() -> Optional[Token]:
    """Read the persisted token, returning ``None`` if absent or corrupt."""
    path = _token_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Token(**data)
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logging.warning("[hf_auth] could not load token at %s: %s", path, e)
        return None


def save_token(token: Token) -> None:
    """Atomically write the token with 0600 permissions."""
    path = _token_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(asdict(token), f)
    os.replace(tmp, path)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        # On Windows / weird filesystems chmod may be a no-op; not fatal.
        logging.debug("[hf_auth] chmod 0600 on %s failed: %s", path, e)


def delete_token() -> None:
    """Remove the persisted token; no-op if it doesn't exist."""
    path = _token_path()
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logging.warning("[hf_auth] could not remove token at %s: %s", path, e)
