"""On-disk OAuth token persistence — one machine-bound blob per provider.

Tokens live under ``folder_paths.get_system_user_directory("download_auth")``,
never in the SQLite DB. Each provider file is written ``0600`` and holds an
opaque blob, not readable JSON: the token JSON is XORed with an HMAC-SHA256
keystream whose key is derived from stable machine/install attributes plus a
per-install random salt.

This is obfuscation, not confidentiality. It stops a token from being read by a
human browsing files, grepped out of a backup, or lifted from a folder copied to
another machine (the blob won't decrypt off its origin machine). It does not
protect against code running inside this process (custom nodes) or an attacker
who reads this source and recomputes the key.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import os
import platform
import secrets
import time
from dataclasses import asdict, dataclass

import folder_paths

_SALT_FILE = ".salt"
_SALT_LEN = 32
_NONCE_LEN = 16
_PBKDF2_ITERS = 200_000


@dataclass
class Token:
    access_token: str
    refresh_token: str | None = None
    # Epoch seconds when the access token expires; 0 means "unknown / no expiry".
    expires_at: int = 0
    token_type: str = "Bearer"
    scope: str | None = None

    def is_expired(self, skew: int = 60) -> bool:
        if not self.expires_at:
            return False
        return time.time() + skew >= self.expires_at


def _auth_dir() -> str:
    path = folder_paths.get_system_user_directory("download_auth")
    os.makedirs(path, exist_ok=True)
    return path


def _token_path(provider: str) -> str:
    return os.path.join(_auth_dir(), f"{provider}.bin")


def _machine_id() -> bytes:
    """A stable per-machine identifier, best-effort across platforms."""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(path, "rb") as f:
                return f.read().strip()
        except OSError:
            pass
    if os.name == "nt":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography"
            )
            try:
                guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                return str(guid).encode("utf-8")
            finally:
                winreg.CloseKey(key)
        except OSError:
            pass
    return platform.node().encode("utf-8")


def _machine_material(auth_dir: str) -> bytes:
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    parts = (_machine_id(), platform.node().encode("utf-8"), user.encode("utf-8"), auth_dir.encode("utf-8"))
    return b"\x00".join(parts)


def _load_or_create_salt(auth_dir: str) -> bytes | None:
    path = os.path.join(auth_dir, _SALT_FILE)
    try:
        with open(path, "rb") as f:
            salt = f.read()
        if len(salt) == _SALT_LEN:
            return salt
    except FileNotFoundError:
        pass
    except OSError:
        return None
    salt = secrets.token_bytes(_SALT_LEN)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(salt)
    os.chmod(path, 0o600)
    return salt


def _derive_key(salt: bytes, auth_dir: str) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", _machine_material(auth_dir), salt, _PBKDF2_ITERS, dklen=32
    )


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out.extend(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(out[:n])


def _xor(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def load(provider: str) -> Token | None:
    auth_dir = _auth_dir()
    try:
        with open(_token_path(provider), "rb") as f:
            blob = base64.b64decode(f.read())
    except FileNotFoundError:
        return None
    except (ValueError, OSError):
        return None
    salt = _load_or_create_salt(auth_dir)
    if salt is None or len(blob) <= _NONCE_LEN:
        return None
    nonce, ciphertext = blob[:_NONCE_LEN], blob[_NONCE_LEN:]
    key = _derive_key(salt, auth_dir)
    plaintext = _xor(ciphertext, _keystream(key, nonce, len(ciphertext)))
    # A wrong machine / corrupt file decrypts to garbage; treat as logged out.
    try:
        data = json.loads(plaintext)
    except ValueError:
        return None
    if not isinstance(data, dict) or "access_token" not in data:
        return None
    return Token(
        access_token=data.get("access_token", ""),
        refresh_token=data.get("refresh_token"),
        expires_at=int(data.get("expires_at", 0) or 0),
        token_type=data.get("token_type", "Bearer"),
        scope=data.get("scope"),
    )


def save(provider: str, token: Token) -> None:
    auth_dir = _auth_dir()
    salt = _load_or_create_salt(auth_dir)
    if salt is None:
        return
    key = _derive_key(salt, auth_dir)
    nonce = secrets.token_bytes(_NONCE_LEN)
    plaintext = json.dumps(asdict(token)).encode("utf-8")
    ciphertext = _xor(plaintext, _keystream(key, nonce, len(plaintext)))
    blob = base64.b64encode(nonce + ciphertext)
    path = _token_path(provider)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
    os.chmod(path, 0o600)


def delete(provider: str) -> None:
    try:
        os.remove(_token_path(provider))
    except FileNotFoundError:
        pass
