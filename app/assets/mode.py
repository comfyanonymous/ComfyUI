"""Runtime hash-mode accessor."""

from __future__ import annotations

from typing import Protocol


class _HashingArguments(Protocol):
    enable_asset_hashing: bool


_args: _HashingArguments | None = None


def init(args: _HashingArguments) -> None:
    global _args
    _args = args


def hashing_enabled() -> bool:
    """Return whether startup enabled asset hashing."""
    return bool(getattr(_args, "enable_asset_hashing", False))
