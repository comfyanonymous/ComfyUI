"""Compatibility shim for the indexed serializer path."""

from __future__ import annotations

from typing import Any


def register_custom_node_serializers(_registry: Any) -> None:
    """Legacy no-op shim.

    Serializer registration now lives directly in the active isolation adapter.
    This module remains importable because the isolation index still references it.
    """
    return None

__all__ = ["register_custom_node_serializers"]
