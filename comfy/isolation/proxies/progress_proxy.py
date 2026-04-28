from __future__ import annotations

import logging
import os
from typing import Any, Optional

try:
    from pyisolate import ProxiedSingleton
except ImportError:

    class ProxiedSingleton:
        pass

from .base import call_singleton_rpc


def _get_progress_state():
    from comfy_execution.progress import get_progress_state

    return get_progress_state()


def _is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"

logger = logging.getLogger(__name__)


class ProgressProxy(ProxiedSingleton):
    _rpc: Optional[Any] = None

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        cls._rpc = rpc.create_caller(cls, cls.get_remote_id())

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None

    @classmethod
    def _get_caller(cls) -> Any:
        if cls._rpc is None:
            raise RuntimeError("ProgressProxy RPC caller is not configured")
        return cls._rpc

    def set_progress(
        self,
        value: float,
        max_value: float,
        node_id: Optional[str] = None,
        image: Any = None,
    ) -> None:
        if _is_child_process():
            call_singleton_rpc(
                self._get_caller(),
                "rpc_set_progress",
                value,
                max_value,
                node_id,
                image,
            )
            return None

        _get_progress_state().update_progress(
            node_id=node_id,
            value=value,
            max_value=max_value,
            image=image,
        )
        return None

    async def rpc_set_progress(
        self,
        value: float,
        max_value: float,
        node_id: Optional[str] = None,
        image: Any = None,
    ) -> None:
        _get_progress_state().update_progress(
            node_id=node_id,
            value=value,
            max_value=max_value,
            image=image,
        )


__all__ = ["ProgressProxy"]
