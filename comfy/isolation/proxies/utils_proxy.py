# pylint: disable=cyclic-import,import-outside-toplevel
from __future__ import annotations

from typing import Optional, Any
from pyisolate import ProxiedSingleton

import os


def _comfy_utils():
    import comfy.utils
    return comfy.utils


class UtilsProxy(ProxiedSingleton):
    """
    Proxy for comfy.utils.
    Primarily handles the PROGRESS_BAR_HOOK to ensure progress updates
    from isolated nodes reach the host.
    """

    # _instance and __new__ removed to rely on SingletonMetaclass
    _rpc: Optional[Any] = None

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        # Create caller using class name as ID (standard for Singletons)
        cls._rpc = rpc.create_caller(cls, "UtilsProxy")

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None

    async def progress_bar_hook(
        self,
        value: int,
        total: int,
        preview: Optional[bytes] = None,
        node_id: Optional[str] = None,
    ) -> Any:
        """
        Host-side implementation: forwards the call to the real global hook.
        Child-side: this method call is intercepted by RPC and sent to host.
        """
        if os.environ.get("PYISOLATE_CHILD") == "1":
            if UtilsProxy._rpc is None:
                raise RuntimeError("UtilsProxy RPC caller is not configured")
            return await UtilsProxy._rpc.progress_bar_hook(
                value, total, preview, node_id
            )

        # Host Execution
        utils = _comfy_utils()
        if utils.PROGRESS_BAR_HOOK is not None:
            return utils.PROGRESS_BAR_HOOK(value, total, preview, node_id)
        return None

    def set_progress_bar_global_hook(self, hook: Any) -> None:
        """Forward hook registration (though usually not needed from child)."""
        if os.environ.get("PYISOLATE_CHILD") == "1":
            raise RuntimeError(
                "UtilsProxy.set_progress_bar_global_hook is not available in child without exact relay support"
            )
        _comfy_utils().set_progress_bar_global_hook(hook)
