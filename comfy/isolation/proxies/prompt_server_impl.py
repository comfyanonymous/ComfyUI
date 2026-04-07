# pylint: disable=import-outside-toplevel,logging-fstring-interpolation,redefined-outer-name,reimported,super-init-not-called
"""Stateless RPC Implementation for PromptServer.

Replaces the legacy PromptServerProxy (Singleton) with a clean Service/Stub architecture.
- Host: PromptServerService (RPC Handler)
- Child: PromptServerStub (Interface Implementation)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional, Callable

import logging

# IMPORTS
from pyisolate import ProxiedSingleton
from .base import call_singleton_rpc

logger = logging.getLogger(__name__)
LOG_PREFIX = "[Isolation:C<->H]"

# ...

# =============================================================================
# CHILD SIDE: PromptServerStub
# =============================================================================


class PromptServerStub:
    """Stateless Stub for PromptServer."""

    # Masquerade as the real server module
    __module__ = "server"

    _instance: Optional["PromptServerStub"] = None
    _rpc: Optional[Any] = None  # This will be the Caller object
    _source_file: Optional[str] = None

    def __init__(self):
        self.routes = RouteStub(self)

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        """Inject RPC client (called by adapter.py or manually)."""
        # Create caller for HOST Service
        # Assuming Host Service is registered as "PromptServerService" (class name)
        # We target the Host Service Class
        target_id = "PromptServerService"
        # We need to pass a class to create_caller? Usually yes.
        # But we don't have the Service class imported here necessarily (if running on child).
        # pyisolate check verify_service type?
        # If we pass PromptServerStub as the 'class', it might mismatch if checking types.
        # But we can try passing PromptServerStub if it mirrors the service name? No, stub is PromptServerStub.
        # We need a dummy class with right name?
        # Or just rely on string ID if create_caller supports it?
        # Standard: rpc.create_caller(PromptServerStub, target_id)
        # But wait, PromptServerStub is the *Local* class.
        # We want to call *Remote* class.
        # If we use PromptServerStub as the type, returning object will be typed as PromptServerStub?
        # The first arg is 'service_cls'.
        cls._rpc = rpc.create_caller(
            PromptServerService, target_id
        )  # We import Service below?

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None

    # We need PromptServerService available for the create_caller call?
    # Or just use the Stub class if ID matches?
    # prompt_server_impl.py defines BOTH. So PromptServerService IS available!

    @property
    def instance(self) -> "PromptServerStub":
        return self

    # ... Compatibility ...
    @classmethod
    def _get_source_file(cls) -> str:
        if cls._source_file is None:
            import folder_paths

            cls._source_file = os.path.join(folder_paths.base_path, "server.py")
        return cls._source_file

    @property
    def __file__(self) -> str:
        return self._get_source_file()

    # --- Properties ---
    @property
    def client_id(self) -> Optional[str]:
        return "isolated_client"

    def supports(self, feature: str) -> bool:
        return True

    @property
    def app(self):
        raise RuntimeError(
            "PromptServer.app is not accessible in isolated nodes. Use RPC routes instead."
        )

    @property
    def prompt_queue(self):
        raise RuntimeError(
            "PromptServer.prompt_queue is not accessible in isolated nodes."
        )

    # --- UI Communication (RPC Delegates) ---
    async def send_sync(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ) -> None:
        if self._rpc:
            await self._rpc.ui_send_sync(event, data, sid)

    async def send(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ) -> None:
        if self._rpc:
            await self._rpc.ui_send(event, data, sid)

    def send_progress_text(self, text: str, node_id: str, sid=None) -> None:
        if self._rpc:
            # Fire and forget likely needed. If method is async on host, caller invocation returns coroutine.
            # We must schedule it?
            # Or use fire_remote equivalent?
            # Caller object usually proxies calls. If host method is async, it returns coro.
            # If we are sync here (send_progress_text checks imply sync usage), we must background it.
            # But UtilsProxy hook wrapper creates task.
            # Does send_progress_text need to be sync? Yes, node code calls it sync.
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._rpc.ui_send_progress_text(text, node_id, sid))
            except RuntimeError:
                call_singleton_rpc(self._rpc, "ui_send_progress_text", text, node_id, sid)

    # --- Route Registration Logic ---
    def register_route(self, method: str, path: str, handler: Callable):
        """Register a route handler via RPC."""
        if not self._rpc:
            logger.error("RPC not initialized in PromptServerStub")
            return

        # Fire registration async
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._rpc.register_route_rpc(method, path, handler))
        except RuntimeError:
            call_singleton_rpc(self._rpc, "register_route_rpc", method, path, handler)


class RouteStub:
    """Simulates aiohttp.web.RouteTableDef."""

    def __init__(self, stub: PromptServerStub):
        self._stub = stub

    def get(self, path: str):
        def decorator(handler):
            self._stub.register_route("GET", path, handler)
            return handler

        return decorator

    def post(self, path: str):
        def decorator(handler):
            self._stub.register_route("POST", path, handler)
            return handler

        return decorator

    def patch(self, path: str):
        def decorator(handler):
            self._stub.register_route("PATCH", path, handler)
            return handler

        return decorator

    def put(self, path: str):
        def decorator(handler):
            self._stub.register_route("PUT", path, handler)
            return handler

        return decorator

    def delete(self, path: str):
        def decorator(handler):
            self._stub.register_route("DELETE", path, handler)
            return handler

        return decorator


# =============================================================================
# HOST SIDE: PromptServerService
# =============================================================================


class PromptServerService(ProxiedSingleton):
    """Host-side RPC Service for PromptServer."""

    def __init__(self):
        # We will bind to the real server instance lazily or via global import
        pass

    @property
    def server(self):
        from server import PromptServer

        return PromptServer.instance

    async def ui_send_sync(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ):
        await self.server.send_sync(event, data, sid)

    async def ui_send(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ):
        await self.server.send(event, data, sid)

    async def ui_send_progress_text(self, text: str, node_id: str, sid=None):
        # Made async to be awaitable by RPC layer
        self.server.send_progress_text(text, node_id, sid)

    async def register_route_rpc(self, method: str, path: str, child_handler_proxy):
        """RPC Target: Register a route that forwards to the Child."""
        from aiohttp import web
        logger.debug(f"{LOG_PREFIX} Registering Isolated Route {method} {path}")

        async def route_wrapper(request: web.Request) -> web.Response:
            # 1. Capture request data
            req_data = {
                "method": request.method,
                "path": request.path,
                "query": dict(request.query),
            }
            if request.can_read_body:
                req_data["text"] = await request.text()

            try:
                # 2. Call Child Handler via RPC (child_handler_proxy is async callable)
                result = await child_handler_proxy(req_data)

                # 3. Serialize Response
                return self._serialize_response(result)
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Isolated Route Error: {e}")
                return web.Response(status=500, text=str(e))

        # Register loop
        self.server.app.router.add_route(method, path, route_wrapper)

    def _serialize_response(self, result: Any) -> Any:
        """Helper to convert Child result -> web.Response"""
        from aiohttp import web
        if isinstance(result, web.Response):
            return result
        # Handle dict (json)
        if isinstance(result, dict):
            return web.json_response(result)
        # Handle string
        if isinstance(result, str):
            return web.Response(text=result)
        # Fallback
        return web.Response(text=str(result))
