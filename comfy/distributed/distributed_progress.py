from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from functools import partial

from typing import Optional, Dict, Any

from aio_pika.patterns import RPC

from ..component_model.executor_types import SendSyncEvent, SendSyncData, ExecutorToClientProgress
from ..component_model.queue_types import BinaryEventTypes


async def _progress(event: SendSyncEvent, data: SendSyncData, user_id: Optional[str] = None,
                    caller_server: Optional[ExecutorToClientProgress] = None) -> None:
    assert caller_server is not None
    assert user_id is not None
    caller_server.send_sync(event, data, sid=user_id)


def _get_name(queue_name: str, user_id: str) -> str:
    return f"{queue_name}.{user_id}.progress"


class DistributedExecutorToClientProgress(ExecutorToClientProgress):
    def __init__(self, rpc: RPC, queue_name: str, loop: AbstractEventLoop):
        self._rpc = rpc
        self._queue_name = queue_name
        self._loop = loop

        self.client_id = None
        self.node_id = None
        self.last_node_id = None

    async def send(self, event: SendSyncEvent, data: SendSyncData, user_id: Optional[str]) -> None:
        # for now, do not send binary data this way, since it cannot be json serialized / it's impractical
        if event == BinaryEventTypes.PREVIEW_IMAGE or event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            return

        if isinstance(data, bytes) or isinstance(data, bytearray):
            return

        if user_id is None:
            # todo: user_id should never be none here
            return

        await self._rpc.call(_get_name(self._queue_name, user_id), {"event": event, "data": data})

    def send_sync(self,
                  event: SendSyncEvent,
                  data: SendSyncData,
                  sid: Optional[str] = None):
        asyncio.run_coroutine_threadsafe(self.send(event, data, sid), self._loop)

    def queue_updated(self):
        # todo: this should gather the global queue data
        pass


class ProgressHandlers:
    def __init__(self, rpc: RPC, caller_server: Optional[ExecutorToClientProgress], queue_name: str):
        self._rpc = rpc
        self._caller_server = caller_server
        self._progress_handlers: Dict[str, Any] = {}
        self._queue_name = queue_name

    async def register_progress(self, user_id: str):
        if user_id in self._progress_handlers:
            return

        handler = partial(_progress, user_id=user_id, caller_server=self._caller_server)
        self._progress_handlers[user_id] = handler
        await self._rpc.register(_get_name(self._queue_name, user_id), handler)

    async def unregister_progress(self, user_id: str):
        if user_id not in self._progress_handlers:
            return
        handler = self._progress_handlers.pop(user_id)
        await self._rpc.unregister(handler)

    async def unregister_all(self):
        await asyncio.gather(*[self._rpc.unregister(handler) for handler in self._progress_handlers.values()])
        self._progress_handlers.clear()
