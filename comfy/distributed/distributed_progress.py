from __future__ import annotations

import asyncio
import base64
from asyncio import AbstractEventLoop
from enum import Enum
from functools import partial
from typing import Optional, Dict, Any

from aio_pika.patterns import RPC

from ..component_model.executor_types import SendSyncEvent, SendSyncData, ExecutorToClientProgress, \
    UnencodedPreviewImageMessage, StatusMessage, QueueInfo, ExecInfo
from ..component_model.queue_types import BinaryEventTypes


async def _progress(event: SendSyncEvent, data: SendSyncData, user_id: Optional[str] = None,
                    caller_server: Optional[ExecutorToClientProgress] = None) -> None:
    assert caller_server is not None
    assert user_id is not None
    if event == BinaryEventTypes.PREVIEW_IMAGE or event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE or isinstance(data, str):
        data: bytes = base64.b64decode(data)
    caller_server.send_sync(event, data, sid=user_id)


def _get_name(queue_name: str, user_id: str) -> str:
    return f"{queue_name}.{user_id}.progress"


class DistributedExecutorToClientProgress(ExecutorToClientProgress):
    def __init__(self, rpc: RPC, queue_name: str, loop: AbstractEventLoop, receive_all_progress_notifications=True):
        self._rpc = rpc
        self._queue_name = queue_name
        self._loop = loop

        self.client_id = None
        self.node_id = None
        self.last_node_id = None
        self.last_prompt_id = None
        self.receive_all_progress_notifications = receive_all_progress_notifications

    async def send(self, event: SendSyncEvent, data: SendSyncData, user_id: Optional[str]) -> None:
        # for now, do not send binary data this way, since it cannot be json serialized / it's impractical
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            from ..cmd.latent_preview_image_encoding import encode_preview_image

            # encode preview image
            event = BinaryEventTypes.PREVIEW_IMAGE.value
            data: UnencodedPreviewImageMessage
            format, pil_image, max_size = data
            data: bytes = encode_preview_image(pil_image, format, max_size)

        if isinstance(data, bytes) or isinstance(data, bytearray):
            if isinstance(event, Enum):
                event: int = event.value
            data: str = base64.b64encode(data).decode()

        if user_id is None:
            # todo: user_id should never be none here
            return

        await self._rpc.call(_get_name(self._queue_name, user_id), {"event": event, "data": data})

    def send_sync(self,
                  event: SendSyncEvent,
                  data: SendSyncData,
                  sid: Optional[str] = None):
        asyncio.run_coroutine_threadsafe(self.send(event, data, sid), self._loop)

    def queue_updated(self, queue_remaining: Optional[int] = None):
        self.send_sync("status", StatusMessage(status=QueueInfo(exec_info=ExecInfo(queue_remaining=queue_remaining))))


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
