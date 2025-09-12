from __future__ import annotations

import asyncio
import base64
import pickle
from asyncio import AbstractEventLoop
from functools import partial
from typing import Optional, Dict, Any, TypeVar, NewType

from aio_pika import DeliveryMode
from aio_pika.patterns import RPC

from ..component_model.executor_types import SendSyncEvent, SendSyncData, ExecutorToClientProgress, \
    StatusMessage, QueueInfo, ExecInfo

T = TypeVar('T')
Base64Pickled = NewType('Base64Pickled', str)


def obj2base64(obj: T) -> Base64Pickled:
    return Base64Pickled(base64.b64encode(pickle.dumps(obj)).decode())


def base642obj(data: Base64Pickled) -> T:
    return pickle.loads(base64.b64decode(data))


async def _progress(event: Base64Pickled, data: Base64Pickled, user_id: Optional[str] = None,
                    caller_server: Optional[ExecutorToClientProgress] = None) -> None:
    assert caller_server is not None
    assert user_id is not None

    caller_server.send_sync(base642obj(event), base642obj(data), sid=user_id)


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
        self.last_prompt_id = None

    @property
    def receive_all_progress_notifications(self) -> bool:
        return True

    async def send(self, event: SendSyncEvent, data: SendSyncData, user_id: Optional[str]) -> None:
        assert user_id is not None, f"event={event} data={data}"
        try:
            # we don't need to await this coroutine
            _ = asyncio.create_task(self._rpc.call(_get_name(self._queue_name, user_id), {"event": obj2base64(event), "data": obj2base64(data)}, expiration=1000, delivery_mode=DeliveryMode.NOT_PERSISTENT))
        except asyncio.TimeoutError:
            # these can gracefully expire
            pass

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
