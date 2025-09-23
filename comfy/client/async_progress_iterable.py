import asyncio
import uuid
from asyncio import Task, Future
from typing import NamedTuple, Optional, AsyncIterable
from typing_extensions import override

from .client_types import V1QueuePromptResponse, ProgressNotification
from ..component_model.executor_types import ExecutorToClientProgress, SendSyncEvent, SendSyncData


class _ProgressNotification(NamedTuple):
    event: SendSyncEvent
    data: SendSyncData
    sid: Optional[str] = None
    complete: bool = False


class QueuePromptWithProgress:
    def __init__(self):
        self._progress_handler = _ProgressHandler()

    def progress(self) -> AsyncIterable[ProgressNotification]:
        return self._progress_handler

    async def get(self) -> V1QueuePromptResponse:
        return await self._progress_handler.fut

    def future(self) -> Future[V1QueuePromptResponse]:
        return self._progress_handler.fut

    @property
    def progress_handler(self) -> ExecutorToClientProgress:
        return self._progress_handler

    def complete(self, task: Task[V1QueuePromptResponse]):
        self._progress_handler.complete(task)


class _ProgressHandler(ExecutorToClientProgress, AsyncIterable[ProgressNotification]):
    def __init__(self, user_id: str = None):
        if user_id is None:
            self.client_id = str(uuid.uuid4())

        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[_ProgressNotification] = asyncio.Queue()
        self.fut: Future[V1QueuePromptResponse] = asyncio.Future()

    @override
    @property
    def receive_all_progress_notifications(self) -> bool:
        return True

    @override
    @receive_all_progress_notifications.setter
    def receive_all_progress_notifications(self, value: bool):
        return

    def send_sync(self,
                  event: SendSyncEvent,
                  data: SendSyncData,
                  sid: Optional[str] = None):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, _ProgressNotification(event, data, sid))

    def complete(self, task: Task[V1QueuePromptResponse]):
        if task.exception() is not None:
            self.fut.set_exception(task.exception())
        else:
            self.fut.set_result(task.result())
        self._queue.put_nowait(_ProgressNotification(None, None, None, complete=True))

    def __aiter__(self):
        return self

    async def __anext__(self):
        result: _ProgressNotification = await self._queue.get()
        self._queue.task_done()

        if result.complete:
            if self.fut.exception() is not None:
                raise self.fut.exception()
            else:
                raise StopAsyncIteration()
        else:
            return ProgressNotification(result.event, result.data, result.sid)
