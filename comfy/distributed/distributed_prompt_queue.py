from __future__ import annotations

import asyncio
import time
import uuid
from asyncio import AbstractEventLoop, Queue, QueueEmpty
from dataclasses import asdict
from time import sleep
from typing import Optional, Dict, List, Mapping, Tuple, Callable

import jwt
from aio_pika import connect_robust
from aio_pika.abc import AbstractConnection, AbstractChannel
from aio_pika.patterns import JsonRPC

from ..cmd.main_pre import tracer
from .distributed_progress import ProgressHandlers
from .distributed_types import RpcRequest, RpcReply
from .history import History
from .server_stub import ServerStub
from ..auth.permissions import jwt_decode
from ..cmd.server import PromptServer
from ..component_model.abstract_prompt_queue import AsyncAbstractPromptQueue, AbstractPromptQueue
from ..component_model.executor_types import ExecutorToClientProgress, SendSyncEvent, SendSyncData, HistoryResultDict
from ..component_model.queue_types import Flags, HistoryEntry, QueueTuple, QueueItem, ExecutionStatus, TaskInvocation, \
    ExecutionError


class DistributedPromptQueue(AbstractPromptQueue, AsyncAbstractPromptQueue):
    """
    A distributed prompt queue for the ComfyUI web client and single-threaded worker.
    """

    def size(self) -> int:
        """
        In a distributed queue, this only returns the client's apparent number of items it is waiting for
        :return:
        """
        return len(self._caller_local_in_progress)

    async def progress(self, event: SendSyncEvent, data: SendSyncData, sid: Optional[str]) -> None:
        self._caller_server.send_sync(event, data, sid=sid)

    @tracer.start_as_current_span("Put Async")
    async def put_async(self, queue_item: QueueItem) -> TaskInvocation | None:
        assert self._is_caller
        assert self._rpc is not None
        reply: TaskInvocation
        if self._closing:
            return None
        self._caller_local_in_progress[queue_item.prompt_id] = queue_item
        if self._caller_server is not None:
            self._caller_server.queue_updated(self.get_tasks_remaining())
        try:
            if "token" in queue_item.extra_data:
                user_token = queue_item.extra_data["token"]
                user_id = jwt_decode(user_token)["sub"]
            else:
                if "client_id" in queue_item.extra_data:
                    user_id = queue_item.extra_data["client_id"]
                elif self._caller_server.client_id is not None:
                    user_id = self._caller_server.client_id
                else:
                    user_id = str(uuid.uuid4())
                    # todo: should we really do this?
                    self._caller_server.client_id = user_id

                # create a stub token
                user_token = jwt.encode({"sub": user_id}, key="", algorithm="none")

            # register callbacks for progress
            assert self._caller_progress_handlers is not None
            await self._caller_progress_handlers.register_progress(user_id)
            request = RpcRequest(prompt_id=queue_item.prompt_id, user_token=user_token, prompt=queue_item.prompt)
            reply = RpcReply(**(await self._rpc.call(self._queue_name, {"request": asdict(request)}))).as_task_invocation()
            self._caller_history.put(queue_item, reply.outputs, reply.status)
            if self._caller_server is not None:
                self._caller_server.queue_updated(self.get_tasks_remaining())

            # if this has a completion future, complete it
            if queue_item.completed is not None:
                queue_item.completed.set_result(reply)
        except Exception as exc:
            # if a caller-side error occurred, use the passed error for the messages
            # we didn't receive any outputs here
            as_exec_exc = ExecutionError(queue_item.prompt_id, exceptions=[exc])
            self._caller_history.put(queue_item, outputs={}, status=as_exec_exc.status)

            # if we have a completer, propagate the exception to it
            if queue_item.completed is not None:
                queue_item.completed.set_exception(as_exec_exc)
            raise as_exec_exc
        finally:
            self._caller_local_in_progress.pop(queue_item.prompt_id)
            if self._caller_server is not None:
                # todo: this ensures that the web ui is notified about the completed task, but it should really be done by worker
                self._caller_server.send_sync("executing", {"node": None, "prompt_id": queue_item.prompt_id}, self._caller_server.client_id)
                self._caller_server.queue_updated(self.get_tasks_remaining())
        return reply

    def put(self, item: QueueItem):
        # caller: execute on main thread
        assert self._is_caller
        if self._closing:
            return
        # this is called by the web server and its event loop is perfectly fine to use
        # the future is now ignored
        asyncio.run_coroutine_threadsafe(self.put_async(item), self._loop)

    async def _callee_do_work_item(self, request: dict) -> dict:
        assert self._is_callee
        request_obj = RpcRequest.from_dict(request)
        item = (await request_obj.as_queue_tuple()).queue_tuple
        item_with_completer = QueueItem(item, self._loop.create_future())
        self._callee_local_in_progress[item_with_completer.prompt_id] = item_with_completer
        # todo: check if we have the local model content needed to execute this request and if not, reject it
        # todo: check if we have enough memory to execute this request, and if not, reject it
        self._callee_local_queue.put_nowait(item)

        # technically this could be messed with or overwritten
        assert item_with_completer.completed is not None
        assert not item_with_completer.completed.done()

        # now we wait for the worker thread to complete the item
        invocation = await item_with_completer.completed
        return asdict(RpcReply.from_task_invocation(invocation, request_obj.user_token))

    def get(self, timeout: float | None = None) -> Optional[Tuple[QueueTuple, str]]:
        # callee: executed on the worker thread
        assert self._is_callee
        # the loop receiving messages must not be mounted on the worker thread
        # otherwise receiving messages will be blocked forever
        try:
            worker_event_loop = asyncio.get_event_loop()
        except RuntimeError:
            worker_event_loop = None
        assert self._loop != worker_event_loop, "get only makes sense in the context of the legacy comfyui prompt worker"
        # spin wait
        timeout = timeout or 30.0
        item = None
        while timeout > 0:
            try:
                item = self._callee_local_queue.get_nowait()
                break
            except QueueEmpty:
                start_time = time.time()
                sleep(0.1)
                timeout -= time.time() - start_time

        if item is None:
            return None

        return item, item[1]

    async def get_async(self, timeout: float | None = None) -> Optional[Tuple[QueueTuple, str]]:
        # callee: executed anywhere
        assert self._is_callee
        try:
            item: QueueTuple = await asyncio.wait_for(self._callee_local_queue.get(), timeout)
        except TimeoutError:
            return None

        return item, item[1]

    def task_done(self, item_id: int, outputs: dict, status: Optional[ExecutionStatus], error_details: Optional['ExecutionErrorMessage'] = None):
        # callee: executed on the worker thread
        if "outputs" in outputs:
            outputs: HistoryResultDict
            outputs = outputs["outputs"]
        assert self._is_callee
        pending = self._callee_local_in_progress.pop(item_id)
        assert pending is not None
        assert pending.completed is not None
        assert not pending.completed.done()
        # finish the task. status will transmit the errors in comfy's domain-specific way
        pending.completed.set_result(TaskInvocation(item_id=item_id, outputs=outputs, status=status, error_details=error_details))
        # todo: the caller is responsible for sending a websocket message right now that the UI expects for updates

    def get_current_queue(self) -> Tuple[List[QueueTuple], List[QueueTuple]]:
        """
        In a distributed queue, all queue items are assumed to be currently in progress
        :return:
        """
        return [], [item.queue_tuple for item in self._caller_local_in_progress.values()]

    def get_tasks_remaining(self) -> int:
        """
        In a distributed queue, shows only the items that this caller is currently waiting for
        :return:
        """
        # caller: executed on main thread
        return len(self._caller_local_in_progress)

    def wipe_queue(self) -> None:
        """
        Does nothing on distributed queues. Once an item has been sent, it cannot be cancelled.
        :return:
        """
        pass

    def delete_queue_item(self, function: Callable[[QueueTuple], bool]) -> bool:
        """
        Does nothing on distributed queues. Once an item has been sent, it cannot be cancelled.
        :param function:
        :return:
        """
        return False

    def get_history(self, prompt_id: Optional[int] = None, max_items=None, offset=None) \
            -> Mapping[str, HistoryEntry]:
        return self._caller_history.copy(prompt_id=prompt_id, max_items=max_items, offset=offset)

    def wipe_history(self):
        self._caller_history.clear()

    def delete_history_item(self, id_to_delete):
        self._caller_history.pop(id_to_delete)

    def set_flag(self, name: str, data: bool) -> None:
        """
        Does nothing on distributed queues. Workers must manage their own memory.
        :param name:
        :param data:
        :return:
        """
        pass

    def get_flags(self, reset=True) -> Flags:
        """
        Does nothing on distributed queues. Workers must manage their own memory.
        :param reset:
        :return:
        """
        return Flags()

    def __init__(self,
                 caller_server: Optional[ExecutorToClientProgress | PromptServer] = None,
                 queue_name: str = "comfyui",
                 connection_uri="amqp://localhost/",
                 is_caller=True,
                 is_callee=True,
                 loop: Optional[AbstractEventLoop] = None):
        super().__init__()
        # this constructor is called on the main thread
        self._loop = loop or asyncio.get_event_loop() or asyncio.new_event_loop()
        self._queue_name = queue_name
        self._connection_uri = connection_uri
        self._connection: Optional[AbstractConnection] = None  # Connection will be set up asynchronously
        self._channel: Optional[AbstractChannel] = None  # Channel will be set up asynchronously
        self._is_caller = is_caller
        self._is_callee = is_callee
        self._closing = False
        self._initialized = False

        # as rpc caller
        self._caller_server = caller_server or ServerStub()
        self._caller_progress_handlers: Optional[ProgressHandlers] = None
        self._caller_local_in_progress: dict[str | int, QueueItem] = {}
        self._caller_history: History = History()

        # as rpc callee
        self._callee_local_queue: Queue = Queue()
        self._callee_local_in_progress: Dict[int | str, QueueItem] = {}
        self._rpc: Optional[JsonRPC] = None

    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def init(self):
        if self._initialized:
            return
        self._connection = await connect_robust(self._connection_uri, loop=self._loop)
        self._channel = await self._connection.channel()
        self._rpc = await JsonRPC.create(channel=self._channel, auto_delete=False, durable=True)
        if self._is_caller:
            self._caller_progress_handlers = ProgressHandlers(self._rpc, self._caller_server, self._queue_name)
        # this makes the queue available to complete work items
        if self._is_callee:
            await self._rpc.register(self._queue_name, self._callee_do_work_item)
        self._initialized = True

    async def close(self):
        if self._closing or not self._initialized:
            return

        self._closing = True
        if self._is_caller:
            await self._caller_progress_handlers.unregister_all()

        await self._rpc.close()
        await self._channel.close()
        await self._connection.close()
        self._initialized = False
        self._closing = False
