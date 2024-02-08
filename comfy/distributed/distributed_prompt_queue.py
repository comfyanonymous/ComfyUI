from __future__ import annotations

import asyncio
import uuid
from asyncio import AbstractEventLoop, Queue
from dataclasses import asdict
from typing import Optional, Dict, List, Mapping, Tuple, Callable

import jwt
from aio_pika import connect_robust
from aio_pika.abc import AbstractConnection, AbstractChannel
from aio_pika.patterns import JsonRPC

from .distributed_types import RpcRequest, RpcReply
from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.queue_types import Flags, HistoryEntry, QueueTuple, QueueItem, ExecutionStatus, TaskInvocation
from .history import History
from ..cmd.server import PromptServer


class DistributedPromptQueue(AbstractPromptQueue):
    """
    A distributed prompt queue for
    """

    def size(self) -> int:
        """
        In a distributed queue, this only returns the client's apparent number of items it is waiting for
        :return:
        """
        return len(self.caller_local_in_progress)

    async def put_async(self, queue_item: QueueItem):
        assert self.is_caller
        if self._closing:
            return
        self.caller_local_in_progress[queue_item.prompt_id] = queue_item
        if self.caller_server is not None:
            self.caller_server.queue_updated()
        try:
            if "token" in queue_item.extra_data:
                user_token = queue_item.extra_data["token"]
            else:
                if "client_id" in queue_item.extra_data:
                    client_id = queue_item.extra_data["client_id"]
                elif self.caller_server.client_id is not None:
                    client_id = self.caller_server.client_id
                else:
                    client_id = str(uuid.uuid4())
                    # todo: should we really do this?
                    self.caller_server.client_id = client_id

                # create a stub token
                user_token = jwt.encode({"sub": client_id}, key="", algorithm="none")
            request = RpcRequest(prompt_id=queue_item.prompt_id, user_token=user_token, prompt=queue_item.prompt)
            assert self.rpc is not None
            res: TaskInvocation = RpcReply(
                **(await self.rpc.call(self.queue_name, {"request": asdict(request)}))).as_task_invocation()

            self.caller_history.put(queue_item, res.outputs, res.status)
            if self.caller_server is not None:
                self.caller_server.queue_updated()

            # if this has a completion future, complete it
            if queue_item.completed is not None:
                queue_item.completed.set_result(res)
            return res
        except Exception as e:
            # if a caller-side error occurred, use the passed error for the messages
            # we didn't receive any outputs here
            self.caller_history.put(queue_item, outputs={},
                                    status=ExecutionStatus(status_str="error", completed=False, messages=[str(e)]))

            # if we have a completer, propoagate the exception to it
            if queue_item.completed is not None:
                queue_item.completed.set_exception(e)
            else:
                # otherwise, this should raise in the event loop, which I suppose isn't handled
                raise e
        finally:
            self.caller_local_in_progress.pop(queue_item.prompt_id)
            if self.caller_server is not None:
                self.caller_server.queue_updated()

    def put(self, item: QueueItem):
        # caller: execute on main thread
        assert self.is_caller
        if self._closing:
            return
        # this is called by the web server and its event loop is perfectly fine to use
        # the future is now ignored
        self.loop.call_soon_threadsafe(self.put_async, item)

    async def _callee_do_work_item(self, request: dict) -> dict:
        assert self.is_callee
        request_obj = RpcRequest.from_dict(request)
        item = request_obj.as_queue_tuple().queue_tuple
        item_with_completer = QueueItem(item, self.loop.create_future())
        self.callee_local_in_progress[item_with_completer.prompt_id] = item_with_completer
        # todo: check if we have the local model content needed to execute this request and if not, reject it
        # todo: check if we have enough memory to execute this request, and if not, reject it
        await self.callee_local_queue.put(item)

        # technically this could be messed with or overwritten
        assert item_with_completer.completed is not None
        assert not item_with_completer.completed.done()

        # now we wait for the worker thread to complete the item
        invocation = await item_with_completer.completed
        return asdict(RpcReply.from_task_invocation(invocation, request_obj.user_token))

    def get(self, timeout: float | None = None) -> Optional[Tuple[QueueTuple, str | int]]:
        # callee: executed on the worker thread
        assert self.is_callee
        try:
            item = asyncio.run_coroutine_threadsafe(self.callee_local_queue.get(), self.loop).result(timeout)
        except TimeoutError:
            return None

        return item, item[1]

    def task_done(self, item_id: int, outputs: dict, status: Optional[ExecutionStatus]):
        # callee: executed on the worker thread
        assert self.is_callee
        pending = self.callee_local_in_progress.pop(item_id)
        assert pending is not None
        assert pending.completed is not None
        assert not pending.completed.done()
        # finish the task. status will transmit the errors in comfy's domain-specific way
        pending.completed.set_result(TaskInvocation(item_id=item_id, outputs=outputs, status=status))

    def get_current_queue(self) -> Tuple[List[QueueTuple], List[QueueTuple]]:
        """
        In a distributed queue, all queue items are assumed to be currently in progress
        :return:
        """
        return [], [item.queue_tuple for item in self.caller_local_in_progress.values()]

    def get_tasks_remaining(self) -> int:
        """
        In a distributed queue, shows only the items that this caller is currently waiting for
        :return:
        """
        # caller: executed on main thread
        return len(self.caller_local_in_progress)

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

    def get_history(self, prompt_id: Optional[int] = None, max_items=None, offset=-1) \
            -> Mapping[str, HistoryEntry]:
        return self.caller_history.copy(prompt_id=prompt_id, max_items=max_items, offset=offset)

    def wipe_history(self):
        self.caller_history.clear()

    def delete_history_item(self, id_to_delete):
        self.caller_history.pop(id_to_delete)

    def set_flag(self, name: str, data: bool) -> None:
        """
        Does nothing on distributed queues. Workers must manage their own memory.
        :param name:
        :param data:
        :return:
        """
        pass

    def get_flags(self, reset) -> Flags:
        """
        Does nothing on distributed queues. Workers must manage their own memory.
        :param reset:
        :return:
        """
        return Flags()

    def __init__(self,
                 server: Optional[ExecutorToClientProgress | PromptServer] = None,
                 queue_name: str = "comfyui",
                 connection_uri="amqp://localhost/",
                 is_caller=True,
                 is_callee=True,
                 loop: Optional[AbstractEventLoop] = None):
        super().__init__()
        # this constructor is called on the main thread
        self.loop = loop or asyncio.get_event_loop() or asyncio.new_event_loop()
        self.queue_name = queue_name
        self.connection_uri = connection_uri
        self.connection: Optional[AbstractConnection] = None  # Connection will be set up asynchronously
        self.channel: Optional[AbstractChannel] = None  # Channel will be set up asynchronously
        self.is_caller = is_caller
        self.is_callee = is_callee
        self._closing = False

        # as rpc caller
        self.caller_server = server
        self.caller_local_in_progress: dict[str | int, QueueItem] = {}
        self.caller_history: History = History()

        # as rpc callee
        self.callee_local_queue: Queue = Queue()
        self.callee_local_in_progress: Dict[int | str, QueueItem] = {}
        self.rpc: Optional[JsonRPC] = None

        # todo: the prompt queue really shouldn't do this
        if server is not None:
            server.prompt_queue = self

    async def init(self):
        self.connection = await connect_robust(self.connection_uri, loop=self.loop)
        self.channel = await self.connection.channel()
        self.rpc = await JsonRPC.create(channel=self.channel)
        self.rpc.host_exceptions = True
        # this makes the queue available to complete work items
        if self.is_callee:
            await self.rpc.register(self.queue_name, self._callee_do_work_item)

    async def close(self):
        self._closing = True
        await self.rpc.close()
        await self.channel.close()
        await self.connection.close()
