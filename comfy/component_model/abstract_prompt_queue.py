from __future__ import annotations

import typing
from abc import ABCMeta, abstractmethod

from .executor_types import HistoryResultDict
from .queue_types import QueueTuple, HistoryEntry, QueueItem, Flags, ExecutionStatus, TaskInvocation, AbstractPromptQueueGetCurrentQueueItems


class AbstractPromptQueue(metaclass=ABCMeta):
    """
    The interface of a queue inside ComfyUI.

    put is intended to be used by a prompt creator.

    get is intended to be used by a worker.
    """

    @abstractmethod
    def size(self) -> int:
        """
        The number of items currently in the queue. Excludes items being processed.
        :return:
        """
        pass

    @abstractmethod
    def put(self, item: QueueItem):
        """
        Puts an item on the queue. Does not block or wait
        :param item: a queue item
        :return:
        """
        pass

    @abstractmethod
    def get(self, timeout: float | None = None) -> typing.Optional[typing.Tuple[QueueTuple, str]]:
        """
        Pops an item off the queue. Blocking. If a timeout is provided, this will return None after
        :param timeout: the number of seconds to time out for a blocking get
        :return: the queue tuple and its item ID, or None if timed out or no item is on the queue
        """
        pass

    @abstractmethod
    def task_done(self, item_id: str, outputs: HistoryResultDict,
                  status: typing.Optional[ExecutionStatus]):
        """
        Signals to the user interface that the task with the specified id is completed
        :param item_id: the ID of the task that should be marked as completed
        :param outputs: an opaque dictionary of outputs
        :param status:
        :return:
        """
        pass

    @abstractmethod
    def get_current_queue(self) -> AbstractPromptQueueGetCurrentQueueItems:
        """
        Gets the current state of the queue
        :return: A tuple containing (the currently running items, the items awaiting execution)
        """
        pass

    @abstractmethod
    def get_tasks_remaining(self) -> int:
        """
        Gets the length of the queue and the currently processing tasks
        :return: an integer count
        """
        pass

    @abstractmethod
    def wipe_queue(self) -> None:
        """
        Deletes all items on the queue
        :return:
        """
        pass

    @abstractmethod
    def delete_queue_item(self, function: typing.Callable[[QueueTuple], bool]) -> bool:
        """
        Deletes the first queue item that satisfies the predicate
        :param function: a predicate that takes queue tuples and returns true if it matches
        :return: True if an item as removed
        """
        pass

    @abstractmethod
    def get_history(self, prompt_id: typing.Optional[str] = None, max_items=None, offset=-1) -> typing.Mapping[
        str, HistoryEntry]:
        """
        Creates a deep copy of the history
        :param prompt_id:
        :param max_items:
        :param offset:
        :return:
        """
        pass

    @abstractmethod
    def wipe_history(self):
        pass

    @abstractmethod
    def delete_history_item(self, id_to_delete: str):
        pass

    @abstractmethod
    def set_flag(self, name: str, data: bool) -> None:
        pass

    @abstractmethod
    def get_flags(self, reset: bool = True) -> Flags:
        """
        Resets the flags for the next model unload or free memory request.
        :param reset:
        :return:
        """
        pass

    def get_current_queue_volatile(self) -> AbstractPromptQueueGetCurrentQueueItems:
        """
        A workaround to "improve performance with large number of queued prompts",
        :return: A tuple containing (the currently running items, the items awaiting execution)
        """
        return self.get_current_queue()


class AsyncAbstractPromptQueue(metaclass=ABCMeta):
    @abstractmethod
    async def put_async(self, queue_item) -> TaskInvocation | None:
        """
        Puts the item on the queue, and waits until it is complete
        :param queue_item:
        :return:
        :raises: ExecutionException when the worker returns an error, which can be cast to a task invocation
        """
        pass

    @abstractmethod
    async def get_async(self, timeout: float | None = None) -> typing.Optional[typing.Tuple[QueueTuple, str]]:
        pass
