from __future__ import annotations

import asyncio
import copy
import time
import typing
from enum import Enum
from typing import NamedTuple, Optional, List, Literal, Sequence

from typing_extensions import NotRequired, TypedDict

from .outputs_types import OutputsDict
from .sensitive_data import SENSITIVE_EXTRA_DATA_KEYS

if typing.TYPE_CHECKING:
    from .executor_types import ExecutionErrorMessage


class QueueTuple(NamedTuple):
    priority: float
    prompt_id: str
    prompt: dict
    extra_data: Optional[ExtraData] = None
    good_outputs: Optional[List[str]] = None
    sensitive: Optional[dict] = None


MAXIMUM_HISTORY_SIZE = 10000


class TaskInvocation(NamedTuple):
    item_id: int | str
    outputs: OutputsDict
    status: Optional[ExecutionStatus]
    error_details: Optional['ExecutionErrorMessage'] = None


class ExecutionStatus(NamedTuple):
    status_str: Literal['success', 'error']
    completed: bool
    messages: List[str]

    def as_dict(self, error_details: Optional['ExecutionErrorMessage'] = None) -> ExecutionStatusAsDict:
        result: ExecutionStatusAsDict = {
            "status_str": self.status_str,
            "completed": self.completed,
            "messages": copy.copy(self.messages),
        }
        if error_details is not None:
            result["error_details"] = error_details
        return result


class ExecutionError(RuntimeError):
    def __init__(self, task_id: int | str, status: Optional[ExecutionStatus] = None, exceptions: Optional[Sequence[Exception]] = None, *args):
        super().__init__(*args)
        self._task_id = task_id
        if status is not None:
            self._status = status
        elif exceptions is not None:
            self._status = ExecutionStatus('error', False, [str(ex) for ex in exceptions])
        else:
            self._status = ExecutionStatus('error', False, [])

    @property
    def status(self) -> ExecutionStatus:
        return self._status

    def as_task_invocation(self) -> TaskInvocation:
        return TaskInvocation(self._task_id, {}, self.status)

    def __str__(self):
        return ",".join(self._status.messages)


class ExecutionStatusAsDict(TypedDict):
    status_str: Literal['success', 'error']
    completed: bool
    messages: List[str]
    error_details: NotRequired[ExecutionErrorMessage]


class Flags(TypedDict, total=False):
    unload_models: NotRequired[bool]
    free_memory: NotRequired[bool]


class HistoryEntry(TypedDict):
    prompt: QueueTuple
    outputs: dict
    status: NotRequired[ExecutionStatusAsDict]


class ExtraData(TypedDict):
    client_id: NotRequired[str]
    extra_pnginfo: NotRequired[str]
    token: NotRequired[str]


class QueueDict(dict):
    """
    A wrapper class for a queue tuple, the object that is given to executors.

    Attributes:
        queue_tuple (QueueTuple): the corresponding queued workflow and other related data
    """
    __slots__ = ('queue_tuple',)

    def __init__(self, queue_tuple: QueueTuple):
        # initialize the dictionary superclass with the data we want to serialize.
        # populate the queue tuple with the appropriate dummy fields
        queue_tuple = QueueTuple(*queue_tuple)
        if queue_tuple.sensitive is None:
            sensitive = {}
            extra_data = queue_tuple.extra_data or {}
            for sensitive_val in SENSITIVE_EXTRA_DATA_KEYS:
                if sensitive_val in extra_data:
                    sensitive[sensitive_val] = extra_data.pop(sensitive_val)
            extra_data["create_time"] = int(time.time() * 1000)  # timestamp in milliseconds
            queue_tuple = QueueTuple(queue_tuple.priority, queue_tuple.prompt_id, queue_tuple.prompt, extra_data, queue_tuple.good_outputs, sensitive)

        super().__init__(
            priority=queue_tuple[0],
            prompt_id=queue_tuple[1],
            prompt=queue_tuple[2],
            extra_data=queue_tuple[3],
            good_outputs=queue_tuple[4],
            sensitive=queue_tuple[5],
        )
        # Store the original tuple in a slot, making it invisible to json.dumps.
        self.queue_tuple = queue_tuple

    @property
    def priority(self) -> float:
        return self.queue_tuple[0]

    @property
    def prompt_id(self) -> str:
        return self.queue_tuple[1]

    @property
    def prompt(self) -> dict:
        return self.queue_tuple[2]

    @property
    def extra_data(self) -> Optional[ExtraData]:
        if len(self.queue_tuple) > 3:
            return self.queue_tuple[3]
        return None

    @property
    def good_outputs(self) -> Optional[List[str]]:
        if len(self.queue_tuple) > 4:
            return self.queue_tuple[4]
        return None

    @property
    def sensitive(self) -> Optional[dict]:
        if len(self.queue_tuple) > 5:
            return self.queue_tuple[5]
        return None


NamedQueueTuple = QueueDict


class QueueItem(QueueDict):
    """
    An item awaiting processing in the queue: a NamedQueueTuple with a future that is completed when the item is done
    processing.

    Attributes:
        completed (Optional[Future[TaskInvocation | dict]]): A future of a task invocation (the signature of the task_done method)
            or a dictionary of outputs
    """
    __slots__ = ('completed',)

    def __init__(self, queue_tuple: QueueTuple, completed: asyncio.Future[TaskInvocation | dict] | None):
        # Initialize the parent, which sets up the dictionary representation.
        super().__init__(queue_tuple=queue_tuple)
        # Store the future in a slot so it won't be serialized.
        self.completed = completed

    def __lt__(self, other: QueueItem):
        if not isinstance(other, QueueItem):
            return NotImplemented
        return self.priority < other.priority


class BinaryEventTypes(Enum):
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    TEXT = 3
    PREVIEW_IMAGE_WITH_METADATA = 4


class ExecutorToClientMessage(TypedDict, total=False):
    node: str
    prompt_id: str
    output: NotRequired[str]


AbstractPromptQueueGetCurrentQueueItems = tuple[list[QueueItem], list[QueueItem]]
