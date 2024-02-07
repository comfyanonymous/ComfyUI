from __future__ import annotations

import asyncio
from enum import Enum
from typing import NamedTuple, Optional, TypedDict, List, Literal, NotRequired
from dataclasses import dataclass
from typing import Tuple

QueueTuple = Tuple[float, str, dict, dict, list]
MAXIMUM_HISTORY_SIZE = 10000


class TaskInvocation(NamedTuple):
    item_id: int | str
    outputs: dict
    status: Optional[ExecutionStatus]


class ExecutionStatus(NamedTuple):
    status_str: Literal['success', 'error']
    completed: bool
    messages: List[str]


class ExecutionStatusAsDict(TypedDict):
    status_str: Literal['success', 'error']
    completed: bool
    messages: List[str]


class Flags(TypedDict, total=False):
    unload_models: NotRequired[bool]
    free_memory: NotRequired[bool]


class HistoryEntry(TypedDict):
    prompt: QueueTuple
    outputs: dict
    status: NotRequired[ExecutionStatusAsDict]


@dataclass
class QueueItem:
    """
    An item awaiting processing in the queue

    Attributes:
        queue_tuple (QueueTuple): the corresponding queued workflow and other related data
        completed (Optional[Future[TaskInvocation | dict]]): A future of a task invocation (the signature of the task_done method)
            or a dictionary of outputs
    """
    queue_tuple: QueueTuple
    completed: asyncio.Future[TaskInvocation | dict] | None

    def __lt__(self, other: QueueItem):
        return self.queue_tuple[0] < other.queue_tuple[0]

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
    def extra_data(self) -> Optional[dict]:
        if len(self.queue_tuple) > 2:
            return self.queue_tuple[3]
        else:
            return None

    @property
    def good_outputs(self) -> Optional[List[str]]:
        if len(self.queue_tuple) > 3:
            return self.queue_tuple[4]
        else:
            return None


class BinaryEventTypes(Enum):
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2


class ExecutorToClientMessage(TypedDict, total=False):
    node: str
    prompt_id: str
    output: NotRequired[str]
