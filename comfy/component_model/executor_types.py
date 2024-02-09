from __future__ import annotations  # for Python 3.7-3.9

from typing_extensions import NotRequired, TypedDict
from typing import Optional, Literal, Protocol, TypeAlias, Union

from comfy.component_model.queue_types import BinaryEventTypes


class ExecInfo(TypedDict):
    queue_remaining: int


class QueueInfo(TypedDict):
    exec_info: ExecInfo


class StatusMessage(TypedDict):
    status: QueueInfo
    sid: NotRequired[str]


class ExecutingMessage(TypedDict):
    node: str | None
    prompt_id: NotRequired[str]
    output: NotRequired[dict]


class ProgressMessage(TypedDict):
    value: float
    max: float
    prompt_id: Optional[str]
    node: Optional[str]


ExecutedMessage: TypeAlias = ExecutingMessage

SendSyncEvent: TypeAlias = Union[Literal["status", "executing", "progress", "executed"], BinaryEventTypes, None]

SendSyncData: TypeAlias = Union[StatusMessage, ExecutingMessage, ProgressMessage, bytes, bytearray, None]


class ExecutorToClientProgress(Protocol):
    """
    Specifies the interface for the dependencies a prompt executor needs from a server.

    Attributes:
        client_id (Optional[str]): the client ID that this object collects feedback for
        last_node_id: (Optional[str]): the most recent node that was processed by the executor
        last_prompt_id: (Optional[str]): the most recent prompt that was processed by the executor
    """

    client_id: Optional[str]
    last_node_id: Optional[str]
    last_prompt_id: Optional[str]

    def send_sync(self,
                  event: SendSyncEvent,
                  data: SendSyncData,
                  sid: Optional[str] = None):
        """
        Sends feedback to the client with the specified ID about a specific node

        :param event: a string event name, BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, BinaryEventTypes.PREVIEW_IMAGE, 0 (?) or None
        :param data: a StatusMessage dict when the event is status; an ExecutingMessage dict when the status is executing, binary bytes with a binary event type, or nothing
        :param sid: websocket ID / the client ID to be responding to
        :return:
        """
        pass

    def queue_updated(self):
        """
        Indicates that the local client's queue has been updated
        :return:
        """
        pass
