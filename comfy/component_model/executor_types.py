from __future__ import annotations  # for Python 3.7-3.9

import concurrent.futures
from enum import Enum
from typing import Optional, Literal, Protocol, Union, NamedTuple, List, runtime_checkable, Iterable, Dict, Any

import PIL.Image
from typing_extensions import NotRequired, TypedDict, Never

from .encode_text_for_progress import encode_text_for_progress
from .outputs_types import OutputsDict
from .queue_types import BinaryEventTypes
from ..cli_args_types import Configuration
from ..nodes.package_typing import InputTypeSpec
from ..progress_types import PreviewImageMetadata


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
    sid: NotRequired[str]


class ProgressMessage(TypedDict):
    value: float
    max: float
    prompt_id: Optional[str]
    node: Optional[str]
    sid: NotRequired[str]
    output: NotRequired[dict]


class UnencodedPreviewImageMessage(NamedTuple):
    format: Literal["JPEG", "PNG"]
    pil_image: PIL.Image.Image
    max_size: int = 512
    node_id: str = ""
    task_id: str = ""


class ExecutionInterruptedMessage(TypedDict):
    prompt_id: str
    node_id: str
    node_type: str
    executed: list[str]


class ExecutionErrorMessage(TypedDict):
    prompt_id: str
    node_id: str
    node_type: str
    executed: list[str]
    exception_message: str
    exception_type: str
    traceback: list[str]
    current_inputs: list[Never] | dict[str, FormattedValue]
    current_outputs: list[str]


class DependencyExecutionErrorMessage(TypedDict):
    node_id: str
    exception_message: str
    exception_type: Literal["graph.DependencyCycleError"]
    traceback: list[Never]
    current_inputs: list[Never]


class ActiveNodeProgressState(TypedDict, total=True):
    value: float
    max: float
    # a string value from the NodeState enum
    state: Literal["pending", "running", "finished", "error"]
    node_id: str
    prompt_id: str
    display_node_id: str
    parent_node_id: str
    real_node_id: str


class ProgressStateMessage(TypedDict, total=True):
    prompt_id: str
    nodes: dict[str, ActiveNodeProgressState]

PreviewImageWithMetadataMessage = tuple[UnencodedPreviewImageMessage, PreviewImageMetadata]

ExecutedMessage = ExecutingMessage

SendSyncEvent = Union[Literal["status", "execution_error", "executing", "progress", "executed", "progress_state"], BinaryEventTypes, None]

SendSyncData = Union[ProgressStateMessage, StatusMessage, ExecutingMessage, DependencyExecutionErrorMessage, ExecutionErrorMessage, ExecutionInterruptedMessage, ProgressMessage, UnencodedPreviewImageMessage, PreviewImageWithMetadataMessage, bytes, bytearray, str, None]


class SocketsMetadata(TypedDict, total=True):
    feature_flags: dict[str, Any]


class DefaultSocketsMetadata(TypedDict, total=True):
    __unimplemented: Literal[True]


SocketsMetadataType = dict[str, SocketsMetadata] | DefaultSocketsMetadata


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

    @property
    def receive_all_progress_notifications(self) -> bool:
        """
        Set to true if this should receive progress bar updates, in addition to the standard execution lifecycle messages
        :return:
        """
        return False

    @receive_all_progress_notifications.setter
    def receive_all_progress_notifications(self, value: bool):
        pass

    @property
    def sockets_metadata(self) -> SocketsMetadataType:
        """
        Metadata about what the socket supports

        Currently used only by the frontend

        :return: in the abstract base class, a static object that is used by the web server to ignore this; in the real classes, sometimes information about connected users
        """
        return {"__unimplemented": True}

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

    def send_progress_text(self, text: Union[bytes, bytearray, str], node_id: str, sid=None):
        """
        Send text to the client
        :param text: the text to send
        :param node_id: the node this belongs to
        :param sid: websocket ID / the client ID to be responding to
        :return:
        """
        message = encode_text_for_progress(node_id, text)

        self.send_sync(BinaryEventTypes.TEXT, message, sid)

    def queue_updated(self, queue_remaining: Optional[int] = None):
        """
        Indicates that the local client's queue has been updated
        :return: nothing
        """
        pass


ExceptionTypes = Literal["custom_validation_failed", "value_not_in_list", "value_bigger_than_max", "value_smaller_than_min", "invalid_input_type", "exception_during_inner_validation", "return_type_mismatch", "bad_linked_input", "required_input_missing", "invalid_prompt", "prompt_no_outputs", "exception_during_validation", "prompt_outputs_failed_validation"]
FormattedValue = str | int | bool | float | None


class ValidationErrorExtraInfoDict(TypedDict, total=False):
    exception_type: NotRequired[str]
    traceback: NotRequired[List[str]]
    dependent_outputs: NotRequired[List[str]]
    class_type: NotRequired[str]
    input_name: NotRequired[str]
    input_config: NotRequired[Dict[str, InputTypeSpec]]
    received_value: NotRequired[Any]
    linked_node: NotRequired[str]
    exception_message: NotRequired[str]
    node_errors: NotRequired[Dict[str, 'NodeErrorsDictValue']]


class ValidationErrorDict(TypedDict):
    type: str
    message: str
    details: str
    extra_info: list[Never] | ValidationErrorExtraInfoDict


class NodeErrorsDictValue(TypedDict, total=False):
    dependent_outputs: NotRequired[List[str]]
    errors: List[ValidationErrorDict]
    class_type: str


class ValidationTuple(NamedTuple):
    valid: bool
    error: Optional[ValidationErrorDict | DependencyExecutionErrorMessage]
    good_output_node_ids: List[str]
    node_errors: list[Never] | Dict[str, NodeErrorsDictValue]


class ValidateInputsTuple(NamedTuple):
    valid: bool
    errors: List[ValidationErrorDict]
    unique_id: str


class RecursiveExecutionErrorDetailsInterrupted(TypedDict, total=True):
    node_id: str


class RecursiveExecutionErrorDetails(TypedDict, total=True):
    node_id: str
    exception_message: str
    exception_type: str
    traceback: list[str]
    current_inputs: NotRequired[dict[str, FormattedValue]]
    current_outputs: NotRequired[dict[str, list[list[FormattedValue]]]]


class RecursiveExecutionTuple(NamedTuple):
    valid: ExecutionResult
    error_details: Optional[RecursiveExecutionErrorDetails | RecursiveExecutionErrorDetailsInterrupted]
    exc_info: Optional[Exception]


class ExecutionResult(Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2

    def __bool__(self):
        return self == 0


class DuplicateNodeError(Exception):
    pass


class HistoryResultDict(TypedDict, total=True):
    outputs: OutputsDict
    meta: OutputsDict


class DependencyCycleError(Exception):
    pass


class NodeInputError(Exception):
    pass


class NodeNotFoundError(Exception):
    pass


class Executor(Protocol):
    def submit(self, fn, /, *args, **kwargs) -> concurrent.futures.Future:
        ...

    def shutdown(self, wait=True, *, cancel_futures=False):
        ...


ExecutePromptArgs = tuple[dict, str, str, dict, ExecutorToClientProgress | None, Configuration | None]


@runtime_checkable
class ComboOptions(Protocol):
    def view_for_validation(self) -> list[str]:
        ...
