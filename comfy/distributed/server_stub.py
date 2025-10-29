from __future__ import annotations

import uuid
from typing import Literal, Optional, Union

from ..component_model.executor_types import ExecutorToClientProgress, StatusMessage, ExecutingMessage
from ..component_model.queue_types import BinaryEventTypes


class ServerStub(ExecutorToClientProgress):
    """
    This class is a stub implementation of ExecutorToClientProgress. This will handle progress events.
    """

    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.last_node_id = None
        self.last_prompt_id = None

    def send_sync(self,
                  event: Literal["status", "executing"] | BinaryEventTypes | str | None,
                  data: StatusMessage | ExecutingMessage | bytes | bytearray | None, sid: str | None = None):
        pass

    def queue_updated(self, queue_remaining: Optional[int] = None):
        pass

    def send_progress_text(self, text: Union[bytes, bytearray, str], node_id: str, sid=None):
        pass

    @property
    def receive_all_progress_notifications(self) -> bool:
        return False

    def add_on_prompt_handler(self, handler):
        pass