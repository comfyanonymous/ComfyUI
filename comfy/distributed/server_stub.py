from __future__ import annotations

import uuid
from typing import Literal

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

    def queue_updated(self):
        pass
