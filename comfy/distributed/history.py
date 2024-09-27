from __future__ import annotations

import copy
import collections
import typing
from itertools import islice

from ..component_model.queue_types import HistoryEntry, QueueItem, ExecutionStatus, MAXIMUM_HISTORY_SIZE


class History:
    def __init__(self):
        self.history: typing.OrderedDict[str, HistoryEntry] = collections.OrderedDict()

    def put(self, queue_item: QueueItem, outputs: dict, status: ExecutionStatus):
        self.history[queue_item.prompt_id] = HistoryEntry(prompt=queue_item.queue_tuple,
                                                          outputs=outputs,
                                                          status=ExecutionStatus(*status)._asdict())

    def copy(self, prompt_id: typing.Optional[str | int] = None, max_items: typing.Optional[int] = None,
             offset: typing.Optional[int] = None) -> dict[str, HistoryEntry]:
        if offset is not None and offset < 0:
            offset = max(len(self.history) + offset, 0)
        max_items = max_items or MAXIMUM_HISTORY_SIZE
        if prompt_id in self.history:
            return {prompt_id: copy.deepcopy(self.history[prompt_id])}
        else:
            ordered_dict = collections.OrderedDict()
            for k in islice(self.history, offset, max_items):
                ordered_dict[k] = copy.deepcopy(self.history[k])
            return ordered_dict

    def clear(self):
        self.history.clear()

    def pop(self, key: str):
        self.history.pop(key)
