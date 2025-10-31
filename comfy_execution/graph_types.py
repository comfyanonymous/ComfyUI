from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Mapping

from frozendict import frozendict, deepfreeze

from comfy.component_model.executor_types import NodeNotFoundError

if typing.TYPE_CHECKING:
    from .graph import TopologicalSort, DynamicPrompt

NodeOutputByIdAndIndex = tuple[str, int]
InputValue = typing.Union[NodeOutputByIdAndIndex, bool, bytes, int, float, str, typing.IO, typing.BinaryIO]
Input = Mapping[str, InputValue]


@dataclass(frozen=True)
class FrozenTopologicalSort:
    dynprompt: "FrozenPrompt"
    pendingNodes: frozendict
    blockCount: frozendict
    blocking: frozendict
    externalBlocks: int

    @classmethod
    def from_topological_sort(cls, ts: "TopologicalSort") -> FrozenTopologicalSort:
        return cls(
            dynprompt=FrozenPrompt.from_dynamic_prompt(ts.dynprompt),
            pendingNodes=deepfreeze(ts.pendingNodes),
            blockCount=deepfreeze(ts.blockCount),
            blocking=deepfreeze(ts.blocking),
            externalBlocks=ts.externalBlocks,
        )


@dataclass(frozen=True)
class FrozenPrompt:
    original_prompt: frozendict
    ephemeral_prompt: frozendict
    ephemeral_parents: frozendict
    ephemeral_display: frozendict

    @classmethod
    def from_dynamic_prompt(cls, dynprompt: "DynamicPrompt") -> FrozenPrompt:
        return cls(
            original_prompt=deepfreeze(dynprompt.original_prompt),
            ephemeral_prompt=deepfreeze(dynprompt.ephemeral_prompt),
            ephemeral_parents=deepfreeze(dynprompt.ephemeral_parents),
            ephemeral_display=deepfreeze(dynprompt.ephemeral_display),
        )

    def get_node(self, node_id):
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        raise NodeNotFoundError(f"Node {node_id} not found")

    def has_node(self, node_id):
        return node_id in self.original_prompt or node_id in self.ephemeral_prompt

    def get_real_node_id(self, node_id):
        while node_id in self.ephemeral_parents:
            node_id = self.ephemeral_parents[node_id]
        return node_id

    def get_parent_node_id(self, node_id):
        return self.ephemeral_parents.get(node_id, None)

    def get_display_node_id(self, node_id):
        while node_id in self.ephemeral_display:
            node_id = self.ephemeral_display[node_id]
        return node_id

    def all_node_ids(self):
        return set(self.original_prompt.keys()).union(set(self.ephemeral_prompt.keys()))

    def get_original_prompt(self):
        return self.original_prompt
