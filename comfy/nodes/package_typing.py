from __future__ import annotations

from typing import Protocol, ClassVar, Tuple, Dict
from dataclasses import dataclass, field


class CustomNode(Protocol):
    @classmethod
    def INPUT_TYPES(cls) -> dict: ...

    RETURN_TYPES: ClassVar[Tuple[str]]
    RETURN_NAMES: ClassVar[Tuple[str]] = None
    OUTPUT_IS_LIST: ClassVar[Tuple[bool]] = None
    INPUT_IS_LIST: ClassVar[bool] = None
    FUNCTION: ClassVar[str]
    CATEGORY: ClassVar[str]
    OUTPUT_NODE: ClassVar[bool] = None


@dataclass
class ExportedNodes:
    NODE_CLASS_MAPPINGS: Dict[str, CustomNode] = field(default_factory=dict)
    NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = field(default_factory=dict)

    def update(self, exported_nodes: ExportedNodes) -> ExportedNodes:
        self.NODE_CLASS_MAPPINGS.update(exported_nodes.NODE_CLASS_MAPPINGS)
        self.NODE_DISPLAY_NAME_MAPPINGS.update(exported_nodes.NODE_DISPLAY_NAME_MAPPINGS)
        return self

    def __len__(self):
        return len(self.NODE_CLASS_MAPPINGS)

    def __sub__(self, other):
        exported_nodes = ExportedNodes().update(self)
        for self_key in exported_nodes.NODE_CLASS_MAPPINGS:
            if self_key in other.NODE_CLASS_MAPPINGS:
                exported_nodes.NODE_CLASS_MAPPINGS.pop(self_key)
            if self_key in other.NODE_DISPLAY_NAME_MAPPINGS:
                exported_nodes.NODE_DISPLAY_NAME_MAPPINGS.pop(self_key)
        return exported_nodes

    def __add__(self, other):
        exported_nodes = ExportedNodes().update(self)
        return exported_nodes.update(other)
