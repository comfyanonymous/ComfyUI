from __future__ import annotations

from typing import Any, TypedDict


class InputMapOldId(TypedDict):
    """Map an old node input to a new node input by ID."""
    new_id: str
    old_id: str


class InputMapSetValue(TypedDict):
    """Set a specific value for a new node input."""
    new_id: str
    set_value: Any


InputMap = InputMapOldId | InputMapSetValue
"""
Input mapping for node replacement. Type is inferred by dictionary keys:
- {"new_id": str, "old_id": str} - maps old input to new input
- {"new_id": str, "set_value": Any} - sets a specific value for new input
"""


class OutputMap(TypedDict):
    """Map outputs of node replacement via indexes."""
    new_idx: int
    old_idx: int


class NodeReplace:
    """
    Defines a possible node replacement, mapping inputs and outputs of the old node to the new node.

    Also supports assigning specific values to the input widgets of the new node.

    Args:
        new_node_id: The class name of the new replacement node.
        old_node_id: The class name of the deprecated node.
        old_widget_ids: Ordered list of input IDs for widgets that may not have an input slot
            connected. The workflow JSON stores widget values by their relative position index,
            not by ID. This list maps those positional indexes to input IDs, enabling the
            replacement system to correctly identify widget values during node migration.
        input_mapping: List of input mappings from old node to new node.
        output_mapping: List of output mappings from old node to new node.
    """
    def __init__(self,
        new_node_id: str,
        old_node_id: str,
        old_widget_ids: list[str] | None=None,
        input_mapping: list[InputMap] | None=None,
        output_mapping: list[OutputMap] | None=None,
    ):
        self.new_node_id = new_node_id
        self.old_node_id = old_node_id
        self.old_widget_ids = old_widget_ids
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

    def as_dict(self):
        """Create serializable representation of the node replacement."""
        return {
            "new_node_id": self.new_node_id,
            "old_node_id": self.old_node_id,
            "old_widget_ids": self.old_widget_ids,
            "input_mapping": list(self.input_mapping) if self.input_mapping else None,
            "output_mapping": list(self.output_mapping) if self.output_mapping else None,
        }
