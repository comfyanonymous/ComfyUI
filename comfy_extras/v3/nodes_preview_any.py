from __future__ import annotations

import json

from comfy_api.v3 import io, ui


class PreviewAny(io.ComfyNodeV3):
    """Originally implement from https://github.com/rgthree/rgthree-comfy/blob/main/py/display_any.py

    upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes"""

    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="PreviewAny_V3",  # frontend expects "PreviewAny" to work
            display_name="Preview Any _V3",  # frontend ignores "display_name" for this node
            description="Preview any type of data by converting it to a readable text format.",
            category="utils",
            inputs=[
                io.AnyType.Input("source"),  # TODO: does not work currently, as `io.AnyType` does not define __ne__
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, source=None) -> io.NodeOutput:
        value = "None"
        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = "source exists, but could not be serialized."

        return io.NodeOutput(ui=ui.PreviewText(value))


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    PreviewAny,
]
