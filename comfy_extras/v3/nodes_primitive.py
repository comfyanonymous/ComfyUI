from __future__ import annotations

import sys

from comfy_api.v3 import io


class String(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="PrimitiveString_V3",
            display_name="String _V3",
            category="utils/primitive",
            inputs=[
                io.String.Input("value"),
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, value: str) -> io.NodeOutput:
        return io.NodeOutput(value)


class StringMultiline(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="PrimitiveStringMultiline_V3",
            display_name="String (Multiline) _V3",
            category="utils/primitive",
            inputs=[
                io.String.Input("value", multiline=True),
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, value: str) -> io.NodeOutput:
        return io.NodeOutput(value)


class Int(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="PrimitiveInt_V3",
            display_name="Int _V3",
            category="utils/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class Float(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="PrimitiveFloat_V3",
            display_name="Float _V3",
            category="utils/primitive",
            inputs=[
                io.Float.Input("value", min=-sys.maxsize, max=sys.maxsize),
            ],
            outputs=[io.Float.Output()],
        )

    @classmethod
    def execute(cls, value: float) -> io.NodeOutput:
        return io.NodeOutput(value)


class Boolean(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="PrimitiveBoolean_V3",
            display_name="Boolean _V3",
            category="utils/primitive",
            inputs=[
                io.Boolean.Input("value"),
            ],
            outputs=[io.Boolean.Output()],
        )

    @classmethod
    def execute(cls, value: bool) -> io.NodeOutput:
        return io.NodeOutput(value)


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    String,
    StringMultiline,
    Int,
    Float,
    Boolean,
]
