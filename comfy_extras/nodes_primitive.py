# Primitive nodes that are evaluated at backend.
from __future__ import annotations

from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class String(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {"value": (IO.STRING, {})},
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "execute"
    CATEGORY = "utils/primitive"

    def execute(self, value: str) -> tuple[str]:
        return (value,)


class Int(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {"value": (IO.INT, {"control_after_generate": True})},
        }

    RETURN_TYPES = (IO.INT,)
    FUNCTION = "execute"
    CATEGORY = "utils/primitive"

    def execute(self, value: int) -> tuple[int]:
        return (value,)


class Float(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {"value": (IO.FLOAT, {})},
        }

    RETURN_TYPES = (IO.FLOAT,)
    FUNCTION = "execute"
    CATEGORY = "utils/primitive"

    def execute(self, value: float) -> tuple[float]:
        return (value,)


class Boolean(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {"value": (IO.BOOLEAN, {})},
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    FUNCTION = "execute"
    CATEGORY = "utils/primitive"

    def execute(self, value: bool) -> tuple[bool]:
        return (value,)


NODE_CLASS_MAPPINGS = {
    "PrimitiveString": String,
    "PrimitiveInt": Int,
    "PrimitiveFloat": Float,
    "PrimitiveBoolean": Boolean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimitiveString": "String",
    "PrimitiveInt": "Int",
    "PrimitiveFloat": "Float",
    "PrimitiveBoolean": "Boolean",
}
