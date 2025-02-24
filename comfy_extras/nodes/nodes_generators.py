import typing
from decimal import Decimal

from comfy.comfy_types import IO
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes


class IntRange(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "start": ("INT", {"default": 0, "step": 0.001}),
                "end": ("INT", {"default": 1, "step": 0.001}),
                "step": ("INT", {"default": 1}),
            }
        }

    CATEGORY = "generators"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, start: int, end: int, step: int) -> tuple[list[int]]:
        return list(range(start, end, step)),


class FloatRange1(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "start": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "end": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "step": ("FLOAT", {"default": 1}),
            }
        }

    CATEGORY = "generators"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, start: float, end: float, step: float) -> tuple[list[float]]:
        if step == 0:
            return [],

        steps = (Decimal(str(end)) - Decimal(str(start))) / Decimal(str(step))
        return [start + i * step for i in range(int(steps))],


class FloatRange2(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "start": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "end": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "fence_posts": ("INT", {"default": 2, "min": 0}),
            }
        }

    CATEGORY = "generators"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, start: float, end: float, fence_posts: int) -> tuple[list[float]]:
        if fence_posts == 0:
            return [],
        elif fence_posts == 1:
            return [start],
        elif fence_posts == 2:
            return [start, end],

        step = (end - start) / (fence_posts - 1)
        return [start + i * step for i in range(fence_posts)],


class FloatRange3(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "start": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "end": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "spans": ("INT", {"default": 1, "min": 0}),
            }
        }

    CATEGORY = "generators"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, start: float, end: float, spans: int) -> tuple[list[float]]:
        if spans == 0:
            return [],
        elif spans == 1:
            return [start],

        span_width = (end - start) / spans
        return [start + i * span_width for i in range(spans)],


class StringSplit(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("STRING", {}),
                "delimiter": ("STRING", {"default": ","}),
            }
        }

    CATEGORY = "generators"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, value: str = "", delimiter: str = ",") -> tuple[list[str]]:
        return value.split(delimiter),


class IterateList(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": (IO.ANY, {})
            }
        }

    CATEGORY = "generators"
    RETURN_TYPES = IO.ANY,
    FUNCTION = "execute"

    def execute(self, value: typing.Any) -> tuple[typing.Any]:
        return value,


export_custom_nodes()
