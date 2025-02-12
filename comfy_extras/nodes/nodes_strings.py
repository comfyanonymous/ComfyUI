from typing import Any

import torch
from natsort import natsorted

from comfy.comfy_types import IO
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult


def format_value(value: Any) -> str | Any:
    """Helper function to format values for string formatting."""
    if value is None:
        return "None"
    elif isinstance(value, torch.Tensor):
        if value.numel() > 10:  # For large tensors
            shape_str = 'x'.join(str(x) for x in value.shape)
            return f"<Tensor shape={shape_str}>"
        else:
            return str(value.tolist())
    return value


class StringFormat(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        optional = {f"value{i}": (IO.ANY, {"default": "", "forceInput": True}) for i in range(5)}
        optional["format"] = (IO.STRING, {"default": "{}", "multiline": True})
        return {
            "required": {},
            "optional": optional
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "strings"
    FUNCTION = "execute"

    def execute(self, format: str = "{}", *args: Any, **kwargs) -> ValidatedNodeResult:
        for k, v in kwargs.items():
            kwargs[k] = format_value(v)
        try:
            if any(f'{{{k}' in format for k in kwargs.keys()):
                return (format.format(**kwargs),)
            else:
                return (format.format(*[kwargs[k] for k in natsorted(kwargs.keys())]),)
        except (IndexError, KeyError) as e:
            return ("Format error: " + str(e),)


class ToString(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": (IO.ANY, {}),
            }
        }

    CATEGORY = "strings"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, value: Any):
        return str(value),


export_custom_nodes()
