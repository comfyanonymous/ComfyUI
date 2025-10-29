import re
from typing import Optional

from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes

MATCH_TYPE_NAME = "MATCH"


class RegexFlags(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        flags_ = {flag_name: ("BOOLEAN", {}) for flag_name in ["ASCII", "IGNORECASE", "LOCALE", "MULTILINE", "DOTALL", "VERBOSE", "UNICODE"]}
        return {
            "required": flags_,
        }

    CATEGORY = "regular_expressions"
    FUNCTION = "execute"
    RETURN_TYPES = ("INT",)

    def execute(self, **kwargs) -> tuple[int]:
        has_noflag = hasattr(re.RegexFlag, "NOFLAG")
        # use getattr for python 3.10 compatibility
        flags = getattr(re.RegexFlag, "NOFLAG") if has_noflag else 0
        for name, on in kwargs.items():
            if on and hasattr(re.RegexFlag, name):
                flags |= int(re.RegexFlag[name])

        return int(flags),


class Regex(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "pattern": ("STRING", {}),
                "string": ("STRING", {}),
            },
            "optional": {
                "flags": ("INT", {"min": 0, "default": 0})
            }
        }

    CATEGORY = "regular_expressions"
    FUNCTION = "execute"
    RETURN_TYPES = (MATCH_TYPE_NAME, "BOOLEAN")
    RETURN_NAMES = ("match", "match found")

    def execute(self, pattern: str = "", string: str = "", flags: int = 0) -> tuple[re.Match, bool]:
        match = re.match(pattern=pattern, string=string, flags=flags)
        return match, match is not None


class RegexMatchGroupByIndex(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "match": (MATCH_TYPE_NAME, {}),
                "index": ("INT", {}),
            },
        }

    CATEGORY = "regular_expressions"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    def execute(self, match: Optional[re.Match], index: int = 0) -> tuple[str]:
        return "" if match is None else match.group(index),


class RegexMatchGroupByName(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "match": (MATCH_TYPE_NAME, {}),
                "name": ("STRING", {}),
            },
        }

    CATEGORY = "regular_expressions"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    def execute(self, match: Optional[re.Match], name: str = "") -> tuple[str]:
        return "" if match is None else match.group(name),


class RegexMatchExpand(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "match": (MATCH_TYPE_NAME, {}),
                "template": ("STRING", {}),
            },
        }

    CATEGORY = "regular_expressions"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)

    def execute(self, match: Optional[re.Match], template: str = "") -> tuple[str]:
        return "" if match is None else match.expand(template),


export_custom_nodes()
