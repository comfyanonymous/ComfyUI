from comfy.comfy_types.node_typing import IO, ComfyNodeABC

class LogicIF(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "if_condition": (IO.BOOLEAN, {}),
                "when_true": (IO.ANY, {})
            },
            "optional": {
                "when_false": (IO.ANY, {})
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/logic"

    def execute(self, if_condition, when_true, when_false, **kwargs):
        if if_condition:
            return when_true,
        else:
            return when_false,

class LogicAND(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_a": (IO.BOOLEAN, {}),
                "input_b": (IO.BOOLEAN, {})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    FUNCTION = "execute"
    CATEGORY = "utils/logic"

    def execute(self, input_a, input_b, **kwargs):
        return input_a and input_b,

class LogicOR(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_a": (IO.BOOLEAN, {}),
                "input_b": (IO.BOOLEAN, {})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    FUNCTION = "execute"
    CATEGORY = "utils/logic"

    def execute(self, input_a, input_b, **kwargs):
        return input_a or input_b,

class LogicNOT(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": (IO.BOOLEAN, {})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    FUNCTION = "execute"
    CATEGORY = "utils/logic"

    def execute(self, input, **kwargs):
        return not input,

class LogicXOR(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_a": (IO.BOOLEAN, {}),
                "input_b": (IO.BOOLEAN, {})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    FUNCTION = "execute"
    CATEGORY = "utils/logic"

    def execute(self, input_a, input_b, **kwargs):
        return input_a != input_b,

NODE_CLASS_MAPPINGS = {
    "LogicIF": LogicIF,
    "LogicAND": LogicAND,
    "LogicOR": LogicOR,
    "LogicNOT": LogicNOT,
    "LogicXOR": LogicXOR
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogicIF": "IF",
    "LogicAND": "AND",
    "LogicOR": "OR",
    "LogicNOT": "NOT",
    "LogicXOR": "XOR"
}
