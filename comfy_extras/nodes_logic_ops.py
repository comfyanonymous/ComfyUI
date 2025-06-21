# Credit to pythongosssss for the AnyType code
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class BooleanLogicGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1": ("BOOLEAN", {"default": True}),
                "value2": ("BOOLEAN", {"default": True}),
                "mode": (["NOT", "AND", "OR", "NAND", "NOR", "XOR", "XNOR"],)
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("output",)
    CATEGORY = "utils/logic"
    FUNCTION = "apply_operation"

    _OPS = {
        "NOT":  lambda a, b: not a,
        "AND":  lambda a, b: a and b,
        "OR":   lambda a, b: a or b,
        "NAND": lambda a, b: not (a and b),
        "NOR":  lambda a, b: not (a or b),
        "XOR":  lambda a, b: a ^ b,
        "XNOR": lambda a, b: not (a ^ b),
    }

    def apply_operation(self, value1: bool, value2: bool, mode: str) -> tuple[bool]:
        return (self._OPS[mode](value1, value2),)

class BooleanSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"switch": ("BOOLEAN", {"default": True})},
            "optional": {"on_true": (any,), "on_false": (any,)}
        }

    RETURN_TYPES = (any,)
    CATEGORY = "utils/logic"
    FUNCTION = "switch"

    def switch(self, switch: bool, on_true=None, on_false=None) -> tuple:
        return ((on_true if switch else on_false),)

class OutputExists:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"variable": (any,)}}

    RETURN_TYPES = ("BOOLEAN",)
    CATEGORY = "utils/logic"
    FUNCTION = "test_existence"

    def test_existence(self, variable=None) -> tuple[bool]:
        return (variable is not None,)

NODE_CLASS_MAPPINGS = {
    "BooleanLogicGate": BooleanLogicGate,
    "BooleanSwitch": BooleanSwitch,
    "OutputExists": OutputExists,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "BooleanLogicGate": "Boolean Logic Gate",
    "BooleanSwitch": "Boolean Switch",
    "OutputExists": "Output Exists",
    }
