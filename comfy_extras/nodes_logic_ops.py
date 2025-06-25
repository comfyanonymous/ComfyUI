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

class CompareValues:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "valueA": (any, ),
                "valueB": (any, ),
                "mode": (["A == B", "A != B", "A > b", "A >= b", "a < B", "a <= B"],)
                }
            }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("output",)
    CATEGORY = "utils/logic"
    FUNCTION = "compare"

    _OPS = {
        "A == B": lambda a, b: a == b,
        "A != B": lambda a, b: a != b,
        "A > b":  lambda a, b: a > b,
        "A >= b": lambda a, b: a >= b,
        "a < B":  lambda a, b: a < b,
        "a <= B": lambda a, b: a <= b,
    }

    def compare(self, valueA, valueB, mode: str) -> tuple[bool]:
        if isinstance(valueA, str) and isinstance(valueB, (int, float)):
            try: valueA = float(valueA)
            except ValueError as e: raise ValueError("Failed to convert valueA to float.") from e
        elif isinstance(valueB, str) and isinstance(valueA, (int, float)):
            try: valueB = float(valueB)
            except ValueError as e: raise ValueError("Failed to convert valueB to float.") from e

        return (self._OPS[mode](valueA, valueB),)

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
    "CompareValues": CompareValues,
    "BooleanSwitch": BooleanSwitch,
    "OutputExists": OutputExists,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "BooleanLogicGate": "Boolean Logic Gate",
    "CompareValues": "Compare Values",
    "BooleanSwitch": "Boolean Switch",
    "OutputExists": "Output Exists",
    }
