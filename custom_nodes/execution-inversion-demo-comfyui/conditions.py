import re
import torch

class IntConditions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "operation": (["==", "!=", "<", ">", "<=", ">="],),
            },
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "int_condition"

    CATEGORY = "Conditions"

    def int_condition(self, a, b, operation):
        if operation == "==":
            return (a == b,)
        elif operation == "!=":
            return (a != b,)
        elif operation == "<":
            return (a < b,)
        elif operation == ">":
            return (a > b,)
        elif operation == "<=":
            return (a <= b,)
        elif operation == ">=":
            return (a >= b,)


class FloatConditions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0, "min": -999999999999.0, "max": 999999999999.0, "step": 1}),
                "b": ("FLOAT", {"default": 0, "min": -999999999999.0, "max": 999999999999.0, "step": 1}),
                "operation": (["==", "!=", "<", ">", "<=", ">="],),
            },
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "float_condition"

    CATEGORY = "Conditions"

    def float_condition(self, a, b, operation):
        if operation == "==":
            return (a == b,)
        elif operation == "!=":
            return (a != b,)
        elif operation == "<":
            return (a < b,)
        elif operation == ">":
            return (a > b,)
        elif operation == "<=":
            return (a <= b,)
        elif operation == ">=":
            return (a >= b,)

class StringConditions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"multiline": False}),
                "b": ("STRING", {"multiline": False}),
                "operation": (["a == b", "a != b", "a IN b", "a MATCH REGEX(b)", "a BEGINSWITH b", "a ENDSWITH b"],),
                "case_sensitive": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "string_condition"

    CATEGORY = "Conditions"

    def string_condition(self, a, b, operation, case_sensitive):
        if not case_sensitive:
            a = a.lower()
            b = b.lower()

        if operation == "a == b":
            return (a == b,)
        elif operation == "a != b":
            return (a != b,)
        elif operation == "a IN b":
            return (a in b,)
        elif operation == "a MATCH REGEX(b)":
            try:
                return (re.match(b, a) is not None,)
            except:
                return (False,)
        elif operation == "a BEGINSWITH b":
            return (a.startswith(b),)
        elif operation == "a ENDSWITH b":
            return (a.endswith(b),)

class ToBoolNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
            },
            "optional": {
                "invert": ("BOOL", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "to_bool"

    CATEGORY = "InversionDemo Nodes"

    def to_bool(self, value, invert = False):
        if isinstance(value, torch.Tensor):
            if value.max().item() == 0 and value.min().item() == 0:
                result = False
            else:
                result = True
        else:
            try:
                result = bool(value)
            except:
                # Can't convert it? Well then it's something or other. I dunno, I'm not a Python programmer.
                result = True

        if invert:
            result = not result

        return (result,)

CONDITION_NODE_CLASS_MAPPINGS = {
    "IntConditions": IntConditions,
    "FloatConditions": FloatConditions,
    "StringConditions": StringConditions,
    "ToBoolNode": ToBoolNode,
}

CONDITION_NODE_DISPLAY_NAME_MAPPINGS = {
    "IntConditions": "Int Condition",
    "FloatConditions": "Float Condition",
    "StringConditions": "String Condition",
    "ToBoolNode": "To Bool",
}
