import re
import torch

class TestIntConditions:
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

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "int_condition"

    CATEGORY = "Testing/Logic"

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


class TestFloatConditions:
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

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "float_condition"

    CATEGORY = "Testing/Logic"

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

class TestStringConditions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("STRING", {"multiline": False}),
                "b": ("STRING", {"multiline": False}),
                "operation": (["a == b", "a != b", "a IN b", "a MATCH REGEX(b)", "a BEGINSWITH b", "a ENDSWITH b"],),
                "case_sensitive": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "string_condition"

    CATEGORY = "Testing/Logic"

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

class TestToBoolNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "to_bool"

    CATEGORY = "Testing/Logic"

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

class TestBoolOperationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("BOOLEAN",),
                "b": ("BOOLEAN",),
                "op": (["a AND b", "a OR b", "a XOR b", "NOT a"],),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "bool_operation"

    CATEGORY = "Testing/Logic"

    def bool_operation(self, a, b, op):
        if op == "a AND b":
            return (a and b,)
        elif op == "a OR b":
            return (a or b,)
        elif op == "a XOR b":
            return (a ^ b,)
        elif op == "NOT a":
            return (not a,)


CONDITION_NODE_CLASS_MAPPINGS = {
    "TestIntConditions": TestIntConditions,
    "TestFloatConditions": TestFloatConditions,
    "TestStringConditions": TestStringConditions,
    "TestToBoolNode": TestToBoolNode,
    "TestBoolOperationNode": TestBoolOperationNode,
}

CONDITION_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestIntConditions": "Int Condition",
    "TestFloatConditions": "Float Condition",
    "TestStringConditions": "String Condition",
    "TestToBoolNode": "To Bool",
    "TestBoolOperationNode": "Bool Operation",
}
