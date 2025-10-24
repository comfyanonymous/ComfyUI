from comfy.comfy_types.node_typing import IO, ComfyNodeABC
import math

class MathAdd(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, **kwargs):
        return value_a + value_b,

class MathSubtract(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, **kwargs):
        return value_a - value_b,

class MathMultiply(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, **kwargs):
        return value_a * value_b,

class MathDivide(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {}),
                "handle_zero": (IO.BOOLEAN, {"default": True})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, handle_zero, **kwargs):
        if value_b == 0:
            if handle_zero:
                return 0,
            else:
                raise ValueError("Division by zero")
        return value_a / value_b,

class MathPower(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base": (IO.NUMBER, {}),
                "exponent": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, base, exponent, **kwargs):
        return base ** exponent,

class MathFloor(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (IO.NUMBER, {}),
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value, **kwargs):
        return math.floor(value),

class MathNumberConvert(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "number_value": (IO.NUMBER, {}),
            }
        }

    RETURN_TYPES = (IO.INT, IO.FLOAT,)
    RETURN_NAMES = ("result_int", "result_float",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, number_value, **kwargs):
        return int(number_value), float(number_value)

class MathCeil(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value, **kwargs):
        return math.ceil(value),

class MathRound(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (IO.NUMBER, {}),
                "decimals": (IO.INT, {"default": 0, "min": 0, "max": 10})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value, decimals, **kwargs):
        return round(value, decimals),

class MathModulo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {}),
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, **kwargs):
        if value_b == 0:
            return 0,
        return value_a % value_b,

class MathAbs(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value, **kwargs):
        return abs(value),

class MathSqrt(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value, **kwargs):
        if value < 0:
            return 0,
        return math.sqrt(value),

class MathSin(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "angle": (IO.NUMBER, {}),
                "unit": (IO.COMBO, {"options": ["Radians", "Degrees"]})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math/trigonometry"

    def execute(self, angle, unit, **kwargs):
        if unit == "Degrees":
            angle = math.radians(angle)
        return math.sin(angle),

class MathCos(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "angle": (IO.NUMBER, {}),
                "unit": (IO.COMBO, {"options": ["Radians", "Degrees"]})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math/trigonometry"

    def execute(self, angle, unit, **kwargs):
        if unit == "Degrees":
            angle = math.radians(angle)
        return math.cos(angle),

class MathTan(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "angle": (IO.NUMBER, {}),
                "unit": (IO.COMBO, {"options": ["Radians", "Degrees"]})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math/trigonometry"

    def execute(self, angle, unit, **kwargs):
        if unit == "Degrees":
            angle = math.radians(angle)
        return math.tan(angle),

class MathMin(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, **kwargs):
        return min(value_a, value_b),

class MathMax(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, **kwargs):
        return max(value_a, value_b),

class MathClamp(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (IO.NUMBER, {}),
                "min_value": (IO.NUMBER, {}),
                "max_value": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value, min_value, max_value, **kwargs):
        return max(min(value, max_value), min_value),

class StringToNumber(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": (IO.STRING, {"multiline": False})
            },
            "optional": {
                "default_value": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.NUMBER,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, string, **kwargs):
        default_value = kwargs.get("default_value")

        if default_value is None:
            default_value = 0

        try:
            if '.' in string:
                return float(string),
            else:
                return int(string),
        except (ValueError, TypeError):
            return default_value,

class NumberToString(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "number": (IO.NUMBER, {})
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, number, **kwargs):
        return str(number),

class MathCompare(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value_a": (IO.NUMBER, {}),
                "value_b": (IO.NUMBER, {}),
                "comparison": (IO.COMBO, {"options": ["Equal", "Not Equal", "Greater Than", "Less Than", "Greater Than or Equal", "Less Than or Equal"]})
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"
    CATEGORY = "utils/math"

    def execute(self, value_a, value_b, comparison, **kwargs):
        if comparison == "Equal":
            return value_a == value_b,
        elif comparison == "Not Equal":
            return value_a != value_b,
        elif comparison == "Greater Than":
            return value_a > value_b,
        elif comparison == "Less Than":
            return value_a < value_b,
        elif comparison == "Greater Than or Equal":
            return value_a >= value_b,
        elif comparison == "Less Than or Equal":
            return value_a <= value_b,
        else:
            return False,

NODE_CLASS_MAPPINGS = {
    "MathAdd": MathAdd,
    "MathSubtract": MathSubtract,
    "MathMultiply": MathMultiply,
    "MathDivide": MathDivide,
    "MathPower": MathPower,
    "MathFloor": MathFloor,
    "MathCeil": MathCeil,
    "MathRound": MathRound,
    "MathModulo": MathModulo,
    "MathAbs": MathAbs,
    "MathSqrt": MathSqrt,
    "MathSin": MathSin,
    "MathCos": MathCos,
    "MathTan": MathTan,
    "MathMin": MathMin,
    "MathMax": MathMax,
    "MathClamp": MathClamp,
    "MathNumberConvert": MathNumberConvert,
    "StringToNumber": StringToNumber,
    "NumberToString": NumberToString,
    "MathCompare": MathCompare
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MathAdd": "Add",
    "MathSubtract": "Subtract",
    "MathMultiply": "Multiply",
    "MathDivide": "Divide",
    "MathPower": "Power",
    "MathFloor": "Floor",
    "MathCeil": "Ceil",
    "MathRound": "Round",
    "MathModulo": "Modulo",
    "MathAbs": "Absolute",
    "MathSqrt": "Square Root",
    "MathSin": "Sine",
    "MathCos": "Cosine",
    "MathTan": "Tangent",
    "MathMin": "Minimum",
    "MathMax": "Maximum",
    "MathClamp": "Clamp",
    "MathNumberConvert": "Number Convert",
    "StringToNumber": "String To Number",
    "NumberToString": "Number To String",
    "MathCompare": "Compare"
}
