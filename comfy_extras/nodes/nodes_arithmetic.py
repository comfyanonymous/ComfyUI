from functools import reduce
from operator import add, mul, pow

from comfy.nodes.package_typing import CustomNode, InputTypes


class FloatAdd(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("FLOAT", {})}
        range_.update({f"value{i}": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (reduce(add, kwargs.values(), 0.0),)


class FloatSubtract(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value0": ("FLOAT", {}),
                "value1": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, value0, value1):
        return (value0 - value1,)


class FloatMultiply(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("FLOAT", {})}
        range_.update({f"value{i}": ("FLOAT", {"default": 1.0}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (reduce(mul, kwargs.values(), 1.0),)


class FloatDivide(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value0": ("FLOAT", {}),
                "value1": ("FLOAT", {"default": 1.0}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, value0, value1):
        return (value0 / value1 if value1 != 0 else float("inf"),)


class FloatPower(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "base": ("FLOAT", {}),
                "exponent": ("FLOAT", {"default": 1.0}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, base, exponent):
        return (pow(base, exponent),)


class IntAdd(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("INT", {})}
        range_.update({f"value{i}": ("INT", {"default": 0}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (reduce(add, kwargs.values(), 0),)


class IntSubtract(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value0": ("INT", {}),
                "value1": ("INT", {"default": 0}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value0, value1):
        return (value0 - value1,)


class IntMultiply(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("INT", {})}
        range_.update({f"value{i}": ("INT", {"default": 1}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (reduce(mul, kwargs.values(), 1),)


class IntDivide(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value0": ("INT", {}),
                "value1": ("INT", {"default": 1}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value0, value1):
        return (value0 // value1 if value1 != 0 else 0,)


class IntMod(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value0": ("INT", {}),
                "value1": ("INT", {"default": 1}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value0, value1):
        return (value0 % value1 if value1 != 0 else 0,)


class IntPower(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "base": ("INT", {}),
                "exponent": ("INT", {"default": 1}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, base, exponent):
        return (pow(base, exponent),)


class FloatMin(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("FLOAT", {})}
        range_.update({f"value{i}": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (min(kwargs.values()),)


class FloatMax(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("FLOAT", {})}
        range_.update({f"value{i}": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (max(kwargs.values()),)


class FloatAbs(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("FLOAT", {})
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (abs(value),)


class FloatAverage(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("FLOAT", {})}
        range_.update({f"value{i}": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (sum(kwargs.values()) / len(kwargs),)


class IntMin(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("INT", {})}
        range_.update({f"value{i}": ("INT", {"default": 0}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (min(kwargs.values()),)


class IntMax(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("INT", {})}
        range_.update({f"value{i}": ("INT", {"default": 0}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (max(kwargs.values()),)


class IntAbs(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("INT", {})
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (abs(value),)


class IntAverage(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        range_ = {"value0": ("INT", {})}
        range_.update({f"value{i}": ("INT", {"default": 0}) for i in range(1, 5)})

        return {
            "required": {},
            "optional": range_
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, **kwargs):
        return (sum(kwargs.values()) // len(kwargs),)


class FloatLerp(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}),
                "b": ("FLOAT", {"default": 1.0}),
                "t": ("FLOAT", {}),
                "clamped": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, a, b, t, clamped):
        value = a + (b - a) * t
        if clamped:
            value = min(max(value, a), b)
        return (value,)


class FloatInverseLerp(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}),
                "b": ("FLOAT", {"default": 1.0}),
                "value": ("FLOAT", {}),
                "clamped": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, a, b, value, clamped):
        if a == b:
            return (0.0,)
        t = (value - a) / (b - a)
        if clamped:
            t = min(max(t, 0.0), 1.0)
        return (t,)


class FloatClamp(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("FLOAT", {}),
                "min": ("FLOAT", {"default": 0.0, "step": 0.01, "round": 0.000001}),
                "max": ("FLOAT", {"default": 1.0}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, value: float = 0, **kwargs):
        v_min: float = kwargs['min']
        v_max: float = kwargs['max']
        return (min(max(value, v_min), v_max),)


class IntLerp(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "a": ("INT", {"default": 0}),
                "b": ("INT", {"default": 10}),
                "t": ("FLOAT", {}),
                "clamped": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, a, b, t, clamped):
        value = int(round(a + (b - a) * t))
        if clamped:
            value = min(max(value, a), b)
        return (value,)


class IntInverseLerp(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "a": ("INT", {"default": 0}),
                "b": ("INT", {"default": 10}),
                "value": ("INT", {}),
                "clamped": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    def execute(self, a, b, value, clamped):
        if a == b:
            return (0.0,)
        t = (value - a) / (b - a)
        if clamped:
            t = min(max(t, 0.0), 1.0)
        return (t,)


class IntClamp(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "value": ("INT", {}),
                "min": ("INT", {"default": 0}),
                "max": ("INT", {"default": 1}),
            }
        }

    CATEGORY = "arithmetic"
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value: int = 0, **kwargs):
        v_min: int = kwargs['min']
        v_max: int = kwargs['max']

        return (min(max(value, v_min), v_max),)


NODE_CLASS_MAPPINGS = {}
for cls in (
        FloatAdd,
        FloatSubtract,
        FloatMultiply,
        FloatDivide,
        FloatPower,
        FloatMin,
        FloatMax,
        FloatAbs,
        FloatAverage,
        FloatLerp,
        FloatInverseLerp,
        FloatClamp,
        IntAdd,
        IntSubtract,
        IntMultiply,
        IntDivide,
        IntMod,
        IntPower,
        IntMin,
        IntMax,
        IntAbs,
        IntAverage,
        IntLerp,
        IntInverseLerp,
        IntClamp,
):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
