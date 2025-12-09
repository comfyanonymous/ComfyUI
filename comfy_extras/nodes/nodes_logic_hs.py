import inspect
import operator
from typing import OrderedDict, Callable, Any

from comfy.comfy_types import IO
from comfy.lazy_helpers import is_input_pending
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes


def takes_n_args(obj, n):
    if not callable(obj):
        return False

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False

    params = sig.parameters.values()

    named_param_count = sum(
        1 for p in params
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD)
    )

    return named_param_count == n


_BINARY_OPS: dict[str, Callable[[Any, Any], Any]] = OrderedDict({
    **{op: getattr(operator, op) for op in dir(operator) if takes_n_args(getattr(operator, op), 2)},
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
})
_UNARY_OPS: dict[str, Callable[[Any], Any]] = {
    **{op: getattr(operator, op) for op in dir(operator) if takes_n_args(getattr(operator, op), 1)},
    "not": lambda a: not a
}


class IsNone(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {},
            "optional": {
                "value": (IO.ANY, {}),
            }
        }

    RETURN_TYPES = (IO.BOOLEAN,)
    FUNCTION = "execute"
    CATEGORY = "logic"

    def execute(self, value=None):
        return (value is None,)


class LazySwitch(CustomNode):
    """
    sherlocked from KJ nodes with fixes
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN",),
            },
            "optional": {
                "on_false": (IO.ANY, {"lazy": True}),
                "on_true": (IO.ANY, {"lazy": True}),
            },
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "execute"
    CATEGORY = "logic"
    DESCRIPTION = "Controls flow of execution based on a boolean switch."

    def check_lazy_status(self, switch, on_false=None, on_true=None):
        try:
            on_false_not_evaluated, on_true_not_evaluated = is_input_pending("on_false", "on_true")
        except LookupError:
            on_false_not_evaluated, on_true_not_evaluated = on_false is None, on_true is None
        if switch and on_true_not_evaluated:
            return ["on_true"]
        if not switch and on_false_not_evaluated:
            return ["on_false"]

    def execute(self, switch, on_false=None, on_true=None):
        value = on_true if switch else on_false
        return (value,)


class UnaryOperation(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {},
            "optional": {
                "value": (IO.ANY, {}),
                "op": (list(_UNARY_OPS.keys()), {"default": "not"})
            }
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "execute"
    CATEGORY = "logic"

    def execute(self, value, op):
        return _UNARY_OPS[op](value),


class BooleanUnaryOperation(UnaryOperation):
    RETURN_TYPES = (IO.BOOLEAN,)

    def execute(self, value, op):
        return bool(super().execute(value, op)[0]),


class BinaryOperation(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {},
            "optional": OrderedDict({
                "lhs": (IO.ANY, {"lazy": True}),
                "op": (list(_BINARY_OPS.keys()), {"default": "eq"}),
                "rhs": (IO.ANY, {"lazy": True}),
            })
        }

    RETURN_TYPES = (IO.ANY,)
    FUNCTION = "execute"
    CATEGORY = "logic"
    DESCRIPTION = ""

    def check_lazy_status(self, lhs=None, op=None, rhs=None) -> list[str]:
        try:
            lhs_not_evaluated, rhs_not_evaluated = is_input_pending("lhs", "rhs")
        except LookupError:
            lhs_not_evaluated, rhs_not_evaluated = lhs is None, rhs is None
        lhs_evaluated, rhs_evaluated = not lhs_not_evaluated, not rhs_not_evaluated
        match op:
            case "and":
                if lhs_not_evaluated:
                    return ["lhs"]
                if lhs_evaluated and lhs is not False and rhs_not_evaluated:
                    return ["rhs"]
                return []
            case "or":
                if lhs_not_evaluated:
                    return ["lhs"]
                if lhs_evaluated and lhs is not True and rhs_not_evaluated:
                    return ["rhs"]
                return []
            case _:
                to_eval = []
                if lhs_not_evaluated:
                    to_eval.append("lhs")
                if rhs_not_evaluated:
                    to_eval.append("rhs")
                return to_eval

    def execute(self, lhs, op, rhs):
        return _BINARY_OPS[op](lhs, rhs),


class BooleanBinaryOperation(BinaryOperation):
    RETURN_TYPES = (IO.BOOLEAN,)

    def execute(self, lhs, op, rhs):
        return bool(super().execute(lhs, op, rhs)[0]),


export_custom_nodes()
