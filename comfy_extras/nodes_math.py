"""Math expression node using simpleeval for safe evaluation.

Provides a ComfyMathExpression node that evaluates math expressions
against dynamically-grown numeric inputs.
"""

from __future__ import annotations

import math

from simpleeval import simple_eval
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


def _variadic_sum(*args):
    """Support both sum(values) and sum(a, b, c)."""
    if len(args) == 1 and hasattr(args[0], "__iter__"):
        return sum(args[0])
    return sum(args)


MATH_FUNCTIONS = {
    "sum": _variadic_sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "pow": pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _positional_alias(index: int) -> str:
    """Convert 0-based index to single letter: 0->a, 1->b, ..., 25->z."""
    return chr(ord("a") + index)


class MathExpressionNode(io.ComfyNode):
    """Evaluates a math expression against dynamically-grown inputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template(
            "num", allowed_types=[io.Float, io.Int]
        )
        autogrow = io.Autogrow.TemplateNames(
            input=io.MatchType.Input("value", template=template),
            names=[_positional_alias(i) for i in range(26)],
            min=1,
        )
        return io.Schema(
            node_id="ComfyMathExpression",
            display_name="Math Expression",
            category="math",
            search_aliases=[
                "expression", "formula", "calculate", "eval", "math"
            ],
            inputs=[
                io.String.Input("expression", default="a + b"),
                io.Autogrow.Input("values", template=autogrow),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template, display_name="result"
                ),
            ],
        )

    @classmethod
    def execute(
        cls, expression: str, values: io.Autogrow.Type
    ) -> io.NodeOutput:
        context: dict = dict(values)
        context["values"] = list(values.values())

        result = simple_eval(expression, names=context, functions=MATH_FUNCTIONS)
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError(
                f"Math Expression '{expression}' must evaluate to a numeric result, "
                f"got {type(result).__name__}: {result!r}"
            )
        return io.NodeOutput(result)


class MathExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MathExpressionNode]


async def comfy_entrypoint() -> MathExtension:
    return MathExtension()
