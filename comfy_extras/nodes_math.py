"""Math expression node using JSONata for evaluation.

Provides a ComfyMathExpression node that evaluates JSONata expressions
against dynamically-grown numeric inputs.
"""

from __future__ import annotations

import jsonata as jsonata_lib
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


def _positional_alias(index: int) -> str:
    """Convert 0-based index to single letter: 0->a, 1->b, ..., 25->z."""
    return chr(ord("a") + index)


class MathExpressionNode(io.ComfyNode):
    """Evaluates a JSONata expression against dynamically-grown inputs."""

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

        result = jsonata_lib.Jsonata(expression).evaluate(context)
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
