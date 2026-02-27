"""Math expression node using JSONata for evaluation.

Provides a ComfyMathExpression node that evaluates JSONata expressions
against dynamically-grown numeric inputs. Supports frontend eager
evaluation via the eager_eval schema field.
"""

from __future__ import annotations

import jsonata as jsonata_lib
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class MathExpressionNode(io.ComfyNode):
    """Evaluates a JSONata expression against dynamically-grown inputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template(
            "num", allowed_types=[io.Float, io.Int]
        )
        autogrow = io.Autogrow.TemplatePrefix(
            input=io.MatchType.Input("value", template=template),
            prefix="value",
            min=1,
            max=100,
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
            is_output_node=True,
            eager_eval=io.EagerEval(expr_widget="expression"),
        )

    @classmethod
    def execute(
        cls, expression: str, values: io.Autogrow.Type
    ) -> io.NodeOutput:
        context: dict = {}
        for i, (key, val) in enumerate(values.items()):
            context[chr(ord("a") + i)] = val  # positional: a, b, c, ...
            context[key] = val  # also by input name: value0, value1, ...
        context["values"] = list(values.values())  # for $sum(values) etc.

        result = jsonata_lib.Jsonata(expression).evaluate(context)
        return io.NodeOutput(
            result, ui={"result": [result], "context": [context]}
        )


class MathExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MathExpressionNode]


async def comfy_entrypoint() -> MathExtension:
    return MathExtension()
