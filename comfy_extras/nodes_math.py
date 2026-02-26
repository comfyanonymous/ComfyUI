from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class Add(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("add")
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.MatchType.Input("operand", template=template),
            prefix="operand",
            min=2,
            max=10
        )
        return io.Schema(
            node_id="Add",
            display_name="Add",
            category="math",
            inputs=[
                io.Autogrow.Input("operands", template=autogrow_template)
            ],
            outputs=[
                io.MatchType.Output(template=template)
            ],
        )

    @classmethod
    def execute(cls, operands: io.Autogrow.Type) -> io.NodeOutput:
        values = list(operands.values())
        result = values[0]
        for value in values[1:]:
            result = result + value
        return io.NodeOutput(result)


class Subtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("subtract")
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.MatchType.Input("operand", template=template),
            prefix="operand",
            min=2,
            max=10
        )
        return io.Schema(
            node_id="Subtract",
            display_name="Subtract",
            category="math",
            inputs=[
                io.Autogrow.Input("operands", template=autogrow_template)
            ],
            outputs=[
                io.MatchType.Output(template=template)
            ],
        )

    @classmethod
    def execute(cls, operands: io.Autogrow.Type) -> io.NodeOutput:
        values = list(operands.values())
        result = values[0]
        for value in values[1:]:
            result = result - value
        return io.NodeOutput(result)


class Multiply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("multiply")
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.MatchType.Input("operand", template=template),
            prefix="operand",
            min=2,
            max=10
        )
        return io.Schema(
            node_id="Multiply",
            display_name="Multiply",
            category="math",
            inputs=[
                io.Autogrow.Input("operands", template=autogrow_template)
            ],
            outputs=[
                io.MatchType.Output(template=template)
            ],
        )

    @classmethod
    def execute(cls, operands: io.Autogrow.Type) -> io.NodeOutput:
        values = list(operands.values())
        result = values[0]
        for value in values[1:]:
            result = result * value
        return io.NodeOutput(result)


class Divide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("divide")
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.MatchType.Input("operand", template=template),
            prefix="operand",
            min=2,
            max=10
        )
        return io.Schema(
            node_id="Divide",
            display_name="Divide",
            category="math",
            inputs=[
                io.Autogrow.Input("operands", template=autogrow_template)
            ],
            outputs=[
                io.MatchType.Output(template=template)
            ],
        )

    @classmethod
    def validate_inputs(cls, operands: io.Autogrow.Type) -> bool:
        values = list(operands.values())
        # Check for division by zero in any divisor (all operands except the first)
        for value in values[1:]:
            if value == 0:
                return "Division by zero is not allowed"
        return True

    @classmethod
    def execute(cls, operands: io.Autogrow.Type) -> io.NodeOutput:
        values = list(operands.values())
        result = values[0]
        for value in values[1:]:
            if value == 0:
                raise ValueError("Division by zero is not allowed")
            result = result / value
        return io.NodeOutput(result)


class MathExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Add,
            Subtract,
            Multiply,
            Divide,
        ]


async def comfy_entrypoint() -> MathExtension:
    return MathExtension()
