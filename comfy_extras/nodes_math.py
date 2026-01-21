from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class IntAdd(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IntAdd",
            display_name="Int Add",
            category="math",
            inputs=[
                io.Int.Input("a"),
                io.Int.Input("b"),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(a + b)


class IntSubtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IntSubtract",
            display_name="Int Subtract",
            category="math",
            inputs=[
                io.Int.Input("a"),
                io.Int.Input("b"),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(a - b)


class IntMultiply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IntMultiply",
            display_name="Int Multiply",
            category="math",
            inputs=[
                io.Int.Input("a"),
                io.Int.Input("b"),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(a * b)


class IntDivide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IntDivide",
            display_name="Int Divide",
            category="math",
            inputs=[
                io.Int.Input("a"),
                io.Int.Input("b"),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def validate_inputs(cls, a: int, b: int) -> bool:
        if b == 0:
            return "Division by zero is not allowed"
        return True

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        # Integer division (floor division)
        return io.NodeOutput(a // b)


class MathExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            IntAdd,
            IntSubtract,
            IntMultiply,
            IntDivide,
        ]


async def comfy_entrypoint() -> MathExtension:
    return MathExtension()
