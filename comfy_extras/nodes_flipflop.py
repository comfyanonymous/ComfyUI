from __future__ import annotations
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class FlipFlop(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FlipFlopNew",
            display_name="FlipFlop (New)",
            category="_for_testing",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="block_percentage", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output()
            ],
            description="Apply FlipFlop transformation to model using setup_flipflop_holders method"
        )

    @classmethod
    def execute(cls, model: io.Model.Type, block_percentage: float) -> io.NodeOutput:
        # NOTE: this is just a hacky prototype still, this would not be exposed as a node.
        # At the moment, this modifies the underlying model with no way to 'unpatch' it.
        model = model.clone()
        if not hasattr(model.model.diffusion_model, "setup_flipflop_holders"):
            raise ValueError("Model does not have flipflop holders; FlipFlop not supported")
        model.model.diffusion_model.setup_flipflop_holders(block_percentage)
        return io.NodeOutput(model)

class FlipFlopExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FlipFlop,
        ]


async def comfy_entrypoint() -> FlipFlopExtension:
    return FlipFlopExtension()
