from __future__ import annotations
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from comfy.ldm.flipflop_transformer import FLIPFLOP_REGISTRY

class FlipFlopOld(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FlipFlop",
            display_name="FlipFlop (Old)",
            category="_for_testing",
            inputs=[
                io.Model.Input(id="model")
            ],
            outputs=[
                io.Model.Output()
            ],
            description="Apply FlipFlop transformation to model using registry-based patching"
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        patch_cls = FLIPFLOP_REGISTRY.get(model.model.diffusion_model.__class__.__name__, None)
        if patch_cls is None:
            raise ValueError(f"Model {model.model.diffusion_model.__class__.__name__} not supported")

        model.model.diffusion_model = patch_cls.patch(model.model.diffusion_model)

        return io.NodeOutput(model)

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
            FlipFlopOld,
            FlipFlop,
        ]


async def comfy_entrypoint() -> FlipFlopExtension:
    return FlipFlopExtension()
