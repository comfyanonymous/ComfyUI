from typing import Callable

import torch
from typing_extensions import override

from comfy.ldm.modules.attention import get_attention_function
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import ComfyExtension, io
from server import PromptServer


class Sage3PatchModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sage3PatchModel",
            display_name="Patch SageAttention 3",
            description="Apply `attention3_sage` to the middle blocks and steps, while using optimized_attention for the first/last blocks and steps",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model: ModelPatcher) -> io.NodeOutput:
        sage3: Callable | None = get_attention_function("sage3", default=None)

        if sage3 is None:
            PromptServer.instance.send_progress_text(
                "`sageattn3` is not installed / available...",
                cls.hidden.unique_id,
            )
            return io.NodeOutput(model)

        def attention_override(func: Callable, *args, **kwargs):
            transformer_options: dict = kwargs.get("transformer_options", {})

            block_index: int = transformer_options.get("block_index", 0)
            total_blocks: int = transformer_options.get("total_blocks", 1)

            if block_index == 0 or block_index >= (total_blocks - 1):
                return func(*args, **kwargs)

            sample_sigmas: torch.Tensor = transformer_options["sample_sigmas"]
            sigmas: torch.Tensor = transformer_options["sigmas"]

            total_steps: int = sample_sigmas.size(0)
            step: int = 0

            for i in range(total_steps):
                if torch.allclose(sample_sigmas[i], sigmas):
                    step = i
                    break

            if step == 0 or step >= (total_steps - 1):
                return func(*args, **kwargs)

            return sage3(*args, **kwargs)

        model = model.clone()
        model.model_options["transformer_options"][
            "optimized_attention_override"
        ] = attention_override

        return io.NodeOutput(model)


class Sage3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Sage3PatchModel]


async def comfy_entrypoint():
    return Sage3Extension()
