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
            description="Patch the model to use `attention3_sage` during the selected blocks and steps",
            category="_for_testing",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "skip_early_block",
                    tooltip="Use the default attention function for the first few Blocks",
                    default=1,
                    min=0,
                    max=99,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    optional=True,
                    advanced=True,
                ),
                io.Int.Input(
                    "skip_last_block",
                    tooltip="Use the default attention function for the last few Blocks",
                    default=1,
                    min=0,
                    max=99,
                    step=1,
                    optional=True,
                    advanced=True,
                ),
                io.Int.Input(
                    "skip_early_step",
                    tooltip="Use the default attention function for the first few Steps",
                    default=1,
                    min=0,
                    max=99,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    optional=True,
                    advanced=True,
                ),
                io.Int.Input(
                    "skip_last_step",
                    tooltip="Use the default attention function for the last few Steps",
                    default=1,
                    min=0,
                    max=99,
                    step=1,
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[io.Model.Output()],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model: ModelPatcher,
        skip_early_block: int = 1,
        skip_last_block: int = 1,
        skip_early_step: int = 1,
        skip_last_step: int = 1,
    ) -> io.NodeOutput:
        sage3: Callable | None = get_attention_function("sage3", default=None)

        if sage3 is None:
            if cls.hidden.unique_id:
                PromptServer.instance.send_progress_text(
                    'To use the "Patch SageAttention 3" node, the `sageattn3` package must be installed first',
                    cls.hidden.unique_id,
                )
            return io.NodeOutput(model)

        def sage_wrapper(model_function, kwargs: dict):
            # parse the current step on every model call instead of every attention call

            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]

            transformer_options: dict = c.get("transformer_options", {})

            sample_sigmas: torch.Tensor = transformer_options.get("sample_sigmas", None)
            sigmas: torch.Tensor = transformer_options.get("sigmas", None)

            if sample_sigmas is None or sigmas is None:
                transformer_options["_sage3"] = False
                return model_function(x, timestep, **c)

            mask: torch.Tensor = (sample_sigmas == sigmas).nonzero(as_tuple=True)[0]

            total_steps: int = sample_sigmas.size(0)
            step: int = mask.item() if mask.numel() > 0 else -1  # [0, N)

            transformer_options["_sage3"] = step > -1 and (
                skip_early_step <= step < total_steps - skip_last_step
            )

            return model_function(x, timestep, **c)

        def attention_override(func: Callable, *args, **kwargs):
            transformer_options: dict = kwargs.get("transformer_options", {})

            if not transformer_options.get("_sage3", False):
                return func(*args, **kwargs)

            total_blocks: int = transformer_options.get("total_blocks", -1)
            block_index: int = transformer_options.get("block_index", -1)  # [0, N)

            if total_blocks > -1 and (
                skip_early_block <= block_index < total_blocks - skip_last_block
            ):
                return sage3(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        model = model.clone()
        model.set_model_unet_function_wrapper(sage_wrapper)
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
