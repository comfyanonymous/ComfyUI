"""
ComfyUI nodes for autoregressive video generation (Causal Forcing, Self-Forcing, etc.).
  - EmptyARVideoLatent: create 5D [B, C, T, H, W] video latent tensors
  - SamplerARVideo: SAMPLER for the block-by-block autoregressive denoising loop
"""

import torch
from typing_extensions import override

import comfy.model_management
import comfy.samplers
from comfy_api.latest import ComfyExtension, io


class EmptyARVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyARVideoLatent",
            category="latent/video",
            inputs=[
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=1024, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[
                io.Latent.Output(display_name="LATENT"),
            ],
        )

    @classmethod
    def execute(cls, width, height, length, batch_size) -> io.NodeOutput:
        lat_t = ((length - 1) // 4) + 1
        latent = torch.zeros(
            [batch_size, 16, lat_t, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": latent})


class SamplerARVideo(io.ComfyNode):
    """Sampler for autoregressive video models (Causal Forcing, Self-Forcing).

    All AR-loop parameters are owned by this node so they live in the workflow.
    Add new widgets here as the AR sampler grows new options.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerARVideo",
            display_name="Sampler AR Video",
            category="sampling/custom_sampling/samplers",
            inputs=[
                io.Int.Input(
                    "num_frame_per_block",
                    default=1, min=1, max=64,
                    tooltip="Frames per autoregressive block. 1 = framewise, "
                            "3 = chunkwise. Must match the checkpoint's training mode.",
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, num_frame_per_block) -> io.NodeOutput:
        extra_options = {
            "num_frame_per_block": num_frame_per_block,
        }
        return io.NodeOutput(comfy.samplers.ksampler("ar_video", extra_options))


class ARVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyARVideoLatent,
            SamplerARVideo,
        ]


async def comfy_entrypoint() -> ARVideoExtension:
    return ARVideoExtension()
