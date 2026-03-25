"""
ComfyUI nodes for autoregressive video generation (Causal Forcing, Self-Forcing, etc.).
  - EmptyARVideoLatent: create 5D [B, C, T, H, W] video latent tensors
"""

import torch
from typing_extensions import override

import comfy.model_management
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


class ARVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyARVideoLatent,
        ]


async def comfy_entrypoint() -> ARVideoExtension:
    return ARVideoExtension()
