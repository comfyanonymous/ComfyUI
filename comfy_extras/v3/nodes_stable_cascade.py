"""
This file is part of ComfyUI.
Copyright (C) 2024 Stability AI

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch

import comfy.utils
import nodes
from comfy_api.v3 import io


class StableCascade_EmptyLatentImage_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StableCascade_EmptyLatentImage_V3",
            category="latent/stable_cascade",
            inputs=[
                io.Int.Input("width", default=1024, min=256, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=1024, min=256, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("compression", default=42, min=4, max=128, step=1),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output("stage_c", display_name="stage_c"),
                io.Latent.Output("stage_b", display_name="stage_b"),
            ],
        )

    @classmethod
    def execute(cls, width, height, compression, batch_size=1):
        c_latent = torch.zeros([batch_size, 16, height // compression, width // compression])
        b_latent = torch.zeros([batch_size, 4, height // 4, width // 4])
        return io.NodeOutput({"samples": c_latent}, {"samples": b_latent})


class StableCascade_StageC_VAEEncode_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StableCascade_StageC_VAEEncode_V3",
            category="latent/stable_cascade",
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.Int.Input("compression", default=42, min=4, max=128, step=1),
            ],
            outputs=[
                io.Latent.Output("stage_c", display_name="stage_c"),
                io.Latent.Output("stage_b", display_name="stage_b"),
            ],
        )

    @classmethod
    def execute(cls, image, vae, compression):
        width = image.shape[-2]
        height = image.shape[-3]
        out_width = (width // compression) * vae.downscale_ratio
        out_height = (height // compression) * vae.downscale_ratio

        s = comfy.utils.common_upscale(image.movedim(-1, 1), out_width, out_height, "bicubic", "center").movedim(1, -1)

        c_latent = vae.encode(s[:, :, :, :3])
        b_latent = torch.zeros([c_latent.shape[0], 4, (height // 8) * 2, (width // 8) * 2])
        return io.NodeOutput({"samples": c_latent}, {"samples": b_latent})


class StableCascade_StageB_Conditioning_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StableCascade_StageB_Conditioning_V3",
            category="conditioning/stable_cascade",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("stage_c"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, stage_c):
        c = []
        for t in conditioning:
            d = t[1].copy()
            d["stable_cascade_prior"] = stage_c["samples"]
            n = [t[0], d]
            c.append(n)
        return io.NodeOutput(c)


class StableCascade_SuperResolutionControlnet_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="StableCascade_SuperResolutionControlnet_V3",
            category="_for_testing/stable_cascade",
            is_experimental=True,
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
            ],
            outputs=[
                io.Image.Output("controlnet_input", display_name="controlnet_input"),
                io.Latent.Output("stage_c", display_name="stage_c"),
                io.Latent.Output("stage_b", display_name="stage_b"),
            ],
        )

    @classmethod
    def execute(cls, image, vae):
        width = image.shape[-2]
        height = image.shape[-3]
        batch_size = image.shape[0]
        controlnet_input = vae.encode(image[:, :, :, :3]).movedim(1, -1)

        c_latent = torch.zeros([batch_size, 16, height // 16, width // 16])
        b_latent = torch.zeros([batch_size, 4, height // 2, width // 2])
        return io.NodeOutput(controlnet_input, {"samples": c_latent}, {"samples": b_latent})


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    StableCascade_EmptyLatentImage_V3,
    StableCascade_StageB_Conditioning_V3,
    StableCascade_StageC_VAEEncode_V3,
    StableCascade_SuperResolutionControlnet_V3,
]
