from __future__ import annotations

import kornia.color
import torch
from kornia.morphology import (
    bottom_hat,
    closing,
    dilation,
    erosion,
    gradient,
    opening,
    top_hat,
)

import comfy.model_management
from comfy_api.v3 import io


class ImageRGBToYUV(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ImageRGBToYUV_V3",
            category="image/batch",
            inputs=[
                io.Image.Input(id="image"),
            ],
            outputs=[
                io.Image.Output(id="Y", display_name="Y"),
                io.Image.Output(id="U", display_name="U"),
                io.Image.Output(id="V", display_name="V"),
            ],
        )

    @classmethod
    def execute(cls, image):
        out = kornia.color.rgb_to_ycbcr(image.movedim(-1, 1)).movedim(1, -1)
        return io.NodeOutput(out[..., 0:1].expand_as(image), out[..., 1:2].expand_as(image), out[..., 2:3].expand_as(image))


class ImageYUVToRGB(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="ImageYUVToRGB_V3",
            category="image/batch",
            inputs=[
                io.Image.Input(id="Y"),
                io.Image.Input(id="U"),
                io.Image.Input(id="V"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, Y, U, V):
        image = torch.cat([torch.mean(Y, dim=-1, keepdim=True), torch.mean(U, dim=-1, keepdim=True), torch.mean(V, dim=-1, keepdim=True)], dim=-1)
        return io.NodeOutput(kornia.color.ycbcr_to_rgb(image.movedim(-1, 1)).movedim(1, -1))


class Morphology(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="Morphology_V3",
            display_name="ImageMorphology _V3",
            category="image/postprocessing",
            inputs=[
                io.Image.Input(id="image"),
                io.Combo.Input(id="operation", options=["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"]),
                io.Int.Input(id="kernel_size", default=3, min=3, max=999, step=1),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, operation, kernel_size):
        device = comfy.model_management.get_torch_device()
        kernel = torch.ones(kernel_size, kernel_size, device=device)
        image_k = image.to(device).movedim(-1, 1)
        if operation == "erode":
            output = erosion(image_k, kernel)
        elif operation == "dilate":
            output = dilation(image_k, kernel)
        elif operation == "open":
            output = opening(image_k, kernel)
        elif operation == "close":
            output = closing(image_k, kernel)
        elif operation == "gradient":
            output = gradient(image_k, kernel)
        elif operation == "top_hat":
            output = top_hat(image_k, kernel)
        elif operation == "bottom_hat":
            output = bottom_hat(image_k, kernel)
        else:
            raise ValueError(f"Invalid operation {operation} for morphology. Must be one of 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'bottomhat'")
        return io.NodeOutput(output.to(comfy.model_management.intermediate_device()).movedim(1, -1))


NODES_LIST = [
    ImageRGBToYUV,
    ImageYUVToRGB,
    Morphology,
]
