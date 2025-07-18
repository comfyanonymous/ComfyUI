from __future__ import annotations

from kornia.filters import canny

import comfy.model_management
from comfy_api.v3 import io


class Canny(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="Canny_V3",
            category="image/preprocessors",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("low_threshold", default=0.4, min=0.01, max=0.99, step=0.01),
                io.Float.Input("high_threshold", default=0.8, min=0.01, max=0.99, step=0.01),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, low_threshold, high_threshold) -> io.NodeOutput:
        output = canny(image.to(comfy.model_management.get_torch_device()).movedim(-1, 1), low_threshold, high_threshold)
        img_out = output[1].to(comfy.model_management.intermediate_device()).repeat(1, 3, 1, 1).movedim(1, -1)
        return io.NodeOutput(img_out)


NODES_LIST = [
    Canny,
]
