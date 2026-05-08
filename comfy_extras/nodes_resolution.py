from __future__ import annotations
import math
from enum import Enum
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_H = "3:2 (Photo)"
    STANDARD_H = "4:3 (Standard)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"
    PHOTO_V = "2:3 (Portrait Photo)"
    STANDARD_V = "3:4 (Portrait Standard)"
    WIDESCREEN_V = "9:16 (Portrait Widescreen)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.ULTRAWIDE_H: (21, 9),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.WIDESCREEN_V: (9, 16),
}


class ResolutionSelector(io.ComfyNode):
    """Calculate width and height from aspect ratio and megapixel target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResolutionSelector",
            display_name="Resolution Selector",
            category="utils",
            description="Calculate width and height from aspect ratio and megapixel target. Useful for setting up Empty Latent Image dimensions.",
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.SQUARE,
                    tooltip="The aspect ratio for the output dimensions.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Target total megapixels. 1.0 MP ≈ 1024×1024 for square.",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "width", tooltip="Calculated width in pixels (multiple of 8)."
                ),
                io.Int.Output(
                    "height", tooltip="Calculated height in pixels (multiple of 8)."
                ),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio: str, megapixels: float) -> io.NodeOutput:
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        width = round(w_ratio * scale / 8) * 8
        height = round(h_ratio * scale / 8) * 8
        return io.NodeOutput(width, height)


class ResolutionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ResolutionSelector,
        ]


async def comfy_entrypoint() -> ResolutionExtension:
    return ResolutionExtension()
