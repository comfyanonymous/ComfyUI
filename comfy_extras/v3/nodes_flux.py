from __future__ import annotations

import comfy.utils
import node_helpers
from comfy_api.v3 import io

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class CLIPTextEncodeFlux(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="CLIPTextEncodeFlux_V3",
            category="advanced/conditioning/flux",
            inputs=[
                io.Clip.Input(id="clip"),
                io.String.Input(id="clip_l", multiline=True, dynamic_prompts=True),
                io.String.Input(id="t5xxl", multiline=True, dynamic_prompts=True),
                io.Float.Input(id="guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, clip_l, t5xxl, guidance):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}))


class FluxDisableGuidance(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="FluxDisableGuidance_V3",
            category="advanced/conditioning/flux",
            description="This node completely disables the guidance embed on Flux and Flux like models",
            inputs=[
                io.Conditioning.Input(id="conditioning"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": None})
        return io.NodeOutput(c)


class FluxGuidance(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="FluxGuidance_V3",
            category="advanced/conditioning/flux",
            inputs=[
                io.Conditioning.Input(id="conditioning"),
                io.Float.Input(id="guidance", default=3.5, min=0.0, max=100.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return io.NodeOutput(c)


class FluxKontextImageScale(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.SchemaV3(
            node_id="FluxKontextImageScale_V3",
            category="advanced/conditioning/flux",
            description="This node resizes the image to one that is more optimal for flux kontext.",
            inputs=[
                io.Image.Input(id="image"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image):
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        return io.NodeOutput(image)


NODES_LIST = [
    CLIPTextEncodeFlux,
    FluxDisableGuidance,
    FluxGuidance,
    FluxKontextImageScale,
]
