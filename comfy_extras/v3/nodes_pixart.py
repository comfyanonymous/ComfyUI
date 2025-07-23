from __future__ import annotations

import nodes
from comfy_api.v3 import io


class CLIPTextEncodePixArtAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodePixArtAlpha_V3",
            category="advanced/conditioning",
            description="Encodes text and sets the resolution conditioning for PixArt Alpha. Does not apply to PixArt Sigma.",
            inputs=[
                io.Int.Input("width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Clip.Input("clip"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, text, clip):
        tokens = clip.tokenize(text)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"width": width, "height": height}))


NODES_LIST = [
    CLIPTextEncodePixArtAlpha,
]
