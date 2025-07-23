from __future__ import annotations

import torch

import comfy.model_management
import node_helpers
from comfy_api.v3 import io


class TextEncodeAceStepAudio(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeAceStepAudio_V3",
            category="conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("tags", multiline=True, dynamic_prompts=True),
                io.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                io.Float.Input("lyrics_strength", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, tags, lyrics, lyrics_strength) -> io.NodeOutput:
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(tags, lyrics=lyrics))
        conditioning = node_helpers.conditioning_set_values(conditioning, {"lyrics_strength": lyrics_strength})
        return io.NodeOutput(conditioning)


class EmptyAceStepLatentAudio(io.ComfyNodeV3):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyAceStepLatentAudio_V3",
            category="latent/audio",
            inputs=[
                io.Float.Input("seconds", default=120.0, min=1.0, max=1000.0, step=0.1),
                io.Int.Input(
                    "batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size) -> io.NodeOutput:
        length = int(seconds * 44100 / 512 / 8)
        latent = torch.zeros([batch_size, 8, 16, length], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "type": "audio"})


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    TextEncodeAceStepAudio,
    EmptyAceStepLatentAudio,
]
