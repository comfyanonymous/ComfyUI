import torch
from typing_extensions import override

import comfy.model_management
import node_helpers
from comfy_api.latest import ComfyExtension, io


class TextEncodeAceStepAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeAceStepAudio",
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
        tokens = clip.tokenize(tags, lyrics=lyrics)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"lyrics_strength": lyrics_strength})
        return io.NodeOutput(conditioning)


class EmptyAceStepLatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyAceStepLatentAudio",
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


class AceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeAceStepAudio,
            EmptyAceStepLatentAudio,
        ]

async def comfy_entrypoint() -> AceExtension:
    return AceExtension()
