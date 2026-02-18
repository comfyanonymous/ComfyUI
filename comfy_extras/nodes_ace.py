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

class TextEncodeAceStepAudio15(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeAceStepAudio1.5",
            category="conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("tags", multiline=True, dynamic_prompts=True),
                io.String.Input("lyrics", multiline=True, dynamic_prompts=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("bpm", default=120, min=10, max=300),
                io.Float.Input("duration", default=120.0, min=0.0, max=2000.0, step=0.1),
                io.Combo.Input("timesignature", options=['2', '3', '4', '6']),
                io.Combo.Input("language", options=["en", "ja", "zh", "es", "de", "fr", "pt", "ru", "it", "nl", "pl", "tr", "vi", "cs", "fa", "id", "ko", "uk", "hu", "ar", "sv", "ro", "el"]),
                io.Combo.Input("keyscale", options=[f"{root} {quality}" for quality in ["major", "minor"] for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]]),
                io.Boolean.Input("generate_audio_codes", default=True, tooltip="Enable the LLM that generates audio codes. This can be slow but will increase the quality of the generated audio. Turn this off if you are giving the model an audio reference.", advanced=True),
                io.Float.Input("cfg_scale", default=2.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Float.Input("temperature", default=0.85, min=0.0, max=2.0, step=0.01, advanced=True),
                io.Float.Input("top_p", default=0.9, min=0.0, max=2000.0, step=0.01, advanced=True),
                io.Int.Input("top_k", default=0, min=0, max=100, advanced=True),
                io.Float.Input("min_p", default=0.000, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("audio_cover_strength", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True, tooltip="Controls how many denoising steps use LM code conditioning. 1.0 = all steps, 0.5 = first half only."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, tags, lyrics, seed, bpm, duration, timesignature, language, keyscale, generate_audio_codes, cfg_scale, temperature, top_p, top_k, min_p, audio_cover_strength) -> io.NodeOutput:
        tokens = clip.tokenize(tags, lyrics=lyrics, bpm=bpm, duration=duration, timesignature=int(timesignature), language=language, keyscale=keyscale, seed=seed, generate_audio_codes=generate_audio_codes, cfg_scale=cfg_scale, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"audio_cover_strength": audio_cover_strength})
        return io.NodeOutput(conditioning)


class EmptyAceStepLatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyAceStepLatentAudio",
            display_name="Empty Ace Step 1.0 Latent Audio",
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


class EmptyAceStep15LatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyAceStep1.5LatentAudio",
            display_name="Empty Ace Step 1.5 Latent Audio",
            category="latent/audio",
            inputs=[
                io.Float.Input("seconds", default=120.0, min=1.0, max=1000.0, step=0.01),
                io.Int.Input(
                    "batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size) -> io.NodeOutput:
        length = round((seconds * 48000 / 1920))
        latent = torch.zeros([batch_size, 64, length], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "type": "audio"})

class ReferenceAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReferenceTimbreAudio",
            display_name="Reference Audio",
            category="advanced/conditioning/audio",
            is_experimental=True,
            description="This node sets the reference audio for ace step 1.5",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, conditioning, latent=None) -> io.NodeOutput:
        if latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_audio_timbre_latents": [latent["samples"]]}, append=True)
        return io.NodeOutput(conditioning)

class AceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeAceStepAudio,
            EmptyAceStepLatentAudio,
            TextEncodeAceStepAudio15,
            EmptyAceStep15LatentAudio,
            ReferenceAudio,
        ]

async def comfy_entrypoint() -> AceExtension:
    return AceExtension()
