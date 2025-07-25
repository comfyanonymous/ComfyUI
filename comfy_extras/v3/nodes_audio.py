from __future__ import annotations

import hashlib
import os

import torch
import torchaudio

import comfy.model_management
import folder_paths
import node_helpers
from comfy_api.latest import io, ui


class ConditioningStableAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConditioningStableAudio_V3",
            category="conditioning",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input("seconds_start", default=0.0, min=0.0, max=1000.0, step=0.1),
                io.Float.Input("seconds_total", default=47.0, min=0.0, max=1000.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, seconds_start, seconds_total) -> io.NodeOutput:
        return io.NodeOutput(
            node_helpers.conditioning_set_values(
                positive, {"seconds_start": seconds_start, "seconds_total": seconds_total}
            ),
            node_helpers.conditioning_set_values(
                negative, {"seconds_start": seconds_start, "seconds_total": seconds_total}
            ),
        )


class EmptyLatentAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyLatentAudio_V3",
            category="latent/audio",
            inputs=[
                io.Float.Input("seconds", default=47.6, min=1.0, max=1000.0, step=0.1),
                io.Int.Input(
                    id="batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size) -> io.NodeOutput:
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "type": "audio"})


class LoadAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadAudio_V3",  # frontend expects "LoadAudio" to work
            display_name="Load Audio _V3",  # frontend ignores "display_name" for this node
            category="audio",
            inputs=[
                io.Combo.Input("audio", upload=io.UploadType.audio, options=cls.get_files_options()),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def get_files_options(cls) -> list[str]:
        input_dir = folder_paths.get_input_directory()
        return sorted(folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"]))

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        waveform, sample_rate = torchaudio.load(folder_paths.get_annotated_filepath(audio))
        return io.NodeOutput({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})

    @classmethod
    def fingerprint_inputs(s, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(s, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True


class PreviewAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PreviewAudio_V3",  # frontend expects "PreviewAudio" to work
            display_name="Preview Audio _V3",  # frontend ignores "display_name" for this node
            category="audio",
            inputs=[
                io.Audio.Input("audio"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        return io.NodeOutput(ui=ui.PreviewAudio(audio, cls=cls))


class SaveAudioMP3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveAudioMP3_V3",  # frontend expects "SaveAudioMP3" to work
            display_name="Save Audio(MP3) _V3",  # frontend ignores "display_name" for this node
            category="audio",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("filename_prefix", default="audio/ComfyUI"),
                io.Combo.Input("quality", options=["V0", "128k", "320k"], default="V0"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="mp3", quality="V0") -> io.NodeOutput:
        return io.NodeOutput(
            ui=ui.AudioSaveHelper.get_save_audio_ui(
                audio, filename_prefix=filename_prefix, cls=cls, format=format, quality=quality
            )
        )


class SaveAudioOpus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveAudioOpus_V3",  # frontend expects "SaveAudioOpus" to work
            display_name="Save Audio(Opus) _V3",  # frontend ignores "display_name" for this node
            category="audio",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("filename_prefix", default="audio/ComfyUI"),
                io.Combo.Input("quality", options=["64k", "96k", "128k", "192k", "320k"], default="128k"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="opus", quality="128k") -> io.NodeOutput:
        return io.NodeOutput(
            ui=ui.AudioSaveHelper.get_save_audio_ui(
                audio, filename_prefix=filename_prefix, cls=cls, format=format, quality=quality
            )
        )


class SaveAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveAudio_V3",  # frontend expects "SaveAudio" to work
            display_name="Save Audio _V3",  # frontend ignores "display_name" for this node
            category="audio",
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("filename_prefix", default="audio/ComfyUI"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="flac") -> io.NodeOutput:
        return io.NodeOutput(
            ui=ui.AudioSaveHelper.get_save_audio_ui(audio, filename_prefix=filename_prefix, cls=cls, format=format)
        )


class VAEDecodeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDecodeAudio_V3",
            category="latent/audio",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, vae, samples) -> io.NodeOutput:
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        return io.NodeOutput({"waveform": audio, "sample_rate": 44100})


class VAEEncodeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEEncodeAudio_V3",
            category="latent/audio",
            inputs=[
                io.Audio.Input("audio"),
                io.Vae.Input("vae"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, audio) -> io.NodeOutput:
        sample_rate = audio["sample_rate"]
        if 44100 != sample_rate:
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
        else:
            waveform = audio["waveform"]
        return io.NodeOutput({"samples": vae.encode(waveform.movedim(1, -1))})


NODES_LIST: list[type[io.ComfyNode]] = [
    ConditioningStableAudio,
    EmptyLatentAudio,
    LoadAudio,
    PreviewAudio,
    SaveAudioMP3,
    SaveAudioOpus,
    SaveAudio,
    VAEDecodeAudio,
    VAEEncodeAudio,
]
