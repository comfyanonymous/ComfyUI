from __future__ import annotations

import torchaudio
import folder_paths
import os
import io
import hashlib
from comfy_api.v3 import io, ui


class PreviewAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="PreviewAudio_V3",
            display_name="Preview Audio _V3",
            category="audio",
            inputs=[
                io.Audio.Input("audio"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio):
        return io.NodeOutput(ui=ui.PreviewAudio(audio, cls=cls))


class LoadAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="LoadAudio_V3",
            display_name="Load Audio _V3",
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


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    # EmptyLatentAudio_V3,
    # VAEEncodeAudio_V3,
    # VAEDecodeAudio_V3,
    # SaveAudio_V3,
    # SaveAudioMP3_V3,
    # SaveAudioOpus_V3,
    LoadAudio_V3,
    PreviewAudio_V3,
    # ConditioningStableAudio_V3,
]
