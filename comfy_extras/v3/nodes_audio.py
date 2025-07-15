from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO

import av
import torch
import torchaudio

import comfy.model_management
import folder_paths
import node_helpers
from comfy.cli_args import args
from comfy_api.v3 import io, ui


class ConditioningStableAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ConditioningStableAudio_V3",
            category="conditioning",
            inputs=[
                io.Conditioning.Input(id="positive"),
                io.Conditioning.Input(id="negative"),
                io.Float.Input(id="seconds_start", default=0.0, min=0.0, max=1000.0, step=0.1),
                io.Float.Input(id="seconds_total", default=47.0, min=0.0, max=1000.0, step=0.1),
            ],
            outputs=[
                io.Conditioning.Output(id="positive_out", display_name="positive"),
                io.Conditioning.Output(id="negative_out", display_name="negative"),
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


class EmptyLatentAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="EmptyLatentAudio_V3",
            category="latent/audio",
            inputs=[
                io.Float.Input(id="seconds", default=47.6, min=1.0, max=1000.0, step=0.1),
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
        return io.NodeOutput({"samples":latent, "type": "audio"})


class LoadAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="LoadAudio_V3",         # frontend expects "LoadAudio" to work
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


class PreviewAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="PreviewAudio_V3",         # frontend expects "PreviewAudio" to work
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


class SaveAudioMP3_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="SaveAudioMP3_V3",           # frontend expects "SaveAudioMP3" to work
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
    def execute(self, audio, filename_prefix="ComfyUI", format="mp3", quality="V0") -> io.NodeOutput:
        return _save_audio(self, audio, filename_prefix, format, quality)


class SaveAudioOpus_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="SaveAudioOpus_V3",           # frontend expects "SaveAudioOpus" to work
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
    def execute(self, audio, filename_prefix="ComfyUI", format="opus", quality="128k") -> io.NodeOutput:
        return _save_audio(self, audio, filename_prefix, format, quality)


class SaveAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="SaveAudio_V3",         # frontend expects "SaveAudio" to work
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
        return _save_audio(cls, audio, filename_prefix, format)


class VAEDecodeAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="VAEDecodeAudio_V3",
            category="latent/audio",
            inputs=[
                io.Latent.Input(id="samples"),
                io.Vae.Input(id="vae"),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, vae, samples) -> io.NodeOutput:
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        std = torch.std(audio, dim=[1,2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        return io.NodeOutput({"waveform": audio, "sample_rate": 44100})


class VAEEncodeAudio_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="VAEEncodeAudio_V3",
            category="latent/audio",
            inputs=[
                io.Audio.Input(id="audio"),
                io.Vae.Input(id="vae"),
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


def _save_audio(cls, audio, filename_prefix="ComfyUI", format="flac", quality="128k") -> io.NodeOutput:
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory()
    )

    # Prepare metadata dictionary
    metadata = {}
    if not args.disable_metadata:
        if cls.hidden.prompt is not None:
            metadata["prompt"] = json.dumps(cls.hidden.prompt)
        if cls.hidden.extra_pnginfo is not None:
            for x in cls.hidden.extra_pnginfo:
                metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

    # Opus supported sample rates
    OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

    results = []
    for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.{format}"
        output_path = os.path.join(full_output_folder, file)

        # Use original sample rate initially
        sample_rate = audio["sample_rate"]

        # Handle Opus sample rate requirements
        if format == "opus":
            if sample_rate > 48000:
                sample_rate = 48000
            elif sample_rate not in OPUS_RATES:
                # Find the next highest supported rate
                for rate in sorted(OPUS_RATES):
                    if rate > sample_rate:
                        sample_rate = rate
                        break
                if sample_rate not in OPUS_RATES:  # Fallback if still not supported
                    sample_rate = 48000

            # Resample if necessary
            if sample_rate != audio["sample_rate"]:
                waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)

        # Create output with specified format
        output_buffer = BytesIO()
        output_container = av.open(output_buffer, mode='w', format=format)

        # Set metadata on the container
        for key, value in metadata.items():
            output_container.metadata[key] = value

        # Set up the output stream with appropriate properties
        if format == "opus":
            out_stream = output_container.add_stream("libopus", rate=sample_rate)
            if quality == "64k":
                out_stream.bit_rate = 64000
            elif quality == "96k":
                out_stream.bit_rate = 96000
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "192k":
                out_stream.bit_rate = 192000
            elif quality == "320k":
                out_stream.bit_rate = 320000
        elif format == "mp3":
            out_stream = output_container.add_stream("libmp3lame", rate=sample_rate)
            if quality == "V0":
                #TODO i would really love to support V3 and V5 but there doesn't seem to be a way to set the qscale level, the property below is a bool
                out_stream.codec_context.qscale = 1
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "320k":
                out_stream.bit_rate = 320000
        else: # format == "flac":
            out_stream = output_container.add_stream("flac", rate=sample_rate)

        frame = av.AudioFrame.from_ndarray(
            waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
            format='flt',
            layout='mono' if waveform.shape[0] == 1 else 'stereo',
        )
        frame.sample_rate = sample_rate
        frame.pts = 0
        output_container.mux(out_stream.encode(frame))

        # Flush encoder
        output_container.mux(out_stream.encode(None))

        # Close containers
        output_container.close()

        # Write the output to file
        output_buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(output_buffer.getbuffer())

        results.append(ui.SavedResult(file, subfolder, io.FolderType.output))
        counter += 1

    return io.NodeOutput(ui={"audio": results})


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    ConditioningStableAudio_V3,
    EmptyLatentAudio_V3,
    LoadAudio_V3,
    PreviewAudio_V3,
    SaveAudioMP3_V3,
    SaveAudioOpus_V3,
    SaveAudio_V3,
    VAEDecodeAudio_V3,
    VAEEncodeAudio_V3,
]
