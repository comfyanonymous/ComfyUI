from __future__ import annotations

import av
import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import hashlib
import node_helpers
import logging
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, UI

class EmptyLatentAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyLatentAudio",
            display_name="Empty Latent Audio",
            category="latent/audio",
            inputs=[
                IO.Float.Input("seconds", default=47.6, min=1.0, max=1000.0, step=0.1),
                IO.Int.Input(
                    "batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."
                ),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, seconds, batch_size) -> IO.NodeOutput:
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=comfy.model_management.intermediate_device())
        return IO.NodeOutput({"samples":latent, "type": "audio"})

    generate = execute  # TODO: remove


class ConditioningStableAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ConditioningStableAudio",
            category="conditioning",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Float.Input("seconds_start", default=0.0, min=0.0, max=1000.0, step=0.1),
                IO.Float.Input("seconds_total", default=47.0, min=0.0, max=1000.0, step=0.1),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, seconds_start, seconds_total) -> IO.NodeOutput:
        positive = node_helpers.conditioning_set_values(positive, {"seconds_start": seconds_start, "seconds_total": seconds_total})
        negative = node_helpers.conditioning_set_values(negative, {"seconds_start": seconds_start, "seconds_total": seconds_total})
        return IO.NodeOutput(positive, negative)

    append = execute  # TODO: remove


class VAEEncodeAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEEncodeAudio",
            display_name="VAE Encode Audio",
            category="latent/audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.Vae.Input("vae"),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, audio) -> IO.NodeOutput:
        sample_rate = audio["sample_rate"]
        if 44100 != sample_rate:
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
        else:
            waveform = audio["waveform"]

        t = vae.encode(waveform.movedim(1, -1))
        return IO.NodeOutput({"samples":t})

    encode = execute  # TODO: remove


class VAEDecodeAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeAudio",
            display_name="VAE Decode Audio",
            category="latent/audio",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, vae, samples) -> IO.NodeOutput:
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        std = torch.std(audio, dim=[1,2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        return IO.NodeOutput({"waveform": audio, "sample_rate": 44100})

    decode = execute  # TODO: remove


class SaveAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAudio",
            display_name="Save Audio (FLAC)",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.String.Input("filename_prefix", default="audio/ComfyUI"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="flac") -> IO.NodeOutput:
        return IO.NodeOutput(
            ui=UI.AudioSaveHelper.get_save_audio_ui(audio, filename_prefix=filename_prefix, cls=cls, format=format)
        )

    save_flac = execute  # TODO: remove


class SaveAudioMP3(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAudioMP3",
            display_name="Save Audio (MP3)",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.String.Input("filename_prefix", default="audio/ComfyUI"),
                IO.Combo.Input("quality", options=["V0", "128k", "320k"], default="V0"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="mp3", quality="128k") -> IO.NodeOutput:
        return IO.NodeOutput(
            ui=UI.AudioSaveHelper.get_save_audio_ui(
                audio, filename_prefix=filename_prefix, cls=cls, format=format, quality=quality
            )
        )

    save_mp3 = execute  # TODO: remove


class SaveAudioOpus(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAudioOpus",
            display_name="Save Audio (Opus)",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.String.Input("filename_prefix", default="audio/ComfyUI"),
                IO.Combo.Input("quality", options=["64k", "96k", "128k", "192k", "320k"], default="128k"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="ComfyUI", format="opus", quality="V3") -> IO.NodeOutput:
        return IO.NodeOutput(
            ui=UI.AudioSaveHelper.get_save_audio_ui(
                audio, filename_prefix=filename_prefix, cls=cls, format=format, quality=quality
            )
        )

    save_opus = execute  # TODO: remove


class PreviewAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PreviewAudio",
            display_name="Preview Audio",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        return IO.NodeOutput(ui=UI.PreviewAudio(audio, cls=cls))

    save_flac = execute  # TODO: remove


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def load(filepath: str) -> tuple[torch.Tensor, int]:
    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()

            frames.append(buf)
            length += buf.shape[1]

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        return wav, sr

class LoadAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return IO.Schema(
            node_id="LoadAudio",
            display_name="Load Audio",
            category="audio",
            inputs=[
                IO.Combo.Input("audio", upload=IO.UploadType.audio, options=sorted(files)),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return IO.NodeOutput(audio)

    @classmethod
    def fingerprint_inputs(cls, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True

    load = execute  # TODO: remove


class RecordAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecordAudio",
            display_name="Record Audio",
            category="audio",
            inputs=[
                IO.Custom("AUDIO_RECORD").Input("audio"),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        audio_path = folder_paths.get_annotated_filepath(audio)

        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return IO.NodeOutput(audio)

    load = execute  # TODO: remove


class TrimAudioDuration(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TrimAudioDuration",
            display_name="Trim Audio Duration",
            description="Trim audio tensor into chosen time range.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.Float.Input(
                    "start_index",
                    default=0.0,
                    min=-0xffffffffffffffff,
                    max=0xffffffffffffffff,
                    step=0.01,
                    tooltip="Start time in seconds, can be negative to count from the end (supports sub-seconds).",
                ),
                IO.Float.Input(
                    "duration",
                    default=60.0,
                    min=0.0,
                    step=0.01,
                    tooltip="Duration in seconds",
                ),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio, start_index, duration) -> IO.NodeOutput:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        audio_length = waveform.shape[-1]

        if start_index < 0:
            start_frame = audio_length + int(round(start_index * sample_rate))
        else:
            start_frame = int(round(start_index * sample_rate))
        start_frame = max(0, min(start_frame, audio_length - 1))

        end_frame = start_frame + int(round(duration * sample_rate))
        end_frame = max(0, min(end_frame, audio_length))

        if start_frame >= end_frame:
            raise ValueError("AudioTrim: Start time must be less than end time and be within the audio length.")

        return IO.NodeOutput({"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate})

    trim = execute  # TODO: remove


class SplitAudioChannels(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplitAudioChannels",
            display_name="Split Audio Channels",
            description="Separates the audio into left and right channels.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
            ],
            outputs=[
                IO.Audio.Output(display_name="left"),
                IO.Audio.Output(display_name="right"),
            ],
        )

    @classmethod
    def execute(cls, audio) -> IO.NodeOutput:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.shape[1] != 2:
            raise ValueError("AudioSplit: Input audio has only one channel.")

        left_channel = waveform[..., 0:1, :]
        right_channel = waveform[..., 1:2, :]

        return IO.NodeOutput({"waveform": left_channel, "sample_rate": sample_rate}, {"waveform": right_channel, "sample_rate": sample_rate})

    separate = execute  # TODO: remove


def match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2):
    if sample_rate_1 != sample_rate_2:
        if sample_rate_1 > sample_rate_2:
            waveform_2 = torchaudio.functional.resample(waveform_2, sample_rate_2, sample_rate_1)
            output_sample_rate = sample_rate_1
            logging.info(f"Resampling audio2 from {sample_rate_2}Hz to {sample_rate_1}Hz for merging.")
        else:
            waveform_1 = torchaudio.functional.resample(waveform_1, sample_rate_1, sample_rate_2)
            output_sample_rate = sample_rate_2
            logging.info(f"Resampling audio1 from {sample_rate_1}Hz to {sample_rate_2}Hz for merging.")
    else:
        output_sample_rate = sample_rate_1
    return waveform_1, waveform_2, output_sample_rate


class AudioConcat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioConcat",
            display_name="Audio Concat",
            description="Concatenates the audio1 to audio2 in the specified direction.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio1"),
                IO.Audio.Input("audio2"),
                IO.Combo.Input(
                    "direction",
                    options=['after', 'before'],
                    default="after",
                    tooltip="Whether to append audio2 after or before audio1.",
                )
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio1, audio2, direction) -> IO.NodeOutput:
        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]

        if waveform_1.shape[1] == 1:
            waveform_1 = waveform_1.repeat(1, 2, 1)
            logging.info("AudioConcat: Converted mono audio1 to stereo by duplicating the channel.")
        if waveform_2.shape[1] == 1:
            waveform_2 = waveform_2.repeat(1, 2, 1)
            logging.info("AudioConcat: Converted mono audio2 to stereo by duplicating the channel.")

        waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)

        if direction == 'after':
            concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
        elif direction == 'before':
            concatenated_audio = torch.cat((waveform_2, waveform_1), dim=2)

        return IO.NodeOutput({"waveform": concatenated_audio, "sample_rate": output_sample_rate})

    concat = execute  # TODO: remove


class AudioMerge(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioMerge",
            display_name="Audio Merge",
            description="Combine two audio tracks by overlaying their waveforms.",
            category="audio",
            inputs=[
                IO.Audio.Input("audio1"),
                IO.Audio.Input("audio2"),
                IO.Combo.Input(
                    "merge_method",
                    options=["add", "mean", "subtract", "multiply"],
                    tooltip="The method used to combine the audio waveforms.",
                )
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio1, audio2, merge_method) -> IO.NodeOutput:
        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]

        waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)

        length_1 = waveform_1.shape[-1]
        length_2 = waveform_2.shape[-1]

        if length_2 > length_1:
            logging.info(f"AudioMerge: Trimming audio2 from {length_2} to {length_1} samples to match audio1 length.")
            waveform_2 = waveform_2[..., :length_1]
        elif length_2 < length_1:
            logging.info(f"AudioMerge: Padding audio2 from {length_2} to {length_1} samples to match audio1 length.")
            pad_shape = list(waveform_2.shape)
            pad_shape[-1] = length_1 - length_2
            pad_tensor = torch.zeros(pad_shape, dtype=waveform_2.dtype, device=waveform_2.device)
            waveform_2 = torch.cat((waveform_2, pad_tensor), dim=-1)

        if merge_method == "add":
            waveform = waveform_1 + waveform_2
        elif merge_method == "subtract":
            waveform = waveform_1 - waveform_2
        elif merge_method == "multiply":
            waveform = waveform_1 * waveform_2
        elif merge_method == "mean":
            waveform = (waveform_1 + waveform_2) / 2

        max_val = waveform.abs().max()
        if max_val > 1.0:
            waveform = waveform / max_val

        return IO.NodeOutput({"waveform": waveform, "sample_rate": output_sample_rate})

    merge = execute  # TODO: remove


class AudioAdjustVolume(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AudioAdjustVolume",
            display_name="Audio Adjust Volume",
            category="audio",
            inputs=[
                IO.Audio.Input("audio"),
                IO.Int.Input(
                    "volume",
                    default=1,
                    min=-100,
                    max=100,
                    tooltip="Volume adjustment in decibels (dB). 0 = no change, +6 = double, -6 = half, etc",
                )
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, audio, volume) -> IO.NodeOutput:
        if volume == 0:
            return IO.NodeOutput(audio)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        gain = 10 ** (volume / 20)
        waveform = waveform * gain

        return IO.NodeOutput({"waveform": waveform, "sample_rate": sample_rate})

    adjust_volume = execute  # TODO: remove


class EmptyAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyAudio",
            display_name="Empty Audio",
            category="audio",
            inputs=[
                IO.Float.Input(
                    "duration",
                    default=60.0,
                    min=0.0,
                    max=0xffffffffffffffff,
                    step=0.01,
                    tooltip="Duration of the empty audio clip in seconds",
                ),
                IO.Int.Input(
                    "sample_rate",
                    default=44100,
                    tooltip="Sample rate of the empty audio clip.",
                    min=1,
                    max=192000,
                ),
                IO.Int.Input(
                    "channels",
                    default=2,
                    min=1,
                    max=2,
                    tooltip="Number of audio channels (1 for mono, 2 for stereo).",
                ),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, duration, sample_rate, channels) -> IO.NodeOutput:
        num_samples = int(round(duration * sample_rate))
        waveform = torch.zeros((1, channels, num_samples), dtype=torch.float32)
        return IO.NodeOutput({"waveform": waveform, "sample_rate": sample_rate})

    create_empty_audio = execute  # TODO: remove


class AudioExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            EmptyLatentAudio,
            VAEEncodeAudio,
            VAEDecodeAudio,
            SaveAudio,
            SaveAudioMP3,
            SaveAudioOpus,
            LoadAudio,
            PreviewAudio,
            ConditioningStableAudio,
            RecordAudio,
            TrimAudioDuration,
            SplitAudioChannels,
            AudioConcat,
            AudioMerge,
            AudioAdjustVolume,
            EmptyAudio,
        ]

async def comfy_entrypoint() -> AudioExtension:
    return AudioExtension()
