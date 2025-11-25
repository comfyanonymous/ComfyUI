from __future__ import annotations

import av
import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import io
import json
import random
import hashlib
import node_helpers
import logging
from comfy.cli_args import args
from comfy.comfy_types import FileLocator

class EmptyLatentAudio:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"seconds": ("FLOAT", {"default": 47.6, "min": 1.0, "max": 1000.0, "step": 0.1}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/audio"

    def generate(self, seconds, batch_size):
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=self.device)
        return ({"samples":latent, "type": "audio"}, )

class ConditioningStableAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "seconds_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                             "seconds_total": ("FLOAT", {"default": 47.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")

    FUNCTION = "append"

    CATEGORY = "conditioning"

    def append(self, positive, negative, seconds_start, seconds_total):
        positive = node_helpers.conditioning_set_values(positive, {"seconds_start": seconds_start, "seconds_total": seconds_total})
        negative = node_helpers.conditioning_set_values(negative, {"seconds_start": seconds_start, "seconds_total": seconds_total})
        return (positive, negative)

class VAEEncodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/audio"

    def encode(self, vae, audio):
        sample_rate = audio["sample_rate"]
        if 44100 != sample_rate:
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
        else:
            waveform = audio["waveform"]

        t = vae.encode(waveform.movedim(1, -1))
        return ({"samples":t}, )

class VAEDecodeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"

    CATEGORY = "latent/audio"

    def decode(self, vae, samples):
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        std = torch.std(audio, dim=[1,2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        return ({"waveform": audio, "sample_rate": 44100}, )


def save_audio(self, audio, filename_prefix="ComfyUI", format="flac", prompt=None, extra_pnginfo=None, quality="128k"):

    filename_prefix += self.prefix_append
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
    results: list[FileLocator] = []

    # Prepare metadata dictionary
    metadata = {}
    if not args.disable_metadata:
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    # Opus supported sample rates
    OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

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
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format=format)

        # Set metadata on the container
        for key, value in metadata.items():
            output_container.metadata[key] = value

        layout = 'mono' if waveform.shape[0] == 1 else 'stereo'
        # Set up the output stream with appropriate properties
        if format == "opus":
            out_stream = output_container.add_stream("libopus", rate=sample_rate, layout=layout)
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
            out_stream = output_container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
            if quality == "V0":
                #TODO i would really love to support V3 and V5 but there doesn't seem to be a way to set the qscale level, the property below is a bool
                out_stream.codec_context.qscale = 1
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "320k":
                out_stream.bit_rate = 320000
        else: #format == "flac":
            out_stream = output_container.add_stream("flac", rate=sample_rate, layout=layout)

        frame = av.AudioFrame.from_ndarray(waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format='flt', layout=layout)
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

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

    return { "ui": { "audio": results } }

class SaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_flac"

    OUTPUT_NODE = True

    CATEGORY = "audio"

    def save_flac(self, audio, filename_prefix="ComfyUI", format="flac", prompt=None, extra_pnginfo=None):
        return save_audio(self, audio, filename_prefix, format, prompt, extra_pnginfo)

class SaveAudioMP3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
                            "quality": (["V0", "128k", "320k"], {"default": "V0"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_mp3"

    OUTPUT_NODE = True

    CATEGORY = "audio"

    def save_mp3(self, audio, filename_prefix="ComfyUI", format="mp3", prompt=None, extra_pnginfo=None, quality="128k"):
        return save_audio(self, audio, filename_prefix, format, prompt, extra_pnginfo, quality)

class SaveAudioOpus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
                            "quality": (["64k", "96k", "128k", "192k", "320k"], {"default": "128k"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_opus"

    OUTPUT_NODE = True

    CATEGORY = "audio"

    def save_opus(self, audio, filename_prefix="ComfyUI", format="opus", prompt=None, extra_pnginfo=None, quality="V3"):
        return save_audio(self, audio, filename_prefix, format, prompt, extra_pnginfo, quality)

class PreviewAudio(SaveAudio):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

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

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return {"required": {"audio": (sorted(files), {"audio_upload": True})}}

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

    @classmethod
    def IS_CHANGED(s, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True

class RecordAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO_RECORD", {})}}

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)

        waveform, sample_rate = load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )


class TrimAudioDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_index": ("FLOAT", {"default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01, "tooltip": "Start time in seconds, can be negative to count from the end (supports sub-seconds)."}),
                "duration": ("FLOAT", {"default": 60.0, "min": 0.0, "step": 0.01, "tooltip": "Duration in seconds"}),
            },
        }

    FUNCTION = "trim"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Trim audio tensor into chosen time range."

    def trim(self, audio, start_index, duration):
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

        return ({"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate},)


class SplitAudioChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("left", "right")
    FUNCTION = "separate"
    CATEGORY = "audio"
    DESCRIPTION = "Separates the audio into left and right channels."

    def separate(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.shape[1] != 2:
            raise ValueError("AudioSplit: Input audio has only one channel.")

        left_channel = waveform[..., 0:1, :]
        right_channel = waveform[..., 1:2, :]

        return ({"waveform": left_channel, "sample_rate": sample_rate}, {"waveform": right_channel, "sample_rate": sample_rate})


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


class AudioConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio1": ("AUDIO",),
            "audio2": ("AUDIO",),
            "direction": (['after', 'before'], {"default": 'after', "tooltip": "Whether to append audio2 after or before audio1."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concat"
    CATEGORY = "audio"
    DESCRIPTION = "Concatenates the audio1 to audio2 in the specified direction."

    def concat(self, audio1, audio2, direction):
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

        return ({"waveform": concatenated_audio, "sample_rate": output_sample_rate},)


class AudioMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "merge_method": (["add", "mean", "subtract", "multiply"], {"tooltip": "The method used to combine the audio waveforms."}),
            },
        }

    FUNCTION = "merge"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Combine two audio tracks by overlaying their waveforms."

    def merge(self, audio1, audio2, merge_method):
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

        return ({"waveform": waveform, "sample_rate": output_sample_rate},)


class AudioAdjustVolume:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
            "volume": ("INT", {"default": 1.0, "min": -100, "max": 100, "tooltip": "Volume adjustment in decibels (dB). 0 = no change, +6 = double, -6 = half, etc"}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "adjust_volume"
    CATEGORY = "audio"

    def adjust_volume(self, audio, volume):
        if volume == 0:
            return (audio,)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        gain = 10 ** (volume / 20)
        waveform = waveform * gain

        return ({"waveform": waveform, "sample_rate": sample_rate},)


class EmptyAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "duration": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 0xffffffffffffffff, "step": 0.01, "tooltip": "Duration of the empty audio clip in seconds"}),
            "sample_rate": ("INT", {"default": 44100, "tooltip": "Sample rate of the empty audio clip."}),
            "channels": ("INT", {"default": 2, "min": 1, "max": 2, "tooltip": "Number of audio channels (1 for mono, 2 for stereo)."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "create_empty_audio"
    CATEGORY = "audio"

    def create_empty_audio(self, duration, sample_rate, channels):
        num_samples = int(round(duration * sample_rate))
        waveform = torch.zeros((1, channels, num_samples), dtype=torch.float32)
        return ({"waveform": waveform, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "EmptyLatentAudio": EmptyLatentAudio,
    "VAEEncodeAudio": VAEEncodeAudio,
    "VAEDecodeAudio": VAEDecodeAudio,
    "SaveAudio": SaveAudio,
    "SaveAudioMP3": SaveAudioMP3,
    "SaveAudioOpus": SaveAudioOpus,
    "LoadAudio": LoadAudio,
    "PreviewAudio": PreviewAudio,
    "ConditioningStableAudio": ConditioningStableAudio,
    "RecordAudio": RecordAudio,
    "TrimAudioDuration": TrimAudioDuration,
    "SplitAudioChannels": SplitAudioChannels,
    "AudioConcat": AudioConcat,
    "AudioMerge": AudioMerge,
    "AudioAdjustVolume": AudioAdjustVolume,
    "EmptyAudio": EmptyAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentAudio": "Empty Latent Audio",
    "VAEEncodeAudio": "VAE Encode Audio",
    "VAEDecodeAudio": "VAE Decode Audio",
    "PreviewAudio": "Preview Audio",
    "LoadAudio": "Load Audio",
    "SaveAudio": "Save Audio (FLAC)",
    "SaveAudioMP3": "Save Audio (MP3)",
    "SaveAudioOpus": "Save Audio (Opus)",
    "RecordAudio": "Record Audio",
    "TrimAudioDuration": "Trim Audio Duration",
    "SplitAudioChannels": "Split Audio Channels",
    "AudioConcat": "Audio Concat",
    "AudioMerge": "Audio Merge",
    "AudioAdjustVolume": "Audio Adjust Volume",
    "EmptyAudio": "Empty Audio",
}
