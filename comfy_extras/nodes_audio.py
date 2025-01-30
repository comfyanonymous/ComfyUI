import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import io
import json
import struct
import random
import hashlib
import node_helpers
from comfy.cli_args import args

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


def create_vorbis_comment_block(comment_dict, last_block):
    vendor_string = b'ComfyUI'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    user_comment_list_length = len(comments)
    user_comments = b''.join(comments)

    comment_data = struct.pack('<I', vendor_length) + vendor_string + struct.pack('<I', user_comment_list_length) + user_comments
    if last_block:
        id = b'\x84'
    else:
        id = b'\x04'
    comment_block = id + struct.pack('>I', len(comment_data))[1:] + comment_data

    return comment_block

def insert_or_replace_vorbis_comment(flac_io, comment_dict):
    if len(comment_dict) == 0:
        return flac_io

    flac_io.seek(4)

    blocks = []
    last_block = False

    while not last_block:
        header = flac_io.read(4)
        last_block = (header[0] & 0x80) != 0
        block_type = header[0] & 0x7F
        block_length = struct.unpack('>I', b'\x00' + header[1:])[0]
        block_data = flac_io.read(block_length)

        if block_type == 4 or block_type == 1:
            pass
        else:
            header = bytes([(header[0] & (~0x80))]) + header[1:]
            blocks.append(header + block_data)

    blocks.append(create_vorbis_comment_block(comment_dict, last_block=True))

    new_flac_io = io.BytesIO()
    new_flac_io.write(b'fLaC')
    for block in blocks:
        new_flac_io.write(block)

    new_flac_io.write(flac_io.read())
    return new_flac_io


class SaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                              "filename_prefix": ("STRING", {"default": "audio/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "audio"

    def save_audio(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

            buff = insert_or_replace_vorbis_comment(buff, metadata)

            with open(os.path.join(full_output_folder, file), 'wb') as f:
                f.write(buff.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "audio": results } }

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
        waveform, sample_rate = torchaudio.load(audio_path)
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

NODE_CLASS_MAPPINGS = {
    "EmptyLatentAudio": EmptyLatentAudio,
    "VAEEncodeAudio": VAEEncodeAudio,
    "VAEDecodeAudio": VAEDecodeAudio,
    "SaveAudio": SaveAudio,
    "LoadAudio": LoadAudio,
    "PreviewAudio": PreviewAudio,
    "ConditioningStableAudio": ConditioningStableAudio,
}
