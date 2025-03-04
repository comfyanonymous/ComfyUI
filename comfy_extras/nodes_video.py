from __future__ import annotations

import os
import av
import torch
import folder_paths
import json
from fractions import Fraction
from comfy.comfy_types import FileLocator


class SaveWEBM:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "codec": (["vp9", "av1"],),
                     "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "crf": ("FLOAT", {"default": 32.0, "min": 0, "max": 63.0, "step": 1, "tooltip": "Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize."}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/video"

    EXPERIMENTAL = True

    def save_images(self, images, codec, fps, filename_prefix, crf, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        file = f"{filename}_{counter:05}_.webm"
        container = av.open(os.path.join(full_output_folder, file), mode="w")

        if prompt is not None:
            container.metadata["prompt"] = json.dumps(prompt)

        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                container.metadata[x] = json.dumps(extra_pnginfo[x])

        codec_map = {"vp9": "libvpx-vp9", "av1": "libaom-av1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}

        for frame in images:
            frame = av.VideoFrame.from_ndarray(torch.clamp(frame[..., :3] * 255, min=0, max=255).to(device=torch.device("cpu"), dtype=torch.uint8).numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        container.mux(stream.encode())
        container.close()

        results: list[FileLocator] = [{
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }]

        return {"ui": {"images": results, "animated": (True,)}}  # TODO: frontend side


NODE_CLASS_MAPPINGS = {
    "SaveWEBM": SaveWEBM,
}
