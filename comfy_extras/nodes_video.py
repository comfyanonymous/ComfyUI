from __future__ import annotations

import os
import av
import torch
import folder_paths
import json
from typing import Optional, Literal
from fractions import Fraction
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC
from comfy_api.latest import Input, InputImpl, Types
from comfy.cli_args import args

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

        codec_map = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}
        stream = container.add_stream(codec_map[codec], rate=Fraction(round(fps * 1000), 1000))
        stream.width = images.shape[-2]
        stream.height = images.shape[-3]
        stream.pix_fmt = "yuv420p10le" if codec == "av1" else "yuv420p"
        stream.bit_rate = 0
        stream.options = {'crf': str(crf)}
        if codec == "av1":
            stream.options["preset"] = "6"

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

class SaveVideo(ComfyNodeABC):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to save."}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "format": (Types.VideoContainer.as_input(), {"default": "auto", "tooltip": "The format to save the video as."}),
                "codec": (Types.VideoCodec.as_input(), {"default": "auto", "tooltip": "The codec to use for the video."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "image/video"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_video(self, video: Input.Video, filename_prefix, format, codec, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            width,
            height
        )
        results: list[FileLocator] = list()
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(format)}"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=format,
            codec=codec,
            metadata=saved_metadata
        )

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

        return { "ui": { "images": results, "animated": (True,) } }

class CreateVideo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"tooltip": "The images to create a video from."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "audio": (IO.AUDIO, {"tooltip": "The audio to add to the video."}),
            }
        }

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "create_video"

    CATEGORY = "image/video"
    DESCRIPTION = "Create a video from images."

    def create_video(self, images: Input.Image, fps: float, audio: Optional[Input.Audio] = None):
        return (InputImpl.VideoFromComponents(
            Types.VideoComponents(
            images=images,
            audio=audio,
            frame_rate=Fraction(fps),
            )
        ),)

class GetVideoComponents(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to extract components from."}),
            }
        }
    RETURN_TYPES = (IO.IMAGE, IO.AUDIO, IO.FLOAT)
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "get_components"

    CATEGORY = "image/video"
    DESCRIPTION = "Extracts all components from a video: frames, audio, and framerate."

    def get_components(self, video: Input.Video):
        components = video.get_components()

        return (components.images, components.audio, float(components.frame_rate))

class LoadVideo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {"required":
                    {"file": (sorted(files), {"video_upload": True})},
                }

    CATEGORY = "image/video"

    RETURN_TYPES = (IO.VIDEO,)
    FUNCTION = "load_video"
    def load_video(self, file):
        video_path = folder_paths.get_annotated_filepath(file)
        return (InputImpl.VideoFromFile(video_path),)

    @classmethod
    def IS_CHANGED(cls, file):
        video_path = folder_paths.get_annotated_filepath(file)
        mod_time = os.path.getmtime(video_path)
        # Instead of hashing the file, we can just use the modification time to avoid
        # rehashing large files.
        return mod_time

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid video file: {}".format(file)

        return True

NODE_CLASS_MAPPINGS = {
    "SaveWEBM": SaveWEBM,
    "SaveVideo": SaveVideo,
    "CreateVideo": CreateVideo,
    "GetVideoComponents": GetVideoComponents,
    "LoadVideo": LoadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideo": "Save Video",
    "CreateVideo": "Create Video",
    "GetVideoComponents": "Get Video Components",
    "LoadVideo": "Load Video",
}

