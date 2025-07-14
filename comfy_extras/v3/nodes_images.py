import json
import os
import torch
import hashlib

import numpy as np
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

from comfy_api.v3 import io, ui
from comfy.cli_args import args
import folder_paths
import node_helpers


class SaveImage_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="SaveImage_V3",
            display_name="Save Image _V3",
            description="Saves the input images to your ComfyUI output directory.",
            category="image",
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images to save.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, filename_prefix="ComfyUI") -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )
        results = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if cls.hidden.prompt is not None:
                    metadata.add_text("prompt", json.dumps(cls.hidden.prompt))
                if cls.hidden.extra_pnginfo is not None:
                    for x in cls.hidden.extra_pnginfo:
                        metadata.add_text(x, json.dumps(cls.hidden.extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append(ui.SavedResult(file, subfolder, io.FolderType.output))
            counter += 1

        return io.NodeOutput(ui={"images": results})


class SaveAnimatedPNG_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="SaveAnimatedPNG_V3",
            display_name="Save Animated PNG _V3",
            category="image/animation",
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Float.Input("fps", default=6.0, min=0.01, max=1000.0, step=0.01),
                io.Int.Input("compress_level", default=4, min=0, max=9),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, fps, compress_level, filename_prefix="ComfyUI") -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0])
        )
        results = []
        pil_images = []
        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if cls.hidden.prompt is not None:
                metadata.add(
                    b"comf", "prompt".encode("latin-1", "strict") + b"\0" + json.dumps(cls.hidden.prompt).encode("latin-1", "strict"), after_idat=True
                )
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata.add(
                        b"comf", x.encode("latin-1", "strict") + b"\0" + json.dumps(cls.hidden.extra_pnginfo[x]).encode("latin-1", "strict"), after_idat=True
                    )

        file = f"{filename}_{counter:05}_.png"
        pil_images[0].save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level, save_all=True, duration=int(1000.0/fps), append_images=pil_images[1:])
        results.append(ui.SavedResult(file, subfolder, io.FolderType.output))

        return io.NodeOutput(ui={"images": results, "animated": (True,) })


class SaveAnimatedWEBP_V3(io.ComfyNodeV3):
    COMPRESS_METHODS = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="SaveAnimatedWEBP_V3",
            display_name="Save Animated WEBP _V3",
            category="image/animation",
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
                io.Float.Input("fps", default=6.0, min=0.01, max=1000.0, step=0.01),
                io.Boolean.Input("lossless", default=True),
                io.Int.Input("quality", default=80, min=0, max=100),
                io.Combo.Input("method", options=list(cls.COMPRESS_METHODS.keys())),
                # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, fps, filename_prefix, lossless, quality, method, num_frames=0) -> io.NodeOutput:
        method = cls.COMPRESS_METHODS.get(method)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0])
        results = []
        pil_images = []
        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if cls.hidden.prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(cls.hidden.prompt))
            if cls.hidden.extra_pnginfo is not None:
                inital_exif = 0x010f
                for x in cls.hidden.extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(cls.hidden.extra_pnginfo[x]))
                    inital_exif -= 1

        if num_frames == 0:
            num_frames = len(pil_images)

        for i in range(0, len(pil_images), num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(
                os.path.join(full_output_folder, file),
                save_all=True, duration=int(1000.0/fps),
                append_images=pil_images[i + 1:i + num_frames],
                exif=metadata,
                lossless=lossless,
                quality=quality,
                method=method,
            )
            results.append(ui.SavedResult(file, subfolder, io.FolderType.output))
            counter += 1

        return io.NodeOutput(ui={"images": results, "animated": (num_frames != 1,)})


class PreviewImage_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="PreviewImage_V3",
            display_name="Preview Image _V3",
            description="Preview the input images.",
            category="image",
            inputs=[
                io.Image.Input("images", tooltip="The images to preview."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        return io.NodeOutput(ui=ui.PreviewImage(images, cls=cls))


class LoadImage_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="LoadImage_V3",
            display_name="Load Image _V3",
            category="image",
            inputs=[
                io.Combo.Input(
                    "image",
                    upload=io.UploadType.image,
                    image_folder=io.FolderType.input,
                    options=cls.get_files_options(),
                ),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
        )

    @classmethod
    def get_files_options(cls) -> list[str]:
        target_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
        return sorted(folder_paths.filter_files_content_types(files, ["image"]))

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        img = node_helpers.pillow(Image.open, folder_paths.get_annotated_filepath(image))

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return io.NodeOutput(output_image, output_mask)

    @classmethod
    def fingerprint_inputs(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class LoadImageOutput_V3(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="LoadImageOutput_V3",
            display_name="Load Image (from Outputs) _V3",
            description="Load an image from the output folder. "
                        "When the refresh button is clicked, the node will update the image list "
                        "and automatically select the first image, allowing for easy iteration.",
            category="image",
            inputs=[
                io.Combo.Input(
                    "image",
                    upload=io.UploadType.image,
                    image_folder=io.FolderType.output,
                    remote=io.RemoteOptions(
                        route="/internal/files/output",
                        refresh_button=True,
                        control_after_refresh="first",
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        img = node_helpers.pillow(Image.open, folder_paths.get_annotated_filepath(image))

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return io.NodeOutput(output_image, output_mask)

    @classmethod
    def fingerprint_inputs(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    SaveAnimatedPNG_V3,
    SaveAnimatedWEBP_V3,
    SaveImage_V3,
    PreviewImage_V3,
    LoadImage_V3,
    LoadImageOutput_V3,
]
