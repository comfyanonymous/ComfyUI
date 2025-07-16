import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import comfy.utils
import folder_paths
import node_helpers
import nodes
from comfy.cli_args import args
from comfy_api.v3 import io, ui
from server import PromptServer


class GetImageSize(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="GetImageSize_V3",
            display_name="Get Image Size _V3",
            description="Returns width and height of the image, and passes it through unchanged.",
            category="image",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="batch_size"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        height = image.shape[1]
        width = image.shape[2]
        batch_size = image.shape[0]

        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(
                f"width: {width}, height: {height}\n batch size: {batch_size}", cls.hidden.unique_id
            )

        return io.NodeOutput(width, height, batch_size)


class ImageAddNoise(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ImageAddNoise_V3",
            display_name="Image Add Noise _V3",
            category="image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                io.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, seed, strength) -> io.NodeOutput:
        generator = torch.manual_seed(seed)
        s = torch.clip(
            (image + strength * torch.randn(image.size(), generator=generator, device="cpu").to(image)),
            min=0.0,
            max=1.0,
        )
        return io.NodeOutput(s)


class ImageCrop(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ImageCrop_V3",
            display_name="Image Crop _V3",
            category="image/transform",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, x, y) -> io.NodeOutput:
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        return io.NodeOutput(image[:, y:to_y, x:to_x, :])


class ImageFlip(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ImageFlip_V3",
            display_name="Image Flip _V3",
            category="image/transform",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("flip_method", options=["x-axis: vertically", "y-axis: horizontally"]),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, flip_method) -> io.NodeOutput:
        if flip_method.startswith("x"):
            image = torch.flip(image, dims=[1])
        elif flip_method.startswith("y"):
            image = torch.flip(image, dims=[2])

        return io.NodeOutput(image)


class ImageFromBatch(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ImageFromBatch_V3",
            display_name="Image From Batch _V3",
            category="image/batch",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("batch_index", default=0, min=0, max=4095),
                io.Int.Input("length", default=1, min=1, max=4096),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, batch_index, length) -> io.NodeOutput:
        s_in = image
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index : batch_index + length].clone()
        return io.NodeOutput(s)


class ImageRotate(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ImageRotate_V3",
            display_name="Image Rotate _V3",
            category="image/transform",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("rotation", options=["none", "90 degrees", "180 degrees", "270 degrees"]),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, rotation) -> io.NodeOutput:
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        return io.NodeOutput(torch.rot90(image, k=rotate_by, dims=[2, 1]))


class ImageStitch(io.ComfyNodeV3):
    """Upstreamed from https://github.com/kijai/ComfyUI-KJNodes"""

    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ImageStitch_V3",
            display_name="Image Stitch _V3",
            description="Stitches image2 to image1 in the specified direction. "
            "If image2 is not provided, returns image1 unchanged. "
            "Optional spacing can be added between images.",
            category="image/transform",
            inputs=[
                io.Image.Input("image1"),
                io.Combo.Input("direction", options=["right", "down", "left", "up"], default="right"),
                io.Boolean.Input("match_image_size", default=True),
                io.Int.Input("spacing_width", default=0, min=0, max=1024, step=2),
                io.Combo.Input("spacing_color", options=["white", "black", "red", "green", "blue"], default="white"),
                io.Image.Input("image2", optional=True),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image1, direction, match_image_size, spacing_width, spacing_color, image2=None) -> io.NodeOutput:
        if image2 is None:
            return io.NodeOutput(image1)

        # Handle batch size differences
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat([image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)])
            if image2.shape[0] < max_batch:
                image2 = torch.cat([image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)])

        # Match image sizes if requested
        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2

            if direction in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:  # up, down
                target_w, target_h = w1, int(w1 / aspect_ratio)

            image2 = comfy.utils.common_upscale(
                image2.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
            ).movedim(1, -1)

        color_map = {
            "white": 1.0,
            "black": 0.0,
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }

        color_val = color_map[spacing_color]

        # When not matching sizes, pad to align non-concat dimensions
        if not match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            pad_value = 0.0
            if not isinstance(color_val, tuple):
                pad_value = color_val

            if direction in ["left", "right"]:
                # For horizontal concat, pad heights to match
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        pad_h = target_h - h1
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image1 = torch.nn.functional.pad(
                            image1, (0, 0, 0, 0, pad_top, pad_bottom), mode="constant", value=pad_value
                        )
                    if h2 < target_h:
                        pad_h = target_h - h2
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image2 = torch.nn.functional.pad(
                            image2, (0, 0, 0, 0, pad_top, pad_bottom), mode="constant", value=pad_value
                        )
            else:  # up, down
                # For vertical concat, pad widths to match
                if w1 != w2:
                    target_w = max(w1, w2)
                    if w1 < target_w:
                        pad_w = target_w - w1
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image1 = torch.nn.functional.pad(
                            image1, (0, 0, pad_left, pad_right), mode="constant", value=pad_value
                        )
                    if w2 < target_w:
                        pad_w = target_w - w2
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image2 = torch.nn.functional.pad(
                            image2, (0, 0, pad_left, pad_right), mode="constant", value=pad_value
                        )

        # Ensure same number of channels
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat(
                    [
                        image1,
                        torch.ones(
                            *image1.shape[:-1],
                            max_channels - image1.shape[-1],
                            device=image1.device,
                        ),
                    ],
                    dim=-1,
                )
            if image2.shape[-1] < max_channels:
                image2 = torch.cat(
                    [
                        image2,
                        torch.ones(
                            *image2.shape[:-1],
                            max_channels - image2.shape[-1],
                            device=image2.device,
                        ),
                    ],
                    dim=-1,
                )

        # Add spacing if specified
        if spacing_width > 0:
            spacing_width = spacing_width + (spacing_width % 2)  # Ensure even

            if direction in ["left", "right"]:
                spacing_shape = (
                    image1.shape[0],
                    max(image1.shape[1], image2.shape[1]),
                    spacing_width,
                    image1.shape[-1],
                )
            else:
                spacing_shape = (
                    image1.shape[0],
                    spacing_width,
                    max(image1.shape[2], image2.shape[2]),
                    image1.shape[-1],
                )

            spacing = torch.full(spacing_shape, 0.0, device=image1.device)
            if isinstance(color_val, tuple):
                for i, c in enumerate(color_val):
                    if i < spacing.shape[-1]:
                        spacing[..., i] = c
                if spacing.shape[-1] == 4:  # Add alpha
                    spacing[..., 3] = 1.0
            else:
                spacing[..., : min(3, spacing.shape[-1])] = color_val
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0

        # Concatenate images
        images = [image2, image1] if direction in ["left", "up"] else [image1, image2]
        if spacing_width > 0:
            images.insert(1, spacing)

        concat_dim = 2 if direction in ["left", "right"] else 1
        return io.NodeOutput(torch.cat(images, dim=concat_dim))


class LoadImage(io.ComfyNodeV3):
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

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
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
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class LoadImageOutput(io.ComfyNodeV3):
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

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
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
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class PreviewImage(io.ComfyNodeV3):
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


class RepeatImageBatch(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="RepeatImageBatch_V3",
            display_name="Repeat Image Batch _V3",
            category="image/batch",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("amount", default=1, min=1, max=4096),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, amount) -> io.NodeOutput:
        return io.NodeOutput(image.repeat((amount, 1, 1, 1)))


class ResizeAndPadImage(io.ComfyNodeV3):
    @classmethod
    def DEFINE_SCHEMA(cls):
        return io.SchemaV3(
            node_id="ResizeAndPadImage_V3",
            display_name="Resize and Pad Image _V3",
            category="image/transform",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("target_width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input("target_height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                io.Combo.Input("padding_color", options=["white", "black"]),
                io.Combo.Input("interpolation", options=["area", "bicubic", "nearest-exact", "bilinear", "lanczos"]),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, target_width, target_height, padding_color, interpolation) -> io.NodeOutput:
        batch_size, orig_height, orig_width, channels = image.shape

        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)

        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image_permuted = image.permute(0, 3, 1, 2)

        resized = comfy.utils.common_upscale(image_permuted, new_width, new_height, interpolation, "disabled")

        pad_value = 0.0 if padding_color == "black" else 1.0
        padded = torch.full(
            (batch_size, channels, target_height, target_width), pad_value, dtype=image.dtype, device=image.device
        )

        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        padded[:, :, y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

        return io.NodeOutput(padded.permute(0, 2, 3, 1))


class SaveAnimatedPNG(io.ComfyNodeV3):
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
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )
        results = []
        pil_images = []
        for image in images:
            img = Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if cls.hidden.prompt is not None:
                metadata.add(
                    b"comf",
                    "prompt".encode("latin-1", "strict")
                    + b"\0"
                    + json.dumps(cls.hidden.prompt).encode("latin-1", "strict"),
                    after_idat=True,
                )
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata.add(
                        b"comf",
                        x.encode("latin-1", "strict")
                        + b"\0"
                        + json.dumps(cls.hidden.extra_pnginfo[x]).encode("latin-1", "strict"),
                        after_idat=True,
                    )

        file = f"{filename}_{counter:05}_.png"
        pil_images[0].save(
            os.path.join(full_output_folder, file),
            pnginfo=metadata,
            compress_level=compress_level,
            save_all=True,
            duration=int(1000.0 / fps),
            append_images=pil_images[1:],
        )
        results.append(ui.SavedResult(file, subfolder, io.FolderType.output))

        return io.NodeOutput(ui={"images": results, "animated": (True,)})


class SaveAnimatedWEBP(io.ComfyNodeV3):
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
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )
        results = []
        pil_images = []
        for image in images:
            img = Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if cls.hidden.prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(cls.hidden.prompt))
            if cls.hidden.extra_pnginfo is not None:
                inital_exif = 0x010F
                for x in cls.hidden.extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(cls.hidden.extra_pnginfo[x]))
                    inital_exif -= 1

        if num_frames == 0:
            num_frames = len(pil_images)

        for i in range(0, len(pil_images), num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(
                os.path.join(full_output_folder, file),
                save_all=True,
                duration=int(1000.0 / fps),
                append_images=pil_images[i + 1 : i + num_frames],
                exif=metadata,
                lossless=lossless,
                quality=quality,
                method=method,
            )
            results.append(ui.SavedResult(file, subfolder, io.FolderType.output))
            counter += 1

        return io.NodeOutput(ui={"images": results, "animated": (num_frames != 1,)})


class SaveImage(io.ComfyNodeV3):
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
                    tooltip="The prefix for the file to save. This may include formatting information "
                    "such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
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
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
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


NODES_LIST: list[type[io.ComfyNodeV3]] = [
    GetImageSize,
    ImageAddNoise,
    ImageCrop,
    ImageFlip,
    ImageFromBatch,
    ImageRotate,
    ImageStitch,
    LoadImage,
    LoadImageOutput,
    PreviewImage,
    RepeatImageBatch,
    ResizeAndPadImage,
    SaveAnimatedPNG,
    SaveAnimatedWEBP,
    SaveImage,
]
