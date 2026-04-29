from __future__ import annotations

import nodes
import folder_paths

import av
import io
import json

import av
import os
import re
import math
import numpy as np
import struct
import torch

import struct
import zlib
import tempfile
import logging
import comfy.utils
import numpy as np
from fractions import Fraction

from server import PromptServer
from comfy_api.latest import ComfyExtension, Input, IO, UI
from comfy.cli_args import args
from typing_extensions import override
from comfy.cli_args import args

SVG = IO.SVG.Type  # TODO: temporary solution for backward compatibility, will be removed later.

MAX_RESOLUTION = nodes.MAX_RESOLUTION

class ImageCrop(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCrop",
            search_aliases=["trim"],
            display_name="Image Crop (Deprecated)",
            category="image/transform",
            is_deprecated=True,
            essentials_category="Image Tools",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, x, y) -> IO.NodeOutput:
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        img = image[:,y:to_y, x:to_x, :]
        return IO.NodeOutput(img)

    crop = execute  # TODO: remove


class ImageCropV2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCropV2",
            search_aliases=["trim"],
            display_name="Image Crop",
            category="image/transform",
            essentials_category="Image Tools",
            has_intermediate_output=True,
            inputs=[
                IO.Image.Input("image"),
                IO.BoundingBox.Input("crop_region", component="ImageCrop"),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, crop_region) -> IO.NodeOutput:
        x = crop_region.get("x", 0)
        y = crop_region.get("y", 0)
        width = crop_region.get("width", 512)
        height = crop_region.get("height", 512)

        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        img = image[:,y:to_y, x:to_x, :]
        return IO.NodeOutput(img, ui=UI.PreviewImage(img))


class BoundingBox(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrimitiveBoundingBox",
            display_name="Bounding Box",
            category="utils/primitive",
            inputs=[
                IO.Int.Input("x", default=0, min=0, max=MAX_RESOLUTION),
                IO.Int.Input("y", default=0, min=0, max=MAX_RESOLUTION),
                IO.Int.Input("width", default=512, min=1, max=MAX_RESOLUTION),
                IO.Int.Input("height", default=512, min=1, max=MAX_RESOLUTION),
            ],
            outputs=[IO.BoundingBox.Output()],
        )

    @classmethod
    def execute(cls, x, y, width, height) -> IO.NodeOutput:
        return IO.NodeOutput({"x": x, "y": y, "width": width, "height": height})


class RepeatImageBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RepeatImageBatch",
            search_aliases=["duplicate image", "clone image"],
            category="image/batch",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("amount", default=1, min=1, max=4096),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, amount) -> IO.NodeOutput:
        s = image.repeat((amount, 1,1,1))
        return IO.NodeOutput(s)

    repeat = execute  # TODO: remove


class ImageFromBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageFromBatch",
            search_aliases=["select image", "pick from batch", "extract image"],
            category="image/batch",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("batch_index", default=0, min=0, max=4095),
                IO.Int.Input("length", default=1, min=1, max=4096),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, batch_index, length) -> IO.NodeOutput:
        s_in = image
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index:batch_index + length].clone()
        return IO.NodeOutput(s)

    frombatch = execute  # TODO: remove


class ImageAddNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageAddNoise",
            search_aliases=["film grain"],
            category="image",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, seed, strength) -> IO.NodeOutput:
        generator = torch.manual_seed(seed)
        s = torch.clip((image + strength * torch.randn(image.size(), generator=generator, device="cpu").to(image)), min=0.0, max=1.0)
        return IO.NodeOutput(s)

    repeat = execute  # TODO: remove


class SaveAnimatedWEBP(IO.ComfyNode):
    COMPRESS_METHODS = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAnimatedWEBP",
            category="image/animation",
            inputs=[
                IO.Image.Input("images"),
                IO.String.Input("filename_prefix", default="ComfyUI"),
                IO.Float.Input("fps", default=6.0, min=0.01, max=1000.0, step=0.01),
                IO.Boolean.Input("lossless", default=True),
                IO.Int.Input("quality", default=80, min=0, max=100),
                IO.Combo.Input("method", options=list(cls.COMPRESS_METHODS.keys())),
                # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, fps, filename_prefix, lossless, quality, method, num_frames=0) -> IO.NodeOutput:
        return IO.NodeOutput(
            ui=UI.ImageSaveHelper.get_save_animated_webp_ui(
                images=images,
                filename_prefix=filename_prefix,
                cls=cls,
                fps=fps,
                lossless=lossless,
                quality=quality,
                method=cls.COMPRESS_METHODS.get(method)
            )
        )

    save_images = execute  # TODO: remove


class SaveAnimatedPNG(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAnimatedPNG",
            category="image/animation",
            inputs=[
                IO.Image.Input("images"),
                IO.String.Input("filename_prefix", default="ComfyUI"),
                IO.Float.Input("fps", default=6.0, min=0.01, max=1000.0, step=0.01),
                IO.Int.Input("compress_level", default=4, min=0, max=9, advanced=True),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, fps, compress_level, filename_prefix="ComfyUI") -> IO.NodeOutput:
        return IO.NodeOutput(
            ui=UI.ImageSaveHelper.get_save_animated_png_ui(
                images=images,
                filename_prefix=filename_prefix,
                cls=cls,
                fps=fps,
                compress_level=compress_level,
            )
        )

    save_images = execute  # TODO: remove


class ImageStitch(IO.ComfyNode):
    """Upstreamed from https://github.com/kijai/ComfyUI-KJNodes"""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageStitch",
            search_aliases=["combine images", "join images", "concatenate images", "side by side"],
            display_name="Image Stitch",
            description="Stitches image2 to image1 in the specified direction.\n"
            "If image2 is not provided, returns image1 unchanged.\n"
            "Optional spacing can be added between images.",
            category="image/transform",
            inputs=[
                IO.Image.Input("image1"),
                IO.Combo.Input("direction", options=["right", "down", "left", "up"], default="right"),
                IO.Boolean.Input("match_image_size", default=True),
                IO.Int.Input("spacing_width", default=0, min=0, max=1024, step=2, advanced=True),
                IO.Combo.Input("spacing_color", options=["white", "black", "red", "green", "blue"], default="white", advanced=True),
                IO.Image.Input("image2", optional=True),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        image1,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        image2=None,
    ) -> IO.NodeOutput:
        if image2 is None:
            return IO.NodeOutput(image1)

        # Handle batch size differences
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat(
                    [image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)]
                )
            if image2.shape[0] < max_batch:
                image2 = torch.cat(
                    [image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)]
                )

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
                        image1 = torch.nn.functional.pad(image1, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
                    if h2 < target_h:
                        pad_h = target_h - h2
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
            else:  # up, down
                # For vertical concat, pad widths to match
                if w1 != w2:
                    target_w = max(w1, w2)
                    if w1 < target_w:
                        pad_w = target_w - w1
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)
                    if w2 < target_w:
                        pad_w = target_w - w2
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)

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
        return IO.NodeOutput(torch.cat(images, dim=concat_dim))

    stitch = execute  # TODO: remove


class ResizeAndPadImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ResizeAndPadImage",
            search_aliases=["fit to size"],
            category="image/transform",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("target_width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("target_height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Combo.Input("padding_color", options=["white", "black"], advanced=True),
                IO.Combo.Input("interpolation", options=["area", "bicubic", "nearest-exact", "bilinear", "lanczos"], advanced=True),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, target_width, target_height, padding_color, interpolation) -> IO.NodeOutput:
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
            (batch_size, channels, target_height, target_width),
            pad_value,
            dtype=image.dtype,
            device=image.device
        )

        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        padded[:, :, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        output = padded.permute(0, 2, 3, 1)
        return IO.NodeOutput(output)

    resize_and_pad = execute  # TODO: remove


class SaveSVGNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveSVGNode",
            search_aliases=["export vector", "save vector graphics"],
            description="Save SVG files on disk.",
            category="image/save",
            inputs=[
                IO.SVG.Input("svg"),
                IO.String.Input(
                    "filename_prefix",
                    default="svg/ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, svg: IO.SVG.Type, filename_prefix="svg/ComfyUI") -> IO.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        results: list[UI.SavedResult] = []

        # Prepare metadata JSON
        metadata_dict = {}
        if cls.hidden.prompt is not None:
            metadata_dict["prompt"] = cls.hidden.prompt
        if cls.hidden.extra_pnginfo is not None:
            metadata_dict.update(cls.hidden.extra_pnginfo)

        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata_dict, indent=2) if metadata_dict else None


        for batch_number, svg_bytes in enumerate(svg.data):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.svg"

            # Read SVG content
            svg_bytes.seek(0)
            svg_content = svg_bytes.read().decode('utf-8')

            # Inject metadata if available
            if metadata_json:
                # Create metadata element with CDATA section
                metadata_element = f"""  <metadata>
                <![CDATA[
            {metadata_json}
                ]]>
            </metadata>
            """
                # Insert metadata after opening svg tag using regex with a replacement function
                def replacement(match):
                    # match.group(1) contains the captured <svg> tag
                    return match.group(1) + '\n' + metadata_element

                # Apply the substitution
                svg_content = re.sub(r'(<svg[^>]*>)', replacement, svg_content, flags=re.UNICODE)

            # Write the modified SVG to file
            with open(os.path.join(full_output_folder, file), 'wb') as svg_file:
                svg_file.write(svg_content.encode('utf-8'))

            results.append(UI.SavedResult(filename=file, subfolder=subfolder, type=IO.FolderType.output))
            counter += 1
        return IO.NodeOutput(ui={"images": results})

    save_svg = execute  # TODO: remove


class GetImageSize(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GetImageSize",
            search_aliases=["dimensions", "resolution", "image info"],
            display_name="Get Image Size",
            description="Returns width and height of the image, and passes it through unchanged.",
            category="image",
            inputs=[
                IO.Image.Input("image"),
            ],
            outputs=[
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
                IO.Int.Output(display_name="batch_size"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image) -> IO.NodeOutput:
        height = image.shape[1]
        width = image.shape[2]
        batch_size = image.shape[0]

        # Send progress text to display size on the node
        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(f"width: {width}, height: {height}\n batch size: {batch_size}", cls.hidden.unique_id)

        return IO.NodeOutput(width, height, batch_size)

    get_size = execute  # TODO: remove


class ImageRotate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageRotate",
            display_name="Image Rotate",
            search_aliases=["turn", "flip orientation"],
            category="image/transform",
            essentials_category="Image Tools",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("rotation", options=["none", "90 degrees", "180 degrees", "270 degrees"]),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, rotation) -> IO.NodeOutput:
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        image = torch.rot90(image, k=rotate_by, dims=[2, 1])
        return IO.NodeOutput(image)

    rotate = execute  # TODO: remove


class ImageFlip(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageFlip",
            search_aliases=["mirror", "reflect"],
            category="image/transform",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("flip_method", options=["x-axis: vertically", "y-axis: horizontally"]),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, flip_method) -> IO.NodeOutput:
        if flip_method.startswith("x"):
            image = torch.flip(image, dims=[1])
        elif flip_method.startswith("y"):
            image = torch.flip(image, dims=[2])

        return IO.NodeOutput(image)

    flip = execute  # TODO: remove


class ImageScaleToMaxDimension(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageScaleToMaxDimension",
            category="image/upscaling",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input(
                    "upscale_method",
                    options=["area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic"],
                ),
                IO.Int.Input("largest_size", default=512, min=0, max=MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, upscale_method, largest_size) -> IO.NodeOutput:
        height = image.shape[1]
        width = image.shape[2]

        if height > width:
            width = round((width / height) * largest_size)
            height = largest_size
        elif width > height:
            height = round((height / width) * largest_size)
            width = largest_size
        else:
            height = largest_size
            width = largest_size

        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return IO.NodeOutput(s)

    upscale = execute    # TODO: remove


class SplitImageToTileList(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplitImageToTileList",
            category="image/batch",
            search_aliases=["split image", "tile image", "slice image"],
            display_name="Split Image into List of Tiles",
            description="Splits an image into a batched list of tiles with a specified overlap.",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("tile_width", default=1024, min=64, max=MAX_RESOLUTION),
                IO.Int.Input("tile_height", default=1024, min=64, max=MAX_RESOLUTION),
                IO.Int.Input("overlap", default=128, min=0, max=4096),
            ],
            outputs=[
                IO.Image.Output(is_output_list=True),
            ],
        )

    @staticmethod
    def get_grid_coords(width, height, tile_width, tile_height, overlap):
        coords = []
        stride_x = round(max(tile_width * 0.25, tile_width - overlap))
        stride_y = round(max(tile_width * 0.25, tile_height - overlap))

        y = 0
        while y < height:
            x = 0
            y_end = min(y + tile_height, height)
            y_start = max(0, y_end - tile_height)

            while x < width:
                x_end = min(x + tile_width, width)
                x_start = max(0, x_end - tile_width)

                coords.append((x_start, y_start, x_end, y_end))

                if x_end >= width:
                    break
                x += stride_x

            if y_end >= height:
                break
            y += stride_y

        return coords

    @classmethod
    def execute(cls, image, tile_width, tile_height, overlap):
        b, h, w, c = image.shape
        coords = cls.get_grid_coords(w, h, tile_width, tile_height, overlap)

        output_list = []
        for (x_start, y_start, x_end, y_end) in coords:
            tile = image[:, y_start:y_end, x_start:x_end, :]
            output_list.append(tile)

        return IO.NodeOutput(output_list)


class ImageMergeTileList(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageMergeTileList",
            display_name="Merge List of Tiles to Image",
            category="image/batch",
            search_aliases=["split image", "tile image", "slice image"],
            is_input_list=True,
            inputs=[
                IO.Image.Input("image_list"),
                IO.Int.Input("final_width", default=1024, min=64, max=32768),
                IO.Int.Input("final_height", default=1024, min=64, max=32768),
                IO.Int.Input("overlap", default=128, min=0, max=4096),
            ],
            outputs=[
                IO.Image.Output(is_output_list=False),
            ],
        )

    @classmethod
    def execute(cls, image_list, final_width, final_height, overlap):
        w = final_width[0]
        h = final_height[0]
        ovlp = overlap[0]
        feather_str = 1.0

        first_tile = image_list[0]
        b, t_h, t_w, c = first_tile.shape
        device = first_tile.device
        dtype = first_tile.dtype

        coords = SplitImageToTileList.get_grid_coords(w, h, t_w, t_h, ovlp)

        canvas = torch.zeros((b, h, w, c), device=device, dtype=dtype)
        weights = torch.zeros((b, h, w, 1), device=device, dtype=dtype)

        if ovlp > 0:
            y_w = torch.sin(math.pi * torch.linspace(0, 1, t_h, device=device, dtype=dtype))
            x_w = torch.sin(math.pi * torch.linspace(0, 1, t_w, device=device, dtype=dtype))
            y_w = torch.clamp(y_w, min=1e-5)
            x_w = torch.clamp(x_w, min=1e-5)

            sine_mask = (y_w.unsqueeze(1) * x_w.unsqueeze(0)).unsqueeze(0).unsqueeze(-1)
            flat_mask = torch.ones_like(sine_mask)

            weight_mask = torch.lerp(flat_mask, sine_mask, feather_str)
        else:
            weight_mask = torch.ones((1, t_h, t_w, 1), device=device, dtype=dtype)

        for i, (x_start, y_start, x_end, y_end) in enumerate(coords):
            if i >= len(image_list):
                break

            tile = image_list[i]

            region_h = y_end - y_start
            region_w = x_end - x_start

            real_h = min(region_h, tile.shape[1])
            real_w = min(region_w, tile.shape[2])

            y_end_actual = y_start + real_h
            x_end_actual = x_start + real_w

            tile_crop = tile[:, :real_h, :real_w, :]
            mask_crop = weight_mask[:, :real_h, :real_w, :]

            canvas[:, y_start:y_end_actual, x_start:x_end_actual, :] += tile_crop * mask_crop
            weights[:, y_start:y_end_actual, x_start:x_end_actual, :] += mask_crop

        weights[weights == 0] = 1.0
        merged_image = canvas / weights

        return IO.NodeOutput(merged_image)



def create_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Creates a valid PNG chunk with Length, Type, Data, and CRC32."""
    chunk = struct.pack('>I', len(data)) + chunk_type + data
    crc = zlib.crc32(chunk_type + data) & 0xffffffff
    return chunk + struct.pack('>I', crc)


def inject_comfy_metadata_png(png_bytes, prompt=None, extra_pnginfo=None):
    # IEND chunk is the last 12 bytes of png files
    content = png_bytes[:-12]
    iend = png_bytes[-12:]

    metadata_chunks = b""

    if prompt is not None:
        payload = b'prompt\x00' + json.dumps(prompt).encode('utf-8')
        metadata_chunks += create_png_chunk(b'tEXt', payload)

    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            payload = k.encode('utf-8') + b'\x00' + json.dumps(v).encode('utf-8')
            metadata_chunks += create_png_chunk(b'tEXt', payload)

    return content + metadata_chunks + iend

def inject_comfy_metadata_exr(exr_bytes: bytes, prompt, extra_pnginfo) -> bytes:
    # skip magic and version
    idx = 8

    # parse through existing attributes to find the end of the header
    while True:
        name_start = idx
        while exr_bytes[idx] != 0:
            idx += 1
        name = exr_bytes[name_start:idx]
        idx += 1

        # empty name means we hit the header terminator
        if len(name) == 0:
            break

        # skip attribute type string
        while exr_bytes[idx] != 0:
            idx += 1
        idx += 1

        # read attribute size and skip the value
        attr_size = struct.unpack('<I', exr_bytes[idx:idx+4])[0]
        idx += 4 + attr_size

    # offset table starts right after the header terminator
    table_start = idx

    # build comfyui metadata payload
    payload = b""
    if prompt is not None:
        prompt_str = json.dumps(prompt).encode('utf-8')
        payload += b"prompt\x00string\x00" + struct.pack('<I', len(prompt_str)) + prompt_str
    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            k_enc = k.encode('utf-8')[:254]
            v_enc = json.dumps(v).encode('utf-8')
            payload += k_enc + b"\x00string\x00" + struct.pack('<I', len(v_enc)) + v_enc

    # find the first pixel offset to calculate the table size
    min_offset = struct.unpack('<Q', exr_bytes[table_start:table_start+8])[0]
    num_entries = 1
    while table_start + num_entries * 8 < min_offset:
        offset = struct.unpack('<Q', exr_bytes[table_start + num_entries*8 : table_start + num_entries*8 + 8])[0]
        if offset < min_offset:
            min_offset = offset
        num_entries += 1

    # shift table pointers by the payload size
    shift_amount = len(payload)
    new_table = bytearray()
    for i in range(num_entries):
        offset = struct.unpack('<Q', exr_bytes[table_start + i*8 : table_start + i*8 + 8])[0]
        new_table.extend(struct.pack('<Q', offset + shift_amount))

    # stitch the file back together with the new header and updated table
    return exr_bytes[:table_start - 1] + payload + b'\x00' + new_table + exr_bytes[table_start + num_entries*8:]

def inject_comfy_metadata_avif(avif_bytes: bytes, prompt, extra_pnginfo) -> bytes:
    metadata = {}
    if prompt is not None:
        metadata["prompt"] = prompt
    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            metadata[k] = v

    payload = json.dumps(metadata).encode('utf-8')

    # 16-byte uuid required by isobmff spec
    # 'comfyui_workflow' is exactly 16 bytes long!
    comfy_uuid = b'comfyui_workflow'

    # box size: 4 (size) + 4 (type) + 16 (uuid) + payload length
    box_size = 4 + 4 + 16 + len(payload)
    uuid_box = struct.pack('>I', box_size) + b'uuid' + comfy_uuid + payload

    # isobmff allows top-level boxes at the end of the file.
    return avif_bytes + uuid_box

class SaveImageAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveImageAdvanced",
            search_aliases=["save", "save image", "export image", "output image", "write image", "download"],
            display_name="Save Image",
            description="Saves the input images to your ComfyUI output directory.",
            category="image",
            essentials_category="Basics",
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip="The images to save."
                ),
                IO.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
                IO.DynamicCombo.Input(
                    "format",
                    options=[
                        IO.DynamicCombo.Option(
                            "png", 
                            [
                                IO.Combo.Input(
                                    "bit_depth",
                                    options=["8-bit", "16-bit"],
                                    default="8-bit",
                                    advanced=True,
                                ),
                                IO.Combo.Input(
                                    "color_space",
                                    options=["Raw/Data", "sRGB"],
                                    default="sRGB",
                                    advanced=True,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "avif",
                            [
                                IO.Combo.Input(
                                    "bit_depth",
                                    options=["8-bit", "10-bit", "12-bit"],
                                    default="8-bit",
                                    advanced=True,
                                ),
                                IO.Combo.Input(
                                    "color_space",
                                    options=["sRGB"],
                                    default="sRGB",
                                    advanced=True,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "exr",
                            [
                                IO.Combo.Input(
                                    "bit_depth",
                                    options=["16-bit (half-float)", "32-bit"],
                                    default="16-bit (half-float)",
                                    advanced=True,
                                ),
                                IO.Combo.Input(
                                    "color_space",
                                    options=["Linear", "Raw/Data"],
                                    default="Linear",
                                    advanced=True,
                                ),
                            ],
                        ),
                    ],
                    tooltip="The file format in which to save the image.",
                ),
                IO.Boolean.Input("embed_workflow", default=True, advanced=True),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: Input.Image,
        filename_prefix: str,
        format: dict,
        embed_workflow: bool,
        prompt=None,
        extra_pnginfo=None
    ) -> IO.NodeOutput:
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        for batch_number, image in enumerate(images):
            # get widget values from dynamic combo
            extension = format["format"]
            bit_depth = format["bit_depth"]
            color_space = format["color_space"]

            img_tensor = image.clone()

            height, width, num_channels = img_tensor.shape
            has_alpha = (num_channels == 4)

            # file pathing
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            file = f"{filename_with_batch_num}_{counter:05}.{file_format}"
            file_path = os.path.join(full_output_folder, file)

            if file_format in ["png", "exr", "avif"]:

                # safe bit downcasting
                if (file_format == "png" or file_format == "avif") and bit_depth == "32-bit":
                    bit_depth = "16-bit"
                if file_format == "exr" and bit_depth == "8-bit":
                    bit_depth = "16-bit"

                if bit_depth == "32-bit":
                    img_np = img_tensor.cpu().numpy().astype(np.float32)
                    av_fmt = 'gbrapf32le' if has_alpha else 'gbrpf32le'
                elif bit_depth == "16-bit":
                    if file_format == "exr":
                        # default pyav build doesn't come with a codec for float16 exr format
                        img_np = img_tensor.cpu().numpy().astype(np.float32)
                        av_fmt = 'gbrapf32le' if has_alpha else 'gbrpf32le'
                    else:
                        img_np = (img_tensor * 65535.0).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                        av_fmt = 'rgba64le' if has_alpha else 'rgb48le'
                else:
                    img_np = (img_tensor * 255.0).clamp(0, 255).to(torch.int32).cpu().numpy().astype(np.uint8)
                    av_fmt = 'rgba' if has_alpha else 'rgb24'

                fd, tmp_path = tempfile.mkstemp(suffix=f".{file_format}")
                os.close(fd)
                container_format = "image2" if file_format in ["png", "exr"] else "avif"
                container = av.open(tmp_path, mode='w', format=container_format)

                if file_format == "exr":
                    stream = container.add_stream('exr', rate=1)
                    stream.pix_fmt = av_fmt

                elif file_format == "avif":
                    try:
                        stream = container.add_stream('libsvtav1', rate=1)
                    except Exception:
                        stream = container.add_stream('av1', rate=1)

                    stream.time_base = Fraction(1, 1)

                    if bit_depth in ["16-bit", "32-bit"]:
                        stream.pix_fmt = 'yuv420p10le'
                    else:
                        stream.pix_fmt = 'yuv420p'

                    stream.codec_context.color_range = 2
                    stream.codec_context.colorspace = 1
                    stream.codec_context.color_primaries = 1
                    stream.codec_context.color_trc = 1

                    stream.options = {
                        'preset': '10',
                        'svtav1-params': 'rc=0:qp=20:color-range=1:color-matrix=1:enable-overlays=1',
                        'g': '1'
                    }

                elif file_format == "png":
                    stream = container.add_stream('png', rate=1)
                    if bit_depth == "16-bit":
                        stream.pix_fmt = 'rgba64be' if has_alpha else 'rgb48be'
                    else:
                        stream.pix_fmt = av_fmt

                stream.width = width
                stream.height = height
                stream.time_base = Fraction(1, 1)

                is_planar = av_fmt.startswith('gbrp') or 'p' in av_fmt.split('rgba')[-1]
                if is_planar:
                    if av_fmt.startswith('gbrp'):
                        img_np = img_np[:, :, [1, 2, 0, 3]] if has_alpha else img_np[:, :, [1, 2, 0]]
                    img_np = img_np.transpose(2, 0, 1)

                try:
                    frame = av.VideoFrame.from_ndarray(img_np, format=av_fmt)
                except ValueError:
                    logging.warning("[WARNING] Current FFMPEG Binary can't save natively. Fallbacking.")
                    img_np = (img_tensor * 65535.0).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                    av_fmt = 'rgba64le' if has_alpha else 'rgb48le'
                    frame = av.VideoFrame.from_ndarray(img_np, format=av_fmt)

                # reformat for both avif and exr to ensure correct internal conversion
                if file_format in ["avif", "exr"] or (file_format == "png" and bit_depth == "16-bit"):
                    reformat_kwargs = {"format": stream.pix_fmt}
                    if file_format == "avif":
                        reformat_kwargs.update({
                            "src_colorspace": 1, "dst_colorspace": 1,
                            "src_color_range": 2, "dst_color_range": 2
                        })
                    frame = frame.reformat(**reformat_kwargs)
                    frame.pts = 0
                    frame.time_base = stream.time_base
                    if file_format == "avif":
                        frame.color_range = 2
                        frame.colorspace = 1

                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

                container.close()

                with open(tmp_path, "rb") as f:
                    final_bytes = f.read()
                os.remove(tmp_path)

                if embed_workflow and not args.disable_metadata:
                    if file_format == "png":
                        final_bytes = inject_comfy_metadata_png(final_bytes, prompt, extra_pnginfo)
                    elif file_format == "exr":
                        final_bytes = inject_comfy_metadata_exr(final_bytes, prompt, extra_pnginfo)
                    else:
                        final_bytes = inject_comfy_metadata_avif(final_bytes, prompt, extra_pnginfo)

                with open(file_path, "wb") as f:
                    f.write(final_bytes)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1

        return IO.NodeOutput(ui={"images": results})

class ImagesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ImageCrop,
            ImageCropV2,
            BoundingBox,
            RepeatImageBatch,
            ImageFromBatch,
            ImageAddNoise,
            SaveAnimatedWEBP,
            SaveAnimatedPNG,
            SaveImageAdvanced,
            SaveSVGNode,
            ImageStitch,
            ResizeAndPadImage,
            GetImageSize,
            ImageRotate,
            ImageFlip,
            ImageScaleToMaxDimension,
            SplitImageToTileList,
            ImageMergeTileList,
            SaveImageAdvanced
        ]


async def comfy_entrypoint() -> ImagesExtension:
    return ImagesExtension()
