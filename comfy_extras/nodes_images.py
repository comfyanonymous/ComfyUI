from __future__ import annotations

import nodes
import folder_paths

import json
import os
import re
import torch
import comfy.utils

from server import PromptServer
from comfy_api.latest import ComfyExtension, IO, UI
from typing_extensions import override

SVG = IO.SVG.Type  # TODO: temporary solution for backward compatibility, will be removed later.

MAX_RESOLUTION = nodes.MAX_RESOLUTION

class ImageCrop(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCrop",
            display_name="Image Crop",
            category="image/transform",
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


class RepeatImageBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RepeatImageBatch",
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
                IO.Int.Input("compress_level", default=4, min=0, max=9),
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
            display_name="Image Stitch",
            description="Stitches image2 to image1 in the specified direction.\n"
            "If image2 is not provided, returns image1 unchanged.\n"
            "Optional spacing can be added between images.",
            category="image/transform",
            inputs=[
                IO.Image.Input("image1"),
                IO.Combo.Input("direction", options=["right", "down", "left", "up"], default="right"),
                IO.Boolean.Input("match_image_size", default=True),
                IO.Int.Input("spacing_width", default=0, min=0, max=1024, step=2),
                IO.Combo.Input("spacing_color", options=["white", "black", "red", "green", "blue"], default="white"),
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
            category="image/transform",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("target_width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("target_height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Combo.Input("padding_color", options=["white", "black"]),
                IO.Combo.Input("interpolation", options=["area", "bicubic", "nearest-exact", "bilinear", "lanczos"]),
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
            category="image/transform",
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


class ImagesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ImageCrop,
            RepeatImageBatch,
            ImageFromBatch,
            ImageAddNoise,
            SaveAnimatedWEBP,
            SaveAnimatedPNG,
            SaveSVGNode,
            ImageStitch,
            ResizeAndPadImage,
            GetImageSize,
            ImageRotate,
            ImageFlip,
            ImageScaleToMaxDimension,
        ]


async def comfy_entrypoint() -> ImagesExtension:
    return ImagesExtension()
