from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
import comfy.utils

from jaxtyping import Float
from torch import Tensor

from comfy.component_model.tensor_types import MaskBatch, ImageBatch
from comfy.nodes.package_typing import CustomNode
from ..constants.resolutions import RESOLUTION_MAP, SD_RESOLUTIONS, RESOLUTION_NAMES


class CompositeContext(NamedTuple):
    x: int
    y: int
    width: int
    height: int


def composite(
        destination: Float[Tensor, "B C H W"],
        source: Float[Tensor, "B C H W"],
        x: int,
        y: int,
        mask: Optional[MaskBatch] = None,
) -> ImageBatch:
    """
    Composites a source image onto a destination image at a given (x, y) coordinate
    using an optional mask.

    This simplified implementation first creates a destination-sized, zero-padded
    version of the source image. This canvas is then blended with the destination,
    which cleanly handles all boundary conditions (e.g., source placed partially
    or fully off-screen).

    Args:
        destination (ImageBatch): The background image tensor in (B, C, H, W) format.
        source (ImageBatch): The foreground image tensor to composite, also (B, C, H, W).
        x (int): The x-coordinate (from left) to place the top-left corner of the source.
        y (int): The y-coordinate (from top) to place the top-left corner of the source.
        mask (Optional[MaskBatch]): An optional luma mask tensor with the same batch size,
                                    height, and width as the destination (B, H, W).
                                    Values of 1.0 indicate using the source pixel, while
                                    0.0 indicates using the destination pixel. If None,
                                    the source is treated as fully opaque.

    Returns:
        ImageBatch: The resulting composited image tensor.
    """
    if not isinstance(destination, torch.Tensor) or not isinstance(source, torch.Tensor):
        raise TypeError("destination and source must be torch.Tensor")
    if destination.dim() != 4 or source.dim() != 4:
        raise ValueError("destination and source must be 4D tensors (B, C, H, W)")

    source = source.to(destination.device)

    if source.shape[0] != destination.shape[0]:
        if destination.shape[0] % source.shape[0] != 0:
            raise ValueError(
                "Destination batch size must be a multiple of source batch size for broadcasting."
            )
        source = source.repeat(destination.shape[0] // source.shape[0], 1, 1, 1)

    dest_b, dest_c, dest_h, dest_w = destination.shape
    src_h, src_w = source.shape[2:]

    dest_y_start = max(0, y)
    dest_y_end = min(dest_h, y + src_h)
    dest_x_start = max(0, x)
    dest_x_end = min(dest_w, x + src_w)

    src_y_start = max(0, -y)
    src_y_end = src_y_start + (dest_y_end - dest_y_start)
    src_x_start = max(0, -x)
    src_x_end = src_x_start + (dest_x_end - dest_x_start)

    if dest_y_start >= dest_y_end or dest_x_start >= dest_x_end:
        return destination
    padded_source = torch.zeros_like(destination)
    padded_source[:, :, dest_y_start:dest_y_end, dest_x_start:dest_x_end] = source[
        :, :, src_y_start:src_y_end, src_x_start:src_x_end
    ]
    if mask is None:
        final_mask = torch.zeros(dest_b, 1, dest_h, dest_w, device=destination.device)
        final_mask[:, :, dest_y_start:dest_y_end, dest_x_start:dest_x_end] = 1.0
    else:
        if mask.dim() != 3 or mask.shape[0] != dest_b or mask.shape[1] != dest_h or mask.shape[2] != dest_w:
            raise ValueError(
                f"Provided mask shape {mask.shape} is invalid. "
                f"Expected (batch, height, width): ({dest_b}, {dest_h}, {dest_w})."
            )
        final_mask = mask.to(destination.device).unsqueeze(1)

    blended_image = padded_source * final_mask + destination * (1.0 - final_mask)

    return blended_image


def parse_margin(margin_str: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in margin_str.strip().split()]
    match len(parts):
        case 1:
            return parts[0], parts[0], parts[0], parts[0]
        case 2:
            return parts[0], parts[1], parts[0], parts[1]
        case 3:
            return parts[0], parts[1], parts[2], parts[1]
        case 4:
            return parts[0], parts[1], parts[2], parts[3]
        case _:
            raise ValueError("Invalid margin format.")


class CropAndFitInpaintToDiffusionSize(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), "mask": ("MASK",),
                "resolutions": (RESOLUTION_NAMES, {"default": RESOLUTION_NAMES[0]}),
                "margin": ("STRING", {"default": "64"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "COMPOSITE_CONTEXT")
    RETURN_NAMES = ("image", "mask", "composite_context")
    FUNCTION = "crop_and_fit"
    CATEGORY = "inpaint"

    def crop_and_fit(self, image: torch.Tensor, mask: MaskBatch, resolutions: str, margin: str):
        if mask.max() == 0.0:
            raise ValueError("Mask is empty (all black).")

        mask_coords = torch.nonzero(mask)
        if mask_coords.numel() == 0:
            raise ValueError("Mask is empty (all black).")

        y_coords, x_coords = mask_coords[:, 1], mask_coords[:, 2]
        y_min, x_min = y_coords.min().item(), x_coords.min().item()
        y_max, x_max = y_coords.max().item(), x_coords.max().item()

        top_m, right_m, bottom_m, left_m = parse_margin(margin)
        x_start_expanded, y_start_expanded = x_min - left_m, y_min - top_m
        x_end_expanded, y_end_expanded = x_max + 1 + right_m, y_max + 1 + bottom_m

        img_h, img_w = image.shape[1:3]

        clamped_x_start = max(0, x_start_expanded)
        clamped_y_start = max(0, y_start_expanded)
        clamped_x_end = min(img_w, x_end_expanded)
        clamped_y_end = min(img_h, y_end_expanded)

        initial_w, initial_h = clamped_x_end - clamped_x_start, clamped_y_end - clamped_y_start
        if initial_w <= 0 or initial_h <= 0:
            raise ValueError("Cropped area has zero dimension.")

        supported_resolutions = RESOLUTION_MAP.get(resolutions, SD_RESOLUTIONS)
        diffs = [(abs(res[0] / res[1] - (initial_w / initial_h)), res) for res in supported_resolutions]
        target_res = min(diffs, key=lambda x: x[0])[1]
        target_ar = target_res[0] / target_res[1]

        current_ar = initial_w / initial_h
        if current_ar > target_ar:
            cover_w, cover_h = float(initial_w), float(initial_w) / target_ar
        else:
            cover_h, cover_w = float(initial_h), float(initial_h) * target_ar

        if cover_w > img_w or cover_h > img_h:
            final_x, final_y, final_w, final_h = 0, 0, img_w, img_h
            full_img_ar = img_w / img_h
            diffs_full = [(abs(res[0] / res[1] - full_img_ar), res) for res in supported_resolutions]
            target_res = min(diffs_full, key=lambda x: x[0])[1]
        else:
            center_x = clamped_x_start + initial_w / 2
            center_y = clamped_y_start + initial_h / 2
            final_x, final_y = center_x - cover_w / 2, center_y - cover_h / 2
            final_w, final_h = cover_w, cover_h

            if final_x < 0:
                final_x = 0
            if final_y < 0:
                final_y = 0
            if final_x + final_w > img_w:
                final_x = img_w - final_w
            if final_y + final_h > img_h:
                final_y = img_h - final_h

        final_x, final_y, final_w, final_h = int(final_x), int(final_y), int(final_w), int(final_h)

        cropped_image = image[:, final_y:final_y + final_h, final_x:final_x + final_w]
        cropped_mask = mask[:, final_y:final_y + final_h, final_x:final_x + final_w]

        resized_image = F.interpolate(cropped_image.permute(0, 3, 1, 2), size=(target_res[1], target_res[0]), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        resized_mask = F.interpolate(cropped_mask.unsqueeze(1), size=(target_res[1], target_res[0]), mode="nearest").squeeze(1)

        composite_context = CompositeContext(x=final_x, y=final_y, width=final_w, height=final_h)
        return (resized_image, resized_mask, composite_context)


class CompositeCroppedAndFittedInpaintResult(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "source_mask": ("MASK",),
                "inpainted_image": ("IMAGE",),
                "composite_context": ("COMPOSITE_CONTEXT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_result"
    CATEGORY = "inpaint"

    def composite_result(self, source_image: ImageBatch, source_mask: MaskBatch, inpainted_image: ImageBatch, composite_context: CompositeContext) -> tuple[ImageBatch]:
        context_x, context_y, context_w, context_h = composite_context

        resized_inpainted = F.interpolate(
            inpainted_image.permute(0, 3, 1, 2),
            size=(context_h, context_w),
            mode="bilinear", align_corners=False
        )

        final_image = composite(
            destination=source_image.clone().permute(0, 3, 1, 2),
            source=resized_inpainted,
            x=context_x,
            y=context_y,
            mask=source_mask
        )

        return final_image.permute(0, 2, 3, 1),


class ImageAndMaskResizeNode:
    """
    Sherlocked from https://github.com/CY-CHENYUE/ComfyUI-InpaintEasy

    MIT License

    Copyright (c) 2024 CYCHENYUE

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    DESCRIPTION = "Resize the image and mask simultaneously (from InpaintEasy- 同时调整图片和蒙版的大小)"
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center", "top_left", "top_right", "bottom_left", "bottom_right"]

    def __init__(self):
        self.type = "ImageMaskResize"
        self.output_node = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "resize_method": (s.upscale_methods, {"default": "lanczos"}),
                "crop": (s.crop_methods, {"default": "disabled"}),
                "mask_blur_radius": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 64,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "resize_image_and_mask"

    CATEGORY = "inpaint"

    def resize_image_and_mask(self, image, mask, width, height, resize_method="lanczos", crop="disabled", mask_blur_radius=0):
        # 处理宽高为0的情况
        if width == 0 and height == 0:
            return (image, mask)

        # 对于图像的处理
        samples = image.movedim(-1, 1)  # NHWC -> NCHW
        if width == 0:
            width = max(1, round(samples.shape[3] * height / samples.shape[2]))
        elif height == 0:
            height = max(1, round(samples.shape[2] * width / samples.shape[3]))

        # 使用 torch.nn.functional 直接进行缩放和裁剪
        if crop != "disabled":
            old_width = samples.shape[3]
            old_height = samples.shape[2]

            # 计算缩放比例
            scale = max(width / old_width, height / old_height)
            scaled_width = int(old_width * scale)
            scaled_height = int(old_height * scale)

            # 使用 common_upscale 进行缩放
            samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, resize_method, crop="disabled")

            # 蒙版始终使用bilinear插值
            mask = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(scaled_height, scaled_width), mode='bilinear', align_corners=True)

            # 计算裁剪位置
            crop_x = 0
            crop_y = 0

            if crop == "center":
                crop_x = (scaled_width - width) // 2
                crop_y = (scaled_height - height) // 2
            elif crop == "top_left":
                crop_x = 0
                crop_y = 0
            elif crop == "top_right":
                crop_x = scaled_width - width
                crop_y = 0
            elif crop == "bottom_left":
                crop_x = 0
                crop_y = scaled_height - height
            elif crop == "bottom_right":
                crop_x = scaled_width - width
                crop_y = scaled_height - height
            elif crop == "random":
                crop_x = torch.randint(0, max(1, scaled_width - width), (1,)).item()
                crop_y = torch.randint(0, max(1, scaled_height - height), (1,)).item()

            # 执行裁剪
            samples = samples[:, :, crop_y:crop_y + height, crop_x:crop_x + width]
            mask = mask[:, :, crop_y:crop_y + height, crop_x:crop_x + width]
        else:
            # 直接使用 common_upscale 调整大小
            samples = comfy.utils.common_upscale(samples, width, height, resize_method, crop="disabled")
            mask = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(height, width), mode='bilinear', align_corners=True)

        image_resized = samples.movedim(1, -1)  # NCHW -> NHWC
        mask_resized = mask.squeeze(1)  # NCHW -> NHW

        # 在返回之前添加高斯模糊处理
        if mask_blur_radius > 0:
            # 创建高斯核
            kernel_size = mask_blur_radius * 2 + 1
            x = torch.arange(kernel_size, dtype=torch.float32, device=mask_resized.device)
            x = x - (kernel_size - 1) / 2
            gaussian = torch.exp(-(x ** 2) / (2 * (mask_blur_radius / 3) ** 2))
            gaussian = gaussian / gaussian.sum()

            # 将kernel转换为2D
            gaussian_2d = gaussian.view(1, -1) * gaussian.view(-1, 1)
            gaussian_2d = gaussian_2d.view(1, 1, kernel_size, kernel_size)

            # 应用高斯模糊
            mask_for_blur = mask_resized.unsqueeze(1)  # Add channel dimension
            # 对边界进行padding，使用reflect模式避免边缘问题
            padding = kernel_size // 2
            mask_padded = F.pad(mask_for_blur, (padding, padding, padding, padding), mode='reflect')
            mask_resized = F.conv2d(mask_padded, gaussian_2d.to(mask_resized.device), padding=0).squeeze(1)

            # 确保值在0-1范围内
            mask_resized = torch.clamp(mask_resized, 0, 1)

        return (image_resized, mask_resized)


NODE_CLASS_MAPPINGS = {
    "CropAndFitInpaintToDiffusionSize": CropAndFitInpaintToDiffusionSize,
    "CompositeCroppedAndFittedInpaintResult": CompositeCroppedAndFittedInpaintResult,
    "ImageAndMaskResizeNode": ImageAndMaskResizeNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAndFitInpaintToDiffusionSize": "Crop & Fit Inpaint Region",
    "CompositeCroppedAndFittedInpaintResult": "Composite Inpaint Result",
    "ImageAndMaskResizeNode": "Image and Mask Resize"
}
