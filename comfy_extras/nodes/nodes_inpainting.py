from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from comfy.component_model.tensor_types import MaskBatch, ImageBatch
from comfy.nodes.package_typing import CustomNode
from comfy_extras.constants.resolutions import RESOLUTION_MAP, SD_RESOLUTIONS, RESOLUTION_NAMES


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


NODE_CLASS_MAPPINGS = {
    "CropAndFitInpaintToDiffusionSize": CropAndFitInpaintToDiffusionSize,
    "CompositeCroppedAndFittedInpaintResult": CompositeCroppedAndFittedInpaintResult,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAndFitInpaintToDiffusionSize": "Crop & Fit Inpaint Region",
    "CompositeCroppedAndFittedInpaintResult": "Composite Inpaint Result",
}
