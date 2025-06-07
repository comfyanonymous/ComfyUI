import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional

from comfy.component_model.tensor_types import MaskBatch, ImageBatch
from comfy.nodes.package_typing import CustomNode
from comfy_extras.constants.resolutions import (
    RESOLUTION_NAMES, SDXL_SD3_FLUX_RESOLUTIONS, SD_RESOLUTIONS, LTVX_RESOLUTIONS,
    IDEOGRAM_RESOLUTIONS, COSMOS_RESOLUTIONS, HUNYUAN_VIDEO_RESOLUTIONS,
    WAN_VIDEO_14B_RESOLUTIONS, WAN_VIDEO_1_3B_RESOLUTIONS,
    WAN_VIDEO_14B_EXTENDED_RESOLUTIONS
)

class CompositeContext(NamedTuple):
    x: int
    y: int
    width: int
    height: int

def composite(destination: ImageBatch, source: ImageBatch, x: int, y: int, mask: Optional[MaskBatch] = None):
    """A robust function to composite a source tensor onto a destination tensor."""
    source = source.to(destination.device)
    if source.shape[0] != destination.shape[0]:
        source = source.repeat(destination.shape[0] // source.shape[0], 1, 1, 1)

    x, y = int(x), int(y)
    left, top = x, y
    right, bottom = left + source.shape[3], top + source.shape[2]

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        if mask.dim() == 2: mask = mask.unsqueeze(0)
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        if mask.shape[0] != source.shape[0]:
            mask = mask.repeat(source.shape[0] // mask.shape[0], 1, 1, 1)

    dest_left, dest_top = max(0, left), max(0, top)
    dest_right, dest_bottom = min(destination.shape[3], right), min(destination.shape[2], bottom)

    if dest_right <= dest_left or dest_bottom <= dest_top: return destination

    src_left, src_top = dest_left - left, dest_top - top
    src_right, src_bottom = dest_right - left, dest_bottom

    destination_portion = destination[:, :, dest_top:dest_bottom, dest_left:dest_right]
    source_portion = source[:, :, src_top:src_bottom, src_left:src_right]

    # The mask must be cropped to the region of interest on the destination.
    mask_portion = mask[:, :, dest_top:dest_bottom, dest_left:dest_right]

    blended_portion = (source_portion * mask_portion) + (destination_portion * (1.0 - mask_portion))
    destination[:, :, dest_top:dest_bottom, dest_left:dest_right] = blended_portion
    return destination

def parse_margin(margin_str: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in margin_str.strip().split()]
    if len(parts) == 1: return parts[0], parts[0], parts[0], parts[0]
    if len(parts) == 2: return parts[0], parts[1], parts[0], parts[1]
    if len(parts) == 3: return parts[0], parts[1], parts[2], parts[1]
    if len(parts) == 4: return parts[0], parts[1], parts[2], parts[3]
    raise ValueError("Invalid margin format.")

class CropAndFitInpaintToDiffusionSize(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "mask": ("MASK",),
            "resolutions": (RESOLUTION_NAMES, {"default": "SD1.5"}),
            "margin": ("STRING", {"default": "64"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "COMPOSITE_CONTEXT")
    RETURN_NAMES = ("image", "mask", "composite_context")
    FUNCTION = "crop_and_fit"
    CATEGORY = "inpaint"

    def crop_and_fit(self, image: torch.Tensor, mask: MaskBatch, resolutions: str, margin: str, aspect_ratio_tolerance=0.05):
        if mask.max() <= 0: raise ValueError("Mask is empty.")
        mask_coords = torch.nonzero(mask)
        if mask_coords.numel() == 0: raise ValueError("Mask is empty.")

        y_coords, x_coords = mask_coords[:, 1], mask_coords[:, 2]
        y_min, x_min = y_coords.min().item(), x_coords.min().item()
        y_max, x_max = y_coords.max().item(), x_coords.max().item()

        top_m, right_m, bottom_m, left_m = parse_margin(margin)
        x_start_expanded, y_start_expanded = x_min - left_m, y_min - top_m
        x_end_expanded, y_end_expanded = x_max + 1 + right_m, y_max + 1 + bottom_m

        img_h, img_w = image.shape[1:3]
        clamped_x_start, clamped_y_start = max(0, x_start_expanded), max(0, y_start_expanded)
        clamped_x_end, clamped_y_end = min(img_w, x_end_expanded), min(img_h, y_end_expanded)

        initial_w, initial_h = clamped_x_end - clamped_x_start, clamped_y_end - clamped_y_start
        if initial_w <= 0 or initial_h <= 0: raise ValueError("Cropped area has zero dimension.")

        res_map = { "SDXL/SD3/Flux": SDXL_SD3_FLUX_RESOLUTIONS, "SD1.5": SD_RESOLUTIONS, "LTXV": LTVX_RESOLUTIONS, "Ideogram": IDEOGRAM_RESOLUTIONS, "Cosmos": COSMOS_RESOLUTIONS, "HunyuanVideo": HUNYUAN_VIDEO_RESOLUTIONS, "WAN 14b": WAN_VIDEO_14B_RESOLUTIONS, "WAN 1.3b": WAN_VIDEO_1_3B_RESOLUTIONS, "WAN 14b with extras": WAN_VIDEO_14B_EXTENDED_RESOLUTIONS }
        supported_resolutions = res_map.get(resolutions, SD_RESOLUTIONS)
        diffs = [(abs(res[0] / res[1] - (initial_w / initial_h)), res) for res in supported_resolutions]
        target_res = min(diffs, key=lambda x: x[0])[1]
        target_ar = target_res[0] / target_res[1]

        current_ar = initial_w / initial_h
        final_x, final_y = float(clamped_x_start), float(clamped_y_start)
        final_w, final_h = float(initial_w), float(initial_h)

        if current_ar > target_ar:
            final_w = initial_h * target_ar
            final_x += (initial_w - final_w) / 2
        else:
            final_h = initial_w / target_ar
            final_y += (initial_h - final_h) / 2

        final_x, final_y, final_w, final_h = int(final_x), int(final_y), int(final_w), int(final_h)

        cropped_image = image[:, final_y:final_y + final_h, final_x:final_x + final_w]
        cropped_mask = mask[:, final_y:final_y + final_h, final_x:final_x + final_w]

        resized_image = F.interpolate(cropped_image.permute(0,3,1,2), size=(target_res[1], target_res[0]), mode="bilinear", align_corners=False).permute(0,2,3,1)
        resized_mask = F.interpolate(cropped_mask.unsqueeze(1), size=(target_res[1], target_res[0]), mode="nearest").squeeze(1)

        composite_context = CompositeContext(x=final_x, y=final_y, width=final_w, height=final_h)
        return (resized_image, resized_mask, composite_context)

class CompositeCroppedAndFittedInpaintResult(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"source_image": ("IMAGE",), "source_mask": ("MASK",), "inpainted_image": ("IMAGE",), "composite_context": ("COMPOSITE_CONTEXT",),}}

    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "composite_result", "inpaint"

    def composite_result(self, source_image: ImageBatch, source_mask: MaskBatch, inpainted_image: ImageBatch, composite_context: CompositeContext):
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

        return (final_image.permute(0, 2, 3, 1),)

NODE_CLASS_MAPPINGS = {"CropAndFitInpaintToDiffusionSize": CropAndFitInpaintToDiffusionSize, "CompositeCroppedAndFittedInpaintResult": CompositeCroppedAndFittedInpaintResult}
NODE_DISPLAY_NAME_MAPPINGS = {"CropAndFitInpaintToDiffusionSize": "Crop & Fit Inpaint Region", "CompositeCroppedAndFittedInpaintResult": "Composite Inpaint Result"}