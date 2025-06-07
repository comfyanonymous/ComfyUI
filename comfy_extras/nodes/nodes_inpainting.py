import torch
import torch.nn.functional as F

from comfy.component_model.tensor_types import MaskBatch
from comfy_extras.constants.resolutions import (
    RESOLUTION_NAMES, SDXL_SD3_FLUX_RESOLUTIONS, SD_RESOLUTIONS, LTVX_RESOLUTIONS,
    IDEOGRAM_RESOLUTIONS, COSMOS_RESOLUTIONS, HUNYUAN_VIDEO_RESOLUTIONS,
    WAN_VIDEO_14B_RESOLUTIONS, WAN_VIDEO_1_3B_RESOLUTIONS,
    WAN_VIDEO_14B_EXTENDED_RESOLUTIONS
)


def composite(destination, source, x, y, mask=None, multiplier=1, resize_source=False):
    source = source.to(destination.device)
    if resize_source:
        source = F.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")
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
    mask_portion = mask[:, :, src_top:src_bottom, src_left:src_right]

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


class CropAndFitInpaintToDiffusionSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "resolutions": (RESOLUTION_NAMES, {"default": RESOLUTION_NAMES[0]}), "margin": ("STRING", {"default": "64"}), "overflow": ("BOOLEAN", {"default": True}), }}

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("IMAGE", "MASK", "COMBO[INT]"), ("image", "mask", "composite_context"), "crop_and_fit", "inpaint"

    def crop_and_fit(self, image: torch.Tensor, mask: MaskBatch, resolutions: str, margin: str, overflow: bool, aspect_ratio_tolerance=0.05):
        if mask.max() <= 0: raise ValueError("Mask is empty.")
        mask_coords = torch.nonzero(mask[0]);
        if mask_coords.numel() == 0: raise ValueError("Mask is empty.")
        y_min, x_min = mask_coords.min(dim=0).values;
        y_max, x_max = mask_coords.max(dim=0).values
        top_m, right_m, bottom_m, left_m = parse_margin(margin)
        x_start_init, y_start_init = x_min.item() - left_m, y_min.item() - top_m
        x_end_init, y_end_init = x_max.item() + 1 + right_m, y_max.item() + 1 + bottom_m
        img_h, img_w = image.shape[1:3]
        pad_image, pad_mask = image, mask
        x_start_crop, y_start_crop = x_start_init, y_start_init
        x_end_crop, y_end_crop = x_end_init, y_end_init
        pad_l, pad_t = -min(0, x_start_init), -min(0, y_start_init)
        pad_r, pad_b = max(0, x_end_init - img_w), max(0, y_end_init - img_h)
        if any([pad_l, pad_t, pad_r, pad_b]) and overflow:
            padding = (pad_l, pad_r, pad_t, pad_b)
            pad_image = F.pad(image.permute(0, 3, 1, 2), padding, "constant", 0.5).permute(0, 2, 3, 1)
            pad_mask = F.pad(mask.unsqueeze(1), padding, "constant", 0).squeeze(1)
            x_start_crop += pad_l;
            y_start_crop += pad_t;
            x_end_crop += pad_l;
            y_end_crop += pad_t
        else:
            x_start_crop, y_start_crop = max(0, x_start_init), max(0, y_start_init)
            x_end_crop, y_end_crop = min(img_w, x_end_init), min(img_h, y_end_init)
        composite_x, composite_y = (x_start_init if overflow else x_start_crop), (y_start_init if overflow else y_start_crop)
        cropped_image = pad_image[:, y_start_crop:y_end_crop, x_start_crop:x_end_crop, :]
        cropped_mask = pad_mask[:, y_start_crop:y_end_crop, x_start_crop:x_end_crop]
        context = {"x": composite_x, "y": composite_y, "width": cropped_image.shape[2], "height": cropped_image.shape[1]}

        rgba_bchw = torch.cat((cropped_image.permute(0, 3, 1, 2), cropped_mask.unsqueeze(1)), dim=1)
        res_map = {"SDXL/SD3/Flux": SDXL_SD3_FLUX_RESOLUTIONS, "SD1.5": SD_RESOLUTIONS, "LTXV": LTVX_RESOLUTIONS, "Ideogram": IDEOGRAM_RESOLUTIONS, "Cosmos": COSMOS_RESOLUTIONS, "HunyuanVideo": HUNYUAN_VIDEO_RESOLUTIONS, "WAN 14b": WAN_VIDEO_14B_RESOLUTIONS, "WAN 1.3b": WAN_VIDEO_1_3B_RESOLUTIONS, "WAN 14b with extras": WAN_VIDEO_14B_EXTENDED_RESOLUTIONS}
        supported_resolutions = res_map.get(resolutions, SD_RESOLUTIONS)
        h, w = cropped_image.shape[1:3]
        current_aspect_ratio = w / h
        diffs = [(abs(res[0] / res[1] - current_aspect_ratio), res) for res in supported_resolutions]
        min_diff = min(diffs, key=lambda x: x[0])[0]
        close_res = [res for diff, res in diffs if diff <= min_diff + aspect_ratio_tolerance]
        target_res = max(close_res, key=lambda r: r[0] * r[1])
        scale = max(target_res[0] / w, target_res[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        upscaled_rgba = F.interpolate(rgba_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        y1, x1 = (new_h - target_res[1]) // 2, (new_w - target_res[0]) // 2
        final_rgba_bchw = upscaled_rgba[:, :, y1:y1 + target_res[1], x1:x1 + target_res[0]]
        final_rgba_bhwc = final_rgba_bchw.permute(0, 2, 3, 1)
        resized_image = final_rgba_bhwc[..., :3]
        resized_mask = (final_rgba_bhwc[..., 3] > 0.5).float()
        return (resized_image, resized_mask, (context["x"], context["y"], context["width"], context["height"]))


class CompositeCroppedAndFittedInpaintResult:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"source_image": ("IMAGE",), "source_mask": ("MASK",), "inpainted_image": ("IMAGE",), "composite_context": ("COMBO[INT]",), }}

    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "composite_result", "inpaint"

    def composite_result(self, source_image: torch.Tensor, source_mask: MaskBatch, inpainted_image: torch.Tensor, composite_context: tuple[int, ...]):
        x, y, width, height = composite_context
        target_size = (height, width)

        resized_inpainted_image = F.interpolate(inpainted_image.permute(0, 3, 1, 2), size=target_size, mode="bilinear", align_corners=False)

        # FIX: The logic for cropping the original mask was flawed.
        # This simpler approach directly crops the relevant section of the original source_mask.
        # It correctly handles negative coordinates from the overflow case.
        crop_x_start = max(0, x)
        crop_y_start = max(0, y)
        crop_x_end = min(source_image.shape[2], x + width)
        crop_y_end = min(source_image.shape[1], y + height)

        # The mask for compositing is a direct, high-resolution crop of the source mask.
        final_compositing_mask = source_mask[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        destination_image = source_image.clone().permute(0, 3, 1, 2)

        # We now pass our perfectly cropped high-res mask to the composite function.
        # Note that the `composite` function handles placing this at the correct sub-region.
        final_image_permuted = composite(destination=destination_image, source=resized_inpainted_image, x=x, y=y, mask=final_compositing_mask)

        return (final_image_permuted.permute(0, 2, 3, 1),)


NODE_CLASS_MAPPINGS = {"CropAndFitInpaintToDiffusionSize": CropAndFitInpaintToDiffusionSize, "CompositeCroppedAndFittedInpaintResult": CompositeCroppedAndFittedInpaintResult}
NODE_DISPLAY_NAME_MAPPINGS = {"CropAndFitInpaintToDiffusionSize": "Crop & Fit Inpaint Region", "CompositeCroppedAndFittedInpaintResult": "Composite Inpaint Result"}
