import torch
import torch
import torch.nn.functional as F

from comfy.component_model.tensor_types import MaskBatch
from comfy_extras.constants.resolutions import RESOLUTION_NAMES
from comfy_extras.nodes.nodes_images import ImageResize


# Helper function from the context to composite images
def composite(destination, source, x, y, mask=None, multiplier=1, resize_source=False):
    # This function is adapted from the provided context code
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    # Ensure source has the same batch size as destination
    if source.shape[0] != destination.shape[0]:
        source = source.repeat(destination.shape[0] // source.shape[0], 1, 1, 1)

    x = int(x)
    y = int(y)

    left, top = (x, y)
    right, bottom = (left + source.shape[3], top + source.shape[2])

    if mask is None:
        # If no mask is provided, create a full-coverage mask
        mask = torch.ones_like(source)
    else:
        # Ensure mask is on the correct device and is the correct size
        mask = mask.to(destination.device, copy=True)
        # Check if the mask is 2D (H, W) or 3D (B, H, W) and unsqueeze if necessary
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # Add channel dimension
        mask = torch.nn.functional.interpolate(mask, size=(source.shape[2], source.shape[3]), mode="bilinear")
        if mask.shape[0] != source.shape[0]:
            mask = mask.repeat(source.shape[0] // mask.shape[0], 1, 1, 1)

    # Define the bounds of the overlapping area
    dest_left = max(0, left)
    dest_top = max(0, top)
    dest_right = min(destination.shape[3], right)
    dest_bottom = min(destination.shape[2], bottom)

    # If there is no overlap, return the original destination
    if dest_right <= dest_left or dest_bottom <= dest_top:
        return destination

    # Calculate the source coordinates corresponding to the overlap
    src_left = dest_left - left
    src_top = dest_top - top
    src_right = dest_right - left
    src_bottom = dest_bottom - top

    # Crop the relevant portions of the destination, source, and mask
    destination_portion = destination[:, :, dest_top:dest_bottom, dest_left:dest_right]
    source_portion = source[:, :, src_top:src_bottom, src_left:src_right]
    mask_portion = mask[:, :, src_top:src_bottom, src_left:src_right]

    inverse_mask_portion = 1.0 - mask_portion

    # Perform the composition
    blended_portion = (source_portion * mask_portion) + (destination_portion * inverse_mask_portion)

    # Place the blended portion back into the destination
    destination[:, :, dest_top:dest_bottom, dest_left:dest_right] = blended_portion

    return destination


def parse_margin(margin_str: str) -> tuple[int, int, int, int]:
    """Parses a CSS-style margin string."""
    parts = [int(p) for p in margin_str.strip().split()]
    if len(parts) == 1:
        return parts[0], parts[0], parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1], parts[0], parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], parts[1]
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    raise ValueError("Invalid margin format. Use 1 to 4 integer values.")


class CropAndFitInpaintToDiffusionSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "resolutions": (RESOLUTION_NAMES, {"default": RESOLUTION_NAMES[0]}),
                "margin": ("STRING", {"default": "64"}),
                "overflow": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "COMBO[INT]")
    RETURN_NAMES = ("image", "mask", "composite_context")
    FUNCTION = "crop_and_fit"
    CATEGORY = "inpaint"

    def crop_and_fit(self, image: torch.Tensor, mask: MaskBatch, resolutions: str, margin: str, overflow: bool):
        # 1. Find bounding box of the mask
        if mask.max() <= 0:
            raise ValueError("Mask is empty, cannot determine bounding box.")

        # Find the coordinates of non-zero mask pixels
        mask_coords = torch.nonzero(mask[0])  # Assuming single batch for mask
        if mask_coords.numel() == 0:
            raise ValueError("Mask is empty, cannot determine bounding box.")

        y_min, x_min = mask_coords.min(dim=0).values
        y_max, x_max = mask_coords.max(dim=0).values

        # 2. Parse and apply margin
        top_margin, right_margin, bottom_margin, left_margin = parse_margin(margin)

        x_start = x_min.item() - left_margin
        y_start = y_min.item() - top_margin
        x_end = x_max.item() + 1 + right_margin
        y_end = y_max.item() + 1 + bottom_margin

        img_height, img_width = image.shape[1:3]

        # Store pre-crop context for the compositor node
        context = {
            "x": x_start,
            "y": y_start,
            "width": x_end - x_start,
            "height": y_end - y_start
        }

        # 3. Handle overflow
        padded_image = image
        padded_mask = mask

        pad_left = -min(0, x_start)
        pad_top = -min(0, y_start)
        pad_right = max(0, x_end - img_width)
        pad_bottom = max(0, y_end - img_height)

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            if not overflow:
                # Crop margin to fit within the image
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(img_width, x_end)
                y_end = min(img_height, y_end)
            else:
                # Extend image and mask
                padding = (pad_left, pad_right, pad_top, pad_bottom)
                # Pad image with gray
                padded_image = F.pad(image.permute(0, 3, 1, 2), padding, "constant", 0.5).permute(0, 2, 3, 1)
                # Pad mask with zeros
                padded_mask = F.pad(mask.unsqueeze(1), padding, "constant", 0).squeeze(1)

                # Adjust coordinates for the new padded space
                x_start += pad_left
                y_start += pad_top
                x_end += pad_left
                y_end += pad_top

        # 4. Crop image and mask
        cropped_image = padded_image[:, y_start:y_end, x_start:x_end, :]
        cropped_mask = padded_mask[:, y_start:y_end, x_start:x_end]

        # 5. Resize to a supported resolution
        resizer = ImageResize()
        resized_image, = resizer.resize_image(cropped_image, "cover", resolutions, "lanczos")

        # Resize mask similarly. Convert to image-like tensor for resizing.
        cropped_mask_as_image = cropped_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        resized_mask_as_image, = resizer.resize_image(cropped_mask_as_image, "cover", resolutions, "lanczos")
        # Convert back to a mask (using the red channel)
        resized_mask = resized_mask_as_image[:, :, :, 0]

        # Pack context into a list of ints for output
        # Format: [x, y, width, height]
        composite_context = (context["x"], context["y"], context["width"], context["height"])

        return (resized_image, resized_mask, composite_context)


class CompositeCroppedAndFittedInpaintResult:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "inpainted_image": ("IMAGE",),
                "inpainted_mask": ("MASK",),
                "composite_context": ("COMBO[INT]",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_result"
    CATEGORY = "inpaint"

    def composite_result(self, source_image: torch.Tensor, inpainted_image: torch.Tensor, inpainted_mask: MaskBatch, composite_context: tuple[int, ...]):
        # Unpack context
        x, y, width, height = composite_context

        # The inpainted image and mask are at a diffusion resolution. Resize them back to the original crop size.
        target_size = (height, width)

        # Resize inpainted image
        inpainted_image_permuted = inpainted_image.movedim(-1, 1)
        resized_inpainted_image = F.interpolate(inpainted_image_permuted, size=target_size, mode="bilinear", align_corners=False)

        # Resize inpainted mask
        # Add channel dim: (B, H, W) -> (B, 1, H, W)
        inpainted_mask_unsqueezed = inpainted_mask.unsqueeze(1)
        resized_inpainted_mask = F.interpolate(inpainted_mask_unsqueezed, size=target_size, mode="bilinear", align_corners=False)

        # Prepare for compositing
        destination_image = source_image.clone().movedim(-1, 1)

        # Composite the resized inpainted image back onto the source image
        final_image_permuted = composite(
            destination=destination_image,
            source=resized_inpainted_image,
            x=x,
            y=y,
            mask=resized_inpainted_mask
        )

        final_image = final_image_permuted.movedim(1, -1)
        return (final_image,)


NODE_CLASS_MAPPINGS = {
    "CropAndFitInpaintToDiffusionSize": CropAndFitInpaintToDiffusionSize,
    "CompositeCroppedAndFittedInpaintResult": CompositeCroppedAndFittedInpaintResult,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAndFitInpaintToDiffusionSize": "Crop & Fit Inpaint Region",
    "CompositeCroppedAndFittedInpaintResult": "Composite Inpaint Result",
}
