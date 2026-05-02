import numpy as np
import scipy.ndimage
import torch
import comfy.utils
import comfy.model_management
import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, UI

import nodes

def composite(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[-1], top + source.shape[-2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[-2], source.shape[-1]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[-1] - left + min(0, x), destination.shape[-2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)

    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[..., :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[..., top:bottom, left:right]

    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination

def video_latent_composite(destination, source, x, y, mask=None, multiplier=8, resize_source=False):
    # destination/source shape: [B, C, F, H, W]
    source = source.to(destination.device)

    # 1. Spatial Resizing for Source
    if resize_source:
        # size=(Frames, Height, Width). We keep source's F, but match destination's H, W
        target_size = (source.shape[2], destination.shape[3], destination.shape[4])
        source = torch.nn.functional.interpolate(
            source, 
            size=target_size, 
            mode="trilinear", 
            align_corners=False
        )

    # 2. Coordinate Scaling
    x_latent = x // multiplier
    y_latent = y // multiplier

    # 3. Mask Processing (Input: [F, H, W])
    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        
        # Convert [F, H, W] -> [1, 1, F, H, W]
        # This allows it to broadcast across any Batch or Channel in 'source'
        mask = mask.unsqueeze(0).unsqueeze(0)
            
        # Resize mask spatially, preserving its frame count
        # size=(mask_frames, source_height, source_width)
        mask_target_size = (mask.shape[2], source.shape[3], source.shape[4])
        mask = torch.nn.functional.interpolate(
            mask, 
            size=mask_target_size, 
            mode="trilinear", 
            align_corners=False
        )

    # 4. Dimension Calculations for Spatial Slicing
    dst_h, dst_w = destination.shape[3], destination.shape[4]
    src_h, src_w = source.shape[3], source.shape[4]

    # Calculate visible overlap region
    visible_h = max(0, min(y_latent + src_h, dst_h) - max(0, y_latent))
    visible_w = max(0, min(x_latent + src_w, dst_w) - max(0, x_latent))

    if visible_h <= 0 or visible_w <= 0:
        return destination

    # Determine slicing offsets
    src_top = max(0, -y_latent)
    src_left = max(0, -x_latent)
    dst_top = max(0, y_latent)
    dst_left = max(0, x_latent)

    # 5. Slicing and Blending
    # destination/source/mask are now all 5D: [B, C, F, H, W]
    # We slice only the H and W dimensions (indices 3 and 4)
    m = mask[:, :, :, src_top:src_top+visible_h, src_left:src_left+visible_w]
    s = source[:, :, :, src_top:src_top+visible_h, src_left:src_left+visible_w]
    d = destination[:, :, :, dst_top:dst_top+visible_h, dst_left:dst_left+visible_w]

    # Combine using the mask
    destination[:, :, :, dst_top:dst_top+visible_h, dst_left:dst_left+visible_w] = (m * s) + ((1.0 - m) * d)
    
    return destination

def convert_rgb_mask_to_latent_mask(
    mask: torch.Tensor, 
    k: int, 
    spatial_downsample_h: int,
    spatial_downsample_w: int
) -> torch.Tensor: 
    """ 
    Converts [T, H, W] mask to [T_latent, H_latent, W_latent].
    Handles non-square spatial downsampling.
    """ 
    # 1. Temporal Sampling
    # Select first frame and every k-th frame thereafter
    mask0 = mask[0:1]           
    mask1 = mask[1::k]          
    sampled = torch.cat([mask0, mask1], dim=0)  # [T_latent, H, W] 

    # 2. Prepare for Spatial Interpolation
    # Shape: [Batch=1, Channels=1, Depth=T_latent, Height=H, Width=W]
    sampled = sampled.unsqueeze(0).unsqueeze(0)

    # 3. Calculate New Spatial Dimensions
    h_latent = sampled.shape[-2] // spatial_downsample_h
    w_latent = sampled.shape[-1] // spatial_downsample_w
    
    # 4. Interpolate
    # We maintain the temporal count (sampled.shape[2]) 
    # but resize H and W independently
    pooled = torch.nn.functional.interpolate(
        sampled, 
        size=(sampled.shape[2], h_latent, w_latent), 
        mode="nearest"
    ) 

    # 5. Return to [T_latent, H_latent, W_latent]
    return pooled.squeeze(0).squeeze(0)

class LatentCompositeMasked(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LatentCompositeMasked",
            search_aliases=["overlay latent", "layer latent", "paste latent", "inpaint latent"],
            category="latent",
            inputs=[
                IO.Latent.Input("destination"),
                IO.Latent.Input("source"),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                IO.Boolean.Input("resize_source", default=False),
                IO.Mask.Input("mask", optional=True),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask = None) -> IO.NodeOutput:
        output = destination.copy()
        destination = destination["samples"].clone()
        source = source["samples"]
        output["samples"] = composite(destination, source, x, y, mask, 8, resize_source)
        return IO.NodeOutput(output)

    composite = execute  # TODO: remove

class VideoLatentCompositeMasked(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VideoLatentCompositeMasked",
            search_aliases=["overlay latent", "layer latent", "paste latent", "inpaint latent"],
            category="latent",
            inputs=[
                IO.Latent.Input("destination"),
                IO.Latent.Input("source"),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                IO.Boolean.Input("resize_source", default=False),
                IO.Mask.Input("mask", optional=True),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask=None) -> IO.NodeOutput:
        output = destination.copy()
        # Ensure we work on a copy of the samples to remain non-destructive
        dst_samples = destination["samples"].clone()
        src_samples = source["samples"]
        
        output["samples"] = video_latent_composite(
            dst_samples, 
            src_samples, 
            x, y, 
            mask, 
            multiplier=8, 
            resize_source=resize_source
        )
        return IO.NodeOutput(output)

class ImageCompositeMasked(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCompositeMasked",
            search_aliases=["paste image", "overlay", "layer"],
            category="image",
            inputs=[
                IO.Image.Input("destination"),
                IO.Image.Input("source"),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Boolean.Input("resize_source", default=False),
                IO.Mask.Input("mask", optional=True),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask = None) -> IO.NodeOutput:
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
        return IO.NodeOutput(output)

    composite = execute  # TODO: remove


class MaskToImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MaskToImage",
            search_aliases=["convert mask"],
            display_name="Convert Mask to Image",
            category="mask",
            inputs=[
                IO.Mask.Input("mask"),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, mask) -> IO.NodeOutput:
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return IO.NodeOutput(result)

    mask_to_image = execute  # TODO: remove


class ImageToMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageToMask",
            search_aliases=["extract channel", "channel to mask"],
            display_name="Convert Image to Mask",
            category="mask",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("channel", options=["red", "green", "blue", "alpha"]),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, image, channel) -> IO.NodeOutput:
        channels = ["red", "green", "blue", "alpha"]
        mask = image[:, :, :, channels.index(channel)]
        return IO.NodeOutput(mask)

    image_to_mask = execute  # TODO: remove


class ImageColorToMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageColorToMask",
            search_aliases=["color keying", "chroma key"],
            category="mask",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("color", default=0, min=0, max=0xFFFFFF, step=1, display_mode=IO.NumberDisplay.number),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, image, color) -> IO.NodeOutput:
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        temp = torch.bitwise_left_shift(temp[:,:,:,0], 16) + torch.bitwise_left_shift(temp[:,:,:,1], 8) + temp[:,:,:,2]
        mask = torch.where(temp == color, 1.0, 0).float()
        return IO.NodeOutput(mask)

    image_to_mask = execute  # TODO: remove


class SolidMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SolidMask",
            category="mask",
            inputs=[
                IO.Float.Input("value", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, value, width, height) -> IO.NodeOutput:
        out = torch.full((1, height, width), value, dtype=torch.float32, device=comfy.model_management.intermediate_device())
        return IO.NodeOutput(out)

    solid = execute  # TODO: remove


class InvertMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="InvertMask",
            search_aliases=["reverse mask", "flip mask"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask"),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask) -> IO.NodeOutput:
        out = 1.0 - mask
        return IO.NodeOutput(out)

    invert = execute  # TODO: remove


class CropMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CropMask",
            search_aliases=["cut mask", "extract mask region", "mask slice"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask"),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, x, y, width, height) -> IO.NodeOutput:
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = mask[:, y:y + height, x:x + width]
        return IO.NodeOutput(out)

    crop = execute  # TODO: remove


class MaskComposite(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MaskComposite",
            search_aliases=["combine masks", "blend masks", "layer masks"],
            category="mask",
            inputs=[
                IO.Mask.Input("destination"),
                IO.Mask.Input("source"),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Combo.Input("operation", options=["multiply", "add", "subtract", "and", "or", "xor"]),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x, y, operation) -> IO.NodeOutput:
        output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
        source = source.reshape((-1, source.shape[-2], source.shape[-1]))
        source = source.to(output.device)

        left, top = (x, y,)
        right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
        visible_width, visible_height = (right - left, bottom - top,)

        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = output[:, top:bottom, left:right]

        if operation == "multiply":
            output[:, top:bottom, left:right] = destination_portion * source_portion
        elif operation == "add":
            output[:, top:bottom, left:right] = destination_portion + source_portion
        elif operation == "subtract":
            output[:, top:bottom, left:right] = destination_portion - source_portion
        elif operation == "and":
            output[:, top:bottom, left:right] = torch.bitwise_and(destination_portion.round().bool(), source_portion.round().bool()).float()
        elif operation == "or":
            output[:, top:bottom, left:right] = torch.bitwise_or(destination_portion.round().bool(), source_portion.round().bool()).float()
        elif operation == "xor":
            output[:, top:bottom, left:right] = torch.bitwise_xor(destination_portion.round().bool(), source_portion.round().bool()).float()

        output = torch.clamp(output, 0.0, 1.0)

        return IO.NodeOutput(output)

    combine = execute  # TODO: remove


class FeatherMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FeatherMask",
            search_aliases=["soft edge mask", "blur mask edges", "gradient mask edge"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask"),
                IO.Int.Input("left", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("top", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("right", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("bottom", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, left, top, right, bottom) -> IO.NodeOutput:
        output = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()

        left = min(left, output.shape[-1])
        right = min(right, output.shape[-1])
        top = min(top, output.shape[-2])
        bottom = min(bottom, output.shape[-2])

        for x in range(left):
            feather_rate = (x + 1.0) / left
            output[:, :, x] *= feather_rate

        for x in range(right):
            feather_rate = (x + 1) / right
            output[:, :, -x] *= feather_rate

        for y in range(top):
            feather_rate = (y + 1) / top
            output[:, y, :] *= feather_rate

        for y in range(bottom):
            feather_rate = (y + 1) / bottom
            output[:, -y, :] *= feather_rate

        return IO.NodeOutput(output)

    feather = execute  # TODO: remove


class GrowMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrowMask",
            search_aliases=["expand mask", "shrink mask"],
            display_name="Grow Mask",
            category="mask",
            inputs=[
                IO.Mask.Input("mask"),
                IO.Int.Input("expand", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION, step=1),
                IO.Boolean.Input("tapered_corners", default=True, advanced=True),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, expand, tapered_corners) -> IO.NodeOutput:
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return IO.NodeOutput(torch.stack(out, dim=0))

    expand_mask = execute  # TODO: remove


class ThresholdMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ThresholdMask",
            search_aliases=["binary mask"],
            category="mask",
            inputs=[
                IO.Mask.Input("mask"),
                IO.Float.Input("value", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, value) -> IO.NodeOutput:
        mask = (mask > value).float()
        return IO.NodeOutput(mask)

    image_to_mask = execute  # TODO: remove

class RGBMaskToLatentMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RGBMasktoLatentMask",
            search_aliases=["rgb mask to latent mask", "rgb mask", "latent mask"],
            description="Helpful for applying masks to video latents if the VAE uses spatial downsampling.",
            category="latent",
            inputs=[
                IO.Mask.Input("mask", optional=False),
                IO.Vae.Input("vae", optional=False),
            ],
            outputs=[IO.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, vae) -> IO.NodeOutput:
        # Ensure we work on a copy of the mask to remain non-destructive
        mask_copy = mask.clone()
        downscale_ratio = vae.downscale_ratio
        k = (mask.shape[0] - 1) // (downscale_ratio[0](mask.shape[0]) - 1) if (downscale_ratio[0](mask.shape[0]) - 1) > 1 else 1
        return IO.NodeOutput(convert_rgb_mask_to_latent_mask(mask_copy, k, spatial_downsample_h = downscale_ratio[1], spatial_downsample_w = downscale_ratio[2]))

# Mask Preview - original implement from
# https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
# upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
class MaskPreview(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MaskPreview",
            search_aliases=["show mask", "view mask", "inspect mask", "debug mask"],
            display_name="Preview Mask",
            category="mask",
            description="Saves the input images to your ComfyUI output directory.",
            inputs=[
                IO.Mask.Input("mask"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, mask, filename_prefix="ComfyUI") -> IO.NodeOutput:
        return IO.NodeOutput(ui=UI.PreviewMask(mask))


class MaskExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            LatentCompositeMasked,
            VideoLatentCompositeMasked,
            ImageCompositeMasked,
            MaskToImage,
            ImageToMask,
            ImageColorToMask,
            SolidMask,
            InvertMask,
            CropMask,
            MaskComposite,
            FeatherMask,
            GrowMask,
            ThresholdMask,
            RGBMaskToLatentMask,
            MaskPreview,
        ]


async def comfy_entrypoint() -> MaskExtension:
    return MaskExtension()
