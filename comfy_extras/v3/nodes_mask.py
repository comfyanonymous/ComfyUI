from __future__ import annotations

import numpy as np
import scipy.ndimage
import torch

import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import io, ui


def composite(destination, source, x, y, mask=None, multiplier=8, resize_source=False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[2], destination.shape[3]), mode="bilinear"
        )

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (
        left + source.shape[3],
        top + source.shape[2],
    )

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[2], source.shape[3]),
            mode="bilinear",
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (
        destination.shape[3] - left + min(0, x),
        destination.shape[2] - top + min(0, y),
    )

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination


class CropMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CropMask_V3",
            display_name="Crop Mask _V3",
            category="mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION),
                io.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, x, y, width, height) -> io.NodeOutput:
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        return io.NodeOutput(mask[:, y : y + height, x : x + width])


class FeatherMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FeatherMask_V3",
            display_name="Feather Mask _V3",
            category="mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("left", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("top", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("right", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("bottom", default=0, min=0, max=nodes.MAX_RESOLUTION),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, left, top, right, bottom) -> io.NodeOutput:
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

        return io.NodeOutput(output)


class GrowMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GrowMask_V3",
            display_name="Grow Mask _V3",
            category="mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("expand", default=0, min=-nodes.MAX_RESOLUTION, max=nodes.MAX_RESOLUTION),
                io.Boolean.Input("tapered_corners", default=True),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, expand, tapered_corners) -> io.NodeOutput:
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
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
        return io.NodeOutput(torch.stack(out, dim=0))


class ImageColorToMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageColorToMask_V3",
            display_name="Image Color to Mask _V3",
            category="mask",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("color", default=0, min=0, max=0xFFFFFF),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, image, color) -> io.NodeOutput:
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        temp = (
            torch.bitwise_left_shift(temp[:, :, :, 0], 16)
            + torch.bitwise_left_shift(temp[:, :, :, 1], 8)
            + temp[:, :, :, 2]
        )
        return io.NodeOutput(torch.where(temp == color, 1.0, 0).float())


class ImageCompositeMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageCompositeMasked_V3",
            display_name="Image Composite Masked _V3",
            category="image",
            inputs=[
                io.Image.Input("destination"),
                io.Image.Input("source"),
                io.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Boolean.Input("resize_source", default=False),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask=None) -> io.NodeOutput:
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
        return io.NodeOutput(output)


class ImageToMask(io.ComfyNode):
    CHANNELS = ["red", "green", "blue", "alpha"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageToMask_V3",
            display_name="Convert Image to Mask _V3",
            category="mask",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("channel", options=cls.CHANNELS),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, image, channel) -> io.NodeOutput:
        return io.NodeOutput(image[:, :, :, cls.CHANNELS.index(channel)])


class InvertMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InvertMask_V3",
            display_name="Invert Mask _V3",
            category="mask",
            inputs=[
                io.Mask.Input("mask"),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask) -> io.NodeOutput:
        return io.NodeOutput(1.0 - mask)


class LatentCompositeMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LatentCompositeMasked_V3",
            display_name="Latent Composite Masked _V3",
            category="latent",
            inputs=[
                io.Latent.Input("destination"),
                io.Latent.Input("source"),
                io.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8),
                io.Boolean.Input("resize_source", default=False),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask=None) -> io.NodeOutput:
        output = destination.copy()
        destination_samples = destination["samples"].clone()
        source_samples = source["samples"]
        output["samples"] = composite(destination_samples, source_samples, x, y, mask, 8, resize_source)
        return io.NodeOutput(output)


class MaskComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskComposite_V3",
            display_name="Mask Composite _V3",
            category="mask",
            inputs=[
                io.Mask.Input("destination"),
                io.Mask.Input("source"),
                io.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION),
                io.Combo.Input("operation", options=["multiply", "add", "subtract", "and", "or", "xor"]),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x, y, operation) -> io.NodeOutput:
        output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
        source = source.reshape((-1, source.shape[-2], source.shape[-1]))

        left, top = (
            x,
            y,
        )
        right, bottom = (
            min(left + source.shape[-1], destination.shape[-1]),
            min(top + source.shape[-2], destination.shape[-2]),
        )
        visible_width, visible_height = (
            right - left,
            bottom - top,
        )

        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = output[:, top:bottom, left:right]

        if operation == "multiply":
            output[:, top:bottom, left:right] = destination_portion * source_portion
        elif operation == "add":
            output[:, top:bottom, left:right] = destination_portion + source_portion
        elif operation == "subtract":
            output[:, top:bottom, left:right] = destination_portion - source_portion
        elif operation == "and":
            output[:, top:bottom, left:right] = torch.bitwise_and(
                destination_portion.round().bool(), source_portion.round().bool()
            ).float()
        elif operation == "or":
            output[:, top:bottom, left:right] = torch.bitwise_or(
                destination_portion.round().bool(), source_portion.round().bool()
            ).float()
        elif operation == "xor":
            output[:, top:bottom, left:right] = torch.bitwise_xor(
                destination_portion.round().bool(), source_portion.round().bool()
            ).float()

        return io.NodeOutput(torch.clamp(output, 0.0, 1.0))


class MaskPreview(io.ComfyNode):
    """Mask Preview - original implement in ComfyUI_essentials.

    https://github.com/cubiq/ComfyUI_essentials/blob/9d9f4bedfc9f0321c19faf71855e228c93bd0dc9/mask.py#L81
    Upstream requested in https://github.com/Kosinkadink/rfcs/blob/main/rfcs/0000-corenodes.md#preview-nodes
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskPreview_V3",
            display_name="Preview Mask _V3",
            category="mask",
            inputs=[
                io.Mask.Input("masks"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, masks):
        return io.NodeOutput(ui=ui.PreviewMask(masks))


class MaskToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskToImage_V3",
            display_name="Convert Mask to Image _V3",
            category="mask",
            inputs=[
                io.Mask.Input("mask"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, mask) -> io.NodeOutput:
        return io.NodeOutput(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3))


class SolidMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SolidMask_V3",
            display_name="Solid Mask _V3",
            category="mask",
            inputs=[
                io.Float.Input("value", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION),
                io.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, value, width, height) -> io.NodeOutput:
        return io.NodeOutput(torch.full((1, height, width), value, dtype=torch.float32, device="cpu"))


class ThresholdMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ThresholdMask_V3",
            display_name="Threshold Mask _V3",
            category="mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input("value", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, mask, value) -> io.NodeOutput:
        return io.NodeOutput((mask > value).float())


NODES_LIST: list[type[io.ComfyNode]] = [
    CropMask,
    FeatherMask,
    GrowMask,
    ImageColorToMask,
    ImageCompositeMasked,
    ImageToMask,
    InvertMask,
    LatentCompositeMasked,
    MaskComposite,
    MaskPreview,
    MaskToImage,
    SolidMask,
    ThresholdMask,
]
