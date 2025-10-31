from enum import Enum

import numpy as np
import torch
from skimage import exposure

import comfy.utils
from comfy.component_model.tensor_types import RGBImageBatch, ImageBatch, MaskBatch
from comfy.nodes.package_typing import CustomNode
from comfy_api.latest import io


def resize_mask(mask, shape):
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)


class PorterDuffMode(Enum):
    ADD = 0
    CLEAR = 1
    DARKEN = 2
    DST = 3
    DST_ATOP = 4
    DST_IN = 5
    DST_OUT = 6
    DST_OVER = 7
    LIGHTEN = 8
    MULTIPLY = 9
    OVERLAY = 10
    SCREEN = 11
    SRC = 12
    SRC_ATOP = 13
    SRC_IN = 14
    SRC_OUT = 15
    SRC_OVER = 16
    XOR = 17


def _porter_duff_composite(src_image: torch.Tensor, src_alpha: torch.Tensor, dst_image: torch.Tensor, dst_alpha: torch.Tensor, mode: PorterDuffMode):
    # premultiply alpha
    src_image = src_image * src_alpha
    dst_image = dst_image * dst_alpha

    # composite ops below assume alpha-premultiplied images
    if mode == PorterDuffMode.ADD:
        out_alpha = torch.clamp(src_alpha + dst_alpha, 0, 1)
        out_image = torch.clamp(src_image + dst_image, 0, 1)
    elif mode == PorterDuffMode.CLEAR:
        out_alpha = torch.zeros_like(dst_alpha)
        out_image = torch.zeros_like(dst_image)
    elif mode == PorterDuffMode.DARKEN:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = (1 - dst_alpha) * src_image + (1 - src_alpha) * dst_image + torch.min(src_image, dst_image)
    elif mode == PorterDuffMode.DST:
        out_alpha = dst_alpha
        out_image = dst_image
    elif mode == PorterDuffMode.DST_ATOP:
        out_alpha = src_alpha
        out_image = src_alpha * dst_image + (1 - dst_alpha) * src_image
    elif mode == PorterDuffMode.DST_IN:
        out_alpha = src_alpha * dst_alpha
        out_image = dst_image * src_alpha
    elif mode == PorterDuffMode.DST_OUT:
        out_alpha = (1 - src_alpha) * dst_alpha
        out_image = (1 - src_alpha) * dst_image
    elif mode == PorterDuffMode.DST_OVER:
        out_alpha = dst_alpha + (1 - dst_alpha) * src_alpha
        out_image = dst_image + (1 - dst_alpha) * src_image
    elif mode == PorterDuffMode.LIGHTEN:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = (1 - dst_alpha) * src_image + (1 - src_alpha) * dst_image + torch.max(src_image, dst_image)
    elif mode == PorterDuffMode.MULTIPLY:
        out_alpha = src_alpha * dst_alpha
        out_image = src_image * dst_image
    elif mode == PorterDuffMode.OVERLAY:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = torch.where(2 * dst_image < dst_alpha, 2 * src_image * dst_image,
                                src_alpha * dst_alpha - 2 * (dst_alpha - src_image) * (src_alpha - dst_image))
    elif mode == PorterDuffMode.SCREEN:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = src_image + dst_image - src_image * dst_image
    elif mode == PorterDuffMode.SRC:
        out_alpha = src_alpha
        out_image = src_image
    elif mode == PorterDuffMode.SRC_ATOP:
        out_alpha = dst_alpha
        out_image = dst_alpha * src_image + (1 - src_alpha) * dst_image
    elif mode == PorterDuffMode.SRC_IN:
        out_alpha = src_alpha * dst_alpha
        out_image = src_image * dst_alpha
    elif mode == PorterDuffMode.SRC_OUT:
        out_alpha = (1 - dst_alpha) * src_alpha
        out_image = (1 - dst_alpha) * src_image
    elif mode == PorterDuffMode.SRC_OVER:
        out_alpha = src_alpha + (1 - src_alpha) * dst_alpha
        out_image = src_image + (1 - src_alpha) * dst_image
    elif mode == PorterDuffMode.XOR:
        out_alpha = (1 - dst_alpha) * src_alpha + (1 - src_alpha) * dst_alpha
        out_image = (1 - dst_alpha) * src_image + (1 - src_alpha) * dst_image
    else:
        return None, None

    # back to non-premultiplied alpha
    out_image = torch.where(out_alpha > 1e-5, out_image / out_alpha, torch.zeros_like(out_image))
    out_image = torch.clamp(out_image, 0, 1)
    # convert alpha to mask
    out_alpha = 1 - out_alpha
    return out_image, out_alpha


class PorterDuffImageCompositeV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("IMAGE",),
                "destination": ("IMAGE",),
                "mode": ([mode.name for mode in PorterDuffMode], {"default": PorterDuffMode.DST.name}),
            },
            "optional": {
                "source_alpha": ("MASK",),
                "destination_alpha": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "composite"
    CATEGORY = "mask/compositing"

    def composite(self, source: RGBImageBatch, destination: RGBImageBatch, mode, source_alpha: MaskBatch = None, destination_alpha: MaskBatch = None) -> tuple[RGBImageBatch, MaskBatch]:
        if source_alpha is None:
            source_alpha = torch.zeros(source.shape[:3])
        if destination_alpha is None:
            destination_alpha = torch.zeros(destination.shape[:3])

        batch_size = min(len(source), len(source_alpha), len(destination), len(destination_alpha))
        out_images = []
        out_alphas = []

        for i in range(batch_size):
            src_image = source[i]
            dst_image = destination[i]

            assert src_image.shape[2] == dst_image.shape[2]  # inputs need to have same number of channels

            src_alpha = source_alpha[i].unsqueeze(2)
            dst_alpha = destination_alpha[i].unsqueeze(2)

            if dst_alpha.shape[:2] != dst_image.shape[:2]:
                upscale_input = dst_alpha.unsqueeze(0).permute(0, 3, 1, 2)
                upscale_output = comfy.utils.common_upscale(upscale_input, dst_image.shape[1], dst_image.shape[0], upscale_method='bicubic', crop='center')
                dst_alpha = upscale_output.permute(0, 2, 3, 1).squeeze(0)
            if src_image.shape != dst_image.shape:
                upscale_input = src_image.unsqueeze(0).permute(0, 3, 1, 2)
                upscale_output = comfy.utils.common_upscale(upscale_input, dst_image.shape[1], dst_image.shape[0], upscale_method='bicubic', crop='center')
                src_image = upscale_output.permute(0, 2, 3, 1).squeeze(0)
            if src_alpha.shape != dst_alpha.shape:
                upscale_input = src_alpha.unsqueeze(0).permute(0, 3, 1, 2)
                upscale_output = comfy.utils.common_upscale(upscale_input, dst_alpha.shape[1], dst_alpha.shape[0], upscale_method='bicubic', crop='center')
                src_alpha = upscale_output.permute(0, 2, 3, 1).squeeze(0)

            out_image, out_alpha = _porter_duff_composite(src_image, src_alpha, dst_image, dst_alpha, PorterDuffMode[mode])

            out_images.append(out_image)
            out_alphas.append(out_alpha.squeeze(2))

        return io.NodeOutput(torch.stack(out_images), torch.stack(out_alphas))


class PorterDuffImageCompositeV1(PorterDuffImageCompositeV2):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("IMAGE",),
                "source_alpha": ("MASK",),
                "destination": ("IMAGE",),
                "destination_alpha": ("MASK",),
                "mode": ([mode.name for mode in PorterDuffMode], {"default": PorterDuffMode.DST.name}),
            },
        }

    FUNCTION = "composite_v1"

    def composite_v1(self, source: torch.Tensor, source_alpha: torch.Tensor, destination: torch.Tensor, destination_alpha: torch.Tensor, mode) -> tuple[RGBImageBatch, MaskBatch]:
        # convert mask to alpha
        source_alpha = 1 - source_alpha
        destination_alpha = 1 - destination_alpha
        return super().composite(source, destination, mode, source_alpha, destination_alpha)


class SplitImageWithAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SplitImageWithAlpha",
            display_name="Split Image with Alpha",
            category="mask/compositing",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor) -> io.NodeOutput:
        out_images = [i[:, :, :3] for i in image]
        out_alphas = [i[:, :, 3] if i.shape[2] > 3 else torch.ones_like(i[:, :, 0]) for i in image]
        return io.NodeOutput(torch.stack(out_images), 1.0 - torch.stack(out_alphas))


class JoinImageWithAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoinImageWithAlpha",
            display_name="Join Image with Alpha",
            category="mask/compositing",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("alpha"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, alpha: torch.Tensor) -> io.NodeOutput:
        batch_size = min(len(image), len(alpha))
        out_images = []

        alpha = 1.0 - resize_mask(alpha, image.shape[1:])
        for i in range(batch_size):
            out_images.append(torch.cat((image[i][:, :, :3], alpha[i].unsqueeze(2)), dim=2))

        return io.NodeOutput(torch.stack(out_images))


class Flatten(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "background_color": ("STRING", {"default": "#FFFFFF"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_rgba_to_rgb"

    CATEGORY = "image/postprocessing"

    def convert_rgba_to_rgb(self, images: ImageBatch, background_color) -> tuple[RGBImageBatch]:
        b, h, w, c = images.shape
        if c == 3:
            return images,
        bg_color = torch.tensor(self.hex_to_rgb(background_color), dtype=torch.float32) / 255.0
        rgb = images[..., :3]
        alpha = images[..., 3:4]
        bg = bg_color.view(1, 1, 1, 3).expand(rgb.shape)
        blended = alpha * rgb + (1 - alpha) * bg

        return blended,

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


class EnhanceContrast(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["Histogram Equalization", "Adaptive Equalization", "Contrast Stretching"],),
                "clip_limit": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lower_percentile": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "upper_percentile": ("FLOAT", {"default": 98.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_contrast"

    CATEGORY = "image/adjustments"

    def enhance_contrast(self, image: torch.Tensor, method: str, clip_limit: float, lower_percentile: float, upper_percentile: float) -> tuple[RGBImageBatch]:
        assert image.dim() == 4 and image.shape[-1] == 3, "Input must be a batch of RGB images"

        image = image.cpu()

        processed_images = []
        for img in image:
            img_np = img.numpy()

            if method == "Histogram Equalization":
                enhanced = exposure.equalize_hist(img_np)
            elif method == "Adaptive Equalization":
                enhanced = exposure.equalize_adapthist(img_np, clip_limit=clip_limit)
            elif method == "Contrast Stretching":
                p_low, p_high = np.percentile(img_np, (lower_percentile, upper_percentile))
                enhanced = exposure.rescale_intensity(img_np, in_range=(p_low, p_high))
            else:
                raise ValueError(f"Unknown method: {method}")

            processed_images.append(torch.from_numpy(enhanced.astype(np.float32)))

        result = torch.stack(processed_images)

        return (result,)


class Posterize(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "levels": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 256,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "posterize"

    CATEGORY = "image/adjustments"

    def posterize(self, image: RGBImageBatch, levels: int) -> tuple[RGBImageBatch]:
        assert image.dim() == 4 and image.shape[-1] == 3, "Input must be a batch of RGB images"
        image = image.cpu()
        scale = (levels - 1) / 255.0
        quantized = torch.round(image * 255.0 * scale) / scale / 255.0
        posterized = torch.clamp(quantized, 0, 1)
        return (posterized,)


NODE_CLASS_MAPPINGS = {
    "PorterDuffImageComposite": PorterDuffImageCompositeV1,
    "PorterDuffImageCompositeV2": PorterDuffImageCompositeV2,
    "SplitImageWithAlpha": SplitImageWithAlpha,
    "JoinImageWithAlpha": JoinImageWithAlpha,
    "EnhanceContrast": EnhanceContrast,
    "Posterize": Posterize,
    "Flatten": Flatten
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PorterDuffImageComposite": "Porter-Duff Image Composite (V1)",
    "PorterDuffImageCompositeV2": "Image Composite",
    "SplitImageWithAlpha": "Split Image with Alpha",
    "JoinImageWithAlpha": "Join Image with Alpha",
}
