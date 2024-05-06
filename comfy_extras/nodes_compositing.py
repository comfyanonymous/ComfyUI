import numpy as np
import torch
import comfy.utils
from enum import Enum

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


def porter_duff_composite(src_image: torch.Tensor, src_alpha: torch.Tensor, dst_image: torch.Tensor, dst_alpha: torch.Tensor, mode: PorterDuffMode):
    if mode == PorterDuffMode.ADD:
        out_alpha = torch.clamp(src_alpha + dst_alpha, 0, 1)
        out_image = torch.clamp(src_image + dst_image, 0, 1)
    elif mode == PorterDuffMode.CLEAR:
        out_alpha = torch.zeros_like(dst_alpha)
        out_image = torch.zeros_like(dst_image)
    elif mode == PorterDuffMode.DARKEN:
        out_alpha = src_alpha + dst_alpha  - src_alpha * dst_alpha
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
        out_alpha = None
        out_image = None
    return out_image, out_alpha


class PorterDuffImageComposite:
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

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "composite"
    CATEGORY = "mask/compositing"

    def composite(self, source: torch.Tensor, source_alpha: torch.Tensor, destination: torch.Tensor, destination_alpha: torch.Tensor, mode):
        batch_size = min(len(source), len(destination))
        if batch_size == 1:
            if len(source) != 1:
                batch_size = len(source)
            elif len(destination) != 1:
                batch_size = len(destination)
        out_images = []
        out_alphas = []

        if batch_size != 1:
            if len(source) == 1:
                source = source.repeat(batch_size, 1, 1, 1)
            if len(destination) == 1:
                destination = destination.repeat(batch_size, 1, 1, 1)
            if len(source_alpha) == 1:
                source_alpha = source_alpha.repeat(batch_size, 1, 1)
            if len(destination_alpha) == 1:
                destination_alpha = destination_alpha.repeat(batch_size, 1, 1)
        for i in range(batch_size):
            src_image = source[i]
            dst_image = destination[i]

            assert src_image.shape[2] == dst_image.shape[2] # inputs need to have same number of channels

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

            out_image, out_alpha = porter_duff_composite(src_image, src_alpha, dst_image, dst_alpha, PorterDuffMode[mode])

            out_images.append(out_image)
            out_alphas.append(out_alpha.squeeze(2))

        result = (torch.stack(out_images), torch.stack(out_alphas))
        return result


class SplitImageWithAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "image": ("IMAGE",),
                }
        }

    CATEGORY = "mask/compositing"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "split_image_with_alpha"

    def split_image_with_alpha(self, image: torch.Tensor):
        out_images = [i[:,:,:3] for i in image]
        out_alphas = [i[:,:,3] if i.shape[2] > 3 else torch.ones_like(i[:,:,0]) for i in image]
        result = (torch.stack(out_images), 1.0 - torch.stack(out_alphas))
        return result


class JoinImageWithAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "image": ("IMAGE",),
                    "alpha": ("MASK",),
                }
        }

    CATEGORY = "mask/compositing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "join_image_with_alpha"

    def join_image_with_alpha(self, image: torch.Tensor, alpha: torch.Tensor):
        batch_size = min(len(image), len(alpha))
        out_images = []

        if len(alpha) == 1 and batch_size != 1:
            alpha = alpha.repeat(batch_size, 1, 1, 1)
        alpha = 1.0 - resize_mask(alpha, image.shape[1:])
        for i in range(batch_size):
           out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

        result = (torch.stack(out_images),)
        return result


NODE_CLASS_MAPPINGS = {
    "PorterDuffImageComposite": PorterDuffImageComposite,
    "SplitImageWithAlpha": SplitImageWithAlpha,
    "JoinImageWithAlpha": JoinImageWithAlpha,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "PorterDuffImageComposite": "Porter-Duff Image Composite",
    "SplitImageWithAlpha": "Split Image with Alpha",
    "JoinImageWithAlpha": "Join Image with Alpha",
}
