# Mara Huldra 2023
# SPDX-License-Identifier: MIT
'''
Extra mask operations.
'''
import numpy as np
import rembg
import torch


class BinarizeMask:
    '''Binarize (threshold) a mask.'''

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("INT", {
                    "default": 250,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "binarize"

    CATEGORY = "mask"

    def binarize(self, mask, threshold):
        t = torch.Tensor([threshold / 255.])
        s = (mask >= t).float()
        return (s,)


class ImageCutout:
    '''Perform basic image cutout (adds alpha channel from mask).'''

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cutout"

    CATEGORY = "image/postprocessing"

    def cutout(self, image, mask):
        # XXX check compatible dimensions.
        o = np.zeros((image.shape[0], image.shape[1], image.shape[2], 4))
        o[:, :, :, 0:3] = image.cpu().numpy()
        o[:, :, :, 3] = mask.cpu().numpy()
        return (torch.from_numpy(o),)


NODE_CLASS_MAPPINGS = {
    "BinarizeMask": BinarizeMask,
    "ImageCutout": ImageCutout,
}

