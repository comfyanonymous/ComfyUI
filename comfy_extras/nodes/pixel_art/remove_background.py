# Mara Huldra 2023
# SPDX-License-Identifier: MIT
'''
Estimate what pixels belong to the background and perform a cut-out, using the 'rembg' models.
'''
import numpy as np
import rembg
import torch


MODELS = rembg.sessions.sessions_names


class ImageRemoveBackground:
    '''Remove background from image (adds an alpha channel)'''

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (MODELS, {
                    "default": "u2net",
                }),
                "alpha_matting": (["disabled", "enabled"], {
                    "default": "disabled",
                }),
                "am_foreground_thr": ("INT", {
                    "default": 240,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "am_background_thr": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
                "am_erode_size": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"

    CATEGORY = "image/postprocessing"

    def remove_background(self, image, model, alpha_matting, am_foreground_thr, am_background_thr, am_erode_size):
        session = rembg.new_session(model)
        results = []

        for i in image:
            i = 255. * i.cpu().numpy()
            i = np.clip(i, 0, 255).astype(np.uint8)
            i = rembg.remove(i,
                    alpha_matting=(alpha_matting == "enabled"),
                    alpha_matting_foreground_threshold=am_foreground_thr,
                    alpha_matting_background_threshold=am_background_thr,
                    alpha_matting_erode_size=am_erode_size,
                    session=session,
                    )
            results.append(i.astype(np.float32) / 255.0)

        s = torch.from_numpy(np.array(results))
        return (s,)

class ImageEstimateForegroundMask:
    '''
    Return a mask of which pixels are estimated to belong to foreground.
    Only estimates the mask, does not perform cutout like
    ImageRemoveBackground.
    '''

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (MODELS, {
                    "default": "u2net",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "estimate_background"

    CATEGORY = "image/postprocessing"

    def estimate_background(self, image, model):
        session = rembg.new_session(model)
        results = []

        for i in image:
            i = 255. * i.cpu().numpy()
            i = np.clip(i, 0, 255).astype(np.uint8)
            i = rembg.remove(i, only_mask=True, session=session)
            results.append(i.astype(np.float32) / 255.0)

        s = torch.from_numpy(np.array(results))
        print(s.shape)
        return (s,)


NODE_CLASS_MAPPINGS = {
    "ImageRemoveBackground": ImageRemoveBackground,
    "ImageEstimateForegroundMask": ImageEstimateForegroundMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRemoveBackground": "Remove Background (rembg)",
    "ImageEstimateForegroundMask": "Estimate Foreground (rembg)",
}
