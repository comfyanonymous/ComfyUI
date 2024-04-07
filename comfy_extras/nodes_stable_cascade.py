"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import nodes
import comfy.utils


class StableCascade_EmptyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024, "min": 256, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 256, "max": nodes.MAX_RESOLUTION, "step": 8}),
            "compression": ("INT", {"default": 42, "min": 4, "max": 128, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
        }}
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "latent/stable_cascade"

    def generate(self, width, height, compression, batch_size=1):
        c_latent = torch.zeros([batch_size, 16, height // compression, width // compression])
        b_latent = torch.zeros([batch_size, 4, height // 4, width // 4])
        return ({
            "samples": c_latent,
        }, {
            "samples": b_latent,
        })

class StableCascade_StageC_VAEEncode:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE", ),
            "compression": ("INT", {"default": 42, "min": 4, "max": 128, "step": 1}),
        }}
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "latent/stable_cascade"

    def generate(self, image, vae, compression):
        width = image.shape[-2]
        height = image.shape[-3]
        out_width = (width // compression) * vae.downscale_ratio
        out_height = (height // compression) * vae.downscale_ratio

        s = comfy.utils.common_upscale(image.movedim(-1,1), out_width, out_height, "bicubic", "center").movedim(1,-1)

        c_latent = vae.encode(s[:,:,:,:3])
        b_latent = torch.zeros([c_latent.shape[0], 4, (height // 8) * 2, (width // 8) * 2])
        return ({
            "samples": c_latent,
        }, {
            "samples": b_latent,
        })

class StableCascade_StageB_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "conditioning": ("CONDITIONING",),
                              "stage_c": ("LATENT",),
                             }}
    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "set_prior"

    CATEGORY = "conditioning/stable_cascade"

    def set_prior(self, conditioning, stage_c):
        c = []
        for t in conditioning:
            d = t[1].copy()
            d['stable_cascade_prior'] = stage_c['samples']
            n = [t[0], d]
            c.append(n)
        return (c, )

class StableCascade_SuperResolutionControlnet:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE", ),
        }}
    RETURN_TYPES = ("IMAGE", "LATENT", "LATENT")
    RETURN_NAMES = ("controlnet_input", "stage_c", "stage_b")
    FUNCTION = "generate"

    CATEGORY = "_for_testing/stable_cascade"

    def generate(self, image, vae):
        width = image.shape[-2]
        height = image.shape[-3]
        batch_size = image.shape[0]
        controlnet_input = vae.encode(image[:,:,:,:3]).movedim(1, -1)

        c_latent = torch.zeros([batch_size, 16, height // 16, width // 16])
        b_latent = torch.zeros([batch_size, 4, height // 2, width // 2])
        return (controlnet_input, {
            "samples": c_latent,
        }, {
            "samples": b_latent,
        })

NODE_CLASS_MAPPINGS = {
    "StableCascade_EmptyLatentImage": StableCascade_EmptyLatentImage,
    "StableCascade_StageB_Conditioning": StableCascade_StageB_Conditioning,
    "StableCascade_StageC_VAEEncode": StableCascade_StageC_VAEEncode,
    "StableCascade_SuperResolutionControlnet": StableCascade_SuperResolutionControlnet,
}
