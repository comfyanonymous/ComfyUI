# Mara Huldra 2023
# SPDX-License-Identifier: MIT
'''
Patches the SD model and VAE to make it possible to generate seamlessly tilable
graphics. Horizontal and vertical direction are configurable separately.
'''
from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


def flatten_modules(m):
    '''Return submodules of module m in flattened form.'''
    yield m
    for c in m.children():
        yield from flatten_modules(c)

# from: https://github.com/Astropulse/stable-diffusion-aseprite/blob/main/scripts/image_server.py
def make_seamless_xy(model, x, y):
    for layer in flatten_modules(model):
        if type(layer) == torch.nn.Conv2d:
            layer.padding_modeX = 'circular' if x else 'constant'
            layer.padding_modeY = 'circular' if y else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            layer._conv_forward = __replacementConv2DConvForward.__get__(layer, torch.nn.Conv2d)

def restore_conv2d_methods(model):
    for layer in flatten_modules(model):
        if type(layer) == torch.nn.Conv2d:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)

def __replacementConv2DConvForward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


class MakeModelTileable:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "tile_x": (["disabled", "enabled"], { "default": "disabled", }),
                "tile_y": (["disabled", "enabled"], { "default": "disabled", }),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "patch_models"

    CATEGORY = "advanced/patchers"

    def patch_models(self, model, vae, tile_x, tile_y):
        tile_x = (tile_x == 'enabled')
        tile_y = (tile_y == 'enabled')
        # XXX ideally, we'd return a clone of the model, not patch the model itself
        #model = model.clone()
        #vae = vae.???()

        restore_conv2d_methods(model.model)
        restore_conv2d_methods(vae.first_stage_model)
        make_seamless_xy(model.model, tile_x, tile_y)
        make_seamless_xy(vae.first_stage_model, tile_x, tile_y)
        return (model, vae)


NODE_CLASS_MAPPINGS = {
    "MakeModelTileable": MakeModelTileable,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MakeModelTileable": "Patch model tileability"
}
